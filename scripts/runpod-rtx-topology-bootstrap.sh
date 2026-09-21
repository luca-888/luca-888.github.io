#!/usr/bin/env bash
# Prepare on CPU first, upload the resulting toolkit, then run once on one eight-GPU Pod.
# This bounds local processes only. The caller must separately arrange Pod stop/delete.
# CPU image: bash runpod-rtx-topology-bootstrap.sh prepare --output /tmp/toolkit-export
# GPU Pod:   bash runpod-rtx-topology-bootstrap.sh run --output /workspace/topology-run \
#              --toolkit /workspace/rtx-topology-toolkit.tar.gz --toolkit-sha256 <verified digest>
set -euo pipefail
export RTX_TOPOLOGY_BOOTSTRAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$@" <<'PY'
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

PINS = {
    "nvbandwidth": "1aa9e818d0728c25c87e121d692d829db2239720",
    "nccl-tests": "2535da805b34e96d1dc08be66289be1a6d57f5ad",
}
NCCL_VERSION = "2.27.7"
NCCL_VERSION_CODE = 22707
EXPECTED_CUDA = (12, 8)
BINARIES = ("nvbandwidth", "all_reduce_perf", "all_gather_perf")
SYSTEM_TOOLS = ("lspci", "numactl", "lscpu")
HOST_LIBRARIES = {"libcuda.so.1", "libnvidia-ml.so.1", "libc.so.6", "libm.so.6", "libmvec.so.1",
                  "libdl.so.2", "libpthread.so.0", "librt.so.1", "libresolv.so.2", "libutil.so.1"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


parser = argparse.ArgumentParser(description="Prepare a pinned CPU-built toolkit, or execute one bounded eight-GPU topology suite. No cloud API calls.")
parser.add_argument("mode", choices=("prepare", "run"), help="prepare exports existing CPU-built tools; run never installs or compiles")
parser.add_argument("--output", required=True, help="new directory for bootstrap logs, metadata, and results/")
parser.add_argument("--toolkit", help="prepared .tar.gz; required for run, generated inside output for prepare")
parser.add_argument("--prebuilt-root", default="/opt", help="CPU image root containing nvbandwidth/, nccl-tests/, and nccl-wheel/")
parser.add_argument("--toolkit-sha256", help="required for run; copy the digest emitted by the trusted prepare run")
parser.add_argument("--suite", default=str(Path(os.environ["RTX_TOPOLOGY_BOOTSTRAP_DIR"]) / "rtx-topology-suite.py"))
parser.add_argument("--work-seconds", type=int, default=660, help="suite budget; no automatic reduction")
parser.add_argument("--total-seconds", type=int, help="default: prepare 180, run 780; process bound, not a Pod shutdown timer")
args = parser.parse_args()
if args.total_seconds is None:
    args.total_seconds = {"prepare": 180, "run": 780}[args.mode]
if not 60 <= args.work_seconds <= 660:
    parser.error("work budget must be 60..660 seconds")
if not 60 <= args.total_seconds <= 900:
    parser.error("total budget must be 60..900 seconds")
if args.mode == "run" and (not args.toolkit or not re.fullmatch(r"[0-9a-fA-F]{64}", args.toolkit_sha256 or "")):
    parser.error("run requires --toolkit and its --toolkit-sha256; it never builds missing tools")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    parser.error("execute only inside Linux x86_64; local macOS preparation is limited to syntax/self-checks")
if sys.version_info < (3, 10):
    parser.error("Python >= 3.10 is required")
suite = Path(args.suite).resolve()
if not suite.is_file():
    parser.error("suite script does not exist")
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
deadline = started + args.total_seconds
active = None
state = {"schema_version": 1, "mode": args.mode, "status": "running", "steps": [],
         "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "tool_commits": PINS, "nccl_version": NCCL_VERSION, "cuda_architecture": "sm_120",
         "total_seconds": args.total_seconds,
         "work_seconds": args.work_seconds, "suite_sha256": sha256(suite),
         "notes": ["No cloud provisioning, credentials, automatic retry, or model loading.",
                   "An independent control-plane guard must stop or delete the Pod.",
                   "PyTorch is not required or imported; an official CUDA 12.8+ PyTorch image can supply the base OS/CUDA/SSH."]}


def checkpoint():
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save_json(output / "bootstrap-metadata.json", state)


def kill_active():
    if active and active.poll() is None:
        try:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def interrupted(signum, _frame):
    kill_active()
    raise TimeoutError("bootstrap global deadline" if signum == signal.SIGALRM else f"bootstrap interrupted by signal {signum}")


def run(name, command, seconds=60, env=None):
    global active
    remaining = deadline - 12 - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"no remaining budget for {name}")
    record = {"name": name, "command": [str(value) for value in command], "status": "running",
              "timeout_seconds": min(seconds, remaining), "stdout": f"{name}.stdout.txt", "stderr": f"{name}.stderr.txt"}
    state["steps"].append(record)
    checkpoint()
    began = time.monotonic()
    try:
        with (output / record["stdout"]).open("w") as stdout, (output / record["stderr"]).open("w") as stderr:
            active = subprocess.Popen(record["command"], stdout=stdout, stderr=stderr, env={**os.environ, **(env or {})}, start_new_session=True)
            try:
                code = active.wait(timeout=record["timeout_seconds"])
            except subprocess.TimeoutExpired:
                kill_active()
                record["status"] = "timeout"
                raise TimeoutError(f"{name} exceeded its stage/global budget")
            record["returncode"] = code
            record["status"] = "ok" if code == 0 else "failed"
            if code:
                raise RuntimeError(f"{name} exited {code}; inspect saved stdout/stderr")
    finally:
        kill_active()
        active = None
        record["elapsed_seconds"] = round(time.monotonic() - began, 3)
        checkpoint()
        print(json.dumps({"event": "bootstrap_step", "step": name, "status": record["status"], "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
    return read(output / record["stdout"]) or ""


def validate_cuda():
    version_text = run("cuda-toolkit", ["nvcc", "--version"], seconds=10)
    match = re.search(r"release (\d+)\.(\d+)", version_text)
    if not match or tuple(map(int, match.groups())) < EXPECTED_CUDA:
        raise RuntimeError("CUDA Toolkit >= 12.8 with sm_120 support required; do not install/replace CUDA on the eight-GPU Pod")
    architectures = run("cuda-supported-architectures", ["nvcc", "--list-gpu-arch"], seconds=10)
    if "compute_120" not in architectures:
        raise RuntimeError("nvcc cannot compile compute_120")
    state["cuda_toolkit"] = version_text
    state["os_release"] = read("/etc/os-release")


def prepare():
    # The existing Modal build_image compiles these pins without a GPU.
    # Export its artifacts instead of repeating the build on an eight-GPU Pod.
    if list(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("prepare is CPU-only; visible NVIDIA device files found")
    validate_cuda()
    prebuilt = Path(args.prebuilt_root).resolve()
    for name, commit in PINS.items():
        observed = run(f"{name}-commit", ["git", "-C", prebuilt / name, "rev-parse", "HEAD"], seconds=5)
        if observed != commit:
            raise RuntimeError(f"{name} source commit mismatch")
    nccl_root = prebuilt / "nccl-wheel/nvidia/nccl"
    header = read(nccl_root / "include/nccl.h") or ""
    versions = [re.search(r"#define\s+NCCL_" + key + r"\s+(\d+)", header) for key in ("MAJOR", "MINOR", "PATCH")]
    if not all(versions) or ".".join(match[1] for match in versions) != NCCL_VERSION:
        raise RuntimeError("prebuilt NCCL header does not match 2.27.7")
    toolkit = output / "toolkit"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "lib").mkdir()
    (toolkit / "libexec").mkdir()
    (toolkit / "share").mkdir()
    for filename in BINARIES:
        source = prebuilt / ("nvbandwidth/build/nvbandwidth" if filename == "nvbandwidth" else f"nccl-tests/build/{filename}")
        shutil.copy2(source, toolkit / "bin" / filename)
        elf = run("cuda-elf-" + filename, ["cuobjdump", "--list-elf", source], seconds=10)
        if "sm_120" not in elf:
            raise RuntimeError(f"prebuilt {filename} does not contain sm_120 CUDA code")
    shutil.copy2(nccl_root / "lib/libnccl.so.2", toolkit / "lib/libnccl.so.2")
    for name in SYSTEM_TOOLS:
        source = shutil.which(name)
        if source is None:
            raise RuntimeError(f"CPU preparation image missing required system tool {name}")
        shutil.copy2(source, toolkit / ("libexec/lspci" if name == "lspci" else "bin/" + name))
    ids_source = next((Path(name) for name in ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids.gz")
                       if Path(name).is_file()), None)
    if ids_source is None:
        raise RuntimeError("CPU image has no PCI IDs database to export")
    if ids_source.suffix == ".gz":
        with gzip.open(ids_source, "rb") as source:
            (toolkit / "share/pci.ids").write_bytes(source.read())
    else:
        shutil.copy2(ids_source, toolkit / "share/pci.ids")
    lspci_wrapper = toolkit / "bin/lspci"
    lspci_wrapper.write_text('#!/bin/sh\nbase=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\nif [ "$#" -eq 1 ] && [ "$1" = "--version" ]; then exec "$base/libexec/lspci" "$@"; fi\nexec "$base/libexec/lspci" -i "$base/share/pci.ids" "$@"\n')
    lspci_wrapper.chmod(0o755)
    # ldd reports transitive dependencies too. Bundle non-glibc dependencies to avoid
    # assuming the Runpod template ships Boost, libpci, libnuma or matching libstdc++.
    external_libraries = set()
    dependencies = {}
    executables = [toolkit / "bin" / name for name in (*BINARIES, "numactl", "lscpu")]
    executables += [toolkit / "libexec/lspci", toolkit / "lib/libnccl.so.2"]
    for binary in executables:
        linked = run("build-libraries-" + binary.name, ["ldd", binary], seconds=10)
        for line in linked.splitlines():
            match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
            if not match:
                continue
            soname, source = match.groups()
            if soname in HOST_LIBRARIES:
                external_libraries.add(soname)
                continue
            if source == "not" or not Path(source).is_file() or "/stubs/" in source:
                raise RuntimeError(f"Unresolved non-host dependency {soname}: {source}")
            target = toolkit / "lib" / soname
            if target.exists() and sha256(target) != sha256(Path(source)):
                raise RuntimeError(f"Conflicting runtime library {soname}")
            if not target.exists():
                shutil.copy2(source, target)
            dependencies[soname] = source
    env = {"LD_LIBRARY_PATH": str(toolkit / "lib") + ":/usr/local/cuda/lib64",
           "PATH": str(toolkit / "bin") + ":" + os.environ.get("PATH", "")}
    run("packed-lspci", [toolkit / "bin/lspci", "--version"], seconds=10, env=env)
    run("packed-lscpu", [toolkit / "bin/lscpu", "--version"], seconds=10, env=env)
    run("packed-numactl", [toolkit / "bin/numactl", "--hardware"], seconds=10, env=env)
    version = run("packed-nccl-version", [sys.executable, "-c",
                  "import ctypes; lib=ctypes.CDLL('libnccl.so.2'); v=ctypes.c_int(); "
                  "rc=lib.ncclGetVersion(ctypes.byref(v)); print(v.value); raise SystemExit(rc)"], seconds=10, env=env)
    if version != str(NCCL_VERSION_CODE):
        raise RuntimeError("Packed runtime NCCL version differs from 2.27.7")
    # Only glibc/loader and driver libraries are supplied by the target host.
    manifest = {"schema_version": 2, "tool_commits": PINS, "nccl_version": NCCL_VERSION, "cuda_architecture": "sm_120",
                "platform": "linux-x86_64", "build_os_release": state["os_release"], "cuda_toolkit": state["cuda_toolkit"],
                "origin": "existing CPU-built artifacts; no rebuild", "system_tools": SYSTEM_TOOLS,
                "bundled_dependencies": dependencies, "host_libraries": sorted(external_libraries), "files": {
                    str(path.relative_to(toolkit)): sha256(path) for path in toolkit.rglob("*") if path.is_file()}}
    save_json(toolkit / "manifest.json", manifest)
    archive = output / "rtx-topology-toolkit.tar.gz"
    run("package-toolkit", ["tar", "-czf", archive, "-C", toolkit, "manifest.json", "bin", "lib", "libexec", "share"], seconds=60)
    state["toolkit_sha256"] = sha256(archive)
    state["toolkit_archive"] = archive.name
    state["toolkit_manifest"] = manifest
    print(json.dumps({"event": "toolkit_prepared", "archive": str(archive), "sha256": state["toolkit_sha256"]}), flush=True)
    return archive, state["toolkit_sha256"]


def unpack(archive, expected_sha):
    if sha256(archive) != expected_sha.lower():
        raise RuntimeError("toolkit archive sha256 mismatch")
    toolkit = output / "runtime-toolkit"
    toolkit.mkdir()
    with tarfile.open(archive, "r:gz") as packed:
        members = packed.getmembers()
        if sum(item.size for item in members) > 2 * 1024**3:
            raise RuntimeError("toolkit exceeds planned 2 GiB extracted size")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unexpected archive member {member.name}; symlinks and special files are disallowed")
        packed.extractall(toolkit)
    manifest = json.loads((toolkit / "manifest.json").read_text())
    if (manifest.get("tool_commits") != PINS or manifest.get("nccl_version") != NCCL_VERSION or
            manifest.get("cuda_architecture") != "sm_120" or manifest.get("platform") != "linux-x86_64"):
        raise RuntimeError("toolkit dependency pins/platform do not match this experiment")
    required = {"bin/" + name for name in (*BINARIES, *SYSTEM_TOOLS)} | {"lib/libnccl.so.2", "libexec/lspci", "share/pci.ids"}
    actual = {str(path.relative_to(toolkit)) for path in toolkit.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if manifest.get("schema_version") != 2 or not required <= set(manifest["files"]) or set(manifest["files"]) != actual:
        raise RuntimeError("toolkit manifest has missing/unexpected files")
    for name, digest in manifest["files"].items():
        if sha256(toolkit / name) != digest:
            raise RuntimeError("toolkit file checksum mismatch: " + name)
    state["toolkit_sha256"] = expected_sha
    state["toolkit_manifest"] = manifest
    save_json(output / "toolkit-manifest.json", manifest)
    return toolkit


def execute(archive, digest):
    validate_cuda()
    for command in ("nvidia-smi", "ldd"):
        if shutil.which(command) is None:
            raise RuntimeError(f"runtime image missing {command}; run mode will not install dependencies on the GPU Pod")
    toolkit = unpack(archive, digest)
    runtime_paths = [str(toolkit / "lib"), "/usr/local/cuda/lib64"]
    runtime_paths.extend(path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path and "/stubs" not in path)
    env = {"LD_LIBRARY_PATH": ":".join(dict.fromkeys(runtime_paths)),
           "PATH": str(toolkit / "bin") + ":" + os.environ.get("PATH", "")}
    # ldd verifies host ABI without running a benchmark or importing PyTorch.
    for filename in (*BINARIES, "numactl", "lscpu", "lspci"):
        binary = toolkit / ("libexec/lspci" if filename == "lspci" else "bin/" + filename)
        libraries = run("runtime-libraries-" + filename, ["ldd", binary], seconds=10, env=env)
        if "not found" in libraries:
            raise RuntimeError(f"runtime dependency/ABI mismatch for {filename}; no GPU-side rebuild")
    version = run("nccl-runtime-version", [sys.executable, "-c",
                  "import ctypes; lib=ctypes.CDLL('libnccl.so.2'); v=ctypes.c_int(); "
                  "rc=lib.ncclGetVersion(ctypes.byref(v)); print(v.value); raise SystemExit(rc)"], seconds=10, env=env)
    if version != str(NCCL_VERSION_CODE):
        raise RuntimeError("runtime NCCL is not 2.27.7")
    # Keep the complete test matrix: insufficient remaining time aborts instead of silently shrinking it.
    if deadline - time.monotonic() < args.work_seconds + 35:
        raise TimeoutError("preparation used the reserved suite budget; abort without starting a shortened experiment")
    results = output / "results"
    run("suite", [sys.executable, "-u", suite, "--output", results, "--budget-seconds", args.work_seconds,
                  "--nvbandwidth", toolkit / "bin/nvbandwidth", "--nccl-tests", toolkit / "bin"],
        seconds=args.work_seconds + 12, env=env)
    result = json.loads((results / "metadata.json").read_text())
    if result.get("status") != "complete":
        raise RuntimeError("suite incomplete; preserve partial results and stop this Pod")
    state["suite_status"] = result["status"]


exitcode = 0
checkpoint()
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
    signal.signal(number, interrupted)
signal.setitimer(signal.ITIMER_REAL, args.total_seconds)
try:
    run("suite-self-check", [sys.executable, suite, "--self-check"], seconds=10)
    if args.mode == "prepare":
        archive, digest = prepare()
    else:
        archive, digest = Path(args.toolkit).resolve(), args.toolkit_sha256
    if args.mode == "run":
        execute(archive, digest)
    state["status"] = "prepared" if args.mode == "prepare" else "complete"
except BaseException as exc:
    kill_active()
    state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    exitcode = 2
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    state["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint()
    # Raw results already exist on disk; never delete them if packaging fails.
    if (output / "results").is_dir() and deadline - time.monotonic() > 3:
        try:
            run("package-results", ["tar", "-czf", output / "results.tar.gz", "-C", output, "results", "bootstrap-metadata.json",
                                    "suite.stdout.txt", "suite.stderr.txt"], seconds=20)
        except BaseException as exc:
            state["archive_error"] = str(exc)
    checkpoint()
    print(json.dumps({"event": "bootstrap_finished", "status": state["status"], "output": str(output),
                      "elapsed_seconds": state["elapsed_seconds"], "pod_cleanup_required": True}), flush=True)
sys.exit(exitcode)
PY
