#!/usr/bin/env bash
# One bounded preparation + complete topology suite on the approved Runpod Pod.
# The external guardian owns the Pod lifetime; this script only bounds processes.
set -euo pipefail
exec python3 - "$@" <<'PY'
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback

PINS = {
    "nvbandwidth": "1aa9e818d0728c25c87e121d692d829db2239720",
    "nccl-tests": "2535da805b34e96d1dc08be66289be1a6d57f5ad",
}
parser = argparse.ArgumentParser(description="Prepare once in 240s, then run the unchanged 660s topology suite.")
parser.add_argument("--output", required=True)
parser.add_argument("--suite", required=True)
parser.add_argument("--base-image", required=True, help="Exact image used for this Pod, recorded with the results")
parser.add_argument("--prepare-seconds", type=int, default=240)
parser.add_argument("--work-seconds", type=int, default=660)
parser.add_argument("--total-seconds", type=int, default=960)
parser.add_argument("--collectives-only", action="store_true")
parser.add_argument("--container-compat", action="store_true")
parser.add_argument("--transport-diagnosis", action="store_true")
parser.add_argument("--same-node-comparison", action="store_true")
args = parser.parse_args()
if (args.prepare_seconds, args.work_seconds, args.total_seconds) != (240, 660, 960):
    parser.error("this one-shot plan requires prepare=240, work=660, total=960 seconds")
if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info < (3, 10):
    parser.error("Linux x86_64 with Python >=3.10 is required")
suite = Path(args.suite).resolve()
if not suite.is_file():
    parser.error("suite file is missing")
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
deadline = started + args.total_seconds
prepare_deadline = started + args.prepare_seconds
build = output.parent.parent / ("rtx-topology-build-" + str(time.time_ns()))
active = None
state = {
    "schema_version": 1, "mode": "direct", "status": "running", "phase": "prepare", "steps": [],
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "base_image": args.base_image,
    "tool_commits": PINS, "nccl_version": "2.27.7", "cuda_architecture": "sm_120",
    "prepare_seconds": args.prepare_seconds, "work_seconds": args.work_seconds,
    "total_seconds": args.total_seconds, "build_directory": str(build),
    "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
    "notes": ["One preparation attempt; no retry and no shortened test matrix.",
              "GPU identity and count are checked by the unchanged suite before its benchmarks.",
              "The external guardian must delete the Pod after this process exits."],
}


def checkpoint():
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    content = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    for name in ("bootstrap-metadata.json", "direct-metadata.json"):
        temporary = output / (name + ".tmp")
        temporary.write_text(content)
        temporary.replace(output / name)


def kill_active():
    if active and active.poll() is None:
        try:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def interrupted(signum, _frame):
    kill_active()
    raise TimeoutError(f"direct entrypoint interrupted: phase={state['phase']}, signal={signum}")


def run(name, command, seconds, env=None):
    global active
    limit = prepare_deadline if state["phase"] == "prepare" else deadline - 10
    remaining = limit - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{state['phase']} deadline reached before {name}")
    record = {"name": name, "command": [str(x) for x in command], "status": "running",
              "timeout_seconds": min(seconds, remaining),
              "stdout": name + ".stdout.txt", "stderr": name + ".stderr.txt"}
    state["steps"].append(record)
    checkpoint()
    began = time.monotonic()
    try:
        with (output / record["stdout"]).open("w") as stdout, (output / record["stderr"]).open("w") as stderr:
            active = subprocess.Popen(record["command"], stdout=stdout, stderr=stderr,
                                      env={**os.environ, **(env or {})}, start_new_session=True)
            try:
                code = active.wait(timeout=record["timeout_seconds"])
            except subprocess.TimeoutExpired as exc:
                record["status"] = "timeout"
                raise TimeoutError(f"{name} exceeded its stage/{state['phase']} deadline") from exc
            record.update(returncode=code, status="ok" if code == 0 else "failed")
            if code:
                raise RuntimeError(f"{name} exited {code}; see saved stdout/stderr")
    finally:
        kill_active()
        active = None
        record["elapsed_seconds"] = round(time.monotonic() - began, 3)
        checkpoint()
        print(json.dumps({"event": "direct_step", "name": name, "status": record["status"],
                          "elapsed_seconds": state["elapsed_seconds"]}), flush=True)
    return (output / record["stdout"]).read_text()


def prepare():
    run("suite-self-check", [sys.executable, suite, "--self-check"], 10)
    cuda = run("cuda-version", ["nvcc", "--version"], 10)
    match = re.search(r"release (\d+)\.(\d+)", cuda)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        raise RuntimeError("CUDA >=12.8 is required; do not replace the planned image")
    if "compute_120" not in run("cuda-architectures", ["nvcc", "--list-gpu-arch"], 10):
        raise RuntimeError("CUDA toolkit has no sm_120 support")
    state["cuda_toolkit"] = cuda
    state["os_release"] = Path("/etc/os-release").read_text()
    build.mkdir(parents=True, exist_ok=False)
    install_env = {"DEBIAN_FRONTEND": "noninteractive", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    run("apt-update", ["apt-get", "-o", "Acquire::Retries=0", "-o", "Acquire::http::Timeout=20", "update"], 60, install_env)
    run("apt-install", ["apt-get", "-o", "Acquire::Retries=0", "-o", "Acquire::http::Timeout=20", "install", "-y", "--no-install-recommends",
                        "build-essential", "cmake", "git", "libboost-program-options-dev", "pciutils", "numactl"], 100, install_env)
    wheel = build / "nccl-wheel"
    run("nccl-install", [sys.executable, "-m", "pip", "install", "--no-deps", "--retries", "0", "--timeout", "30",
                         "--target", wheel, "nvidia-nccl-cu12==2.27.7"], 90, install_env)
    nccl = wheel / "nvidia/nccl"
    (nccl / "lib/libnccl.so").symlink_to("libnccl.so.2")
    header = (nccl / "include/nccl.h").read_text()
    actual = [re.search(r"#define\s+NCCL_" + key + r"\s+(\d+)", header) for key in ("MAJOR", "MINOR", "PATCH")]
    if not all(actual) or ".".join(x[1] for x in actual) != "2.27.7":
        raise RuntimeError("NCCL header version mismatch")
    for name, commit in PINS.items():
        path = build / name
        run(name + "-init", ["git", "init", "-q", path], 5)
        run(name + "-fetch", ["git", "-C", path, "fetch", "--depth", "1", "https://github.com/NVIDIA/" + name + ".git", commit], 30)
        run(name + "-checkout", ["git", "-C", path, "checkout", "--detach", "FETCH_HEAD"], 5)
        if run(name + "-commit", ["git", "-C", path, "rev-parse", "HEAD"], 5).strip() != commit:
            raise RuntimeError(name + " commit mismatch")
    jobs = min(8, len(os.sched_getaffinity(0)))
    nvbandwidth = build / "nvbandwidth"
    run("nvbandwidth-configure", ["cmake", "-S", nvbandwidth, "-B", nvbandwidth / "build",
                                 "-DCMAKE_CUDA_ARCHITECTURES=120", "-DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs"], 60)
    run("nvbandwidth-build", ["cmake", "--build", nvbandwidth / "build", "-j", jobs], 120)
    run("nccl-tests-build", ["make", "-C", build / "nccl-tests", "-j", jobs, "MPI=0", "CUDA_HOME=/usr/local/cuda",
                             "NCCL_HOME=" + str(nccl), "NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120"], 120)
    libraries = [str(nccl / "lib"), "/usr/local/cuda/lib64"]
    libraries.extend(p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "/stubs" not in p)
    env = {"LD_LIBRARY_PATH": ":".join(dict.fromkeys(libraries))}
    version = run("nccl-runtime-version", [sys.executable, "-c",
                  "import ctypes; l=ctypes.CDLL('libnccl.so.2'); v=ctypes.c_int(); "
                  "r=l.ncclGetVersion(ctypes.byref(v)); print(v.value); raise SystemExit(r)"], 10, env)
    if version.strip() != "22707":
        raise RuntimeError("runtime NCCL version mismatch")
    state["prepare_elapsed_seconds"] = round(time.monotonic() - started, 3)
    return env


checkpoint()
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
    signal.signal(number, interrupted)
signal.setitimer(signal.ITIMER_REAL, args.total_seconds)
code = 0
try:
    env = prepare()
    if time.monotonic() > prepare_deadline:
        raise TimeoutError("preparation exceeded 240 seconds")
    if deadline - time.monotonic() < args.work_seconds + 30:
        raise TimeoutError("insufficient time for the complete suite; abort without shrinking tests")
    state["phase"] = "suite"
    run("suite", [sys.executable, "-u", suite, "--output", output / "results", "--budget-seconds", args.work_seconds,
                  "--nvbandwidth", build / "nvbandwidth/build/nvbandwidth", "--nccl-tests", build / "nccl-tests/build",
                  *(["--collectives-only"] if args.collectives_only else []),
                  *(["--container-compat"] if args.container_compat else []),
                  *(["--transport-diagnosis"] if args.transport_diagnosis else []),
                  *(["--same-node-comparison"] if args.same_node_comparison else [])],
        args.work_seconds + 12, env)
    result = json.loads((output / "results/metadata.json").read_text())
    if result.get("status") != "complete":
        raise RuntimeError("suite metadata does not report complete")
    state.update(status="complete", suite_status="complete", phase="finished")
except BaseException as exc:
    kill_active()
    state.update(status="failed", error=str(exc) or type(exc).__name__, traceback=traceback.format_exc())
    code = 2
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    state["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint()
    print(json.dumps({"event": "direct_finished", "status": state["status"],
                      "elapsed_seconds": state["elapsed_seconds"], "pod_cleanup_required": True}), flush=True)
sys.exit(code)
PY
