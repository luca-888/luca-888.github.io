#!/usr/bin/env python3
"""One temporary Runpod, one experiment, with a separate local cleanup watchdog.

Default is an offline dry run. --execute is the only path that creates a Pod.
The watchdog survives this process/SSH exiting, but cannot survive laptop shutdown
or guarantee a billing cap while Runpod's control plane is unreachable.
"""

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
REMOTE = "/workspace/rtx-topology-run"
TOTAL_SECONDS, READY_SECONDS, WORK_SECONDS = 1320, 240, 660
PREPARE_SECONDS, ENTRY_SECONDS, ENTRY_WITH_RECOVERY_SECONDS = 240, 960, 1050
CREATE_REJECTED_CODES = {"bad_request", "forbidden", "unauthorized", "no_credentials", "usage_error", "capacity_unavailable"}
CAPACITY_REJECTION = ("failed to create pod: graphql error: There are no longer any instances available "
                      "with the requested specifications. Please refresh and try again.")


def create_rejection_code(payload):
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if code in CREATE_REJECTED_CODES - {"capacity_unavailable"}:
        return code
    if code == "graphql_error" and payload.get("error") == CAPACITY_REJECTION and "id" not in payload:
        return "capacity_unavailable"
    return None


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(source)


def digest_stream(source):
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class Run:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.spec = json.loads((self.directory / "run.json").read_text())

    def event(self, event, **fields):
        record = {"time": time.time(), "event": event, **fields}
        with (self.directory / f"events-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")

    def cli(self, *args, timeout=25):
        # Capture raw responses only in the private run directory; never print them.
        stem = self.directory / f"api-{time.time_ns()}-{os.getpid()}"
        try:
            result = subprocess.run([self.spec["cli"], *args], capture_output=True, timeout=timeout)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr, code = error.stdout or b"", error.stderr or b"", 124
        stem.with_suffix(".stdout").write_bytes(stdout)
        stem.with_suffix(".stderr").write_bytes(stderr)
        try:
            payload = json.loads(stdout if code == 0 else stderr)
        except (ValueError, UnicodeDecodeError):
            payload = {}
            # CLI notes can precede its structured error on stderr.
            for line in (stdout if code == 0 else stderr).decode(errors="replace").splitlines():
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and candidate.get("code"):
                    payload = candidate
        return code, payload

    def list_owned(self):
        code, payload = self.cli("pod", "list", "--all", "--name", self.spec["name"])
        if code or not isinstance(payload, list):
            raise RuntimeError("Unable to verify temporary Pod inventory")
        matches = [row for row in payload if row and row.get("name") == self.spec["name"]]
        if len(matches) > 1:
            raise RuntimeError("Ambiguous unique Pod name; manual resource review required")
        return matches

    def pod_id(self):
        path = self.directory / "pod.json"
        return json.loads(path.read_text())["id"] if path.exists() else None

    def claim(self, pod):
        if not pod.get("id") or pod.get("name") != self.spec["name"]:
            raise RuntimeError("Pod ownership could not be established")
        previous = self.pod_id()
        if previous and previous != pod["id"]:
            raise RuntimeError("Refusing a second Pod")
        atomic_json(self.directory / "pod.json", {"id": pod["id"], "name": pod["name"]})
        return pod["id"]

    def cleanup(self):
        """Idempotent deletion; only a confirmed absence counts as fully cleaned."""
        with (self.directory / "cleanup.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if (self.directory / "cleaned.json").exists():
                return True
            try:
                identifier = self.pod_id()
                if identifier is None:
                    matches = self.list_owned()
                    if matches:
                        identifier = self.claim(matches[0])
                    else:
                        rejection = self.directory / "create-rejected.json"
                        if rejection.exists() and json.loads(rejection.read_text()).get("code") in CREATE_REJECTED_CODES:
                            atomic_json(self.directory / "cleaned.json", {
                                "time": time.time(), "state": "verified-no-pod-after-rejection",
                            })
                            return True
                        # A timed-out create may still be committing. An empty
                        # snapshot is not a rejected create; watch through the
                        # original deadline plus a control-plane settling window.
                        if time.monotonic() < self.spec["deadline_mono"] + 120:
                            return False
                        atomic_json(self.directory / "cleaned.json", {"time": time.time(), "state": "no_pod_found_after_create_grace"})
                        return True
                self.event("delete_requested")
                code, _ = self.cli("pod", "delete", identifier)
                for _ in range(3):
                    matches = self.list_owned()
                    if not matches:
                        get_code, detail = self.cli("pod", "get", identifier)
                        if get_code and detail.get("code") == "not_found":
                            atomic_json(self.directory / "cleaned.json", {"time": time.time(), "state": "deleted_and_verified", "delete_returncode": code})
                            return True
                    time.sleep(2)
                # An error/uncertain delete must still attempt to stop compute.
                stop_code, _ = self.cli("pod", "stop", identifier)
                check_code, detail = self.cli("pod", "get", identifier)
                state = detail.get("runtimeStatus") if check_code == 0 else "unknown"
                atomic_json(self.directory / "cleanup-pending.json", {
                    "time": time.time(), "state": state, "stop_returncode": stop_code,
                    "storage_may_still_bill": True, "watchdog_will_retry_delete": True,
                })
                self.event("delete_unconfirmed", runtime_status=state, storage_may_still_bill=True)
            except Exception as error:
                self.event("cleanup_error", error=type(error).__name__)
                # A failed inventory read must not skip stopping a known owned Pod.
                identifier = self.pod_id()
                if identifier:
                    self.cli("pod", "stop", identifier)
                atomic_json(self.directory / "cleanup-pending.json", {
                    "time": time.time(), "state": "unknown", "storage_may_still_bill": True,
                    "watchdog_will_retry_delete": True,
                })
            return False

    def remaining(self, reserve=60):
        seconds = self.spec["deadline_mono"] - time.monotonic() - reserve
        if seconds <= 0:
            raise TimeoutError("Pod deadline reached; begin cleanup")
        return seconds

    def command(self, args, timeout=30, stdout=subprocess.PIPE):
        result = subprocess.run(args, stdout=stdout, stderr=subprocess.PIPE, timeout=min(timeout, self.remaining()))
        if result.returncode:
            (self.directory / f"command-error-{time.time_ns()}.log").write_bytes(result.stderr)
            raise RuntimeError(f"{Path(args[0]).name} exited {result.returncode}")
        return result.stdout

    def ssh_args(self, endpoint, scp=False):
        return ["scp" if scp else "ssh", "-F", "/dev/null", "-i", self.spec["key"],
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no",
                "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={self.directory / 'known_hosts'}",
                "-P" if scp else "-p", str(endpoint["port"])]

    def ssh(self, endpoint, command, timeout=30):
        return self.command(self.ssh_args(endpoint) + [f"root@{endpoint['ip']}", command], timeout)

    def validate_pod(self, pod):
        # CLI 2.14's actual --include-machine response uses machine.gpuId;
        # gpuTypeId exists in its Go type but may be omitted by the backend.
        model_fields = {"gpuTypeId": pod.get("gpuTypeId"), "gpuId": pod.get("gpuId"),
                        "machine.gpuId": (pod.get("machine") or {}).get("gpuId")}
        model_fields = {key: value for key, value in model_fields.items() if value}
        if pod.get("gpuCount") != 8 or not model_fields or any(value != GPU for value in model_fields.values()):
            raise RuntimeError("Allocated GPU model/count differs from the plan")
        if pod.get("networkVolumeId") or pod.get("volumeInGb", 0) != 0:
            raise RuntimeError("Unexpected persistent storage")
        if pod.get("vcpuCount", 0) < 8 or pod.get("memoryInGb", 0) < 16:
            raise RuntimeError("CPU/memory allocation is below the plan or unavailable")
        rate = pod.get("costPerHr", 0)
        if not isinstance(rate, (int, float)) or not 0 < rate <= self.spec["max_hourly_rate"]:
            raise RuntimeError("Pod hourly quote is missing or over the approved cap")
        atomic_json(self.directory / "allocation.json", {
            **{key: pod[key] for key in ("gpuCount", "vcpuCount", "memoryInGb", "costPerHr")},
            "gpuTypeId": GPU, "gpu_model_fields": model_fields,
        })

    def await_ready(self, identifier):
        while time.monotonic() < self.spec["ready_deadline_mono"]:
            code, pod = self.cli("pod", "get", identifier, "--include-machine", timeout=20)
            if code == 0:
                self.validate_pod(pod)
                endpoint = pod.get("ssh", {})
                if pod.get("runtimeStatus") == "running" and endpoint.get("ip") and endpoint.get("port"):
                    ipaddress.ip_address(endpoint["ip"])
                    if not 1 <= int(endpoint["port"]) <= 65535:
                        raise RuntimeError("Invalid SSH endpoint")
                    try:
                        self.ssh(endpoint, "true", timeout=12)
                    except (RuntimeError, subprocess.TimeoutExpired):
                        pass
                    else:
                        atomic_json(self.directory / "ready.json", {"time": time.time(), "endpoint": endpoint})
                        self.event("ssh_ready")
                        return endpoint
            time.sleep(5)
        raise TimeoutError("SSH did not become ready within 4 minutes")

    def pull(self, endpoint, final=False):
        label = "final" if final else f"partial-{time.time_ns()}"
        archive = self.directory / f"{label}.tar.gz"
        remote_archive = f"{REMOTE}/{label}.tar.gz"
        # tar may return 1 when a live file changes; the resulting partial archive
        # is still useful. A final archive must be made after the SSH job exits.
        allowed = "0" if final else "0 1"
        command = (f"tar --exclude=./results.tar.gz "
                   f"-czf {shlex.quote(remote_archive)} -C {REMOTE}/results .; "
                   f"rc=$?; case ' {allowed} ' in *\" $rc \"*) ;; *) exit $rc;; esac; "
                   f"sha256sum {shlex.quote(remote_archive)}")
        expected = self.ssh(endpoint, command, timeout=30).decode().split()[0]
        self.command(self.ssh_args(endpoint, scp=True) + [f"root@{endpoint['ip']}:{remote_archive}", str(archive)], timeout=45)
        if len(expected) != 64 or sha256(archive) != expected:
            raise RuntimeError("Downloaded archive checksum mismatch")
        # Verify all compressed bytes and reject special files/path traversal.
        with tarfile.open(archive) as contents:
            for member in contents:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                    raise RuntimeError("Unexpected result archive entry")
                if member.isfile():
                    with contents.extractfile(member) as source:
                        digest_stream(source)
        atomic_json(self.directory / f"{label}.verified.json", {"sha256": expected, "bytes": archive.stat().st_size})
        if final:
            target = self.directory / "results"
            target.mkdir()
            with tarfile.open(archive) as contents:
                contents.extractall(target)
        self.event("archive_verified", final=final, bytes=archive.stat().st_size)


def watchdog(directory):
    run = Run(directory)
    atomic_json(run.directory / "watchdog-ready.json", {"pid": os.getpid()})
    while not (run.directory / "cleaned.json").exists():
        try:
            os.kill(run.spec["parent_pid"], 0)
            parent_alive = True
        except ProcessLookupError:
            parent_alive = False
        expired = time.monotonic() >= run.spec["deadline_mono"]
        not_ready = not (run.directory / "ready.json").exists() and time.monotonic() >= run.spec["ready_deadline_mono"]
        if not parent_alive or expired or not_ready or (run.directory / "cleanup-requested").exists():
            run.event("watchdog_cleanup", parent_alive=parent_alive, deadline=expired, readiness_timeout=not_ready)
            if run.cleanup():
                break
            time.sleep(10)
        else:
            time.sleep(2)


def fixed_image(value):
    if any(char.isspace() for char in value) or ":" not in value.rsplit("/", 1)[-1]:
        raise argparse.ArgumentTypeError("Specify an explicit versioned image tag or digest")
    if value.rsplit(":", 1)[-1].lower() in ("", "latest"):
        raise argparse.ArgumentTypeError("An empty or latest image tag is not allowed")
    return value


def create_arguments(name, public_key, image=IMAGE):
    return ["pod", "create", "--name", name, "--gpu-id", GPU, "--gpu-count", "8",
            "--cloud-type", "SECURE", "--image", fixed_image(image), "--container-disk-in-gb", "40",
            "--volume-in-gb", "0", "--ports", "22/tcp", "--ssh",
            "--country-code", "US", "--min-cuda-version", "12.8",
            "--env", json.dumps({"PUBLIC_KEY": public_key})]


def validate_inputs(args):
    fixed_image(args.image)
    paths = {name: Path(getattr(args, name)).expanduser().resolve() for name in ("cli", "direct", "suite", "key")}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing local {name} file")
    public_key = Path(str(paths["key"]) + ".pub").read_text().strip()
    expected = public_key.split()[:2]
    if len(expected) != 2 or "\n" in public_key:
        raise ValueError("Expected one existing OpenSSH public key")
    derived = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", str(paths["key"])], capture_output=True, timeout=5)
    if derived.returncode == 0:
        keys = [derived.stdout.decode().split()[:2]]
    else:
        agent = subprocess.run(["ssh-add", "-L"], capture_output=True, timeout=5)
        keys = [line.split()[:2] for line in agent.stdout.decode().splitlines()] if agent.returncode == 0 else []
    if expected not in keys:
        raise ValueError("Existing private key must be usable without prompting (or loaded in the SSH agent)")
    if not 0 < args.max_hourly_rate <= 20:
        raise ValueError("This experiment permits a total Pod quote of at most $20/hour")
    return paths, public_key


def execute(args):
    paths, public_key = validate_inputs(args)
    if not args.execute:
        print(json.dumps({"dry_run": True, "gpu": GPU, "gpu_count": 8, "image": args.image,
                          "country_code": "US", "create_calls": 1, "ready_seconds": READY_SECONDS, "total_seconds": TOTAL_SECONDS,
                          "prepare_seconds": PREPARE_SECONDS, "work_seconds": WORK_SECONDS,
                          "entry_seconds": ENTRY_SECONDS, "required_remaining_seconds": ENTRY_WITH_RECOVERY_SECONDS,
                          "maximum_pod_hourly_rate": args.max_hourly_rate,
                          "script_sha256": {name: sha256(paths[name]) for name in ("suite", "direct")},
                          "remote_env_names": ["PUBLIC_KEY"]}))
        return
    directory = Path(args.output).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if directory == repository or repository in directory.parents:
        raise ValueError("Private run output must be outside the repository")
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=False)
    inputs = directory / "inputs"
    inputs.mkdir()
    for name, filename in (("suite", "rtx-topology-suite.py"), ("direct", "direct.sh")):
        shutil.copyfile(paths[name], inputs / filename)
    started = time.monotonic()
    spec = {"name": f"blog-rtx-topology-{uuid.uuid4().hex[:20]}", "parent_pid": os.getpid(),
            "image": args.image, "country_code": "US",
            "created_unix": time.time(), "created_mono": started, "deadline_mono": started + TOTAL_SECONDS,
            "ready_deadline_mono": started + READY_SECONDS, "max_hourly_rate": args.max_hourly_rate,
            **{name: str(path) for name, path in paths.items()},
            "script_sha256": {name: sha256(paths[name]) for name in ("suite", "direct")}}
    atomic_json(directory / "run.json", spec)
    run = Run(directory)
    # Verify the UUID name is unused before arming the create/recovery mechanism.
    if run.list_owned():
        raise RuntimeError("Unique Pod name already exists; no create issued")
    with (directory / "watchdog.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(directory)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    for _ in range(50):
        if (directory / "watchdog-ready.json").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Watchdog did not arm; no create issued")
    job = None
    endpoint = None
    failure = None
    try:
        (directory / "create-attempted").touch(exist_ok=False)
        run.event("create_requested")
        code, created = run.cli(*create_arguments(spec["name"], public_key, spec["image"]), timeout=45)
        # Only this one create call exists. Uncertain returns recover by UUID name.
        if code == 0 and isinstance(created, dict) and created.get("id"):
            if not created.get("name"):
                created["name"] = spec["name"]
            run.claim(created)
        while run.pod_id() is None and time.monotonic() < spec["ready_deadline_mono"]:
            matches = run.list_owned()
            if matches:
                run.claim(matches[0])
                break
            rejection = create_rejection_code(created) if code else None
            if rejection:
                atomic_json(directory / "create-rejected.json", {
                    "time": time.time(), "code": rejection, "returncode": code,
                })
                raise RuntimeError("Create was rejected; no replacement will be attempted")
            time.sleep(5)
        if run.pod_id() is None:
            raise TimeoutError("No owned Pod found before readiness deadline")
        endpoint = run.await_ready(run.pod_id())
        run.ssh(endpoint, f"mkdir -p {REMOTE}")
        run.command(run.ssh_args(endpoint, scp=True) + [str(inputs / "rtx-topology-suite.py"), str(inputs / "direct.sh"),
                    f"root@{endpoint['ip']}:{REMOTE}/"], timeout=30)
        command = shlex.join(["bash", f"{REMOTE}/direct.sh", "--output", f"{REMOTE}/results",
                              "--base-image", spec["image"],
                              "--suite", f"{REMOTE}/rtx-topology-suite.py", "--prepare-seconds", str(PREPARE_SECONDS),
                              "--work-seconds", str(WORK_SECONDS), "--total-seconds", str(ENTRY_SECONDS)])
        if run.remaining(reserve=0) < ENTRY_WITH_RECOVERY_SECONDS:
            raise TimeoutError("Less than 1050s remains for the complete direct entry, result recovery and cleanup")
        with (directory / "remote.log").open("wb") as log:
            job = subprocess.Popen(run.ssh_args(endpoint) + [f"root@{endpoint['ip']}", command], stdout=log, stderr=subprocess.STDOUT)
            run.event("suite_started")
            next_pull = time.monotonic() + 30
            while job.poll() is None:
                run.remaining()
                if time.monotonic() >= next_pull:
                    try:
                        run.pull(endpoint)
                    except Exception as error:
                        run.event("partial_pull_failed", error=type(error).__name__)
                    next_pull = time.monotonic() + 30
                time.sleep(1)
            run.event("suite_exited", returncode=job.returncode)
        run.pull(endpoint, final=True)
        if job.returncode:
            raise RuntimeError(f"Remote suite exited {job.returncode}; partial results preserved")
        direct = json.loads((directory / "results/direct-metadata.json").read_text())
        suite = json.loads((directory / "results/results/metadata.json").read_text())
        if direct.get("status") != "complete" or suite.get("status") != "complete":
            raise RuntimeError("Final archive reports incomplete results")
    except BaseException as error:
        failure = error
        run.event("run_failed", error=type(error).__name__, detail=str(error))
        # Recover already-written files if budget permits, without rerunning work.
        if endpoint and spec["deadline_mono"] - time.monotonic() > 90:
            try:
                run.pull(endpoint)
            except Exception:
                pass
    finally:
        if job is not None and job.poll() is None:
            job.terminate()
            try:
                job.wait(timeout=3)
            except subprocess.TimeoutExpired:
                job.kill()
                job.wait()
        (directory / "cleanup-requested").touch()
        cleaned = run.cleanup()
        atomic_json(directory / "local-run.json", {"ended_unix": time.time(),
                    "elapsed_seconds": time.monotonic() - started, "state": "failed" if failure else "finished",
                    "cleanup_verified": cleaned, "watchdog_active_if_needed": not cleaned,
                    "error": str(failure) if failure else None})
    print(json.dumps({"state": "failed" if failure else "finished", "cleanup_verified": cleaned,
                      "output": str(directory), "storage_may_still_bill": not cleaned}))
    if failure or not cleaned:
        raise SystemExit(1)


def self_test():
    # No CLI/SSH/network calls: exercise ownership, ambiguous create and cleanup.
    creation = create_arguments("offline-test", "ssh-ed25519 offline-test")
    assert creation[creation.index("--country-code") + 1] == "US"
    assert "--data-center-ids" not in creation
    assert creation[creation.index("--gpu-count") + 1] == "8"
    assert READY_SECONDS + ENTRY_WITH_RECOVERY_SECONDS <= TOTAL_SECONDS
    assert PREPARE_SECONDS + WORK_SECONDS <= ENTRY_SECONDS
    assert ENTRY_WITH_RECOVERY_SECONDS - ENTRY_SECONDS >= 90
    with tempfile.TemporaryDirectory(prefix="runpod-supervisor-test-") as folder:
        root = Path(folder)
        atomic_json(root / "run.json", {"name": "unique-test", "created_mono": time.monotonic() - 100,
                                       "deadline_mono": time.monotonic() - 200})
        run = Run(root)
        calls = []
        present = [True]
        def fake(*args, **kwargs):
            calls.append(args)
            if args[:2] == ("pod", "list"):
                return 0, [{"name": "unique-test", "id": "fake"}] if present[0] else []
            if args[:2] == ("pod", "delete"):
                present[0] = False
                return 0, {}
            if args[:2] == ("pod", "get"):
                return 1, {"code": "not_found"}
            raise AssertionError(args)
        run.cli = fake
        assert run.cleanup() and run.pod_id() == "fake"
        assert run.cleanup() and sum(call[:2] == ("pod", "delete") for call in calls) == 1
        assert not any(call[:2] == ("pod", "create") for call in calls)
        (root / "cleaned.json").unlink()
        def failed_delete(*args, **kwargs):
            calls.append(args)
            if args[:2] == ("pod", "list"):
                return 0, [{"name": "unique-test", "id": "fake"}]
            if args[:2] == ("pod", "get"):
                return 0, {"runtimeStatus": "stopped"}
            return (1, {}) if args[:2] == ("pod", "delete") else (0, {})
        run.cli = failed_delete
        assert not run.cleanup()
        assert any(call[:2] == ("pod", "stop") for call in calls)
        assert json.loads((root / "cleanup-pending.json").read_text())["storage_may_still_bill"]
        # Real detached watchdog processes, using a file-backed fake CLI only.
        # This verifies cleanup is independent of the SSH/main process lifetime.
        fake_cli = root / "fake-runpodctl"
        fake_cli.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
p = Path(__file__).with_name("fake-cloud.json")
s = json.loads(p.read_text())
a = sys.argv[1:]
if a[:2] == ["pod", "list"]:
    print(json.dumps([{"id": "fake", "name": s["name"]}] if s["present"] else []))
elif a[:2] == ["pod", "delete"]:
    s["present"] = False
    p.write_text(json.dumps(s))
    print("{}")
elif a[:2] == ["pod", "get"]:
    print(json.dumps({"code": "not_found"}), file=sys.stderr)
    sys.exit(1)
else:
    raise SystemExit("Unexpected fake command; cloud calls forbidden")
''')
        fake_cli.chmod(0o700)
        for reason in ("parent_exit", "readiness_timeout", "total_deadline"):
            case = root / reason
            case.mkdir()
            moment = time.monotonic()
            atomic_json(root / "fake-cloud.json", {"name": "test-" + reason, "present": True})
            atomic_json(case / "run.json", {"name": "test-" + reason, "created_mono": moment,
                        "parent_pid": 2**30 if reason == "parent_exit" else os.getpid(),
                        "ready_deadline_mono": moment - 1 if reason == "readiness_timeout" else moment + 60,
                        "deadline_mono": moment - 1 if reason == "total_deadline" else moment + 90,
                        "cli": str(fake_cli)})
            if reason == "total_deadline":
                atomic_json(case / "ready.json", {"time": time.time()})
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--watchdog", str(case)],
                                       capture_output=True, timeout=12, start_new_session=True)
            assert completed.returncode == 0, completed.stderr.decode()
            assert json.loads((case / "cleaned.json").read_text())["state"] == "deleted_and_verified"
            assert not json.loads((root / "fake-cloud.json").read_text())["present"]
    print("Offline checks passed: ownership recovery, delete idempotence, stop fallback, parent-exit watchdog, readiness timeout, total deadline; no cloud calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create exactly one paid eight-GPU Pod")
    parser.add_argument("--dry-run", action="store_true", help="Offline plan validation (also the default)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--watchdog", metavar="PRIVATE_DIRECTORY", help=argparse.SUPPRESS)
    parser.add_argument("--output", help="New private run directory outside the repository")
    parser.add_argument("--direct", default=str(Path(__file__).with_name("runpod-rtx-topology-direct.sh")))
    parser.add_argument("--suite", default=str(Path(__file__).with_name("rtx-topology-suite.py")))
    parser.add_argument("--cli", default="~/.local/bin/runpodctl")
    parser.add_argument("--key", default="~/.ssh/id_ed25519")
    parser.add_argument("--image", type=fixed_image, default=IMAGE, help="Explicit fixed image tag; untagged/latest rejected")
    parser.add_argument("--max-hourly-rate", type=float, default=20.0)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.self_test:
        self_test()
    else:
        if args.dry_run and args.execute:
            parser.error("--dry-run and --execute are mutually exclusive")
        if args.execute and not args.output:
            parser.error("--execute requires --output")
        # SIGTERM/SIGINT enter the same finally cleanup as Python exceptions.
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        execute(args)


if __name__ == "__main__":
    main()
