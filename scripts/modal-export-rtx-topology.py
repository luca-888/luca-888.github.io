"""Export the existing CPU-built toolkit once; never requests a GPU."""

import importlib.util
from pathlib import Path
import sys

import modal

scripts = Path(__file__).parent
spec = importlib.util.spec_from_file_location("rtx_topology_build_source", scripts / "modal-rtx-topology.py")
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
app = modal.App("blog-rtx-topology-cpu-export")
image = (source.build_image
         .add_local_file(scripts / "runpod-rtx-topology-bootstrap.sh", "/opt/runpod-rtx-topology-bootstrap.sh")
         .add_local_file(scripts / "rtx-topology-suite.py", "/opt/rtx-topology-suite.py"))


@app.function(image=image, cpu=(4, 4), memory=(4096, 4096), timeout=540, startup_timeout=60,
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
              single_use_containers=True, include_source=False, serialized=True)
def export_toolkit():
    import hashlib
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import tempfile
    import time
    import uuid

    attempt = uuid.uuid4().hex
    started = time.monotonic()
    output = Path(tempfile.mkdtemp(prefix="rtx-cpu-export-")) / "export"
    yield {"event": "cpu_started", "attempt_id": attempt, "gpu_requested": False}
    process = subprocess.Popen(
        ["bash", "/opt/runpod-rtx-topology-bootstrap.sh", "prepare", "--output", str(output),
         "--suite", "/opt/rtx-topology-suite.py", "--total-seconds", "180"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"event": "prepare_output", "text": line.rstrip()}
            yield {**event, "attempt_id": attempt}
        code = process.wait(timeout=10)
        paths = sorted(path for path in output.glob("*") if path.is_file() and path.suffix != ".tmp")
        manifest = output / "toolkit/manifest.json"
        if manifest.is_file():
            paths.append(manifest)
        for path in paths:
            relative = str(path.relative_to(output))
            size = path.stat().st_size
            yield {"event": "file_start", "path": relative, "size": size, "attempt_id": attempt}
            digest = hashlib.sha256()
            offset = 0
            with path.open("rb") as stream:
                while chunk := stream.read(4 * 1024**2):
                    digest.update(chunk)
                    yield {"event": "file_chunk", "path": relative, "offset": offset, "data": chunk, "attempt_id": attempt}
                    offset += len(chunk)
            yield {"event": "file_end", "path": relative, "sha256": digest.hexdigest(), "attempt_id": attempt}
        yield {"event": "cpu_finished", "returncode": code, "elapsed_seconds": round(time.monotonic() - started, 3), "attempt_id": attempt}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@app.local_entrypoint()
def main(output: str = "/private/tmp/rtx-topology-runpod-20260918-prep"):
    import asyncio
    import hashlib
    import json
    import time

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    status = {"state": "submitted", "submitted_unix": time.time(), "gpu_requested": False,
              "local_limit_seconds": 600, "files": {}}

    def save():
        status["elapsed_seconds"] = round(time.monotonic() - began, 3)
        (destination / "local-export.json").write_text(json.dumps(status, indent=2) + "\n")

    def target(path):
        resolved = (destination / path).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError("Unexpected artifact path")
        return resolved

    async def collect():
        stream = export_toolkit.remote_gen.aio()
        received = {}
        try:
            while True:
                remaining = 600 - (time.monotonic() - began)
                if remaining <= 0:
                    raise TimeoutError("CPU-only export exceeded its 600s submission/prepare/transfer limit")
                try:
                    event = await asyncio.wait_for(anext(stream), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError("CPU-only export exceeded its 600s submission/prepare/transfer limit") from exc
                if status.get("attempt_id") not in (None, event["attempt_id"]):
                    raise RuntimeError("CPU input replay detected; no automatic second attempt")
                status["attempt_id"] = event["attempt_id"]
                kind = event["event"]
                if kind == "file_start":
                    path = target(event["path"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.with_name(path.name + ".part").write_bytes(b"")
                    received[event["path"]] = {"offset": 0, "digest": hashlib.sha256(), "size": event["size"]}
                elif kind == "file_chunk":
                    entry = received[event["path"]]
                    if event["offset"] != entry["offset"]:
                        raise ValueError("Artifact chunk offset mismatch")
                    path = target(event["path"])
                    with path.with_name(path.name + ".part").open("ab") as file:
                        file.write(event["data"])
                    entry["digest"].update(event["data"])
                    entry["offset"] += len(event["data"])
                    continue
                elif kind == "file_end":
                    entry = received[event["path"]]
                    if entry["offset"] != entry["size"] or entry["digest"].hexdigest() != event["sha256"]:
                        raise ValueError("Artifact size/checksum mismatch")
                    path = target(event["path"])
                    path.with_name(path.name + ".part").replace(path)
                    status["files"][event["path"]] = {"size": entry["size"], "sha256": event["sha256"]}
                elif kind == "cpu_started":
                    if status.get("cpu_started_unix") is not None:
                        raise RuntimeError("CPU input restarted; stop instead of resetting budget")
                    status.update(state="running", cpu_started_unix=time.time())
                elif kind == "cpu_finished":
                    status.update(state="finished", returncode=event["returncode"], cpu_elapsed_seconds=event["elapsed_seconds"])
                print(json.dumps(event, ensure_ascii=False), flush=True)
                save()
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await asyncio.wait_for(close(), timeout=5)

    save()
    try:
        asyncio.run(collect())
        if status.get("returncode") != 0 or "rtx-topology-toolkit.tar.gz" not in status["files"]:
            raise RuntimeError("CPU export did not produce a verified toolkit; inspect saved logs")
    except BaseException as exc:
        status.update(state="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        status["ended_unix"] = time.time()
        save()
