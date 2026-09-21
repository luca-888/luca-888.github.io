"""One bounded eight-GPU run; see docs/rtx-pro-6000-experiment.md before invoking."""

from pathlib import Path
import modal

app = modal.App("blog-rtx-pro-6000-topology")
suite_path = Path(__file__).with_name("rtx-topology-suite.py")

build_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .apt_install("build-essential", "cmake", "git", "libboost-program-options-dev", "pciutils", "numactl")
    .run_commands(
        "python -m pip install --no-cache-dir --target /opt/nccl-wheel nvidia-nccl-cu12==2.27.7",
        "ln -sf libnccl.so.2 /opt/nccl-wheel/nvidia/nccl/lib/libnccl.so",
        "git clone https://github.com/NVIDIA/nvbandwidth.git /opt/nvbandwidth && git -C /opt/nvbandwidth checkout 1aa9e818d0728c25c87e121d692d829db2239720",
        "cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64/stubs && cmake --build /opt/nvbandwidth/build -j 4",
        "git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests && git -C /opt/nccl-tests checkout 2535da805b34e96d1dc08be66289be1a6d57f5ad",
        "make -C /opt/nccl-tests -j4 MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/nccl-wheel/nvidia/nccl NVCC_GENCODE='-gencode=arch=compute_120,code=sm_120'",
    )
    .env({"LD_LIBRARY_PATH": "/opt/nccl-wheel/nvidia/nccl/lib:/usr/local/cuda/lib64"})
)
image = build_image.add_local_file(suite_path, "/opt/rtx-topology-suite.py")


@app.function(
    image=image, gpu="RTX-PRO-6000:8", cpu=(8, 8), memory=(16384, 16384),
    timeout=720, startup_timeout=120, max_containers=1, min_containers=0,
    buffer_containers=0, scaledown_window=2, single_use_containers=True,
    include_source=False, serialized=True,
)
def experiment_suite():
    import json
    import os
    from pathlib import Path
    import signal
    import subprocess
    import tempfile
    import time
    import uuid

    output = Path(tempfile.mkdtemp(prefix="rtx-topology-"))
    started = time.monotonic()
    attempt_id = uuid.uuid4().hex
    yield {"event": "gpu_started", "unix_time": time.time(), "attempt_id": attempt_id}
    command = ["python", "-u", "/opt/rtx-topology-suite.py", "--output", str(output), "--budget-seconds", "660"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    sent = {}

    def artifacts():
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            stat = path.stat()
            relative = str(path.relative_to(output))
            signature = (stat.st_size, stat.st_mtime_ns)
            if sent.get(relative) != signature:
                sent[relative] = signature
                yield {"event": "artifact", "path": relative, "data": path.read_bytes(), "attempt_id": attempt_id}

    try:
        with (output / "suite-events.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"event": "suite_output", "text": line.rstrip()}
                yield {**event, "attempt_id": attempt_id}
                yield from artifacts()
        returncode = process.wait(timeout=10)
        yield from artifacts()
        yield {"event": "gpu_finished", "returncode": returncode, "elapsed_seconds": round(time.monotonic() - started, 3), "attempt_id": attempt_id}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@app.local_entrypoint()
def main(output: str):
    import asyncio
    import json
    import time
    import sys

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    submitted_mono, submitted_wall = time.monotonic(), time.time()
    first_event_limit, absolute_limit, occupied_limit = 1800, 2700, 900
    status = {"state": "submitted", "submitted_unix": submitted_wall, "gpu_started_unix": None,
              "first_event_limit_seconds": first_event_limit, "absolute_limit_seconds": absolute_limit,
              "occupied_limit_seconds": occupied_limit, "occupied_anchor_elapsed_seconds": None}
    status_path = destination / "local-run.json"

    def save_status():
        status["elapsed_seconds"] = round(time.monotonic() - submitted_mono, 3)
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    # Persist before constructing/awaiting the single remote stream.
    save_status()

    async def collect():
        stream = experiment_suite.remote_gen.aio()
        next_event = asyncio.create_task(anext(stream))
        sample_task = None
        next_sample_mono = submitted_mono
        last_sample_mono = submitted_mono
        occupied_anchor = None
        first_event_received = False

        def lock_occupied(anchor, reason):
            nonlocal occupied_anchor
            # Earlier evidence can shorten this deadline; retries can never extend it.
            if occupied_anchor is None or anchor < occupied_anchor:
                occupied_anchor = anchor
                status["occupied_anchor_elapsed_seconds"] = round(anchor - submitted_mono, 3)
                status["occupied_anchor_unix_estimate"] = submitted_wall + anchor - submitted_mono
                status["occupied_anchor_reason"] = reason
                save_status()

        # Isolate private RPC state from this Modal CLI process. The child uses the
        # same synchronization boundary as Modal's official container-list command.
        task_query_python = """
import json
import sys
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2

app_id = sys.stdin.read().strip()
@synchronizer.create_blocking
async def query():
    client = await _Client.from_env()
    response = await client.stub.TaskList(api_pb2.TaskListRequest(app_id=app_id))
    print(json.dumps([{"started_at": row.started_at, "enqueued_at": row.enqueued_at} for row in response.tasks]))
query()
"""

        async def read_tasks():
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", task_query_python,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _stderr = await process.communicate(input=app.app_id.encode())
                if process.returncode:
                    raise RuntimeError(f"Isolated TaskList query exited {process.returncode}")
                return json.loads(stdout)
            finally:
                # The surrounding 10s wait_for cancels this coroutine, never the stream.
                if process is not None and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

        async def sample(sample_mono, sample_wall, previous_sample_mono):
            # Bound the two independent reads separately; never wait_for(anext(stream)).
            stats, tasks = await asyncio.gather(
                asyncio.wait_for(experiment_suite.get_current_stats.aio(), timeout=10),
                asyncio.wait_for(read_tasks(), timeout=10), return_exceptions=True,
            )
            snapshot = {"event": "queue_snapshot", "sampled_unix": sample_wall,
                        "sample_elapsed_seconds": round(sample_mono - submitted_mono, 3),
                        "backlog": None, "num_total_runners": None, "num_running_inputs": None,
                        "container_count": None, "pending_count": None, "containers": [], "query_errors": {}}
            if isinstance(stats, BaseException):
                snapshot["query_errors"]["function_stats"] = type(stats).__name__
            else:
                snapshot.update(backlog=stats.backlog, num_total_runners=stats.num_total_runners,
                                num_running_inputs=stats.num_running_inputs)
            if isinstance(tasks, BaseException):
                snapshot["query_errors"]["task_list"] = type(tasks).__name__
            else:
                # Private application/container IDs are deliberately excluded from snapshots.
                snapshot["containers"] = sorted(
                    [{"started_at": task["started_at"], "enqueued_at": task["enqueued_at"]} for task in tasks],
                    key=lambda row: (row["started_at"], row["enqueued_at"]),
                )
                snapshot["container_count"] = len(snapshot["containers"])
                snapshot["pending_count"] = sum(row["started_at"] <= 0 for row in snapshot["containers"])
            if (snapshot["container_count"] or 0) > 1 or (snapshot["num_total_runners"] or 0) > 1:
                raise RuntimeError("Single-container limit exceeded: "
                                   f"runners={snapshot['num_total_runners']}, containers={snapshot['container_count']}")
            starts = [row["started_at"] for row in snapshot["containers"] if row["started_at"] > 0]
            if starts:
                # Convert provider wall timestamps to this process's monotonic clock once.
                earliest = min(starts)
                lock_occupied(sample_mono - max(0, sample_wall - earliest), "TaskStats.started_at")
            if snapshot["query_errors"]:
                lock_occupied(previous_sample_mono, "telemetry unavailable; conservative prior-sample bound")
            elif (snapshot["num_total_runners"] > 0 or snapshot["container_count"] > 0
                  or snapshot["num_running_inputs"] > 0):
                # Pending/started_at describe readiness, not a trustworthy billing boundary.
                lock_occupied(previous_sample_mono, "container/runner observed, including Pending; conservative bound")
            snapshot["occupied_anchor_elapsed_seconds"] = status["occupied_anchor_elapsed_seconds"]
            snapshot["observed_unix"] = time.time()
            return snapshot, sample_mono

        def deadline():
            candidates = [(submitted_mono + absolute_limit, "absolute", absolute_limit)]
            if not first_event_received:
                candidates.append((submitted_mono + first_event_limit, "first_event", first_event_limit))
            if occupied_anchor is not None:
                candidates.append((occupied_anchor + occupied_limit, "occupied", occupied_limit))
            return min(candidates, key=lambda item: item[0])

        try:
            with (destination / "queue-snapshots.jsonl").open("a") as snapshots:
                while True:
                    now = time.monotonic()
                    cutoff, phase, limit = deadline()
                    if now >= cutoff:
                        message = (f"One-shot run phase={phase} exceeded {limit}s limit; "
                                   f"elapsed={now - submitted_mono:.3f}s since submission")
                        if occupied_anchor is not None:
                            message += f"; occupied_elapsed={now - occupied_anchor:.3f}s"
                        raise TimeoutError(message)
                    if sample_task is None and now >= next_sample_mono:
                        sample_task = asyncio.create_task(sample(now, time.time(), last_sample_mono))
                        next_sample_mono = now + 30
                    waiting = {next_event}
                    if sample_task is not None:
                        waiting.add(sample_task)
                    wakeup = min(cutoff, next_sample_mono) if sample_task is None else cutoff
                    done, _ = await asyncio.wait(waiting, timeout=max(0, wakeup - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
                    if sample_task in done:
                        snapshot, last_sample_mono = sample_task.result()
                        sample_task = None
                        status["last_queue_snapshot"] = snapshot
                        snapshots.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
                        snapshots.flush()
                        print(json.dumps(snapshot, ensure_ascii=False), flush=True)
                        save_status()
                    if next_event not in done:
                        continue
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        break
                    first_event_received = True
                    if status.get("attempt_id") not in (None, event["attempt_id"]):
                        raise RuntimeError("Platform restarted the GPU input; stop without mixing results or resetting deadlines")
                    status["attempt_id"] = event["attempt_id"]
                    if event.get("event") == "artifact":
                        target = (destination / event["path"]).resolve()
                        if not target.is_relative_to(destination):
                            raise ValueError("Unexpected artifact path")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(event["data"])
                    else:
                        if event.get("event") == "gpu_started":
                            if status["gpu_started_unix"] is not None:
                                raise RuntimeError("Platform repeated gpu_started; stop without resetting the occupied deadline")
                            lock_occupied(last_sample_mono, "gpu_started; conservative prior-sample bound")
                            status["gpu_started_unix"] = time.time()
                            status["state"] = "running"
                        elif event.get("event") == "gpu_finished":
                            status.update(state="finished", returncode=event["returncode"], gpu_elapsed_seconds=event["elapsed_seconds"])
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    save_status()
                    # Only advance the stream after its previous anext actually returned.
                    next_event = asyncio.create_task(anext(stream))
        finally:
            # Cancellation happens only during teardown, never for a polling timeout.
            pending = [task for task in (next_event, sample_task) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            close = getattr(stream, "aclose", None)
            if close is not None:
                try:
                    await asyncio.wait_for(close(), timeout=5)
                except (Exception, asyncio.CancelledError) as error:
                    status["stream_close_error"] = type(error).__name__

    try:
        asyncio.run(collect())
        if status.get("returncode") != 0:
            raise RuntimeError("Experiment did not complete successfully; inspect saved partial results")
    except BaseException as error:
        status.update(state="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        status["ended_unix"] = time.time()
        save_status()
