"""Compare ordinary submissions and replay for the same two-stream diamond."""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from example import prepare, workload


def measure(fn, iterations):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1000


@torch.no_grad()
def run_case(iterations, rounds):
    buffers, capture_stream, side = prepare()
    y = buffers[-1]
    start = time.perf_counter_ns()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        workload(*buffers, side)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter_ns() - start) / 1e6
    assert torch.count_nonzero(y).item() == 0
    graph.replay()
    assert torch.equal(y, torch.full_like(y, 31))
    workload(*buffers, side)
    assert torch.equal(y, torch.full_like(y, 31))

    calls = {"ordinary": lambda: workload(*buffers, side), "graph": graph.replay}
    samples = {name: [] for name in calls}
    orders = []
    for round_index in range(rounds):
        order = ["ordinary", "graph"] if round_index % 2 == 0 else ["graph", "ordinary"]
        orders.append(order)
        for name in order:
            samples[name].append(measure(calls[name], iterations))
            assert torch.equal(y, torch.full_like(y, 31))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {"kernel_count": 4, "edge_count": 4, "shape": [4096], "correctness_passed": True,
            "capture_and_instantiate_ms": capture_ms, "round_orders": orders,
            "ordinary_us": medians["ordinary"], "graph_us": medians["graph"],
            "speedup": medians["ordinary"] / medians["graph"], "samples_us": samples}


def environment():
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.mem,pstate",
                          "--format=csv"], capture_output=True, text=True, check=True).stdout.strip()
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": props.name, "gpu_total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor], "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "python": sys.version, "cpu": cpu,
            "platform": platform.platform(), "nvidia_smi": smi, "torch_num_threads": 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"environment": environment(), "method": {
        "operation": "A:x=2*input; fork B:b=x+1 / C:c=x+2; join D:y=b+c; same buffers and dependencies",
        "shape": [4096], "dtype": "torch.float32", "iterations_per_round": args.iterations,
        "rounds": args.rounds, "warmup_calls_per_measurement": 20,
        "timer": "perf_counter_ns; synchronize before/after batch; median of round means",
        "round_order": "Alternate ordinary->graph and graph->ordinary",
        "timed_region": "Ordinary workload including stream contexts/event waits, or graph.replay; no copy, clone or per-call synchronization",
        "capture_cost": "One CUDAGraph creation/capture/default instantiation/trailing sync, before first replay",
        "cache_and_clocks": "No cache flush or clock locking",
        "correctness": "Capture leaves y=0; ordinary and graph paths each produce y=31 from input=7"
    }, "results": [run_case(args.iterations, args.rounds)]}
    record = result["results"][0]
    print(f"4-node diamond: ordinary {record['ordinary_us']:.2f} us, "
          f"graph {record['graph_us']:.2f} us, {record['speedup']:.2f}x", flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
