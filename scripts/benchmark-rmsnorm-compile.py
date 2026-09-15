"""Measure chapter-three kernels with the chapter-two reference and protocol.

Run with the same CUDA/PyTorch environment used for rmsnorm-eager.json.
Full environment, first-call costs, validation, and profiler samples stay in
the raw record; the article only presents steady-state results.
"""

import gzip
import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from torch._inductor.utils import run_and_get_code
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.benchmark import Timer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "public/measurements/rmsnorm-compile"
OUTPUT = ROOT / "content/data/rmsnorm-compile.json"
EAGER = json.loads((ROOT / "content/data/rmsnorm-eager.json").read_text())
assert torch.__version__ == EAGER["torch"]
assert platform.python_version() == EAGER["python"]
assert torch.cuda.get_device_name() == EAGER["gpu"]
SOURCE = "\n\n".join(re.findall(
    r"```python\n(.*?)```", (ROOT / "content/rmsnorm.md").read_text(), re.S
)[:2])
assert hashlib.sha256(SOURCE.encode()).hexdigest() == EAGER["reference_sha256"]
namespace = {"torch": torch}
exec(compile(SOURCE, "rmsnorm_reference.py", "exec"), namespace)
forward = namespace["rmsnorm_forward_reference"]
backward = namespace["rmsnorm_backward_reference"]
torch.set_num_threads(1)
torch.manual_seed(20260915)
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def latency(fn):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    samples = [Timer(stmt="fn()", globals={"fn": fn}, num_threads=1)
               .blocked_autorange(min_run_time=0.3) for _ in range(3)]
    return {
        "median_us": statistics.median(m.median for m in samples) * 1e6,
        "round_medians_us": [m.median * 1e6 for m in samples],
        "round_iqr_us": [m.iqr * 1e6 for m in samples],
        "calls_per_block": [m.number_per_run for m in samples],
        "blocks": [len(m.raw_times) for m in samples],
    }


def memory(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        samples.append({
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak,
            "peak_increment_bytes": peak - baseline,
            "output_live_increment_bytes": torch.cuda.memory_allocated() - baseline,
        })
        del output
    return {"peak_increment_bytes": max(s["peak_increment_bytes"] for s in samples),
            "samples": samples}


def validate(actual, expected, names):
    result = {}
    for name, a, b in zip(names, actual, expected):
        rtol, atol = (1e-5, 1e-6) if name == "r" else (0.016, 0.001)
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        diff = (a.float() - b.float()).abs()
        result[name] = {
            "passed": True, "dtype": str(a.dtype), "rtol": rtol, "atol": atol,
            "max_abs_error": diff.max().item(),
            "relative_l2_error": (diff.norm() / b.float().norm().clamp_min(1e-30)).item(),
        }
    return result


def profile_call(fn, path):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        for i in range(3):
            with record_function(f"measured_{i}"):
                fn()
                torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    samples = []
    for i in range(3):
        region = next(e for e in events if e.get("name") == f"measured_{i}")
        kernels = sorted((e for e in events if e.get("cat") == "kernel"
                          and region["ts"] <= e["ts"] < region["ts"] + region["dur"]),
                         key=lambda e: e["ts"])
        origin = kernels[0]["ts"]
        samples.append({
            "kernel_count": len(kernels),
            "kernel_duration_us": sum(e["dur"] for e in kernels),
            "span_us": kernels[-1]["ts"] + kernels[-1]["dur"] - origin,
            "kernels": [{"name": e["name"], "start_us": e["ts"] - origin,
                         "duration_us": e["dur"]} for e in kernels],
        })
    compressed = path.with_suffix(".json.gz")
    with gzip.open(compressed, "wb") as f:
        f.write(path.read_bytes())
    path.unlink()
    return {"samples": samples, "trace": str(compressed.relative_to(ROOT)),
            "trace_sha256": hashlib.sha256(compressed.read_bytes()).hexdigest()}


result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    "triton": triton.__version__, "cuda": torch.version.cuda,
    "driver": subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
    ], text=True).strip(),
    "python": platform.python_version(), "cpu": platform.processor(),
    "reference_sha256": EAGER["reference_sha256"], "reference_source": SOURCE,
    "compile": {"backend": "inductor", "fullgraph": True, "dynamic": False,
                "mode": "default", "cuda_graphs": False},
    "method": EAGER["method"], "memory_method": EAGER["memory"]["metric"],
    "profile_method": EAGER["profile_method"],
    "first_call_note": "Wall time including capture, compilation/cache loading and first execution; not a cold-cache compiler benchmark. Excluded from latency and memory measurements.",
    "validation_method": "Compare y/r to eager; compare dx/dgamma using identical eager r, and additionally run backward with compiled r. BF16 rtol=0.016 atol=0.001, FP32 r rtol=1e-5 atol=1e-6. Random normal inputs, seed 20260915; representative shape also checked for zero, 1e-3 and 1e2 scale inputs.",
    "measurements": [],
}

with torch.no_grad():
    for base_row in EAGER["measurements"]:
        if base_row["dtype"] != "torch.bfloat16":
            continue
        M, N = base_row["M"], base_row["N"]
        folder = ARTIFACTS / f"{M}x{N}"
        folder.mkdir(exist_ok=True)
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        g = torch.randn_like(x)
        ref_y, r = forward(x, gamma)
        row = {"M": M, "N": N, "dtype": "torch.bfloat16", "validation": {}}
        compiled_functions = {}
        for direction, reference, args, names in [
            ("forward", forward, (x, gamma), ("y", "r")),
            ("backward", backward, (x, gamma, g, r), ("dx", "dgamma")),
        ]:
            compiled = torch.compile(reference, backend="inductor", fullgraph=True, dynamic=False)
            compiled_functions[direction] = compiled
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual, codes = run_and_get_code(compiled, *args)
            torch.cuda.synchronize()
            first_call_s = time.perf_counter() - start
            assert len(codes) == 1, (M, N, direction, len(codes))
            code_path = folder / f"{direction}.py"
            code_path.write_text(codes[0])
            row["validation"].update(validate(actual, reference(*args), names))
            del actual
            fn = lambda: compiled(*args)
            row[direction] = latency(fn)
            row[direction]["memory"] = memory(fn)
            row[direction]["first_call_s"] = first_call_s
            row[direction]["code"] = str(code_path.relative_to(ROOT))
            row[direction]["code_sha256"] = hashlib.sha256(codes[0].encode()).hexdigest()
            row[direction]["profile"] = profile_call(fn, folder / f"{direction}-trace.json")
            print(json.dumps({"M": M, "N": N, "direction": direction,
                              "us": row[direction]["median_us"],
                              "MiB": row[direction]["memory"]["peak_increment_bytes"] / 2**20,
                              "kernels": row[direction]["profile"]["samples"][0]["kernel_count"]}), flush=True)

        yc, rc = compiled_functions["forward"](x, gamma)
        row["validation"]["composed"] = validate(
            compiled_functions["backward"](x, gamma, g, rc),
            backward(x, gamma, g, r), ("dx", "dgamma"))
        if M == 1024 and N == 4096:
            row["validation"]["scaled_inputs"] = []
            for scale in (0.0, 1e-3, 1e2):
                xs = x * scale
                expected_y, expected_r = forward(xs, gamma)
                actual_y, actual_r = compiled_functions["forward"](xs, gamma)
                checks = validate((actual_y, actual_r), (expected_y, expected_r), ("y", "r"))
                checks.update(validate(compiled_functions["backward"](xs, gamma, g, actual_r),
                                       backward(xs, gamma, g, expected_r), ("dx", "dgamma")))
                row["validation"]["scaled_inputs"].append({"scale": scale, "checks": checks})
                del xs, expected_y, expected_r, actual_y, actual_r
        result["measurements"].append(row)
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        del x, gamma, g, r, ref_y, yc, rc, args, fn, compiled, compiled_functions

print(f"Saved {OUTPUT}", flush=True)
