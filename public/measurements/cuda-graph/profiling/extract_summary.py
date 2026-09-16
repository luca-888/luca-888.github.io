"""Verify and summarize the unmodified Nsight Systems SQLite export."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3


folder = Path(__file__).resolve().parent
connection = sqlite3.connect(f"file:{folder / 'diamond.sqlite'}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row
names = dict(connection.execute("SELECT id, value FROM StringIds"))
metadata = dict(connection.execute("SELECT name, value FROM META_DATA_EXPORT"))
log = (folder / "collection.log").read_text()
checks = json.loads(re.search(r"(?m)^\{\n.*?^\}", log, re.DOTALL)[0])
assert checks["capture_left_output_unchanged"] and checks["final_value"] == 31

all_kernels = list(connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
ranges = []
for nvtx in connection.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
    apis = list(connection.execute(
        "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= ? AND end <= ? AND globalTid = ? ORDER BY start",
        (nvtx["start"], nvtx["end"], nvtx["globalTid"]),
    ))
    kernels = [k for k in all_kernels if nvtx["start"] <= k["start"] and k["end"] <= nvtx["end"]]
    launches = [a for a in apis if names[a["nameId"]].startswith(("cudaLaunchKernel_", "cudaGraphLaunch_"))]
    correlations = {a["correlationId"] for a in launches}
    correlated = [k for k in all_kernels if k["correlationId"] in correlations]
    row = {
        "name": nvtx["text"],
        "start_ns": nvtx["start"],
        "end_ns": nvtx["end"],
        "start_ms": nvtx["start"] / 1e6,
        "end_ms": nvtx["end"] / 1e6,
        "duration_us": (nvtx["end"] - nvtx["start"]) / 1e3,
        "cuda_api_counts": dict(collections.Counter(names[a["nameId"]] for a in apis)),
        "gpu_kernel_count_in_range": len(kernels),
        "gpu_kernel_count_by_launch_correlation": len(correlated),
        "gpu_kernel_names": dict(collections.Counter(names[k["demangledName"]] for k in kernels)),
        "launch_correlations": [{
            "api": names[a["nameId"]],
            "correlation_id": a["correlationId"],
            "gpu_kernel_count": sum(k["correlationId"] == a["correlationId"] for k in correlated),
        } for a in launches],
        "kernels": [{
            "start_ns": k["start"],
            "end_ns": k["end"],
            "correlation_id": k["correlationId"],
            "graph_node_id": k["graphNodeId"],
            "stream_id": k["streamId"],
            "name": names[k["demangledName"]],
        } for k in kernels],
    }
    if nvtx["text"] == "capture/diamond":
        assert len(launches) == 4 and len(kernels) == len(correlated) == 0
    elif nvtx["text"].startswith("ordinary/iteration_"):
        assert len(launches) == len(kernels) == len(correlated) == 4
        assert all(names[a["nameId"]].startswith("cudaLaunchKernel_") for a in launches)
        assert all(item["gpu_kernel_count"] == 1 for item in row["launch_correlations"])
    elif nvtx["text"].startswith("graph/iteration_"):
        assert len(launches) == 1 and len(kernels) == len(correlated) == 4
        assert names[launches[0]["nameId"]].startswith("cudaGraphLaunch_")
        assert len({k["graphNodeId"] for k in kernels}) == 4
        assert all(k["graphNodeId"] is not None for k in kernels)
    if "/iteration_" in nvtx["text"]:
        a = [k for k in kernels if "MulFunctor<float>" in names[k["demangledName"]]]
        branches = [k for k in kernels if "CUDAFunctorOnSelf_add<float>" in names[k["demangledName"]]]
        d = [k for k in kernels if "CUDAFunctor_add<float>" in names[k["demangledName"]]]
        assert len(a) == len(d) == 1 and len(branches) == 2
        assert a[0]["end"] <= min(k["start"] for k in branches)
        assert max(k["end"] for k in branches) <= d[0]["start"]
        row["dependency_timestamps_passed"] = True
        row["branch_overlap_ns"] = max(0, min(k["end"] for k in branches) - max(k["start"] for k in branches))
    ranges.append(row)
    print(f"{nvtx['text']}: {row['start_ms']:.6f}–{row['end_ms']:.6f} ms; "
          f"launches {len(launches)}; GPU kernels {len(kernels)}")

begin_capture = connection.execute(
    "SELECT r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
    "WHERE s.value LIKE 'cudaStreamBeginCapture_%'"
).fetchone()[0]
preparation = [k for k in all_kernels if k["end"] < begin_capture]
assert len(preparation) == 2
assert all("FillFunctor<long>" in names[k["demangledName"]] for k in preparation)
assert len([r for r in ranges if r["name"].startswith("ordinary/iteration_")]) == 3
assert len([r for r in ranges if r["name"].startswith("graph/iteration_")]) == 3

summary = {
    "source": f"{metadata['EXPORT_PRODUCT_NAME']} {metadata['EXPORT_PRODUCT_VERSION']}, exported SQLite",
    "device": checks["device"],
    "torch": checks["torch"],
    "cuda": checks["cuda"],
    "shape": [checks["numel"]],
    "dtype": checks["dtype"],
    "kernel_count": checks["kernel_count"],
    "not_benchmark": "Node-level profiling adds overhead; durations are trace coordinates, not benchmark measurements.",
    "correctness": {key: checks[key] for key in ("capture_left_output_unchanged", "final_value")},
    "timestamps": "Nanoseconds from Nsight Systems report origin; divide by 1e6 for timeline milliseconds.",
    "files": {name: {"sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
              for name in ("diamond.nsys-rep", "diamond.sqlite", "../profile.py", "../example.py", "extract_summary.py")},
    "capture_note": "PyTorch executes two FillFunctor<long> metadata-initialization kernels before cudaStreamBeginCapture; "
                    "capture/diamond contains 4 captured launch API calls and zero GPU kernels.",
    "ranges": ranges,
}
(folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("PASS: capture, launch correlations, four-node diamond timestamp dependencies and correctness.")
