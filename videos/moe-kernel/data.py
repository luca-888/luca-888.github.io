"""python3 data.py -> data.js：从实测结果抽出影片要用的数据（H100，vLLM 0.31.0，2026-10-08，第 24 层）。
原始数据：public/measurements/moe-kernel/20261008-105322/results.json，测量方法见 docs/moe-kernel-notes.md。"""
import json, re
from pathlib import Path

here = Path(__file__).resolve().parent
r = json.loads((here / "../../public/measurements/moe-kernel/20261008-105322/results.json").read_text())
d = r["decode"]

def short(name):
    for k, v in [("moe_align_block_size", "align"), ("count_and_sort", "count_sort"), ("fused_moe_kernel", "fused_moe_kernel"), ("act_and_mul", "silu_and_mul"),
                 ("moe_sum", "moe_sum"), ("copy", "copy"), ("gemm", "GEMM"), ("Gemm", "GEMM"), ("cutlass", "GEMM"), ("nvjet", "GEMM"), ("Memcpy", "Memcpy DtoH")]:
        if k in name: return v
    return "other"

def kernels(ks): return [[short(n), round(t, 2), round(dt, 2)] for n, t, dt in ks]
loop = r["trace"]["16"]["loop"]["kernels"]
out = {
    "layer": r["layer"], "step": r["step"], "model": r["model"],
    "ids": d["ids"], "w": d["w"], "probs0": d["probs0"], "next": d["next"],
    "active": {m: {int(n): round(v["layer"], 1) for n, v in r["sweep"][m].items() if isinstance(v, dict) and "layer" in v} for m in ("decode", "prefill")},
    "timing": {int(n): {k: round(v[k]["median_ms"], 3) for k in ("loop", "grouped", "fused")} | {"active": v["active"], "loop_kernels": v["loop_kernels"]} for n, v in r["timing"].items()},
    "fused16": kernels(r["trace"]["16"]["fused"]["kernels"]), "fused1024": kernels(r["trace"]["1024"]["fused"]["kernels"]),
    "loop16": kernels(loop), "loop16_calls": r["trace"]["16"]["loop"]["runtime_calls"],
    "config": {int(n): [c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"]] for n, c in r["config"].items()},
    "align16": {k: r["align16"][k] for k in ("block", "num_tokens_post_padded", "sorted_token_ids", "expert_ids")},
}
(here / "data.js").write_text("// 由 data.py 生成：H100 实测（vLLM 0.31.0，第 24 层），原始数据见 public/measurements/moe-kernel/20261008-105322/\nconst DATA = "
                              + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
print("ok", len(out["loop16"]), "loop kernels;", out["fused16"])
from collections import Counter; print(Counter(k[0] for k in out["loop16"]))
