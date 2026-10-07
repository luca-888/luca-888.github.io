# Qwen3.8-27B（BF16）真实权重的分布，以及几种“档位放法”在真实权重上的舍入误差。只用 CPU，不占 GPU。
#   modal run scripts/llm-quantization/weight_hist.py      -> public/measurements/llm-quantization/weights.json
# 做法：不下载整个 checkpoint。读每个 safetensors 分片的 header，再用 HTTP Range 只取选中的几个 tensor。
# 选第 0、31、63 层里名字带 mlp 的二维权重，外加第 31 层的其余二维权重（attention 等）。
# 误差口径：只做 RTN（取最近的一档），不做 GPTQ / AWQ；相对误差 = RMS(w − q) / std(w)。
import json
import pathlib

import modal

REPO = "Qwen/Qwen3.8-27B"
LAYERS = (0, 31, 63)
image = modal.Image.debian_slim().pip_install("numpy", "huggingface_hub", "requests")
app = modal.App("quant-weight-hist", image=image)


@app.function(cpu=4, memory=16384, timeout=1800)
def probe():
    import re

    import numpy as np
    import requests
    from huggingface_hub import hf_hub_url, list_repo_files

    def rng(url, a, b):
        r = requests.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=600)
        r.raise_for_status()
        return r.content

    index = {}
    for f in sorted(x for x in list_repo_files(REPO) if x.endswith(".safetensors")):
        url = hf_hub_url(REPO, f)
        n = int.from_bytes(rng(url, 0, 8), "little")
        for name, meta in json.loads(rng(url, 8, 8 + n)).items():
            if name != "__metadata__":
                index[name] = (url, 8 + n, meta)

    def layer_of(name):
        m = re.search(r"layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    two_d = {k: v for k, v in index.items() if len(v[2]["shape"]) == 2 and v[2]["dtype"] == "BF16" and "visual" not in k and "vision" not in k}
    picked = [k for k in two_d if layer_of(k) in LAYERS and "mlp" in k] + [k for k in two_d if layer_of(k) == 31 and "mlp" not in k]

    e2m1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    fp8 = [0.0] + [m / 8 * 2.0**-6 for m in range(1, 8)] + [(1 + m / 8) * 2.0 ** (e - 7) for e in range(1, 16) for m in range(8)]
    fp8 = np.array([v for v in fp8 if v <= 448], dtype=np.float32)

    def nearest(levels, a):  # a >= 0，levels 升序
        mid = (levels[1:] + levels[:-1]) / 2
        return levels[np.searchsorted(mid, a)]

    def rel(w, q, std):
        return float(np.sqrt(np.mean((w - q) ** 2)) / std)

    def int_q(w, scale, lo, hi):
        return np.clip(np.rint(w / scale), lo, hi) * scale

    out = []
    for name in picked:
        url, base, meta = two_d[name]
        a, b = meta["data_offsets"]
        raw = np.frombuffer(rng(url, base + a, base + b), dtype="<u2")
        w = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
        std, amax = float(w.std()), float(np.abs(w).max())
        absw = np.abs(w)
        g128 = absw.reshape(w.shape[0], -1, 128).max(-1, keepdims=True).repeat(128, -1).reshape(w.shape)
        g16 = absw.reshape(w.shape[0], -1, 16).max(-1, keepdims=True).repeat(16, -1).reshape(w.shape)
        g16 = np.maximum(g16, 1e-12)
        g128 = np.maximum(g128, 1e-12)
        counts, _ = np.histogram(w, bins=161, range=(-8 * std, 8 * std))
        out.append({
            "name": name, "shape": meta["shape"], "std": std, "absmax": amax, "absmax_over_std": amax / std,
            "abs_percentiles": {str(p): float(np.percentile(absw[::7, ::7], p)) for p in (50, 90, 99, 99.9, 99.99)},
            "frac_within_absmax_16th": float((absw < amax / 16).mean()),
            "hist_range_std": 8, "hist": counts.tolist(), "outside_hist": int(w.size - counts.sum()),
            "rel_err": {
                "int8_per_tensor": rel(w, int_q(w, amax / 127, -127, 127), std),
                "fp8_per_tensor": rel(w, np.sign(w) * nearest(fp8, absw / (amax / 448)) * (amax / 448), std),
                "int4_per_tensor": rel(w, int_q(w, amax / 7, -8, 7), std),
                "int4_group128": rel(w, int_q(w, g128 / 7, -8, 7), std),
                "nvfp4_block16": rel(w, np.sign(w) * nearest(e2m1, absw / (g16 / 6)) * (g16 / 6), std),
            },
        })
    return {"repo": REPO, "n_tensors_total": len(index), "layers": LAYERS, "tensors": out}


@app.local_entrypoint()
def main():
    res = probe.remote()
    path = pathlib.Path("public/measurements/llm-quantization/weights.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for t in res["tensors"]:
        e = t["rel_err"]
        print(f'{t["name"]:70s} max/std {t["absmax_over_std"]:7.1f}  <max/16: {t["frac_within_absmax_16th"]:.4f}  '
              f'int8 {e["int8_per_tensor"]:.3f} fp8 {e["fp8_per_tensor"]:.3f} int4/tensor {e["int4_per_tensor"]:.3f} int4/128 {e["int4_group128"]:.3f} nvfp4/16 {e["nvfp4_block16"]:.3f}')
