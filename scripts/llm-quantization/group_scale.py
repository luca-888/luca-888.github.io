# Qwen3.8-27B 第 0 层 down_proj：每组一个 scale 时，各组的 scale 有多大（视频第 6 幕用）。只用 CPU。
#   modal run scripts/llm-quantization/group_scale.py      -> public/measurements/llm-quantization/group-scales.json
# 分组沿一行连续取（与 weight_hist.py 相同）。INT4：scale = 组内绝对值最大 ÷ 7；NVFP4：÷ 6（FP4 最大的刻度是 6）。
import json
import pathlib

import modal

REPO, NAME = "Qwen/Qwen3.8-27B", "model.language_model.layers.0.mlp.down_proj.weight"
image = modal.Image.debian_slim().pip_install("numpy", "huggingface_hub", "requests")
app = modal.App("quant-group-scale", image=image)


@app.function(cpu=4, memory=16384, timeout=1800)
def probe():
    import numpy as np
    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url

    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    url = hf_hub_url(REPO, idx["weight_map"][NAME])
    rng = lambda a, b: requests.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=600).content
    n = int.from_bytes(rng(0, 8), "little")
    meta = json.loads(rng(8, 8 + n))[NAME]
    a, b = meta["data_offsets"]
    raw = np.frombuffer(rng(8 + n + a, 8 + n + b), dtype="<u2")
    w = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    absw = np.abs(w)
    r, c = np.unravel_index(int(absw.argmax()), w.shape)

    def stats(g, div):
        gm = absw.reshape(w.shape[0], -1, g).max(-1)
        k = c // g
        grp = w[r, k * g:(k + 1) * g]
        rest = np.sort(np.abs(grp))[::-1]
        return {
            "group": g, "divisor": div,
            "lamp_group": {"row": int(r), "col0": int(k * g), "absmax": float(rest[0]), "second_absmax": float(rest[1]),
                           "median_abs": float(np.median(rest)), "scale": float(rest[0] / div)},
            "group_absmax_percentiles": {str(p): float(np.percentile(gm, p)) for p in (1, 10, 50, 90, 99, 99.99)},
            "median_scale": float(np.median(gm) / div),
            "frac_groups_scale_over_10x_median": float((gm > 10 * np.median(gm)).mean()),
        }

    return {"repo": REPO, "tensor": NAME, "shape": meta["shape"], "absmax": float(absw.max()), "argmax": [int(r), int(c)],
            "per_tensor_int4_scale": float(absw.max() / 7), "int4_g128": stats(128, 7), "nvfp4_g16": stats(16, 6)}


@app.local_entrypoint()
def main():
    res = probe.remote()
    path = pathlib.Path("public/measurements/llm-quantization/group-scales.json")
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
