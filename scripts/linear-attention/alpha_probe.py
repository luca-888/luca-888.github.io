# Qwen3.5-0.8B 读一段文本时，各 GDN 层、各 head 在每个 token 上的衰减系数 α 与写入强度 β。只用 CPU，不占 GPU。
#   modal run scripts/linear-attention/alpha_probe.py
#   -> public/measurements/linear-attention/<时间>/results.json（token、α、β、环境）
#   -> videos/linear-attention/data.js（视频读取的同一份数据）
# 做法：Modal 上 CPU、FP32 前向，走 transformers 的纯 PyTorch 实现（torch_chunk_gated_delta_rule）。
# α_t = exp(g_t)，g_t = −exp(A_log) · softplus(a_t + dt_bias)；β_t = sigmoid(b_t)。两者在调用 kernel 前算好、作为参数传入，
# 这里替换模块里的 torch_chunk_gated_delta_rule，调用时记下 g 与 beta，再原样调用。结果先写入 Volume，再返回本地。
import json
import pathlib
import time

import modal

MODEL = "Qwen/Qwen3.5-0.8B"
TEXT = "小明把钥匙放在厨房的抽屉里，然后出门上班了。下午他回到家，却怎么也找不到钥匙。最后他想起来，钥匙就在抽屉里。"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("transformers", "safetensors", "huggingface_hub", "accelerate"))
vol = modal.Volume.from_name("linear-attention-measure", create_if_missing=True)
app = modal.App("linear-attention-alpha", image=image)


@app.function(cpu=4, memory=16384, timeout=1800, volumes={"/out": vol})
def probe(stamp: str):
    import platform

    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    captured = []
    orig = M.torch_chunk_gated_delta_rule

    def hook(query, key, value, g, beta, **kw):
        captured.append((g[0].detach().float(), beta[0].detach().float()))   # [T, H]
        return orig(query, key, value, g=g, beta=beta, **kw)

    M.torch_chunk_gated_delta_rule = hook
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = M.Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config.text_config
    ids = tok(TEXT, return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        model(input_ids=ids)
    elapsed = time.time() - t0

    gdn_layers = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    assert len(captured) == len(gdn_layers), (len(captured), len(gdn_layers))
    out = {
        "model": MODEL, "text": TEXT, "tokens": [tok.decode([i]) for i in ids[0].tolist()], "gdn_layers": gdn_layers,
        "num_heads": int(captured[0][0].shape[1]),
        "alpha": [[[round(float(x), 5) for x in row] for row in g.exp().T] for g, _ in captured],   # [层][head][token]
        "beta": [[[round(float(x), 5) for x in row] for row in b.T] for _, b in captured],
        "env": {"where": "Modal CPU (cpu=4)", "dtype": "float32", "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "forward_seconds": round(elapsed, 2)},
    }
    pathlib.Path(f"/out/{stamp}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/out/{stamp}/results.json").write_text(json.dumps(out, ensure_ascii=False))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = probe.remote(stamp)
    d = root / "public/measurements/linear-attention" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(out, ensure_ascii=False))
    (root / "videos/linear-attention/data.js").write_text(
        "// 由 scripts/linear-attention/alpha_probe.py 生成：Qwen3.5-0.8B 各 GDN 层、各 head 在每个 token 上的 α 与 β（Modal CPU、FP32 实测）\n"
        f"const ALPHA_DATA = {json.dumps(out, ensure_ascii=False)};\n")
    print(f"{len(out['tokens'])} tokens, {len(out['gdn_layers'])} GDN layers × {out['num_heads']} heads, forward {out['env']['forward_seconds']}s -> {d}")
