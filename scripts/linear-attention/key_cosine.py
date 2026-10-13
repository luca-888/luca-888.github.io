# Qwen3.5-0.8B 各 GDN 层、各 head 中，L2 归一化后的 key 两两之间的余弦相似度，与同维度随机单位向量对比。只用 CPU，不占 GPU。
#   modal run scripts/linear-attention/key_cosine.py
#   -> public/measurements/linear-attention/<时间>-keys/results.json（每层每个 head 的 32 × 32 余弦矩阵、统计、环境）
#   -> videos/linear-attention/data-keys.js（视频用的摘要：选中 head 的矩阵与各层统计）
# 做法同 alpha_probe.py：替换 torch_chunk_gated_delta_rule，调用时记下传入的 key（kernel 内才做 L2 归一化，这里先归一化再算余弦）。
# 文本与 alpha_probe.py 相同。结果先写入 Volume，再返回本地。
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
app = modal.App("linear-attention-keys", image=image)


@app.function(cpu=4, memory=16384, timeout=1800, volumes={"/out": vol})
def probe(stamp: str):
    import platform

    import torch
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    keys = []
    orig = M.torch_chunk_gated_delta_rule

    def hook(query, key, value, g, beta, **kw):
        keys.append(torch.nn.functional.normalize(key[0].detach().float(), dim=-1))   # [T, H, d]
        return orig(query, key, value, g=g, beta=beta, **kw)

    M.torch_chunk_gated_delta_rule = hook
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = M.Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config.text_config
    ids = tok(TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        model(input_ids=ids)
    T = ids.shape[1]
    off = ~torch.eye(T, dtype=torch.bool)
    cos = [torch.einsum("thd,shd->hts", k, k) for k in keys]                         # [H, T, T]
    # 同维度随机单位向量的对照：T 个向量两两余弦的绝对值均值
    d = keys[0].shape[-1]
    rnd = torch.nn.functional.normalize(torch.randn(2000, T, d), dim=-1)
    rnd_abs = float(torch.einsum("ntd,nsd->nts", rnd, rnd)[:, off].abs().mean())
    out = {
        "model": MODEL, "text": TEXT, "tokens": [tok.decode([i]) for i in ids[0].tolist()],
        "gdn_layers": [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"], "head_dim": d,
        "cos": [[[[round(float(x), 3) for x in row] for row in h] for h in c] for c in cos],      # [层][head][T][T]
        "mean_abs_offdiag": [[round(float(h[off].abs().mean()), 4) for h in c] for c in cos],     # [层][head]
        "mean_offdiag": [[round(float(h[off].mean()), 4) for h in c] for c in cos],
        "random_mean_abs": round(rnd_abs, 4),
        "env": {"where": "Modal CPU (cpu=4)", "dtype": "float32", "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version()},
    }
    pathlib.Path(f"/out/{stamp}").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/out/{stamp}/results.json").write_text(json.dumps(out, ensure_ascii=False))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-keys"
    out = probe.remote(stamp)
    d = root / "public/measurements/linear-attention" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(out, ensure_ascii=False))
    m = [x for L in out["mean_abs_offdiag"] for x in L]
    print(f"{len(out['tokens'])} tokens; |cos| off-diagonal per head: min {min(m):.3f} median {sorted(m)[len(m) // 2]:.3f} max {max(m):.3f}; random {out['random_mean_abs']:.3f} -> {d}")
