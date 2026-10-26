"""MoE Kernel 视频实测：Modal 单卡 H100，Qwen3-30B-A3B。

用法：
  modal run scripts/moe-kernel/modal_measure.py::prepare   # CPU 容器：构建镜像、下载权重到 Volume、导出要用到的源码，不占卡
  modal run scripts/moe-kernel/modal_measure.py::measure   # 申请 H100，一次跑完全部矩阵
结果写到 public/measurements/moe-kernel/<时间戳>/，远端在 Volume moe-kernel-results 留一份。计划与预算见 docs/moe-kernel-notes.md。
"""
import json
import time
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-30B-A3B"
VLLM = "0.31.0"
MODEL_DIR = "/hf/Qwen3-30B-A3B"
SRC = Path(__file__).resolve()

app = modal.App("moe-kernel-measure")
hf = modal.Volume.from_name("moe-kernel-hf", create_if_missing=True)
out = modal.Volume.from_name("moe-kernel-results", create_if_missing=True)   # 结果在远端留一份，本地取回失败时不必重跑
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(f"vllm=={VLLM}", "datasets", "hf_transfer")
         .pip_install("accelerate")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))


def _env():
    import importlib.metadata as md
    import subprocess

    import torch

    e = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)}
    for p in ["vllm", "triton", "transformers"]:
        try:
            e[p] = md.version(p)
        except md.PackageNotFoundError:
            e[p] = None
    if torch.cuda.is_available():
        e["gpu"] = torch.cuda.get_device_name()
        e["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()
    return e


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/hf": hf})
def prepare_remote():
    import inspect
    import os
    import struct

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, local_dir=MODEL_DIR, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    hf.commit()
    r = {"env": _env(), "files": sorted((f, os.path.getsize(f"{MODEL_DIR}/{f}")) for f in os.listdir(MODEL_DIR)),
         "config": json.load(open(f"{MODEL_DIR}/config.json")), "src": {}, "sig": {}, "keys": []}
    first = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors"))[0]
    with open(f"{MODEL_DIR}/{first}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    r["keys"] = sorted((k, v["dtype"], v["shape"]) for k, v in head.items() if "layers.0." in k)

    def grab(name, obj):
        try:
            r["src"][name] = inspect.getsource(obj)
            if callable(obj) and not inspect.isclass(obj):
                r["sig"][name] = str(inspect.signature(obj))
        except Exception as e:
            r["src"][name] = "ERR " + repr(e)

    try:
        import vllm.model_executor.layers.fused_moe.fused_moe as fm
        for n in ["fused_experts", "fused_experts_impl", "get_default_config", "try_get_optimal_moe_config", "fused_moe_kernel",
                  "invoke_fused_moe_kernel", "_prepare_expert_assignment", "get_config_file_name", "get_moe_configs"]:
            if hasattr(fm, n):
                grab("vllm." + n, getattr(fm, n))
        r["fm_names"] = [n for n in dir(fm) if not n.startswith("__")]
        import vllm.model_executor.layers.fused_moe.moe_align_block_size as ma
        r["src"]["vllm.moe_align_block_size.py"] = inspect.getsource(ma)
        cdir = os.path.join(os.path.dirname(fm.__file__), "configs")
        r["configs"] = sorted(f for f in os.listdir(cdir) if "E=128" in f)
    except Exception as e:
        r["vllm_err"] = repr(e)
    try:
        import transformers.models.qwen3_moe.modeling_qwen3_moe as q
        r["src"]["hf.modeling_qwen3_moe"] = inspect.getsource(q)
    except Exception as e:
        r["hf_err"] = repr(e)
    return r


@app.local_entrypoint()
def prepare():
    r = prepare_remote.remote()
    d = SRC.parents[2] / "public/measurements/moe-kernel/prepare"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prepare.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(r["env"], r.get("vllm_err"), r.get("hf_err"))
    print("files:", len(r["files"]), sum(s for _, s in r["files"]) / 1e9, "GB")
    print("sig:", json.dumps(r["sig"], indent=1))
    print("configs:", r.get("configs"))
    for k in r["keys"]:
        print(k)


LAYER, STEP = 24, 20          # 片中跟的那一层与 decode 的那一步
NS = [1, 4, 16, 64, 256, 1024, 4096]
DEADLINE_S = 22 * 60          # 超过后不再启动新的测试项，保存已有结果
PROMPTS = [
    "用三句话解释什么是 Transformer。", "写一个 Python 函数，判断一个字符串是不是回文。", "为什么天空是蓝色的？",
    "Explain how a hash table handles collisions.", "把这句话翻译成英文：今天的会议改到下午三点。", "Write a haiku about autumn rain.",
    "列出学习线性代数的五个建议。", "What is the difference between TCP and UDP?", "用通俗的语言解释通货膨胀。",
    "Write a SQL query that returns the top 5 customers by total order amount.", "介绍一下长城的历史。",
    "How does garbage collection work in Java?", "给我一个三天的东京旅行计划。", "Summarize the plot of Romeo and Juliet in two sentences.",
    "解释一下什么是梯度下降。", "Write a bash one-liner that counts lines in all .py files.",
]


def _run(stamp, tiny):
    import inspect
    import os
    import random
    import statistics
    import traceback
    from collections import Counter

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dev = "cpu" if tiny else "cuda"
    t0 = time.time()
    log, R = [], {"errors": [], "layer": LAYER, "step": STEP}

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(f"[{time.time() - t0:7.1f}s] {s}")

    def save():
        R["log"], R["gpu_seconds"] = log, time.time() - t0
        os.makedirs(f"/out/{stamp}", exist_ok=True)
        json.dump(R, open(f"/out/{stamp}/results.json", "w"), ensure_ascii=False)
        out.commit()

    def section(name, fn):
        if time.time() - t0 > DEADLINE_S:
            R["errors"].append(f"{name}: skipped, past deadline")
            return
        try:
            fn()
        except Exception:
            R["errors"].append(f"{name}: {traceback.format_exc()[-1500:]}")
            say("ERROR in", name, R["errors"][-1][-400:])
        save()

    R["env"] = _env()
    say("env", R["env"])
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    if tiny:      # 同样的结构，hidden 缩到 64，随机权重：只用来在 CPU 上检查代码路径
        c = AutoConfig.from_pretrained(MODEL_DIR)
        c.hidden_size, c.moe_intermediate_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 64, 32, 128, 4, 2, 16
        model = AutoModelForCausalLM.from_config(c).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
        say(f"model loaded, {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")
    layers = model.model.layers
    cfg = model.config
    E, K, H = cfg.num_experts, cfg.num_experts_per_tok, cfg.hidden_size
    R["model"] = {"layers": len(layers), "E": E, "K": K, "H": H, "I": cfg.moe_intermediate_size}

    # 每层 MoE 的输入：记录 router 选中的 expert；第 LAYER 层另存输入本身
    cap = {"calls": None}

    def hook(i):
        def f(mod, args):
            if cap["calls"] is None:
                return
            x = args[0].detach()
            flat = x.reshape(-1, H) if cap["all_pos"] else x[:, -1]
            _, w, ids = mod.gate(flat)
            cap["calls"][i].append((ids.to(torch.int16).cpu(), w.float().cpu(), flat.clone() if i == LAYER else None))
        return f

    for i, l in enumerate(layers):
        l.mlp.register_forward_pre_hook(hook(i))

    def capture(fn, all_pos):
        cap["calls"], cap["all_pos"] = [[] for _ in layers], all_pos
        with torch.no_grad():
            r = fn()
        c, cap["calls"] = cap["calls"], None
        return r, c

    pool = {}

    def decode():
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False) for p in PROMPTS]
        enc = tok(msgs, return_tensors="pt", padding=True).to(dev)
        o, calls = capture(lambda: model.generate(**enc, max_new_tokens=32, min_new_tokens=32, do_sample=False), False)
        T = enc.input_ids.shape[1]
        ids, w, x = calls[LAYER][STEP]
        pool["hero"] = (x, ids.long().to(dev), w.to(dev))
        R["decode"] = {
            "prompts": PROMPTS, "texts": [tok.decode(s[T:], skip_special_tokens=True) for s in o],
            "tokens": [tok.decode(s[T + STEP - 1]) for s in o], "next": [tok.decode(s[T + STEP]) for s in o],
            "ids": ids.tolist(), "w": [[round(v, 4) for v in r] for r in w.tolist()],
            "probs0": [round(v, 5) for v in torch.softmax(torch.nn.functional.linear(x[0], layers[LAYER].mlp.gate.weight).float(), -1).tolist()],
            # 每层、每步选中的 expert：[layer][step][request][k]
            "all_ids": [[c[0].tolist() for c in calls[i]] for i in range(len(layers))],
        }
        cnt = Counter(v for r in ids.tolist() for v in r)
        say(f"decode: 16 requests, step {STEP}, layer {LAYER}: {len(cnt)} experts active, max {max(cnt.values())} tokens,",
            f"{sum(1 for v in cnt.values() if v <= 2)} experts with 1-2 tokens")
        say("tokens at this step:", R["decode"]["tokens"])

    def sweep():
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            paras = [t.strip() for t in ds["text"] if len(t.strip()) > 400]
            R["sweep_text"] = "wikitext-2-raw-v1 test"
        except Exception as e:
            paras = [(p + " ") * 40 for p in PROMPTS] * 8
            R["sweep_text"] = "fallback: repeated prompts (" + repr(e)[:200] + ")"
        say("sweep text:", R["sweep_text"], len(paras), "paragraphs")
        chunks = torch.stack([tok(p, return_tensors="pt").input_ids[0, :64] for p in paras
                              if tok(p, return_tensors="pt").input_ids.shape[1] >= 64][:128]).to(dev)
        _, c1 = capture(lambda: model.model(input_ids=chunks), False)
        long = tok("\n\n".join(paras), return_tensors="pt").input_ids[:, :4096].to(dev)
        _, c2 = capture(lambda: model.model(input_ids=long), True)
        dec = torch.stack([c1[i][0][0] for i in range(len(layers))]).long()     # [layer, 128, K]
        pre = torch.stack([c2[i][0][0] for i in range(len(layers))]).long()     # [layer, 4096, K]
        pool["dec"] = (c1[LAYER][0][2], dec[LAYER].to(dev), c1[LAYER][0][1].to(dev))
        pool["pre"] = (c2[LAYER][0][2], pre[LAYER].to(dev), c2[LAYER][0][1].to(dev))
        rng = random.Random(0)

        def active(ids, pick):          # ids [layer, n, K] → 各层激活 expert 数（对若干组取均值）
            acc = torch.zeros(len(layers))
            for idx in pick:
                sel = ids[:, idx].reshape(len(layers), -1)
                acc += torch.tensor([len(set(r.tolist())) for r in sel]).float()
            return (acc / len(pick)).tolist()

        R["sweep"] = {"decode": {}, "prefill": {}}
        for n in [1, 2, 4, 8, 16, 32, 64, 128]:
            a = active(dec, [rng.sample(range(dec.shape[1]), n) for _ in range(1 if n == dec.shape[1] else 20)])
            R["sweep"]["decode"][n] = {"per_layer": [round(v, 2) for v in a], "mean": sum(a) / len(a), "layer": a[LAYER]}
        for n in [16, 64, 256, 1024, 4096]:
            starts = [0] if n == pre.shape[1] else [rng.randrange(0, pre.shape[1] - n) for _ in range(20)]
            a = active(pre, [list(range(s, s + n)) for s in starts])
            R["sweep"]["prefill"][n] = {"per_layer": [round(v, 2) for v in a], "mean": sum(a) / len(a), "layer": a[LAYER]}
        R["sweep"]["decode_ids_layer"] = dec[LAYER].tolist()
        R["sweep"]["shapes"] = {"decode": list(dec.shape), "prefill": list(pre.shape)}
        for kind in ["decode", "prefill"]:
            say(kind, "active experts (layer %d / mean of layers):" % LAYER,
                {n: f"{v['layer']:.1f} / {v['mean']:.1f}" for n, v in R["sweep"][kind].items()})

    def timing():
        from torch.profiler import ProfilerActivity, profile

        import vllm.model_executor.layers.fused_moe.fused_moe as fm
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

        mlp = layers[LAYER].mlp
        ex = mlp.experts
        w1, w2 = ex.gate_up_proj.data, ex.down_proj.data
        say("w1", tuple(w1.shape), "w2", tuple(w2.shape), w1.dtype)
        params = inspect.signature(fm.fused_experts).parameters

        def inputs(n):
            x, ids, w = pool["hero"] if n == 16 else pool["dec"] if n <= 128 else pool["pre"]
            return x[:n].contiguous(), ids[:n].contiguous(), w[:n].contiguous()

        R["hf_default_impl"] = ex.config._experts_implementation
        say("transformers experts implementation (default):", R["hf_default_impl"])

        def hf(impl):             # transformers 的 Qwen3MoeExperts：eager 是逐个 expert 循环，grouped_mm 走 torch 的 grouped_mm
            def f(x, ids, w):
                ex.config._experts_implementation_internal = impl
                try:
                    return ex(x, ids, w.to(x.dtype))
                finally:
                    ex.config._experts_implementation_internal = R["hf_default_impl"]
            return f

        loop, grouped = hf("eager"), hf("grouped_mm")

        def fused(x, ids, w):
            kw = {"inplace": False} if "inplace" in params else {}
            return fm.fused_experts(x, w1, w2, w.float(), ids.to(torch.int32), **kw)

        def bench(fn, a, reps=50):
            for _ in range(5):
                fn(*a)
            ts = []
            for _ in range(reps):
                torch.cuda.synchronize()
                s = time.perf_counter()
                fn(*a)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - s) * 1e3)
            return {"median_ms": statistics.median(ts), "min_ms": min(ts), "p90_ms": sorted(ts)[int(reps * 0.9)]}

        def trace(fn, a):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn(*a)
                torch.cuda.synchronize()
            ks, rt = [], Counter()
            for e in prof.events():
                if "cuda" in str(e.device_type).lower():
                    ks.append([e.name, e.time_range.start, e.time_range.end])
                elif e.name.startswith("cuda"):
                    rt[e.name] += 1
            ks.sort(key=lambda k: k[1])
            base = ks[0][1] if ks else 0
            return {"kernels": [[k[0][:120], k[1] - base, k[2] - k[1]] for k in ks], "n_kernels": sum(1 for k in ks if "mem" not in k[0].lower()),
                    "runtime_calls": dict(rt), "table": prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60)}

        R["timing"], R["check"], R["trace"] = {}, {}, {}
        with torch.no_grad():
            for n in NS:
                a = inputs(n)
                ya, yb = loop(*a), fused(*a)
                R["check"][n] = {"fused_vs_loop_rel": ((ya.float() - yb.float()).abs().max() / ya.float().abs().max()).item()}
                if n <= 128:   # 循环实现与 transformers 这一层 MoE 模块的输出比较
                    ym = mlp(a[0][None])[0]
                    R["check"][n]["loop_vs_hf_rel"] = ((ya.float() - ym.float()).abs().max() / ym.float().abs().max()).item()
                R["timing"][n] = {"loop": bench(loop, a), "fused": bench(fused, a), "active": len(set(a[1].flatten().tolist()))}
                tl, tf = trace(loop, a), trace(fused, a)
                R["timing"][n].update(loop_kernels=tl["n_kernels"], fused_kernels=tf["n_kernels"],
                                      loop_runtime=tl["runtime_calls"], fused_runtime=tf["runtime_calls"])
                if n in (16, 1024):
                    R["trace"][n] = {"loop": tl, "fused": tf}
                try:              # 第三种实现：torch 的 grouped_mm（transformers 的另一条路径）
                    yg = grouped(*a)
                    R["check"][n]["grouped_vs_loop_rel"] = ((ya.float() - yg.float()).abs().max() / ya.float().abs().max()).item()
                    tg = trace(grouped, a)
                    R["timing"][n].update(grouped=bench(grouped, a), grouped_kernels=tg["n_kernels"], grouped_runtime=tg["runtime_calls"])
                    if n in (16, 1024):
                        R["trace"][n]["grouped"] = tg
                    say(f"        grouped_mm {R['timing'][n]['grouped']['median_ms']:.3f} ms, {tg['n_kernels']} kernels")
                except Exception as e:
                    R["errors"].append(f"grouped_mm N={n}: {e!r}"[:500])
                say(f"N={n:5d} active={R['timing'][n]['active']:3d}  loop {R['timing'][n]['loop']['median_ms']:.3f} ms, {tl['n_kernels']} kernels"
                    f"  |  fused {R['timing'][n]['fused']['median_ms']:.3f} ms, {tf['n_kernels']} kernels  |  rel err {R['check'][n]['fused_vs_loop_rel']:.2e}")
            say("trace (fused, N=16):\n" + R["trace"][16]["fused"]["table"])
            # vLLM 对各 token 数选用的配置，以及对 16 个 token 的排队结果
            R["config"] = {}
            for n in [1, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 4096]:
                R["config"][n] = dict(fm.try_get_optimal_moe_config(w1.shape, w2.shape, K, None, n))
            try:
                R["config_file"] = {"name": fm.get_config_file_name(E, w2.shape[2], None), "tuned_M": sorted(fm.get_moe_configs(E, w2.shape[2], None) or {})}
            except Exception as e:
                R["config_file"] = repr(e)
            say("config:", {n: (c.get("BLOCK_SIZE_M"), c.get("BLOCK_SIZE_N"), c.get("BLOCK_SIZE_K")) for n, c in R["config"].items()}, R["config_file"])
            x, ids, w = inputs(16)
            bm = R["config"][16]["BLOCK_SIZE_M"]
            s, e_ids, npad = moe_align_block_size(ids.to(torch.int32), bm, E)[:3]
            R["align16"] = {"block": bm, "num_tokens_post_padded": int(npad.item()), "sorted_token_ids": s[: int(npad.item())].tolist(),
                            "expert_ids": e_ids.tolist(), "sorted_len": int(s.numel())}
            say(f"moe_align_block_size (16 tokens, block {bm}): {int(npad.item())} rows after padding, {16 * K} real, grid {int(s.numel())} rows")

    section("decode", decode)
    section("sweep", sweep)
    if not tiny:
        section("timing", timing)
    say("done", f"{time.time() - t0:.0f} s", "errors:", len(R["errors"]))
    save()
    return R


@app.function(image=image, gpu="H100!", cpu=8, memory=98304, timeout=1800, max_containers=1, retries=0,
              volumes={"/hf": hf, "/out": out})
def measure_remote(stamp: str):
    return _run(stamp, False)


@app.function(image=image, cpu=8, memory=32768, timeout=1800, volumes={"/hf": hf, "/out": out})
def dryrun_remote():
    return _run("dryrun", True)


@app.local_entrypoint()
def dryrun():
    r = dryrun_remote.remote()
    print("\n".join(r["log"]))
    print("errors", r["errors"])


@app.local_entrypoint()
def measure():
    stamp = time.strftime("%Y%m%d-%H%M%S")
    r = measure_remote.remote(stamp)
    d = SRC.parents[2] / "public/measurements/moe-kernel" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(r, ensure_ascii=False))
    (d / "log.txt").write_text("\n".join(r["log"]))
    print("saved to", d, "gpu seconds", round(r["gpu_seconds"]), "errors", r["errors"])
