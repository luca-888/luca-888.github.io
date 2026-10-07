"""Fused Linear Cross Entropy 视频实测：Modal 单卡 H100。

用法（modal 在 ~/.venvs/modal/bin）：
  modal run scripts/fused-linear-cross-entropy/modal_measure.py::prepare   # CPU 容器：构建镜像、取 config，不占卡
  modal run scripts/fused-linear-cross-entropy/modal_measure.py::measure   # 申请 H100，一次跑完全部矩阵
结果写到 public/measurements/fused-linear-cross-entropy/<时间戳>/。计划与预算见 docs/fused-linear-cross-entropy-notes.md。
"""
import json
import os
import time

import modal

LIGER_COMMIT = "b5cdbf7e346d0d4da7aff36f2b219be5b01179c0"
H, V = 2048, 151936  # Qwen3-1.7B 的 lm_head，与端到端测试同一个模型
BTS = [4096, 8192, 16384, 32768, 65536]
METHODS = ["eager", "liger_ce", "flce", "cce"]
SWEEP_BT, SWEEP_C = 16384, [1, 2, 4, 8, 16, 32, 128]  # chunk 256 → 16384（不切分）
E2E_MODEL = "Qwen/Qwen3-1.7B"
E2E_SEQS = [4096, 8192, 16384, 32768, 65536, 131072]
TIMELINE_SEQ = 16384
DEADLINE_S = 25 * 60  # 超过后不再启动新测试项，保存已有结果

app = modal.App("flce-measure")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "transformers", "huggingface_hub", "cut-cross-entropy")
    .pip_install(f"liger-kernel[cutedsl] @ git+https://github.com/linkedin/Liger-Kernel@{LIGER_COMMIT}")
)


def _env():
    import importlib.metadata as md
    import subprocess

    import torch

    out = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda)}
    for p in ["triton", "transformers", "liger-kernel", "cut-cross-entropy", "nvidia-cutlass-dsl"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["liger_commit"] = LIGER_COMMIT
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
        out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out


def _loss_fn(name, c=1):
    import torch.nn.functional as F

    if name == "eager":  # HF ForCausalLMLoss 的做法：logits 升到 FP32 再算 CE
        return lambda x, w, y: F.cross_entropy((x @ w.t()).float(), y)
    if name == "eager_bf16":
        return lambda x, w, y: F.cross_entropy(x @ w.t(), y)
    if name == "liger_ce":
        from liger_kernel.transformers.functional import liger_cross_entropy

        return lambda x, w, y: liger_cross_entropy(x @ w.t(), y)
    if name == "flce":  # ce_impl=None：内层走 Triton CE kernel
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

        return lambda x, w, y: liger_fused_linear_cross_entropy(x, w, y, chunk_mem_const=c)
    if name == "flce_sm90":
        from liger_kernel.ops.cutedsl.ops.fused_linear_cross_entropy_sm90 import liger_fused_linear_cross_entropy_sm90

        return lambda x, w, y: liger_fused_linear_cross_entropy_sm90(x, w, y)
    if name == "cce":
        from cut_cross_entropy import linear_cross_entropy

        return lambda x, w, y: linear_cross_entropy(x, w, y)
    raise ValueError(name)


def _inputs(bt, seed=0, h=H, v=V):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    # 量级接近真实模型：hidden 经过 final norm，权重 std ≈ 0.02
    x = torch.randn(bt, h, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_()
    w = (torch.randn(v, h, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02).requires_grad_()
    y = torch.randint(0, v, (bt,), device="cuda", generator=g)
    return x, w, y


def _bench(fn, x, w, y, warmup=3, iters=10):
    import torch

    def step():
        x.grad = w.grad = None
        fn(x, w, y).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    # 显存：输入与参数之外，一步 forward + backward 的峰值增量（含 dX、dW 本身）
    x.grad = w.grad = None
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    x.grad = w.grad = None
    return {"peak_bytes": peak, "ms_median": times[len(times) // 2], "ms_all": times}


def _run_case(fn, bt, h=H, v=V):
    import torch

    x, w, y = _inputs(bt, h=h, v=v)
    try:
        return _bench(fn, x, w, y)
    except torch.OutOfMemoryError as e:
        return {"oom": True, "error": str(e)[:200]}
    except Exception as e:  # 实现不支持该形状等
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        del x, w, y
        torch.cuda.empty_cache()


def correctness():
    import torch

    bt = 4096
    x, w, y = _inputs(bt, seed=1)
    xr, wr = x.detach().float().requires_grad_(), w.detach().float().requires_grad_()
    ref = torch.nn.functional.cross_entropy(xr @ wr.t(), y)
    ref.backward()
    out = {"ref_loss": ref.item()}
    for m in ["eager", "eager_bf16", "liger_ce", "flce", "cce"]:
        try:
            x.grad = w.grad = None
            loss = _loss_fn(m)(x, w, y)
            loss.backward()

            def rel(a, b):
                return ((a.float() - b).norm() / b.norm()).item()

            out[m] = {"loss": loss.item(), "loss_abs_err": abs(loss.item() - ref.item()),
                      "dx_rel_err": rel(x.grad, xr.grad), "dw_rel_err": rel(w.grad, wr.grad)}
        except Exception as e:
            out[m] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    del x, w, y, xr, wr
    torch.cuda.empty_cache()
    return out


def kernel_breakdown(bt=SWEEP_BT):
    """一次 forward + backward 里各类 kernel 的 GPU 时间（ms）。"""
    import torch
    from torch.profiler import ProfilerActivity, profile

    res = {}
    for m in ["eager", "flce"]:
        x, w, y = _inputs(bt)
        fn = _loss_fn(m)
        for _ in range(2):
            x.grad = w.grad = None
            fn(x, w, y).backward()
        torch.cuda.synchronize()
        x.grad = w.grad = None
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(x, w, y).backward()
            torch.cuda.synchronize()
        kernels = {}
        for ev in prof.events():
            if ev.device_type == torch.autograd.DeviceType.CUDA:
                kernels.setdefault(ev.name, [0, 0.0])
                kernels[ev.name][0] += 1
                kernels[ev.name][1] += ev.device_time_total / 1000
        res[m] = sorted(([k, n, round(t, 3)] for k, (n, t) in kernels.items()), key=lambda r: -r[2])
        del x, w, y
        torch.cuda.empty_cache()
    return res


def _e2e_model():
    import torch
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(E2E_MODEL)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = AutoModel.from_config(cfg, dtype=torch.bfloat16).cuda()  # 随机初始化，显存与权重值无关
    model.gradient_checkpointing_enable()
    model.train()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16, device="cuda")
    if cfg.tie_word_embeddings:
        lm_head.weight = model.embed_tokens.weight
    params = list(model.parameters()) + ([] if cfg.tie_word_embeddings else [lm_head.weight])
    opt = torch.optim.AdamW(params, lr=1e-5, fused=True)  # 纯 BF16 参数与状态；第一步之后状态常驻
    return cfg, model, lm_head, opt


def _e2e_step(cfg, model, lm_head, opt, ids, method):
    import torch.nn.functional as F

    h = model(input_ids=ids).last_hidden_state[:, :-1].reshape(-1, cfg.hidden_size)
    labels = ids[:, 1:].reshape(-1)
    if method == "eager":
        loss = F.cross_entropy(lm_head(h).float(), labels)
    else:
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

        loss = liger_fused_linear_cross_entropy(h, lm_head.weight, labels)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def end_to_end(deadline):
    import torch

    cfg, model, lm_head, opt = _e2e_model()
    out = {"model": E2E_MODEL, "hidden": cfg.hidden_size, "vocab": cfg.vocab_size, "layers": cfg.num_hidden_layers,
           "params_bytes": sum(p.numel() * p.element_size() for p in model.parameters())
           + (0 if cfg.tie_word_embeddings else lm_head.weight.numel() * 2),
           "runs": {}}
    for method in ["eager", "flce"]:
        for seq in E2E_SEQS:
            if time.time() > deadline:
                out["runs"][f"{method}/{seq}"] = {"skipped": "deadline"}
                continue
            ids = torch.randint(0, cfg.vocab_size, (1, seq), device="cuda")
            try:
                _e2e_step(cfg, model, lm_head, opt, ids, method)  # warmup
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                _e2e_step(cfg, model, lm_head, opt, ids, method)
                torch.cuda.synchronize()
                out["runs"][f"{method}/{seq}"] = {"peak_bytes": torch.cuda.max_memory_allocated(),
                                                  "base_bytes": base, "step_s": time.perf_counter() - t0}
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                out["runs"][f"{method}/{seq}"] = {"oom": True}
            del ids
            torch.cuda.empty_cache()
            print(method, seq, out["runs"][f"{method}/{seq}"], flush=True)
    # 一步训练的显存时间线（allocated 总量随时间变化）
    out["timeline"] = {}
    for method in ["eager", "flce"]:
        ids = torch.randint(0, cfg.vocab_size, (1, TIMELINE_SEQ), device="cuda")
        _e2e_step(cfg, model, lm_head, opt, ids, method)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(max_entries=2_000_000)
        _e2e_step(cfg, model, lm_head, opt, ids, method)
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        events = [e for e in snap["device_traces"][0] if e["action"] in ("alloc", "free_completed")]
        start = torch.cuda.memory_allocated() - sum(
            (e["size"] if e["action"] == "alloc" else -e["size"]) for e in events)
        cur, pts = start, []
        t0 = events[0]["time_us"] if events else 0
        for e in events:
            cur += e["size"] if e["action"] == "alloc" else -e["size"]
            pts.append([e["time_us"] - t0, cur])
        # 降采样到约 600 个点，保留每段的最大值
        step = max(1, len(pts) // 600)
        ds = [max(pts[i:i + step], key=lambda p: p[1]) for i in range(0, len(pts), step)]
        big = sorted(({"size": e["size"], "t": e["time_us"] - t0,
                       "frame": next((f["name"] for f in e.get("frames", []) if "cross_entropy" in f["name"]
                                      or "linear" in f["name"] or "mm" in f["name"]), None)}
                      for e in events if e["action"] == "alloc" and e["size"] >= 256 << 20), key=lambda d: d["t"])
        out["timeline"][method] = {"seq": TIMELINE_SEQ, "start_bytes": start, "points": ds, "big_allocs": big[:60]}
        del ids, snap
        torch.cuda.empty_cache()
    return out


@app.function(image=image, timeout=1800, cpu=2, memory=8192)
def prepare_remote():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(E2E_MODEL)
    import liger_kernel.ops.fused_linear_cross_entropy  # noqa: F401  确认能 import
    from cut_cross_entropy import linear_cross_entropy  # noqa: F401

    # 返回 JSON 字符串：本地没有 torch，不能反序列化 torch 类型
    return json.dumps({"env": _env(), "qwen3": {"hidden": cfg.hidden_size, "vocab": cfg.vocab_size,
                                                "layers": cfg.num_hidden_layers, "tie": cfg.tie_word_embeddings}})


@app.function(image=image, gpu="H100", timeout=30 * 60, cpu=8, memory=65536, retries=0, max_containers=1)
def measure_remote():
    import torch

    deadline = time.time() + DEADLINE_S
    res = {"env": _env(), "shape": {"H": H, "V": V}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def section(name, f):
        if time.time() > deadline:
            res[name] = {"skipped": "deadline"}
            return
        t0 = time.time()
        try:
            res[name] = f()
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:500]}"}
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)

    section("correctness", correctness)

    def micro():
        out = {}
        for m in METHODS:
            fn = _loss_fn(m)
            for bt in BTS:
                if time.time() > deadline:
                    break
                out[f"{m}/{bt}"] = _run_case(fn, bt)
                print(m, bt, {k: v for k, v in out[f"{m}/{bt}"].items() if k != "ms_all"}, flush=True)
        return out

    section("micro", micro)
    section("sweep", lambda: {str(c): _run_case(_loss_fn("flce", c), SWEEP_BT) for c in SWEEP_C})
    section("kernels", kernel_breakdown)
    # CuTe SM90 实现要求 V 是 256 的倍数（151936 不满足），在它 docstring 的形状上单独比较
    section("sm90_shape", lambda: {m: _run_case(_loss_fn(m), 4096, 4096, 128256)
                                   for m in ["eager", "flce", "flce_sm90", "cce"]})
    section("e2e", lambda: end_to_end(deadline))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return json.dumps(res, default=str)


@app.local_entrypoint()
def prepare():
    print(json.dumps(json.loads(prepare_remote.remote()), indent=2))


@app.local_entrypoint()
def measure():
    t0 = time.time()
    res = json.loads(measure_remote.remote())
    res["client_wall_s"] = time.time() - t0
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.join(os.path.dirname(__file__), "../../public/measurements/fused-linear-cross-entropy", stamp)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.abspath(out), f"wall {res['client_wall_s']:.0f}s")
