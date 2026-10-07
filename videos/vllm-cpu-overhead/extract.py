"""从 vLLM engine 实测 trace（H100 · Qwen3-8B · 并发 8）取连续几个稳态 decode step，导出视频用的数据。

每步按 kernel 名切成语义阶段（µs，相对第一步 GPU 注释开始）：
- prep：输入准备的零星小 kernel（MRV2 在 GPU 上生成 positions、block table 等）
- layers：37 个边界，切出 36 层；以每层唯一的 reshape_and_cache kernel 为锚点
- lm：lm_head GEMM；samp：sampling 到 _post_update 结束
另输出 gaps（GPU 上没有任何 kernel 的区间，> 60 µs）、CPU execute_model 区间，
以及每个 execute_model 内 cudaGraphLaunch 的起止（graph），用于把空转拆成处理输出 / 准备输入 / 提交三段。
用法：python videos/vllm-cpu-overhead/extract.py
"""
import glob, gzip, json, re

RUN = "public/measurements/vllm-cpu-overhead/20260930-094235"
PAT = re.compile(r"execute_context_0\(0\)_generation_8\(8\)")   # 8 个 decode 请求的纯 decode step
FIRST, N = 20, 4                                                  # 取稳态中段的连续 4 步
IDLE_MIN = 60


def step_phases(ks):
    """ks：一步内的 kernel [(start, end, name)]，按时间排序。"""
    emb = next(i for i, k in enumerate(ks) if "embedding" in k[2])
    anchors = [k[0] for k in ks if "reshape_and_cache" in k[2]]
    assert len(anchors) == 36, len(anchors)
    off = anchors[0] - ks[emb][0]
    lm_i = max((i for i, k in enumerate(ks) if k[0] > anchors[-1]), key=lambda i: ks[i][1] - ks[i][0])
    post = next(i for i, k in enumerate(ks) if "_post_update" in k[2])
    layers = [a - off for a in anchors] + [ks[lm_i - 1][1]]            # 最后一层到 lm_head 前的 gather 为止
    return {
        "prep": [[k[0], k[1]] for k in ks[:emb]],
        "layers": layers,
        "lm": [ks[lm_i][0], ks[lm_i][1]],
        "samp": [ks[lm_i][1], ks[post][1]],
    }


def extract(cfg):
    path = sorted(glob.glob(f"{RUN}/{cfg}/qwen3-8b/traces/rank0.*"))[0]   # 第一个 trace 是并发 8
    ev = json.load(gzip.open(path))["traceEvents"]
    gpu = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and PAT.fullmatch(e["name"])), key=lambda e: e["ts"])
    cpu = sorted((e for e in ev if e.get("cat") == "user_annotation" and PAT.fullmatch(e["name"])), key=lambda e: e["ts"])
    g, c = gpu[FIRST:FIRST + N + 1], cpu[FIRST:FIRST + N + 1]
    t0, t1 = g[0]["ts"], g[N]["ts"]                  # 窗口：第 1 步 GPU 开始到第 N+1 步 GPU 开始
    r = lambda x: round(x - t0, 1)
    kern = sorted((e["ts"], min(e["ts"] + e["dur"], t1), e["name"]) for e in ev
                  if e.get("cat") == "kernel" and t0 <= e["ts"] < t1)
    steps = []
    for i in range(N):
        lo, hi = g[i]["ts"], g[i + 1]["ts"]
        ph = step_phases([k for k in kern if lo <= k[0] < hi])
        steps.append({k: ([[r(a), r(b)] for a, b in v] if k == "prep" else [r(x) for x in v]) for k, v in ph.items()})
    gl = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch")
    graph = []
    for x in c:
        inside = [g_ for g_ in gl if x["ts"] <= g_[0] <= x["ts"] + x["dur"]]
        graph.append([r(inside[0][0]), r(inside[0][1])] if inside else None)
    gaps, end = [], t0
    for s, e, _ in kern:
        if s - end > IDLE_MIN: gaps.append([r(end), r(s)])
        end = max(end, e)
    if t1 - end > IDLE_MIN: gaps.append([r(end), r(t1)])
    busy = sum(e - s for s, e, _ in kern)
    return {"steps": steps, "gaps": gaps, "cpu": [[r(x["ts"]), r(x["ts"] + x["dur"])] for x in c], "graph": graph,
            "window": r(t1), "busy": round(busy, 1), "kernels_per_step": round(len(kern) / N)}


out = {k: extract(k) for k in ("no_async", "default")}
for k, v in out.items():
    print(k, "window", v["window"], "µs, busy", v["busy"], "kernels/step", v["kernels_per_step"],
          "idle/step", round((v["window"] - v["busy"]) / N, 1))
with open("videos/vllm-cpu-overhead/data.js", "w") as f:
    f.write("// 由 extract.py 生成，勿手改\nwindow.TRACE = " + json.dumps(out, separators=(",", ":")) + ";\n")
