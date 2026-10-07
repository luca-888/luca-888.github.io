"""解析 vLLM engine 实测的 torch profiler trace，输出每个配置 / 并发下 decode 稳态 step 的时间分解。

每个 decode step 在 trace 里有一对 `execute_context_*_generation_N(N)` 注释：
CPU 线程上的 user_annotation（execute_model 的 CPU 耗时）和 GPU 上的 gpu_user_annotation（本步 GPU 区间）。
- period：相邻两步 GPU 区间起点之差（一步的墙钟时间）
- gpu_busy：本步周期（本步注释开始到下一步注释开始）内全部 kernel 时间的并集。
  注释区间之外还有 sampling、lm_head 之后的收尾与下一步的输入准备 kernel，必须算进来
- idle：period − gpu_busy，即 GPU 上没有任何 kernel 在跑的时间（GPU 空转）
- kernels：本步周期内的 kernel 数
- cpu_exec：CPU 侧 execute_model 注释的耗时；launches：本步 cudaLaunchKernel 次数
用法：python scripts/vllm-cpu-overhead/analyze_traces.py
trace 原件（28M）不在工作区，重跑前先从 git 历史取回：git checkout 11e29f2 -- public/measurements/vllm-cpu-overhead
"""
import bisect, glob, gzip, json, os, re, statistics as st

ROOT = "public/measurements/vllm-cpu-overhead"
PAT = re.compile(r"execute_context_0\(0\)_generation_(\d+)\((\d+)\)")


def union_us(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def analyze(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    cpu = sorted((e for e in ev if e.get("cat") == "user_annotation" and PAT.fullmatch(e["name"])), key=lambda e: e["ts"])
    gpu = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and PAT.fullmatch(e["name"])), key=lambda e: e["ts"])
    kern = sorted(((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") == "kernel"))
    kstarts = [k[0] for k in kern]
    launches = sorted(e["ts"] for e in ev if e.get("cat") == "cuda_runtime" and e["name"] in ("cudaLaunchKernel", "cudaGraphLaunch", "cuLaunchKernelEx", "cudaLaunchKernelExC"))
    n = min(len(cpu), len(gpu))
    rows = []
    for i in range(n - 1):
        g, gn = gpu[i], gpu[i + 1]
        lo, hi = g["ts"], gn["ts"]
        a, b = bisect.bisect_left(kstarts, lo), bisect.bisect_left(kstarts, hi)
        busy = union_us([(max(s, lo), min(e, hi)) for s, e in kern[a:b]])
        c = cpu[i]
        nl = bisect.bisect_right(launches, c["ts"] + c["dur"]) - bisect.bisect_left(launches, c["ts"])
        rows.append({
            "period": hi - lo, "gpu_span": g["dur"], "gpu_busy": busy,
            "idle": hi - lo - busy, "cpu_exec": c["dur"], "launches": nl, "kernels": b - a,
        })
    # 稳态：去掉首尾各 20% 的 step
    k = len(rows) // 5
    rows = rows[k: len(rows) - k] or rows
    med = lambda key: st.median(r[key] for r in rows)
    return {"steps": len(rows), **{key: round(med(key), 1) for key in ("period", "gpu_span", "gpu_busy", "idle", "cpu_exec", "launches", "kernels")}}


def main():
    out = {}
    for run in sorted(glob.glob(f"{ROOT}/2026*")):
        for cfg_dir in sorted(glob.glob(f"{run}/*/qwen3-8b")):
            cfg = cfg_dir.split("/")[-2]
            traces = sorted(glob.glob(f"{cfg_dir}/traces/rank0.*.pt.trace.json.gz"), key=lambda p: int(re.search(r"rank0\.(\d+)\.", p).group(1)))
            for conc, tp in zip((8, 64), traces):  # profile 先跑并发 8，再跑并发 64
                out[f"{cfg}/c{conc}"] = analyze(tp)
    json.dump(out, open(f"{ROOT}/step_breakdown.json", "w"), indent=1)
    print(f"{'config':<18}{'steps':>6}{'period':>9}{'gpu_busy':>10}{'idle':>8}{'cpu_exec':>10}{'launch':>8}{'kernels':>9}   (µs)")
    for k, v in out.items():
        print(f"{k:<18}{v['steps']:>6}{v['period']:>9}{v['gpu_busy']:>10}{v['idle']:>8}{v['cpu_exec']:>10}{v['launches']:>8}{v['kernels']:>9}")


if __name__ == "__main__":
    main()
