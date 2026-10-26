"""python3 -I data.py [profile-data 目录] -> data.js
从 DeepSeek 公开的 H800 推理 profile（github.com/deepseek-ai/profile-data，commit 4496024）里各取一层的 kernel 时间线：
decode（EP128，每卡 128 个请求）与 prefill（EP32，每卡两个 8K token 的 micro-batch）。kernel 按名字归为通信、attention、expert 等几类，
时间相对窗口起点，单位 µs。profile 由 DeepSeek 用均衡的路由模拟得到（见其 README）。"""
import json, re, statistics, sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Documents/ChatGPT/reference-repos/profile-data")

def kind(n):
    if re.search(r"internode::(dispatch|notify_dispatch|get_dispatch_layout)|cuda::get_dispatch_layout", n): return "dispatch"
    if re.search(r"internode::(combine|cached_notify)", n): return "combine"
    if re.search(r"flash|rotary|layer_norm|2112u, 7168u|24576u, 1536u|7168u, 16384u|cutlass_80_tensorop_bf16|top2_sum_gate|cutlass_80_tensorop_s16816", n): return "attn"
    if re.search(r"4096u, 7168u|7168u, 2048u|swiglu|grouped_gemm|expand_to_fused|reduce_fused|get_fused_mapping|clean_and_count", n): return "moe"
    return "other"

def short(n):
    n = re.sub(r"^void ", "", n); n = re.sub(r"\(.*", "", n)
    m = re.search(r"internode::(\w+)", n)
    if m: return m.group(1)
    m = re.search(r"fp8_gemm_kernel<(\d+)u, (\d+)u", n)
    if m: return f"fp8_gemm {m.group(1)}×{m.group(2)}"
    if "flash_fwd_splitkv_mla_kernel" in n: return "FlashMLA"
    if "compute_attn_ws" in n: return "flash attention"
    return re.sub(r"<.*", "", n).split("::")[-1][:40]

def load(name):
    ev = json.loads((src / name).read_text())["traceEvents"]
    return sorted((e for e in ev if e.get("cat") == "kernel" and e["tid"] in (7, 16)), key=lambda e: e["ts"])

def window(ks, t0, span):
    out = [{"t": round(e["ts"] - t0, 1), "d": e["dur"], "s": e["tid"], "k": kind(e["name"]), "n": short(e["name"])}
           for e in ks if t0 <= e["ts"] < t0 + span]
    return [x for x in out if x["d"] >= 1]

# decode：通信也在计算流（stream 7）上，是很短的发出 / 接收 kernel；取 trace 中段一个 dispatch 接收（较长的 dispatch_ll）为起点
dec = load("decode.json")
dll = [e for e in dec if "dispatch_ll" in e["name"] and e["tid"] == 7]
recv = [e for e in dll if e["dur"] > 45]
d0 = recv[len(recv) // 2]["ts"]
decode = window(dec, d0, 1460)
# prefill：通信在单独的 stream 16（24 个 SM），计算在 stream 7（108 个 SM）；以中段一个 dispatch 为起点
pre = load("prefill.json")
pd = [e for e in pre if re.search(r"internode::dispatch<", e["name"])]
p0 = pd[len(pd) // 2]["ts"]
prefill = window(pre, p0, 34000)

# decode 一整步：计算 stream 上全部 kernel 的起点（ms，相对第一个 kernel），用来画“一步有多少个 kernel”
ks7 = [e for e in dec if e["tid"] == 7]
step = [round((e["ts"] - ks7[0]["ts"]) / 1000, 2) for e in ks7]

# prefill 每个 MoE 层的平均：第一个到最后一个 dispatch 之间，两条 stream 上 kernel 时长之和 ÷ 层数（每层 2 个 micro-batch 各一次 dispatch）
t_a, t_b = pd[0]["ts"], pd[-1]["ts"]
nl = (len(pd) - 1) / 2
per_layer = {s: round(sum(e["dur"] for e in pre if e["tid"] == s and t_a <= e["ts"] < t_b) / nl / 1000, 2) for s in (7, 16)}

def med(ks, pat): return round(statistics.median(e["dur"] for e in ks if re.search(pat, e["name"])), 1)
stats = {"decode": {"flashmla": med(dec, "flash_fwd_splitkv_mla_kernel"), "dispatch_ll": med(dec, "dispatch_ll"), "combine_ll": med(dec, "combine_ll")},
         "prefill": {"dispatch": med(pre, r"internode::dispatch<"), "combine": med(pre, r"internode::combine<"), "notify": med(pre, "notify_dispatch"),
                     "attn": med(pre, "compute_attn_ws"), "layer_compute_ms": per_layer[7], "layer_comm_ms": per_layer[16], "layers": nl}}
Path(__file__).with_name("data.js").write_text("// 由 data.py 生成：DeepSeek profile-data（H800）中各取一层的 kernel 时间线，t / d 单位 µs，s 为 CUDA stream\n"
    + "const TRACE = " + json.dumps({"decode": decode, "prefill": prefill, "step": step, "stats": stats}, ensure_ascii=False, separators=(",", ":")) + ";\n")
print(stats, len(decode), len(prefill), len(step), step[-1])
