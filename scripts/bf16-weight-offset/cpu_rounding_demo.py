"""CPU 数值演示：Gemma RMSNorm 的 (1 + w) 在 BF16 与 FP32 中相加的差别。

不是 GPU 实测。RMSNorm 用 float64 计算，只在两处做 BF16 舍入：
  bf16_preadd：先把 1 + w 舍入为 BF16（修复前的 CuTe DSL Gemma 路径）
  fp32_add   ：w 转 FP32 后再加 1（修复后，与 HF / Triton 一致）
最后 Y 与 dX 各舍入一次到 BF16。用法：python3 cpu_rounding_demo.py
"""
import math

import torch

bf = torch.bfloat16


def ulp(v):
    return 2.0 ** (math.floor(math.log2(abs(v))) - 7)


print("== BF16 刻度间隔（相邻可表示数之差）")
for v in [0.01, 0.5, 1.0, 4.0, 8.0]:
    print(f"  |v| 在 {v} 附近：{ulp(v):.3e}")

print("\n== 1 + w：BF16 中相加 vs FP32 中相加")
for w in [0.00390625, 0.003, 0.005, -0.003, 0.01]:
    wb = torch.tensor(w).to(bf)
    a = (wb + 1).item()
    b = wb.float().item() + 1.0
    print(f"  w={w:+.8f} (BF16 存为 {wb.item():+.10f})  BF16 加={a:.8f}  FP32 加={b:.8f}  差={a - b:+.3e}")


def run(x, w, dy, eps=1e-6):
    xf = x.double()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    xhat = xf * rstd
    out = {}
    for name, we in [("fp32_add", w.double() + 1.0), ("bf16_preadd", (w + 1.0).double())]:
        y = (xhat * we).to(bf)
        g = dy.double() * we
        dx = ((g - xhat * (g * xhat).mean(-1, keepdim=True)) * rstd).to(bf)
        out[name] = (y, dx)
    return out


torch.manual_seed(42)
M, N = 128, 4096
x = torch.randn(M, N).to(bf)
dy = torch.randn(M, N).to(bf)
rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
for label, w in [("w ~ randn（Liger parity 测试的分布）", torch.randn(N).to(bf)),
                 ("w ~ 0.01 * randn（PR 复现脚本的分布）", (torch.randn(N) * 0.01).to(bf))]:
    o = run(x, w, dy)
    (y0, dx0), (y1, dx1) = o["fp32_add"], o["bf16_preadd"]
    d = ((w + 1.0).double() - (w.double() + 1.0)).abs()
    print(f"\n== {label}, M={M}, N={N}")
    print(f"  1+w 被舍入的比例 {(d > 0).float().mean().item():.3f}，最大舍入误差 {d.max().item():.3e}")
    print(f"  Y : max|差| {(y1.double() - y0.double()).abs().max().item():.6g}，相对 L2 {rel(y1, y0):.3e}，"
          f"不同元素比例 {(y1 != y0).float().mean().item():.3f}")
    print(f"  dX: max|差| {(dx1.double() - dx0.double()).abs().max().item():.6g}，相对 L2 {rel(dx1, dx0):.3e}")
