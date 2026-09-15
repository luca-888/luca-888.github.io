import torch
import triton
import triton.language as tl


@triton.jit
def dg_finish(P, DG, N: tl.constexpr, GROUPS: tl.constexpr, COLS: tl.constexpr):
    groups = tl.arange(0, GROUPS)
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    partial = tl.load(P + groups[:, None] * N + cols[None, :])
    tl.store(DG + cols, tl.sum(partial, 0))


@triton.jit
def fused_backward(X, W, G, R, DX, P, N: tl.constexpr, ROWS: tl.constexpr):
    group = tl.program_id(0)
    col = tl.arange(0, N)
    gamma = tl.load(W + col).to(tl.float32)
    dg = tl.full((N,), 0, tl.float32)
    for offset in range(ROWS):
        row = group * ROWS + offset
        x = tl.load(X + row * N + col).to(tl.float32)
        g = tl.load(G + row * N + col).to(tl.float32)
        r = tl.load(R + row)
        dx_hat = g * gamma
        dr = tl.sum(dx_hat * x, 0)
        dx = r * dx_hat - x * (r * r * r) * dr / N
        tl.store(DX + row * N + col, dx)
        dg += g * (x * r)
    tl.store(P + group * N + col, dg)



def rmsnorm_backward(x, gamma, g, r):
    """Measured specialization: contiguous 4096x8192 BF16; r is FP32."""
    assert x.shape == (4096, 8192)
    M, N = x.shape
    dx = torch.empty_like(x)
    dgamma = torch.empty_like(gamma)
    partial = torch.empty((M // 32, N), device=x.device, dtype=torch.float32)
    fused_backward[(M // 32,)](x, gamma, g, r, dx, partial, N, 32, num_warps=8, num_stages=1)
    dg_finish[(N // 128,)](partial, dgamma, N, M // 32, 128, num_warps=4)
    return dx, dgamma
