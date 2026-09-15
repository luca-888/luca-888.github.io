"""Derived from the saved 4096x8192 TorchInductor output.

dx_reuse requires XBLOCK=1 and R0_BLOCK=N (the whole row).
The wrappers support only the five contiguous BF16 article workloads.
See changes.diff for the exact changes from the generated kernel bodies.
"""
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

@triton.jit
def dgamma_direct(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + xnumel*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + xnumel*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tmp8.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp10, None)

@triton.jit
def dx_original(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp11 * tmp11
        tmp21 = tmp20 * tmp11
        tmp22 = tmp19 * tmp21
        tmp23 = tmp22 * tmp9
        tmp24 = (1.0 / r0_numel)
        tmp25 = tmp23 * tmp24
        tmp26 = tmp17 - tmp25
        tmp27 = tmp26.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + r0_numel*x0), tmp27, r0_mask)

@triton.jit
def dx_reuse(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    tl.static_assert(XBLOCK == 1 and R0_BLOCK == r0_numel)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    r0_offset = 0
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + r0_numel*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = _tmp9 + tmp8
    _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    r0_offset = 0
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp12 = tmp0
    tmp14 = tmp2
    tmp18 = tmp5
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp4
    tmp17 = tmp11 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp11 * tmp11
    tmp21 = tmp20 * tmp11
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = (1.0 / r0_numel)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 - tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + r0_numel*x0), tmp27, r0_mask)


@triton.jit
def dgamma_partial(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr, GROUP_ROWS: tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, GROUP_ROWS, R0_BLOCK):
        r0_index = tl.program_id(1) * GROUP_ROWS + r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + xnumel*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + xnumel*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tmp8.to(tl.float32)
    tl.store(out_ptr1 + (tl.program_id(1) * xnumel + x0), tmp10, None)

@triton.jit
def dgamma_finish(partial_ptr, dgamma_ptr, N: tl.constexpr, GROUPS: tl.constexpr, GROUP_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    cols = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    groups = tl.arange(0, GROUP_BLOCK)
    partial = tl.load(partial_ptr + groups[:, None] * N + cols[None, :],
                      (groups[:, None] < GROUPS) & (cols[None, :] < N), other=0)
    dgamma = tl.sum(partial, axis=0)
    tl.store(dgamma_ptr + cols, dgamma, cols < N)

