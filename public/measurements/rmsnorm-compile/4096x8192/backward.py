# AOT ID: ['13_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/px/cpxedpwcpu7y6bkhd4iuykx2lue6h6n3qjz7r6az6p3eh3titt6i.py
# Topologically Sorted Source Nodes: [g, x, x_hat, mul_6, dgamma, to_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   dgamma => sum_2
#   g => convert_element_type_2
#   mul_6 => mul_6
#   to_1 => convert_element_type_4
#   x => convert_element_type
#   x_hat => mul
# Graph fragment:
#   %arg2_1 : Tensor "bf16[4096, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg0_1 : Tensor "bf16[4096, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[4096, 1][1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sum_2 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=sum_2]
#   %convert_element_type_2 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %convert_element_type : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %mul : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %arg3_1), kwargs = {})
#   %mul_6 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %mul), kwargs = {})
#   %sum_2 : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [0]), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.bfloat16), kwargs = {})
#   return %sum_2,%convert_element_type_4
triton_red_fused__to_copy_mul_sum_0 = async_compile.triton('triton_red_fused__to_copy_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 134250496, 'r0_': 16384}}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 4096
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
        tmp0 = tl.load(in_ptr0 + (x0 + 8192*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 8192*r0_1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tn/ctnxrxsiuruo6l667h2asbge6sjuoy2gycsd4qekeobapvw4djxb.py
# Topologically Sorted Source Nodes: [g, gamma, dx_hat, mul_3, x, pow_1, mul_4, mul_2, dr, mul_5, truediv, dx, to], Original ATen: [aten._to_copy, aten.mul, aten.pow, aten.sum, aten.div, aten.sub]
# Source node to ATen node mapping:
#   dr => sum_1
#   dx => sub
#   dx_hat => mul_1
#   g => convert_element_type_2
#   gamma => convert_element_type_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   pow_1 => pow_1
#   to => convert_element_type_3
#   truediv => div
#   x => convert_element_type
# Graph fragment:
#   %arg2_1 : Tensor "bf16[4096, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg1_1 : Tensor "bf16[8192][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg0_1 : Tensor "bf16[4096, 8192][8192, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[4096, 1][1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %sum_1 : Tensor "f32[4096, 1][1, 4096]cuda:0" = PlaceHolder[target=sum_1]
#   %convert_element_type_2 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[8192][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %mul_1 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %convert_element_type_1), kwargs = {})
#   %mul_3 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg3_1, %mul_1), kwargs = {})
#   %convert_element_type : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg3_1, 3), kwargs = {})
#   %mul_4 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %pow_1), kwargs = {})
#   %mul_2 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %convert_element_type), kwargs = {})
#   %sum_1 : Tensor "f32[4096, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [1], True), kwargs = {})
#   %mul_5 : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %sum_1), kwargs = {})
#   %div : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_5, 8192), kwargs = {})
#   %sub : Tensor "f32[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %div), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[4096, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.bfloat16), kwargs = {})
#   return %sum_1,%convert_element_type_3
triton_red_fused__to_copy_div_mul_pow_sub_sum_1 = async_compile.triton('triton_red_fused__to_copy_div_mul_pow_sub_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_div_mul_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'D54CF9E23D08230976451A78161C34C47AAE433535668E89333E7039B8F7F720', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 16384, 'r0_': 268451840}}
)
@triton.jit
def triton_red_fused__to_copy_div_mul_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 8192
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        tmp12 = tl.load(in_ptr0 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 8192*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp13 * tmp15
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp11 * tmp11
        tmp21 = tmp20 * tmp11
        tmp22 = tmp19 * tmp21
        tmp23 = tmp22 * tmp9
        tmp24 = 0.0001220703125
        tmp25 = tmp23 * tmp24
        tmp26 = tmp17 - tmp25
        tmp27 = tmp26.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp27, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (4096, 8192), (8192, 1))
        assert_size_stride(arg1_1, (8192, ), (1, ))
        assert_size_stride(arg2_1, (4096, 8192), (8192, 1))
        assert_size_stride(arg3_1, (4096, 1), (1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((8192, ), (1, ), torch.bfloat16)
            # Topologically Sorted Source Nodes: [g, x, x_hat, mul_6, dgamma, to_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_0.run(arg2_1, arg0_1, arg3_1, buf3, 8192, 4096, stream=stream0)
            buf1 = empty_strided_cuda((4096, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [g, gamma, dx_hat, mul_3, x, pow_1, mul_4, mul_2, dr, mul_5, truediv, dx, to], Original ATen: [aten._to_copy, aten.mul, aten.pow, aten.sum, aten.div, aten.sub]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_div_mul_pow_sub_sum_1.run(arg2_1, arg1_1, arg0_1, arg3_1, buf1, 4096, 8192, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
        return (buf1, buf3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4096, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
