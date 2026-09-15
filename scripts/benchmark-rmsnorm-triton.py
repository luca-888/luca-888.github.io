"""Controlled edits to saved Inductor kernels; chapter-four raw measurements.

Run prepare-rmsnorm-triton.py first, then this script in the original torch env.
GPU comparisons use the same CUDA-graph timing and preallocated output buffers.
Full-call comparisons independently use the chapter-two torch Timer protocol.
"""
import hashlib
import importlib.util
import json
import platform
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from torch.utils.benchmark import Timer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'public/measurements/rmsnorm-triton'
OUTPUT = ROOT / 'content/data/rmsnorm-triton.json'
SHAPES = [(1, 4096), (128, 4096), (1024, 4096), (4096, 4096), (4096, 8192)]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


k = module(ART / 'kernels.py', 'rmsnorm_variants')
source = '\n\n'.join(re.findall(r'```python\n(.*?)```', (ROOT / 'content/rmsnorm.md').read_text(), re.S)[:2])
scope = {'torch': torch}
exec(compile(source, 'rmsnorm_reference.py', 'exec'), scope)
forward = scope['rmsnorm_forward_reference']
backward = scope['rmsnorm_backward_reference']


def check(actual, expected, names=('dx', 'dgamma')):
    result = {}
    for name, a, b in zip(names, actual, expected):
        rtol, atol = (1e-5, 1e-6) if name == 'r' else (0.016, 0.001)
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        diff = (a.float() - b.float()).abs()
        result[name] = dict(passed=True, rtol=rtol, atol=atol,
            max_abs_error=diff.max().item(),
            relative_l2_error=(diff.norm() / b.float().norm().clamp_min(1e-30)).item())
    return result


def gpu_times(functions):
    samples = {name: [] for name in functions}
    items = list(functions.items())
    for _, fn in items:
        fn()
    torch.cuda.synchronize()
    # Rotate order in each round; all candidates use the same measurement path.
    for i in range(3):
        for name, fn in items[i:] + items[:i]:
            values = triton.testing.do_bench_cudagraph(fn, rep=50, return_mode='all')
            samples[name].append([v * 1000 for v in values])
    return {name: dict(median_us=statistics.median(map(statistics.median, rounds)),
                      round_medians_us=list(map(statistics.median, rounds)), samples_us=rounds)
            for name, rounds in samples.items()}


def call_times(functions):
    samples = {name: [] for name in functions}
    items = list(functions.items())
    for _, fn in items:
        for _ in range(30):
            fn()
    torch.cuda.synchronize()
    for i in range(3):
        for name, fn in items[i:] + items[:i]:
            m = Timer(stmt='fn()', globals={'fn': fn}, num_threads=1).blocked_autorange(min_run_time=0.3)
            samples[name].append(dict(median_us=m.median * 1e6, iqr_us=m.iqr * 1e6,
                                     calls_per_block=m.number_per_run, raw_times_s=m.raw_times))
    return {name: dict(median_us=statistics.median(r['median_us'] for r in rounds), rounds=rounds)
            for name, rounds in samples.items()}


def memory(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = fn()
        torch.cuda.synchronize()
        samples.append(dict(baseline_allocated_bytes=baseline,
            peak_increment_bytes=torch.cuda.max_memory_allocated() - baseline,
            output_live_increment_bytes=torch.cuda.memory_allocated() - baseline))
        del output
    return dict(peak_increment_bytes=max(s['peak_increment_bytes'] for s in samples), samples=samples)


def dx_launch(x, gamma, g, r, out, warps, reuse=True):
    M, N = x.shape
    fn = k.dx_reuse if reuse else k.dx_original
    return fn[(M,)](g, gamma, x, r, out, M, N, XBLOCK=1, R0_BLOCK=N,
                    num_warps=warps, num_stages=1, enable_fp_fusion=True)


def dg_launch(x, g, r, out, partial, group_rows):
    M, N = x.shape
    if group_rows == 0:
        return k.dgamma_direct[(triton.cdiv(N, 64),)](g, x, r, out, N, M,
                XBLOCK=64, R0_BLOCK=64, num_warps=16, num_stages=1)
    groups = triton.cdiv(M, group_rows)
    first = k.dgamma_partial[(triton.cdiv(N, 64), groups)](g, x, r, partial, N, M,
            XBLOCK=64, R0_BLOCK=64, GROUP_ROWS=group_rows, num_warps=16, num_stages=1)
    last = k.dgamma_finish[(triton.cdiv(N, 128),)](partial, out, N, groups,
            triton.next_power_of_2(groups), 128, num_warps=4, num_stages=1)
    return first, last


def optimized(x, gamma, g, r, warps, group_rows):
    M, N = x.shape
    dg = torch.empty_like(gamma)
    partial = torch.empty((triton.cdiv(M, group_rows), N), device=x.device, dtype=torch.float32) if group_rows else None
    dg_launch(x, g, r, dg, partial, group_rows)
    del partial
    dx = torch.empty_like(x)
    dx_launch(x, gamma, g, r, dx, warps)
    return dx, dg


def resources(compiled, name):
    if not isinstance(compiled, tuple):
        compiled = (compiled,)
    result = []
    for i, kernel in enumerate(compiled):
        path = ART / f'{name}-{i}.ptx'
        path.write_text(kernel.asm['ptx'])
        result.append(dict(registers_per_thread=kernel.n_regs, spills=kernel.n_spills,
                           shared_bytes=kernel.metadata.shared, ptx=path.name))
    return result


def save(result):
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    (ART / 'measurements.json').write_bytes(OUTPUT.read_bytes())


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260915)
    baseline = json.loads((ROOT / 'content/data/rmsnorm-eager.json').read_text())
    assert baseline['python'] == platform.python_version()
    assert baseline['torch'] == torch.__version__
    assert baseline['gpu'] == torch.cuda.get_device_name()
    assert hashlib.sha256(source.encode()).hexdigest() == baseline['reference_sha256']
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, triton=triton.__version__, python=platform.python_version(),
        reference_sha256=baseline['reference_sha256'], reference_source=source,
        gpu_method='triton.testing.do_bench_cudagraph, rep=50 ms, 10 event samples per round, 3 rotated rounds; median of round medians. Ablations use preallocated buffers. Final GPU pipelines capture full callables with allocations in graph private pool. Repeated same inputs, no explicit cache flush.',
        call_method='torch.utils.benchmark.Timer, 30 warmup calls, 3 rotated rounds, blocked_autorange(min_run_time=0.3), 1 CPU thread; median of round medians; includes allocations and Python wrapper, excludes first-call compilation. Forward callable is shared exactly.',
        memory_method='Peak torch.cuda.memory_allocated increment above live inputs and r; includes outputs and temporary tensors; 3 samples; same CUDA stream, partial freed before dx allocation; excludes graph pool and compiler caches.',
        validation_method='Eager BF16 reference: rtol=.016, atol=.001; r FP32 rtol=1e-5 atol=1e-6. Shared eager r plus composed compiled r. All five shapes and zero/1e-3/1e2 scaled x on main shape.',
        ablations={}, measurements=[])
    if '--final-only' in sys.argv:
        result = json.loads(OUTPUT.read_text())
        result['measurements'] = []
        result['final_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        measure_shapes(result)
        return
    M, N = SHAPES[-1]
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    gamma = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    g = torch.randn_like(x)
    _, r = forward(x, gamma)
    expected = backward(x, gamma, g, r)
    original = module(ROOT / 'public/measurements/rmsnorm-compile/4096x8192/backward.py', 'saved_backward')
    check(original.call([x, gamma, g, r]), expected)
    dx = torch.empty_like(x)
    dg = torch.empty_like(gamma)
    actual_dx = getattr(original, 'triton_red_fused__to_copy_div_mul_pow_sub_sum_1')
    actual_dg = getattr(original, 'triton_red_fused__to_copy_mul_sum_0')
    dx_fns = {'generated': lambda: actual_dx.run(g, gamma, x, r, dx, M, N, stream=torch.cuda.current_stream().cuda_stream)}
    dx_meta = {}
    for reuse, warps in [(False,16),(True,16),(True,8),(True,4)]:
        name = f'{"reuse" if reuse else "copy"}_{warps}'
        fn = lambda w=warps, use=reuse: dx_launch(x, gamma, g, r, dx, w, use)
        compiled = fn()
        dx_meta[name] = dict(warps=warps, reuse=reuse, resources=resources(compiled, f'dx-{name}'),
                             validation=check((dx,), expected[:1], ('dx',)))
        dx_fns[name] = fn
    dx_results = gpu_times(dx_fns)
    result['ablations']['dx'] = {name: {**data, **dx_meta.get(name, {})} for name, data in dx_results.items()}
    print('DX', json.dumps({name: data['median_us'] for name,data in dx_results.items()}), flush=True)
    # Independent extraction control: the unchanged copy should match generated execution.
    dg_fns = {'generated': lambda: actual_dg.run(g, x, r, dg, N, M, stream=torch.cuda.current_stream().cuda_stream)}
    dg_meta = {}
    for group in [0,128,256,512]:
        name = f'group_{group}' if group else 'direct'
        partial = torch.empty((triton.cdiv(M, group), N), device='cuda', dtype=torch.float32) if group else None
        fn = lambda p=partial, rows=group: dg_launch(x, g, r, dg, p, rows)
        compiled = fn()
        dg_meta[name] = dict(group_rows=group, workspace_bytes=partial.numel()*4 if group else 0,
            kernel_count=2 if group else 1, resources=resources(compiled, f'dgamma-{name}'),
            validation=check((dg,), expected[1:], ('dgamma',)))
        dg_fns[name] = fn
    dg_results = gpu_times(dg_fns)
    result['ablations']['dgamma'] = {name: {**data, **dg_meta.get(name, {})} for name, data in dg_results.items()}
    print('DGAMMA', json.dumps({name: data['median_us'] for name,data in dg_results.items()}), flush=True)
    dx_choice = min(dx_meta, key=lambda name: dx_results[name]['median_us'] if dx_meta[name]['reuse'] else float('inf'))
    dg_choice = min(dg_meta, key=lambda name: dg_results[name]['median_us'])
    warps, group_rows = dx_meta[dx_choice]['warps'], dg_meta[dg_choice]['group_rows']
    result['selection'] = dict(dx_warps=warps, dgamma_group_rows=group_rows,
        rule='Choose lowest main-shape GPU median among measured reuse warps and dgamma strategies; use exactly this configuration on all five shapes, no per-shape fallback or tuning.')
    save(result)
    del dx_fns, dg_fns, expected, dx, dg, x, gamma, g, r, partial
    measure_shapes(result)


def measure_shapes(result):
    warps = result['selection']['dx_warps']
    group_rows = result['selection']['dgamma_group_rows']
    for M,N in SHAPES:
        x = torch.randn(M,N,device='cuda',dtype=torch.bfloat16)
        gamma = torch.randn(N,device='cuda',dtype=torch.bfloat16)
        g = torch.randn_like(x)
        y,r = forward(x,gamma)
        torch._dynamo.reset()
        compiled_forward = torch.compile(forward, fullgraph=True, dynamic=False)
        compiled_backward = torch.compile(backward, fullgraph=True, dynamic=False)
        cy, cr = compiled_forward(x,gamma)
        expected = backward(x,gamma,g,r)
        opt = lambda: optimized(x,gamma,g,r,warps,group_rows)
        funcs = {'eager':lambda:backward(x,gamma,g,r),
                 'compile':lambda:compiled_backward(x,gamma,g,r), 'triton':opt}
        row = dict(M=M,N=N,dtype=str(x.dtype), validation={
            'forward':check((cy,cr),(y,r),('y','r')),
            'compile':check(funcs['compile'](),expected),
            'triton':check(opt(),expected),
            'composed':check(optimized(x,gamma,g,cr,warps,group_rows),expected)},
            workspace_bytes=triton.cdiv(M,group_rows)*N*4 if group_rows else 0)
        row['backward'] = call_times(funcs)
        for name,fn in funcs.items():
            row['backward'][name]['memory'] = memory(fn)
        row['backward_gpu'] = gpu_times({name:fn for name,fn in funcs.items() if name != 'eager'})
        fwd_fns = {'eager':lambda:forward(x,gamma), 'compile':lambda:compiled_forward(x,gamma)}
        row['forward'] = call_times(fwd_fns)
        for name,fn in fwd_fns.items():
            row['forward'][name]['memory'] = memory(fn)
        # No independent Triton Forward: it is the identical compile callable.
        if (M,N) == SHAPES[-1]:
            row['validation']['scaled_x'] = []
            for scale in [0.,1e-3,1e2]:
                sx=x*scale
                sy,sr=forward(sx,gamma)
                scy,scr=compiled_forward(sx,gamma)
                row['validation']['scaled_x'].append(dict(scale=scale,
                    forward=check((scy,scr),(sy,sr),('y','r')),
                    backward=check(optimized(sx,gamma,g,scr,warps,group_rows),backward(sx,gamma,g,sr))))
        result['measurements'].append(row)
        save(result)
        print('FINAL', M,N, 'call', {n:round(d['median_us'],2) for n,d in row['backward'].items()},
              'gpu', {n:round(d['median_us'],2) for n,d in row['backward_gpu'].items()}, flush=True)
    result['complete'] = True
    save(result)


if __name__ == '__main__':
    with torch.no_grad():
        main()
