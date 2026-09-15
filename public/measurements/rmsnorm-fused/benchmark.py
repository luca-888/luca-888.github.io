"""Isolated hand-written BF16 RMSNorm Backward experiment, RTX 4090, 4096x8192."""
import importlib.util
import json
import re
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.utils.benchmark import Timer

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
M, N = 4096, 8192


@triton.jit
def dx_kernel(X, W, G, R, DX, N: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, N)
    x = tl.load(X + row * N + col).to(tl.float32)
    gamma = tl.load(W + col).to(tl.float32)
    g = tl.load(G + row * N + col).to(tl.float32)
    r = tl.load(R + row)
    dx_hat = g * gamma
    dr = tl.sum(dx_hat * x, 0)
    dx = r * dx_hat - x * (r * r * r) * dr / N
    tl.store(DX + row * N + col, dx)


@triton.jit
def dg_partial(X, G, R, P, M: tl.constexpr, N: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    x = tl.load(X + rows[:, None] * N + cols[None, :], rows[:, None] < M, other=0).to(tl.float32)
    g = tl.load(G + rows[:, None] * N + cols[None, :], rows[:, None] < M, other=0).to(tl.float32)
    r = tl.load(R + rows, rows < M, other=0)
    dg = tl.sum(g * (x * r[:, None]), 0)
    tl.store(P + tl.program_id(0) * N + cols, dg)


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


def validate(actual, expected):
    records = []
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=.016, atol=.001)
        d = (a.float()-b.float()).abs()
        records.append(dict(passed=True, max_abs_error=d.max().item(),
            relative_l2_error=(d.norm()/b.float().norm().clamp_min(1e-30)).item()))
    return records


def gpu_times(fns, rounds=3):
    samples = {name: [] for name in fns}
    items = list(fns.items())
    for _, fn in items:
        fn()
    torch.cuda.synchronize()
    for i in range(rounds):
        for name, fn in items[i:]+items[:i]:
            times = triton.testing.do_bench_cudagraph(fn, rep=30, return_mode='all')
            samples[name].append([v*1000 for v in times])
    return {name:dict(median_us=statistics.median(map(statistics.median, values)),
                     rounds_us=values) for name,values in samples.items()}


def resource(kernel):
    return dict(registers=kernel.n_regs, spills=kernel.n_spills, shared_bytes=kernel.metadata.shared)


def main():
    torch.manual_seed(20260915)
    torch.set_num_threads(1)
    source = '\n\n'.join(re.findall(r'```python\n(.*?)```', (ROOT/'content/rmsnorm.md').read_text(), re.S)[:2])
    ns = {'torch':torch}
    exec(source, ns)
    forward, backward = ns['rmsnorm_forward_reference'], ns['rmsnorm_backward_reference']
    x = torch.randn(M,N,device='cuda',dtype=torch.bfloat16)
    gamma = torch.randn(N,device='cuda',dtype=torch.bfloat16)
    g = torch.randn_like(x)
    _, r = forward(x,gamma)
    expected = backward(x,gamma,g,r)
    spec = importlib.util.spec_from_file_location('original', ROOT/'public/measurements/rmsnorm-compile/4096x8192/backward.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    validate(original.call([x,gamma,g,r]), expected)
    dx, dg = torch.empty_like(x), torch.empty_like(gamma)
    generated_dx = lambda:original.triton_red_fused__to_copy_div_mul_pow_sub_sum_1.run(g,gamma,x,r,dx,M,N,stream=torch.cuda.current_stream().cuda_stream)
    generated_dg = lambda:original.triton_red_fused__to_copy_mul_sum_0.run(g,x,r,dg,N,M,stream=torch.cuda.current_stream().cuda_stream)
    def generated_both():
        generated_dg()
        generated_dx()
    result = dict(gpu=torch.cuda.get_device_name(),shape=[M,N],torch=torch.__version__,triton=triton.__version__,
        reference_source=source,
        gpu_method='Preallocated identical inputs/outputs, CUDA graph replay, rep=30ms, 10 samples x 3 rotating rounds, median of round medians, no cache flush; grouped dgamma includes its final reduction.',
        call_method='torch.utils.benchmark.Timer, 30 warmups, blocked_autorange(.3s), 3 rotating rounds, includes allocations; actual torch.compile callable compared with handwritten wrapper.',
        dtype='BF16 x/g/gamma/dx/dgamma; FP32 r and intermediate values',validations={}, resources={})
    dx_fns = {'compile_dx':generated_dx}
    for warps in [4,8,16]:
        name=f'dx_w{warps}'
        fn=lambda w=warps:dx_kernel[(M,)](x,gamma,g,r,dx,N,num_warps=w,num_stages=1)
        result['resources'][name] = resource(fn())
        result['validations'][name] = validate((dx,),expected[:1])
        dx_fns[name]=fn
    result['dx']=gpu_times(dx_fns)
    print('DX', {n:round(d['median_us'],2) for n,d in result['dx'].items()},flush=True)
    best_dx=min((n for n in dx_fns if n!='compile_dx'),key=lambda n:result['dx'][n]['median_us'])
    dg_fns = {'compile_dgamma':generated_dg}
    dg_configs = {}
    for rows,cols,warps in [(32,128,4),(64,128,4),(128,128,8),(256,128,8),(128,256,8)]:
        name=f'dg_r{rows}_c{cols}_w{warps}'
        p=torch.empty(M//rows,N,device='cuda',dtype=torch.float32)
        def fn(p=p,rows=rows,cols=cols,warps=warps):
            first=dg_partial[(M//rows,N//cols)](x,g,r,p,M,N,rows,cols,num_warps=warps)
            last=dg_finish[(N//128,)](p,dg,N,M//rows,128,num_warps=4)
            return first,last
        result['resources'][name]=[resource(v) for v in fn()]
        result['validations'][name]=validate((dg,),expected[1:])
        dg_fns[name]=fn
        dg_configs[name]=(rows,cols,warps)
    result['dgamma']=gpu_times(dg_fns)
    print('DG', {n:round(d['median_us'],2) for n,d in result['dgamma'].items()},flush=True)
    best_dg=min(dg_configs,key=lambda n:result['dgamma'][n]['median_us'])
    def separate():
        dg_fns[best_dg]()
        dx_fns[best_dx]()
    all_fns={'compile':generated_both,'separate':separate}
    fused_configs={}
    for rows in [4,8,16,32]:
        for warps in [4,8]:
            name=f'fused_r{rows}_w{warps}'
            p=torch.empty(M//rows,N,device='cuda',dtype=torch.float32)
            def fn(p=p,rows=rows,warps=warps):
                first=fused_backward[(M//rows,)](x,gamma,g,r,dx,p,N,rows,num_warps=warps,num_stages=1)
                last=dg_finish[(N//128,)](p,dg,N,M//rows,128,num_warps=4)
                return first,last
            result['resources'][name]=[resource(v) for v in fn()]
            result['validations'][name]=validate((dx,dg),expected)
            all_fns[name]=fn
            fused_configs[name]=(rows,warps)
    result['backward_gpu']=gpu_times(all_fns)
    print('BACKWARD', {n:round(d['median_us'],2) for n,d in result['backward_gpu'].items()},flush=True)
    best=min((n for n in all_fns if n!='compile'),key=lambda n:result['backward_gpu'][n]['median_us'])
    result['selected']=best
    result['selected_separate']=[best_dx,best_dg]
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    def handmade(x,gamma,g,r):
        dx=torch.empty_like(x)
        dg=torch.empty_like(gamma)
        if best=='separate':
            rows,cols,warps=dg_configs[best_dg]
            p=torch.empty(M//rows,N,device='cuda',dtype=torch.float32)
            dg_partial[(M//rows,N//cols)](x,g,r,p,M,N,rows,cols,num_warps=warps)
            dg_finish[(N//128,)](p,dg,N,M//rows,128,num_warps=4)
            dx_kernel[(M,)](x,gamma,g,r,dx,N,num_warps=int(best_dx.split('w')[1]),num_stages=1)
        else:
            rows,warps=fused_configs[best]
            p=torch.empty(M//rows,N,device='cuda',dtype=torch.float32)
            fused_backward[(M//rows,)](x,gamma,g,r,dx,p,N,rows,num_warps=warps,num_stages=1)
            dg_finish[(N//128,)](p,dg,N,M//rows,128,num_warps=4)
        return dx,dg
    compiled=torch.compile(backward,fullgraph=True,dynamic=False)
    funcs={'compile':lambda:compiled(x,gamma,g,r),'handwritten':lambda:handmade(x,gamma,g,r)}
    result['call_times']={n:[] for n in funcs}
    result['memory']={}
    for name,fn in funcs.items():
        validate(fn(),expected)
        for _ in range(30):fn()
    for i in range(3):
        items=list(funcs.items())
        for name,fn in items[i%2:]+items[:i%2]:
            m=Timer(stmt='fn()',globals={'fn':fn},num_threads=1).blocked_autorange(min_run_time=.3)
            result['call_times'][name].append(m.median*1e6)
    for name,fn in funcs.items():
        peaks=[]
        for _ in range(3):
            torch.cuda.synchronize()
            base=torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output=fn()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated()-base)
            del output
        result['memory'][name]=dict(peak_increment_bytes=max(peaks),samples=peaks)
    result['edge_validation']=[]
    for scale in [0.,1e-3,1e2]:
        sx=x*scale
        _,sr=forward(sx,gamma)
        result['edge_validation'].append(dict(scale=scale,validation=validate(handmade(sx,gamma,g,sr),backward(sx,gamma,g,sr))))
    result['selected_gpu_recheck']=gpu_times({'compile':generated_both,'handwritten':all_fns[best]})
    result['complete']=True
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL',best,'CALL',result['call_times'],'MEMORY',result['memory'],flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
