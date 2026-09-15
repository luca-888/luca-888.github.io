"""Record PTX changes and verify the final backward kernel sequences."""
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'public/measurements/rmsnorm-triton'
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'scripts/benchmark-rmsnorm-triton.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
data = json.loads(b.OUTPUT.read_text())
assert data['complete']
warps = data['selection']['dx_warps']
group_rows = data['selection']['dgamma_group_rows']
result = {'ptx': [], 'profiles': [], 'note': 'PTX instruction counts are static source counts, not dynamic DRAM transactions. Profile timings are diagnostic only; tables use measurements.json.'}
files = [ROOT / 'public/measurements/rmsnorm-compile/4096x8192/triton_red_fused__to_copy_div_mul_pow_sub_sum_1.ptx', *sorted(ART.glob('dx*.ptx'))]
for path in files:
    source = path.read_text()
    result['ptx'].append(dict(file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        global_load_instructions=len(re.findall(r'\bld\.global', source)),
        vector_global_load_instructions=len(re.findall(r'\bld\.global[^\n]*\.v4\.b32', source))))

torch.set_num_threads(1)
torch.manual_seed(20260916)
with torch.no_grad():
    for M,N in b.SHAPES:
        torch._dynamo.reset()
        x = torch.randn(M,N,device='cuda',dtype=torch.bfloat16)
        gamma = torch.randn(N,device='cuda',dtype=torch.bfloat16)
        g = torch.randn_like(x)
        _,r = b.forward(x,gamma)
        compiled = torch.compile(b.backward,fullgraph=True,dynamic=False)
        expected = b.backward(x,gamma,g,r)
        for name,fn in [('compile',lambda:compiled(x,gamma,g,r)),
                        ('triton',lambda:b.optimized(x,gamma,g,r,warps,group_rows))]:
            validation = b.check(fn(),expected)
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
                for i in range(3):
                    with record_function(f'measured_{i}'):
                        fn()
                        torch.cuda.synchronize()
            path = ART / f'{M}x{N}-{name}-trace.json'
            p.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
            samples = []
            for i in range(3):
                region = next(e for e in events if e.get('name') == f'measured_{i}')
                kernels = sorted((e for e in events if e.get('cat')=='kernel'
                    and region['ts'] <= e['ts'] < region['ts']+region['dur']),key=lambda e:e['ts'])
                samples.append(dict(kernel_count=len(kernels), kernels=[dict(name=e['name'],duration_us=e['dur']) for e in kernels]))
            packed = path.with_suffix('.json.gz')
            with gzip.open(packed,'wb') as f:
                f.write(path.read_bytes())
            path.unlink()
            result['profiles'].append(dict(M=M,N=N,implementation=name,validation=validation,
                samples=samples,trace=packed.name,sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
            print(M,N,name,[s['kernel_count'] for s in samples],flush=True)
(ART / 'inspection.json').write_text(json.dumps(result,indent=2)+'\n')
for row in data['measurements']:
    for name in ('compile', 'triton'):
        record = next(p for p in result['profiles'] if p['M'] == row['M'] and p['N'] == row['N'] and p['implementation'] == name)
        counts = [s['kernel_count'] for s in record['samples']]
        assert len(set(counts)) == 1
        row['backward_gpu'][name]['kernel_count'] = counts[0]
b.save(data)
