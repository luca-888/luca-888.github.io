"""Extract real Inductor kernels, then make small, recorded source changes."""
import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
original = ROOT / 'public/measurements/rmsnorm-compile/4096x8192/backward.py'
folder = ROOT / 'public/measurements/rmsnorm-triton'
folder.mkdir(exist_ok=True)
source = original.read_text()
kernels = re.findall(r"@triton\.jit\n[\s\S]*?(?=\n''', device_str=)", source)
assert len(kernels) == 2
names = [re.search(r'def (\w+)\(', k).group(1) for k in kernels]
# Keep the generated math, indexing, masks and cache hints. Expose only shape
# constants so the same strategy can be checked on the five article shapes.
dg = kernels[0].replace(names[0], 'dgamma_direct')
dg = dg.replace('xnumel, r0_numel, XBLOCK', 'xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK')
dg = dg.replace('    xnumel = 8192\n    r0_numel = 4096\n', '')
dg = dg.replace('8192*r0_1', 'xnumel*r0_1')
dx = kernels[1].replace(names[1], 'dx_original')
dx = dx.replace('xnumel, r0_numel, XBLOCK', 'xnumel: tl.constexpr, r0_numel: tl.constexpr, XBLOCK')
dx = dx.replace('    xnumel = 4096\n    r0_numel = 8192\n', '')
dx = dx.replace('8192*x0', 'r0_numel*x0').replace('0.0001220703125', '(1.0 / r0_numel)')
reuse = dx.replace('def dx_original(', 'def dx_reuse(') + '\n'
# Each loop has exactly one iteration for a full row. Move its body into the
# same scope so the second phase can use the first phase's loaded values.
reuse = re.sub(r'    for r0_offset in range\(0, r0_numel, R0_BLOCK\):\n((?:        [^\n]*\n)+)',
               lambda m: '    r0_offset = 0\n' + ''.join(line[4:] for line in m[1].splitlines(True)), reuse)
reuse = reuse.replace('    xoffset =', '    tl.static_assert(XBLOCK == 1 and R0_BLOCK == r0_numel)\n    xoffset =', 1)
for new, old in [('tmp12', 'tmp0'), ('tmp14', 'tmp2'), ('tmp18', 'tmp5')]:
    reuse, count = re.subn(rf'    {new} = tl.load[^\n]+', f'    {new} = {old}', reuse)
    assert count == 1
reuse = reuse.replace('tmp16 = tmp13 * tmp15', 'tmp16 = tmp4')
# Full-row R0_BLOCK is required: otherwise tmp0/tmp2/tmp5 refer only to the
# final reduction tile. Both runners always pass R0_BLOCK=N, XBLOCK=1.
partial = dg.replace('def dgamma_direct(', 'def dgamma_partial(')
partial = partial.replace('R0_BLOCK : tl.constexpr):', 'R0_BLOCK : tl.constexpr, GROUP_ROWS: tl.constexpr):')
partial = partial.replace('range(0, r0_numel, R0_BLOCK)', 'range(0, GROUP_ROWS, R0_BLOCK)')
partial = partial.replace('r0_index = r0_offset + r0_base', 'r0_index = tl.program_id(1) * GROUP_ROWS + r0_offset + r0_base')
partial = partial.replace('out_ptr1 + (x0)', 'out_ptr1 + (tl.program_id(1) * xnumel + x0)')
finish = '''@triton.jit
def dgamma_finish(partial_ptr, dgamma_ptr, N: tl.constexpr, GROUPS: tl.constexpr, GROUP_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    cols = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    groups = tl.arange(0, GROUP_BLOCK)
    partial = tl.load(partial_ptr + groups[:, None] * N + cols[None, :],
                      (groups[:, None] < GROUPS) & (cols[None, :] < N), other=0)
    dgamma = tl.sum(partial, axis=0)
    tl.store(dgamma_ptr + cols, dgamma, cols < N)
'''
header = '''"""Derived from the saved 4096x8192 TorchInductor output.

dx_reuse requires XBLOCK=1 and R0_BLOCK=N (the whole row).
The wrappers support only the five contiguous BF16 article workloads.
See changes.diff for the exact changes from the generated kernel bodies.
"""
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

'''
(folder / 'kernels.py').write_text(header + '\n\n'.join([dg, dx, reuse, partial, finish]) + '\n')
changes = []
for before, after, a, b in [(kernels[0], dg, 'generated_dgamma', 'dgamma_direct'),
                           (kernels[1], dx, 'generated_dx', 'dx_original'),
                           (dx, reuse, 'dx_original', 'dx_reuse'),
                           (dg, partial, 'dgamma_direct', 'dgamma_partial')]:
    changes.extend(difflib.unified_diff((before.rstrip() + '\n').splitlines(True),
                                      (after.rstrip() + '\n').splitlines(True), fromfile=a, tofile=b))
(folder / 'changes.diff').write_text(''.join(changes))
(folder / 'origin.json').write_text(json.dumps({'source': str(original.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
    'kernels_sha256': hashlib.sha256((folder / 'kernels.py').read_bytes()).hexdigest(),
    'supported_shapes': [[1,4096],[128,4096],[1024,4096],[4096,4096],[4096,8192]],
    'constraint': 'dx_reuse: XBLOCK=1, R0_BLOCK=N; BF16 x/g/gamma/dx/dgamma and FP32 r/intermediates; contiguous input; no autograd integration.'}, indent=2)+'\n')
print('Saved derived kernels and exact source diffs.')
