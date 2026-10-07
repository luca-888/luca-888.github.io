# python3 sheet.py out.png a.png b.png …  -> 两列的总览图（每格 960 × 540）
import subprocess, sys
out, files = sys.argv[1], sys.argv[2:]
ins = [x for f in files for x in ('-i', f)]
flt = ''.join(f'[{i}:v]scale=960:540[v{i}];' for i in range(len(files)))
lay = '|'.join(f'{(i % 2) * 960}_{(i // 2) * 540}' for i in range(len(files)))
flt += ''.join(f'[v{i}]' for i in range(len(files))) + f'xstack=inputs={len(files)}:layout={lay}:fill=white'
if len(files) == 1: flt = '[0:v]scale=960:540'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *ins, '-filter_complex', flt, out], check=True)
