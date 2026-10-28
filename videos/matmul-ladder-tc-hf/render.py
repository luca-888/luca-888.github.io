"""python3 render.py [输出.mp4]  —— 本机没有 ffprobe 时出带配音的成片（默认 build/matmul-ladder-tc-hf.mp4，1080p 30 fps）：
1. 暂时去掉 index.html 里的 <audio>，用 hyperframes 渲染无声画面
   （环境变量 HYPERFRAMES_FFPROBE_PATH 指向一个只回答版本与基本信息的 ffprobe 替身，渲染前的环境检查要用）；
2. 按各 <audio> 的 data-start 把配音放到时间轴上，用 ffmpeg 混成一条音轨，响度标准化到 -16 LUFS；
3. 合成最终 mp4，并恢复 index.html。"""
import re, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else here / 'build/matmul-ladder-tc-hf.mp4').resolve()
out.parent.mkdir(parents=True, exist_ok=True)
idx = here / 'index.html'
src = idx.read_text()
clips = [(float(m.group(2)), m.group(1)) for m in re.finditer(r'<audio[^>]*src="([^"]+)"[^>]*data-start="([\d.]+)"', src)]
total = float(re.search(r'id="root"[^>]*data-duration="([\d.]+)"', src).group(1))
silent = out.with_name(out.stem + '.silent.mp4')
try:
    idx.write_text(re.sub(r'\s*<audio[^>]*></audio>', '', src))
    subprocess.run(['npx', '--yes', 'hyperframes@0.8.200', 'render', '--fps', '30', '--quality', 'delivery', '--output', str(silent), '--quiet'], cwd=here, check=True)
finally:
    idx.write_text(src)
args, flt = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(silent)], []
for i, (t, f) in enumerate(clips):
    args += ['-i', str(here / f)]
    flt.append(f'[{i + 1}:a]adelay={int(t * 1000)}|{int(t * 1000)}[a{i}]')
flt.append(''.join(f'[a{i}]' for i in range(len(clips))) + f'amix=inputs={len(clips)}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=0:{total}[aout]')
args += ['-filter_complex', ';'.join(flt), '-map', '0:v', '-map', '[aout]', '-c:v', 'copy', '-c:a', 'aac', '-ar', '48000', '-b:a', '192k', '-shortest', str(out)]
subprocess.run(args, check=True)
silent.unlink()
print(f'{out}：{len(clips)} 句配音，{total:.1f} s')
