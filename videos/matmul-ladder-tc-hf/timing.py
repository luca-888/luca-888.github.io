"""python3 timing.py [--vo] [--only s01,s02]  —— 按 narration.md 排每句的起点（节拍），重写 beats.json、index.html 的幕宿主、compositions/captions.html，
并把各幕 compositions/<幕>.html 里的 `const B = [...]`（每句起点）与 `const DUR = …` 改成新值：画面动画都写成 B[i] + 偏移，跟着配音走。
无 --vo：按字数估计每句时长（无声预览）；有 --vo：按 assets/vo/vo.json 的实测时长，并放入逐句 <audio>。--only 只排列出的幕。"""
import json, re, sys
from pathlib import Path
here = Path(__file__).resolve().parent
spk = lambda x: len(re.findall(r'[一-鿿]', x)) * 0.21 + len(re.findall(r'[A-Za-z0-9][A-Za-z0-9._\-]*', x)) * 0.32 + len(re.findall(r'[，；：、]', x)) * 0.15
lines, cur = {}, None
for l in (here / 'narration.md').read_text().splitlines():
    if l.startswith('## s'): cur = l[3:].strip(); lines[cur] = []
    elif l.startswith('## '): cur = None
    elif cur and l.strip() and not l.startswith('#'): lines[cur].append(l.strip())
vo = json.loads((here / 'assets/vo/vo.json').read_text())['scenes'] if '--vo' in sys.argv else {}
only = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else list(lines)
B = {}

def pieces(text, dur):
    parts = [p for p in re.split(r'(?<=[。；])', text) if p.strip()]
    out = []
    for p in parts:
        if len(p) <= 26: out.append(p); continue
        c = ''
        for seg in re.split(r'(?<=[，：])', p):
            if c and len(c + seg) > 26: out.append(c); c = seg
            else: c += seg
        if c: out.append(c)
    w = [max(spk(p), 0.3) for p in out]; sw = sum(w); t = 0; res = []
    for p, x in zip(out, w):
        d = dur * x / sw; res.append((re.sub(r'[。；，：]$', '', p), t, d)); t += d
    return res

hosts, caps, audio, t0 = [], [], [], 0.0
for k in only:
    says = [vo[k][i]['dur'] if vo else spk(l) * 0.8 + 0.3 for i, l in enumerate(lines[k])]
    beats = [0.6]   # 每句说完停 0.6 s 再进下一句
    for d in says[:-1]: beats.append(round(beats[-1] + d + 0.6, 2))
    dur = round(beats[-1] + says[-1] + 1.5, 2); B[k] = {'beats': beats, 'dur': dur}
    f = here / f'compositions/{k}.html'
    if f.exists():
        c = re.sub(r'const B = \[[^\]]*\];', f'const B = {json.dumps(beats)};', f.read_text())
        f.write_text(re.sub(r'const DUR = [\d.]+;', f'const DUR = {dur};', c))
    else: print(f'  {k}: compositions/{k}.html 还不存在，只排字幕')
    hosts.append(f'      <div id="{k}" data-composition-id="{k}" data-composition-src="compositions/{k}.html" data-start="{t0:.2f}" data-duration="{dur}" data-track-index="1" data-track-kind="graphics" data-width="1920" data-height="1080"></div>')
    for i, (line, b) in enumerate(zip(lines[k], beats)):
        nxt = beats[i + 1] if i + 1 < len(beats) else dur
        ps = pieces(line, says[i])
        for j, (p, pt, pd) in enumerate(ps):
            st = t0 + b + pt; en = t0 + min(nxt - 0.1, b + says[i] + 1.0) if j == len(ps) - 1 else st + pd
            caps.append(f'    <div id="cap-{k}-{i}-{j}" class="clip kit-sub" data-start="{st:.2f}" data-duration="{en - st:.2f}" data-track-index="1"><span>{p}</span></div>')
        if vo:
            r = vo[k][i]
            audio.append(f'      <audio id="vo-{k}-{i}" class="clip" src="{r["file"]}" data-start="{t0 + b - r["head"]:.2f}" data-duration="{r["len"]}" data-track-index="3" data-volume="1"></audio>')
    t0 += dur
total = round(t0, 2)
hosts.append(f'      <div id="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="{total}" data-track-index="2" data-track-kind="captions" style="z-index:100" data-width="1920" data-height="1080"></div>')
(here / 'compositions/captions.html').write_text(f'''<template>
  <style>
    #root {{ position: absolute; inset: 0; }}
  </style>
  <!-- 由 timing.py 生成：改配音稿后重跑 -->
  <div id="root" data-composition-id="captions" data-width="1920" data-height="1080" data-layout-allow-caption-zone>
{chr(10).join(caps)}
  </div>
  <script>
    window.__timelines["captions"] = gsap.timeline({{ paused: true }}).set({{}}, {{}}, {total});
  </script>
</template>
''')
s = (here / 'index.html').read_text()
a = re.search(r'<div[^>]*\bid="root"', s).start(); z = s.index('    <script>', a)
s = s[:a] + f'<div id="root" data-composition-id="main" data-start="0" data-duration="{total}" data-width="1920" data-height="1080">\n' + '\n'.join(hosts + audio) + '\n    </div>\n' + s[z:]
(here / 'index.html').write_text(s)
(here / 'beats.json').write_text(json.dumps(B, indent=1, ensure_ascii=False))
print(f'{len(B)} 幕，共 {total} s；字幕 {len(caps)} 条' + (f'；配音 {len(audio)} 句' if vo else '（无声版，按估计排）'))
