"""python3 timing.py  —— 按 beats.json（每幕的节拍起点与幕长）和 narration.md 重写 index.html 的幕宿主与 compositions/captions.html。
录好配音（assets/vo/vo.json）后加 --vo：每句按实测时长重排节拍、放入逐句 <audio>，并把新的节拍写回 beats.json。
画面里的动画按各幕自己的秒数写，重排后需要按 beats.json 的新起点核对。"""
import json, re, sys
from pathlib import Path
here = Path(__file__).resolve().parent
spk = lambda x: len(re.findall(r'[一-鿿]', x)) * 0.21 + len(re.findall(r'[A-Za-z0-9][A-Za-z0-9._\-]*', x)) * 0.32 + len(re.findall(r'[，；：、]', x)) * 0.15
B = json.loads((here / 'beats.json').read_text())
lines, cur = {}, None
for l in (here / 'narration.md').read_text().splitlines():
    if l.startswith('## s'): cur = l[3:].strip(); lines[cur] = []
    elif l.startswith('## '): cur = None
    elif cur and l.strip(): lines[cur].append(l.strip())
vo = json.loads((here / 'assets/vo/vo.json').read_text())['scenes'] if '--vo' in sys.argv else {}

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
for k in B:
    beats, dur = B[k]['beats'], B[k]['dur']
    says = [vo[k][i]['dur'] if vo else spk(l) + 0.4 for i, l in enumerate(lines[k])]
    if vo:   # 按实测时长保证每句说完再进下一拍（只往后推，不往前拉）
        for i in range(1, len(beats)): beats[i] = round(max(beats[i], beats[i - 1] + says[i - 1] + 0.5), 2)
        dur = round(max(dur, beats[-1] + says[-1] + 1.0), 2); B[k] = {'beats': beats, 'dur': dur}
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
hosts.append(f'      <div id="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0" data-duration="{total}" data-track-index="2" data-track-kind="captions" data-width="1920" data-height="1080"></div>')
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
if vo: (here / 'beats.json').write_text(json.dumps(B, indent=1, ensure_ascii=False))
print(f'{len(B)} 幕，共 {total} s；字幕 {len(caps)} 条' + (f'；配音 {len(audio)} 句' if vo else '（无声版，按估计排）'))
