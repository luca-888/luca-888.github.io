"""python3 warp.py —— 各幕画面按 beats-authored.json（写画面时的节拍）制作；配音重排后 beats.json 变了，
本脚本把每幕的时间按节拍分段线性映射到新节拍：
  1. 计时元素的 data-start / data-duration（原值保存在 data-a-start / data-a-duration，可重复运行）；
  2. GSAP 时间线：注册处换成一条新时长的外层时间线，按映射 seek 原时间线。"""
import json, re
from pathlib import Path
here = Path(__file__).resolve().parent
A = json.loads((here / 'beats-authored.json').read_text()); B = json.loads((here / 'beats.json').read_text())

def mapper(a, b):
    P = [(0.0, 0.0)] + list(zip(a['beats'], b['beats'])) + [(a['dur'], b['dur'])]
    def f(t):
        for (x0, y0), (x1, y1) in zip(P, P[1:]):
            if t <= x1: return y0 + (t - x0) * (y1 - y0) / (x1 - x0) if x1 > x0 else y1
        return P[-1][1] + (t - P[-1][0])
    return P, f

for k in A:
    p = here / f'compositions/{k}.html'; s = p.read_text()
    P, f = mapper(A[k], B[k])
    def clip(m):
        tag = m.group(0)
        st = re.search(r'data-a-start="([\d.]+)"', tag) or re.search(r'data-start="([\d.]+)"', tag)
        du = re.search(r'data-a-duration="([\d.]+)"', tag) or re.search(r'data-duration="([\d.]+)"', tag)
        if not st or not du: return tag
        s0, d0 = float(st.group(1)), float(du.group(1))
        ns, ne = f(s0), f(s0 + d0)
        tag = re.sub(r'\sdata-a-(start|duration)="[\d.]+"', '', tag)
        tag = re.sub(r'data-start="[\d.]+"', f'data-start="{ns:.2f}" data-a-start="{s0}"', tag)
        tag = re.sub(r'data-duration="[\d.]+"', f'data-duration="{ne - ns:.2f}" data-a-duration="{d0}"', tag)
        return tag
    s = re.sub(r'<(div|span|svg|canvas|img|section|p|h1|h2)\b[^>]*\bdata-start="[^>]*>', clip, s)
    reg = f'window.__timelines["{k}"] = tl;'
    warp = (f'/* warp */ {{ const P = {json.dumps([[round(x, 3), round(y, 3)] for x, y in P])}; '
            'const inv = (y) => { for (let i = 1; i < P.length; i++) if (y <= P[i][1] || i === P.length - 1) { const [x0, y0] = P[i - 1], [x1, y1] = P[i]; return y1 > y0 ? x0 + (y - y0) * (x1 - x0) / (y1 - y0) : x1; } return y; }; '
            f'tl.pause(); const st = {{ t: 0 }}; const outer = gsap.timeline({{ paused: true }}); '
            f'outer.fromTo(st, {{ t: 0 }}, {{ t: {B[k]["dur"]}, duration: {B[k]["dur"]}, ease: "none", immediateRender: true, onUpdate: () => tl.seek(inv(st.t), false) }}, 0); '
            f'window.__timelines["{k}"] = outer; }} /* /warp */')
    s = re.sub(r'/\* warp \*/.*?/\* /warp \*/', reg, s, flags=re.S)   # 先还原
    assert reg in s, k
    s = s.replace(reg, warp)
    p.write_text(s)
    print(k, 'map', [(round(x, 1), round(y, 1)) for x, y in P][1:-1])
