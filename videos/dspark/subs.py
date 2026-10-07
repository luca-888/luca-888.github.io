"""python3 subs.py  -> subs.js（字幕）、beats.js（节拍：每句配音的字幕与时长）并同步 review.html 里的配音稿。
时长在合成配音前按字数估算（汉字 4.2 字/秒，英文词与数字各 0.4 秒）；合成后改用实测时长。
来源是 script.md 的“v7 配音稿”表格：每幕一组，按标点切成不超过一行的短句。封面与第 1 幕同在 sOpen；“拟新增”一幕暂不入片。"""
import json, re
s = open('script.md').read(); s = s[s.index('## v7 配音稿'):]
rows = [l for l in s.splitlines() if l.startswith('| ') and not l.startswith('| 幕') and not l.startswith('| ---')]
sc = []
for l in rows:
    c = [x.strip() for x in l.strip('|').split('|')]
    if c[0]: sc.append([c[0], []])
    sc[-1][1].append([c[2], c[3]])
def cues(text, lim=24):
    out = []
    for p in (p.strip() for p in re.findall(r"[^，。：；？！]+[，。：；？！]?", text) if p.strip()):
        t = p.rstrip('，。：；')
        if out and len(out[-1]) + len(t) + 1 <= lim and not out[-1].endswith(('？', '！')): out[-1] += t if out[-1].endswith('：') else '，' + t
        else: out.append(p if p.endswith('：') else t)
    return out
by = {k: [c for t, _ in v for c in cues(t)] for k, v in sc}
names = [k for k, _ in sc]
film = [by[names[0]] + by[names[1]]] + [by[n] for n in names[2:] if not n.startswith('（拟新增）')]
assert len(film) == 14, len(film)
def est(t): return round(len(re.findall(r'[\u4e00-\u9fff]', t)) / 4.2 + len(re.findall(r'[A-Za-z0-9][A-Za-z0-9.,%\-]*', t)) * .4, 2)
bk = {k: [{'text': t, 'cues': cues(t), 'dur': est(t)} for t, _ in v] for k, v in sc}
beats = [bk[names[0]] + bk[names[1]]] + [bk[n] for n in names[2:] if not n.startswith('（拟新增）')]
open('beats.js', 'w').write("// 由 subs.py 生成：每幕的节拍（一句配音一个节拍）。dur 为该句时长，配音合成前按字数估算\nwindow.BEATS = " + json.dumps(beats, ensure_ascii=False) + ";\n")
open('subs.js', 'w').write("// 由 subs.py 从 script.md 的 v7 配音稿生成：每幕一组字幕（配音合成前按字数均分时间，合成后改用逐字时间戳）\nwindow.SUBS = " + json.dumps(film, ensure_ascii=False) + ";\n")
h = open('review.html').read(); a = h.index('const VO = '); b = h.index(';\n', a)
open('review.html', 'w').write(h[:a] + 'const VO = ' + json.dumps(sc, ensure_ascii=False) + h[b:])
