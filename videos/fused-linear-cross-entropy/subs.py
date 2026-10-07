"""python3 subs.py  -> beats.js：每幕的节拍（一句配音一个节拍，带切好的字幕短句与按字数估算的时长）。
来源是 script.md“配音稿”表格：第一列非空的行开始新的一幕，第三列是这一句配音。配音合成后 vo.js 用实测时长覆盖 dur。"""
import json, re
from pathlib import Path

here = Path(__file__).resolve().parent
s = (here / "script.md").read_text(); s = s[s.index("## 配音稿"):]
rows = [l for l in s.splitlines() if l.startswith("| ") and not l.startswith("| 幕") and not l.startswith("| ---")]
scenes = []
for l in rows:
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if c[0]: scenes.append([])
    scenes[-1].append(c[2])


def cues(text, lim=26):
    out = []
    for p in (p.strip() for p in re.findall(r"[^，。：；？！]+[，。：；？！]?", text) if p.strip()):
        t = p.rstrip("，。；")
        if out and len(out[-1]) + len(t) + 1 <= lim and not out[-1].endswith(("？", "！")):
            out[-1] += t if out[-1].endswith("：") else "，" + t
        else:
            out.append(t)
    return [o.rstrip("：") if o.endswith("：") and o is out[-1] else o for o in out]


def est(t):
    return round(len(re.findall(r"[一-鿿]", t)) / 4.2 + len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%\-]*", t)) * .4, 2)


beats = [[{"text": t, "cues": cues(t), "dur": est(t)} for t in g] for g in scenes]
(here / "beats.js").write_text("// 由 subs.py 生成：每幕的节拍（一句配音一个节拍）。dur 在配音合成前按字数估算\nwindow.BEATS = "
                              + json.dumps(beats, ensure_ascii=False) + ";\n")
todo = [t for g in scenes for t in g if "【" in t]
print(f"{len(beats)} scenes, {sum(map(len, beats))} beats, ~{sum(b['dur'] for g in beats for b in g):.0f}s speech;"
      f" {len(todo)} beats still have placeholders")
