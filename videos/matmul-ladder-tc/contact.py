"""python3 contact.py -> keyframes/v2/contact-sheet.png：每幕一行（机制帧、幕尾帧），左侧标幕号与标题，右侧附这一幕的配音稿。"""
import json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

here = Path(__file__).resolve().parent
kf = here / "keyframes/v2"
script = json.loads(subprocess.run(["node", "-e", "const fs=require('fs');eval(fs.readFileSync('script.js','utf8')+';console.log(JSON.stringify(SCRIPT))')"], cwd=here, capture_output=True, text=True, check=True).stdout)
TW, TH = 640, 360
font = lambda s: ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", s)
rows = len(script); W = 120 + 2 * (TW + 16) + 700; H = 90 + rows * (TH + 24)
sheet = Image.new("RGB", (W, H), "#e6e0d3"); d = ImageDraw.Draw(sheet)
total = sum(s["dur"] for s in script)
d.text((24, 24), f"H100 矩阵乘法 · 下（Tensor Core）v2 关键帧总览　{len(script)} 幕　预计 {int(total // 60)}:{int(total % 60):02d}（按配音估计）", fill="#141414", font=font(30))
t0 = 0
for k, s in enumerate(script):
    y = 90 + k * (TH + 24)
    d.text((24, y + 6), f"{k + 1}", fill="#141414", font=font(44))
    d.text((24, y + 64), f"{int(t0 // 60)}:{int(t0 % 60):02d}", fill="#8a8474", font=font(20))
    d.text((24, y + 92), f"{s['dur']:.0f} s", fill="#8a8474", font=font(20))
    for j, name in enumerate(["mid", "end"]):
        f = kf / f"s{k + 1:02d}-{name}.png"
        x = 120 + j * (TW + 16)
        if f.exists(): sheet.paste(Image.open(f).convert("RGB").resize((TW, TH), Image.LANCZOS), (x, y))
        else: d.rectangle([x, y, x + TW, y + TH], outline="#cfc8b8")
    tx, ty = 120 + 2 * (TW + 16) + 10, y
    d.text((tx, ty), s["tag"], fill="#141414", font=font(22))
    ty += 34
    for b in s["beats"]:
        line = b[2] or "（留白）"
        buf = ""
        for ch in line:
            buf += ch
            if d.textlength(buf, font=font(17)) > 670: d.text((tx, ty), buf, fill="#3a3a3a", font=font(17)); ty += 23; buf = ""
        if buf: d.text((tx, ty), buf, fill="#3a3a3a", font=font(17)); ty += 23
        ty += 6
    t0 += s["dur"]
sheet.save(kf / "contact-sheet.png")
print(kf / "contact-sheet.png", sheet.size)
