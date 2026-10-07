"""python3 bgm.py  -> build/bgm.wav：一段简单的 8-bit 风格循环（方波琶音 + 低音 + 噪声镲），时长取 timing.js。
节拍 96 bpm，四个和弦一轮（Am F C G）；对战中间加镲，开头与结尾只有琶音。film.mjs 混音时把它压在配音下面。"""
import json, math, random, struct, wave
from pathlib import Path

here = Path(__file__).resolve().parent
tm = json.loads((here / "timing.js").read_text().replace("window.TIMING = ", "").rstrip().rstrip(";"))
DUR, SR, BPM = tm["duration"] + 1.0, 48000, 96   # 132 时用户觉得偏快（成片再乘 1.1 倍速后更快）
step = 60 / BPM / 4                      # 十六分音符
N = lambda n: 440 * 2 ** ((n - 69) / 12)  # MIDI 音高 -> Hz
CHORDS = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]]   # Am F C G
PAT = [0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 3, 1, 2]
sq = lambda ph, duty: 1.0 if (ph % 1.0) < duty else -1.0
random.seed(3)
out = []
for i in range(int(DUR * SR)):
    t = i / SR; k = int(t / step); bar = (k // 16) % 4; pos = k % 16; u = t - k * step
    chord = CHORDS[bar]
    arp = sq(t * N(chord[PAT[pos]] + 12), .25) * math.exp(-u * 9) * .5
    bass = sq(t * N(chord[0] - 12), .5) * (math.exp(-(t % (step * 4)) * 5)) * .42
    busy = 9.1 < t < 49.6
    hat = (random.random() * 2 - 1) * math.exp(-u * 60) * (.22 if busy and pos % 2 == 0 else 0)
    kick_t = t % (step * 8); kick = math.sin(2 * math.pi * (50 + 90 * math.exp(-kick_t * 30)) * kick_t) * math.exp(-kick_t * 14) * (.5 if busy else 0)
    fade = min(1, t / .4) * min(1, max(0, (DUR - t) / 1.5))
    out.append((arp + bass + hat + kick) * .45 * fade)
# 简单低通，去掉方波的刺耳高频
y, a = 0.0, .22
(here / "build").mkdir(exist_ok=True)
with wave.open(str(here / "build/bgm.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    buf = bytearray()
    for x in out:
        y += a * (x - y); buf += struct.pack("<h", int(max(-1, min(1, y)) * 32000))
    w.writeframes(bytes(buf))
print(f"bgm {DUR:.1f}s")
