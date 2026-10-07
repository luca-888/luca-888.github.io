"""python3 vo.py [--force]  -> build/vo/full.mp3（ElevenLabs TTS，整篇一次合成，语气连贯）、各段 wav 与 timing.js。
整篇稿子一次请求并取回逐字时间戳；录音整条原样使用（不变速、不剪、不切段），时间戳只用来排画面和字幕。
key 从仓库根目录 .env.local 的 ELEVENLABS_API_KEY 读取；稿子没变时不重复合成。"""
import base64, json, re, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
cfg = json.loads((here / "narration.json").read_text())
out = here / "build/vo"; out.mkdir(parents=True, exist_ok=True)
texts = [s["text"] for s in cfg["segments"]]; full = "\n".join(texts)
meta = out / "full.json"
if not meta.exists() or json.loads(meta.read_text())["text"] != full or "--force" in sys.argv:
    key = next((l.split("=", 1)[1].strip() for l in (here / "../../.env.local").read_text().splitlines() if l.startswith("ELEVENLABS_API_KEY=")), "")
    if not key: sys.exit("ELEVENLABS_API_KEY missing in .env.local")
    payload = {"text": full, "model_id": cfg["model_id"], "language_code": "zh"}
    if cfg.get("voice_settings"): payload["voice_settings"] = cfg["voice_settings"]
    def ask(p):
        req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{cfg['voice_id']}/with-timestamps?output_format=mp3_44100_128", data=json.dumps(p).encode(), headers={"xi-api-key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
    try: res = ask(payload)
    except urllib.error.HTTPError as e:   # 模型不接受 voice_settings 时退回默认设置
        print("request rejected:", e.code, e.read().decode()[:300]); payload.pop("voice_settings", None); res = ask(payload)
    (out / "full.mp3").write_bytes(base64.b64decode(res["audio_base64"]))
    meta.write_text(json.dumps({"text": full, "alignment": res["alignment"]}, ensure_ascii=False))
    print("characters synthesized this run:", len(full))
al = json.loads(meta.read_text())["alignment"]
chars, st, en = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
assert "".join(chars) == full, "alignment text differs from the script"
total = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out / "full.mp3")]))

# 每段在整篇中的字符区间与起止时间；切点取相邻两段之间停顿的中点
spans, pos = [], 0
for t in texts: spans.append((pos, pos + len(t))); pos += len(t) + 1
cuts = [0.0] + [(en[spans[i][1] - 1] + st[spans[i + 1][0]]) / 2 for i in range(len(spans) - 1)] + [total]

# 配音整条原样使用：不变速、不剪停顿、不切段。这里只算时间轴：影片的每一段拉伸到对应那句话的时长上。
t, segs = 0.0, []
for i, (seg, (c0, c1)) in enumerate(zip(cfg["segments"], spans)):
    a, b = seg["film"]; t0, t1 = cuts[i], cuts[i + 1]
    # 字幕：按标点切成短句，起止取逐字时间戳（时间戳比实际发声略晚，起点提前 0.12 秒）
    cues = []
    for m in re.finditer(r"[^。？！，：]+[。？！，：]?", seg["text"]):
        p0, p1 = c0 + m.start(), c0 + m.end(); txt = m.group().strip().rstrip("，。：")
        if cues and len(cues[-1][2]) + len(txt) <= 12: cues[-1][1] = round(en[p1 - 1], 2); cues[-1][2] += txt if cues[-1][2].endswith(("？", "！")) else "，" + txt
        else: cues.append([round(max(0, st[p0] - .12), 2), round(en[p1 - 1], 2), txt])
    segs.append({"film": [a, b], "out": [round(t0, 3), round(t1, 3)], "dur": round(t1 - t0, 2), "cues": cues})
(here / "timing.js").write_text("window.TIMING = " + json.dumps({"duration": round(total, 3), "voice": "build/vo/full.mp3", "segs": segs}, ensure_ascii=False, indent=1) + ";\n")
print(f"voice {total:.1f}s")
for s in segs: print(f"  {s['film']} -> {s['out']}  speed x{(s['film'][1]-s['film'][0])/(s['out'][1]-s['out'][0]):.2f}")
