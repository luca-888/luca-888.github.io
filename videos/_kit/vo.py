"""python3 videos/_kit/vo.py <视频目录> [--speed S] [--redo s01 s03 ...] [--dry-run]
  读 narration.md（每幕一个 "## 幕 id"，每行一句；可选的 "## 读法" 一节每行 "原文 → 读法"）
  -> assets/vo/<幕>-<句>.wav（每句一个）与 assets/vo/vo.json（每句的文字、文件、实测时长、开头静音），录音缓存在 assets/vo/cache/。
  之后在 index.html 里按 vo.json 放 <audio>、按句写字幕（HyperFrames 的 captions 做法）。
MiniMax TTS（音色与模型见 AGENTS.md，读法规范见 docs/tts.md）。按幕合成：一幕要新录的句子一次送出，句间插入停顿标记，返回的字幕文件按标记分句。
缓存按句复用：改过的句子才重录，没变的沿用原录音；--redo s03 整幕重录。--dry-run 只列出要合成的句子与预计计费，不调用接口。
读法：通用规则在 videos/lib/tts.py，本片的术语读法写在 narration.md 的 "## 读法"（原文按字面匹配）。
key 从仓库根目录 .env.local 的 MINIMAX_API_KEY 读取。"""
import hashlib, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT.parent / "lib"))
from tts import say

VOICE, MODEL, PAUSE = "ttv-voice-2026100701390026-S2EIdGRY", "speech-2.8-hd", "<#0.5#>"
PRICE = 3.5 / 10000   # 元 / 计费字符；1 个汉字计 2 个字符

args = sys.argv[1:]
if not args or args[0].startswith("--"): sys.exit(__doc__)
here = Path(args[0]).resolve()
speed = float(args[args.index("--speed") + 1]) if "--speed" in args else 1.15
redo = set(args[args.index("--redo") + 1:]) - {"--dry-run"} if "--redo" in args else set()
dry = "--dry-run" in args
# narration.md → 幕列表与读法
scenes, extra, section = [], [], None
for line in (here / "narration.md").read_text().splitlines():
    line = line.strip()
    if line.startswith("## "):
        section = line[3:].strip()
        if section != "读法": scenes.append({"id": section, "lines": []})
    elif not line or line.startswith("#") or section is None: continue
    elif section == "读法":
        if "→" in line: a, b = [x.strip() for x in line.split("→", 1)]; extra.append((re.escape(a), b))
    else: scenes[-1]["lines"].append(line)
out = here / "assets/vo"; cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
key = lambda text: hashlib.sha1(f"{VOICE}|{MODEL}|{speed}|{text}".encode()).hexdigest()[:16]

def duration(f):
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", subprocess.run(["ffmpeg", "-i", str(f)], capture_output=True, text=True).stderr)
    return int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))

def synth(text):
    env = KIT.parent.parent / ".env.local"
    api_key = next((l.split("=", 1)[1].strip() for l in env.read_text().splitlines() if l.startswith("MINIMAX_API_KEY=")), "")
    if not api_key: sys.exit("MINIMAX_API_KEY missing in .env.local")
    payload = {"model": MODEL, "text": text, "stream": False, "language_boost": "Chinese", "subtitle_enable": True,
        "voice_setting": {"voice_id": VOICE, "speed": speed, "vol": 1.0, "pitch": 0}, "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3", "channel": 1}}
    req = urllib.request.Request("https://api.minimaxi.com/v1/t2a_v2", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: res = json.loads(r.read())
    if res["base_resp"]["status_code"]: sys.exit(f"MiniMax: {res['base_resp']}")
    for i in range(5):   # 字幕文件下载偶尔断连：重试，避免重复付费合成
        try:
            with urllib.request.urlopen(res["data"]["subtitle_file"], timeout=60) as r: sub = json.loads(r.read()); break
        except Exception as e:
            if i == 4: raise
            print("  字幕下载失败，重试：", e); time.sleep(3)
    return bytes.fromhex(res["data"]["audio"]), [[x["time_begin"] / 1000, x["time_end"] / 1000] for x in sub], res["extra_info"]["usage_characters"]

# 缓存里每段录音按停顿标记分句，得到“送 TTS 的句子 → (录音, 人声起止, 可裁剪的前后界)”
def segments(h):
    m = json.loads((cache / f"{h}.json").read_text()); bounds, total = m["bounds"], duration(cache / f"{h}.mp3")
    parts = m["text"].split(PAUSE); assert len(parts) == len(bounds), h
    cuts = [0.0] + [(bounds[i][1] + bounds[i + 1][0]) / 2 for i in range(len(parts) - 1)] + [total]
    return {t: (cache / f"{h}.mp3", *bounds[i], cuts[i], cuts[i + 1]) for i, t in enumerate(parts)}
SENT = {}
for mf in sorted(cache.glob("*.json"), key=lambda p: p.stat().st_mtime):   # 只认同一音色、模型与语速的录音（文件名即哈希）
    if key(json.loads(mf.read_text())["text"]) == mf.stem: SENT.update(segments(mf.stem))

lines, billed, plan = {}, 0, []
for sc in scenes:
    texts = sc["lines"]
    says = [say(t, extra) for t in texts]
    need = says if sc["id"] in redo else [s for s in says if s not in SENT]
    if need:
        text = PAUSE.join(need)
        chars = sum(2 if "一" <= c <= "鿿" else 1 for c in text.replace(PAUSE, ""))
        plan.append((sc["id"], len(need), chars))
        if dry: continue
        audio, bounds, used = synth(text)
        assert len(bounds) == len(need), (sc["id"], len(bounds), len(need))
        h = key(text)
        (cache / f"{h}.mp3").write_bytes(audio); (cache / f"{h}.json").write_text(json.dumps({"text": text, "bounds": bounds, "usage": used}, ensure_ascii=False))
        SENT.update(segments(h)); billed += used; print(f"  {sc['id']}: 新合成 {len(need)} 句，计费 {used} 字符")
    if dry: continue
    lines[sc["id"]] = []
    for n, (t, s) in enumerate(zip(texts, says), 1):
        src, t0, t1, c0, c1 = SENT[s]; a, z = max(c0, t0 - .2), min(c1, t1 + .3)
        wav = out / f"{sc['id']}-{n:02d}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{z:.3f}", "-i", str(src), "-af",
            f"afade=t=in:d=0.03,afade=t=out:st={z - a - .05:.3f}:d=0.05", "-ar", "48000", "-ac", "2", str(wav)], check=True)
        lines[sc["id"]].append({"text": t, "file": f"assets/vo/{wav.name}", "dur": round(t1 - t0, 2), "head": round(t0 - a, 2), "len": round(z - a, 2)})

if dry:
    total = sum(c for *_, c in plan)
    for sid, n, c in plan: print(f"  {sid}: 要合成 {n} 句，约 {c} 计费字符")
    print(f"共约 {total} 计费字符，约 {total * PRICE:.2f} 元" if plan else "全部句子都有录音，不需要合成")
    sys.exit()
(out / "vo.json").write_text(json.dumps({"speed": speed, "voice": VOICE, "model": MODEL, "scenes": lines}, ensure_ascii=False, indent=1))
print(f"speed {speed}；{sum(len(v) for v in lines.values())} 句；本次计费 {billed} 字符（约 {billed * PRICE:.2f} 元）→ assets/vo/vo.json")
