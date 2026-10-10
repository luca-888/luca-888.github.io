"""python3 vo.py [--speed S] [--redo N ...]
  -> build/vo/cache/<hash>.mp3（每幕一段，按文案与语速缓存）、每句一个 build/vo/b-<幕>-<句>.wav，以及 vo.js（每句的实测时长）。
MiniMax TTS（音色与模型见 AGENTS.md，读法规范见 docs/tts.md）。按幕合成：一幕的配音稿（script.js 的节拍）一次送出，
句间插入停顿标记，返回的字幕文件按标记分句。文案与语速没变的幕复用缓存；--redo 3 5 只重录第 3、5 幕（从 1 数）。
合成后核对每句是否在下一个节拍之前说完，超出的列出来，改 script.js 的节拍而不是加快语速。
key 从仓库根目录 .env.local 的 MINIMAX_API_KEY 读取。"""
import hashlib, json, re, subprocess, sys, urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "lib"))
from tts import say

VOICE, MODEL, PAUSE = "ttv-voice-2026100701390026-S2EIdGRY", "speech-2.8-hd", "<#0.5#>"
# 本片自己的术语读法（通用规则在 videos/lib/tts.py）
EXTRA = []
args = sys.argv[1:]
speed = float(args[args.index("--speed") + 1]) if "--speed" in args else 1.15
redo = {int(a) - 1 for a in args[args.index("--redo") + 1:] if a.isdigit()} if "--redo" in args else set()
script = json.loads(subprocess.run(["node", "-e", "const fs=require('fs');eval(fs.readFileSync('script.js','utf8')+';console.log(JSON.stringify(SCRIPT))')"], cwd=here, capture_output=True, text=True, check=True).stdout)
out = here / "build/vo"; cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)

def duration(f):
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", subprocess.run(["ffmpeg", "-i", str(f)], capture_output=True, text=True).stderr)
    return int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))

def synth(text):
    api_key = next((l.split("=", 1)[1].strip() for l in (here / "../../.env.local").read_text().splitlines() if l.startswith("MINIMAX_API_KEY=")), "")
    if not api_key: sys.exit("MINIMAX_API_KEY missing in .env.local")
    payload = {"model": MODEL, "text": text, "stream": False, "language_boost": "Chinese", "subtitle_enable": True,
        "voice_setting": {"voice_id": VOICE, "speed": speed, "vol": 1.0, "pitch": 0}, "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3", "channel": 1}}
    req = urllib.request.Request("https://api.minimaxi.com/v1/t2a_v2", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: res = json.loads(r.read())
    if res["base_resp"]["status_code"]: print(f"  MiniMax: {res['base_resp']}"); return None
    with urllib.request.urlopen(res["data"]["subtitle_file"], timeout=60) as r: sub = json.loads(r.read())
    return bytes.fromhex(res["data"]["audio"]), [[x["time_begin"] / 1000, x["time_end"] / 1000] for x in sub], res["extra_info"]["usage_characters"]

vo, billed, late = [], 0, []
for k, sc in enumerate(script):
    lines = [b[2] for b in sc["beats"]]
    text = PAUSE.join(say(l, EXTRA) for l in lines)
    h = hashlib.sha1(f"{VOICE}|{MODEL}|{speed}|{text}".encode()).hexdigest()[:16]; mp3, meta = cache / f"{h}.mp3", cache / f"{h}.json"
    if not meta.exists() or k in redo:
        got = synth(text)
        if got is None:   # 合成失败（如余额不足）：这一幕先不配音，按字数估时长，其余幕照常
            vo.append([{"text": l, "dur": round(len(l) * 0.17, 2), "head": 0, "file": ""} for l in lines]); print(f"  幕 {k + 1}: 未合成"); continue
        audio, bounds, used = got
        mp3.write_bytes(audio); meta.write_text(json.dumps({"text": text, "bounds": bounds, "usage": used}, ensure_ascii=False))
        billed += used; print(f"  幕 {k + 1}: 新合成，计费 {used} 字符")
    bounds = json.loads(meta.read_text())["bounds"]   # 每句发声的起止，相对这一幕的录音
    assert len(bounds) == len(lines), (k, len(bounds), len(lines))
    total = duration(mp3)
    cuts = [0.0] + [(bounds[i][1] + bounds[i + 1][0]) / 2 for i in range(len(lines) - 1)] + [total]
    row = []
    for j, l in enumerate(lines):
        t0, t1 = bounds[j]; a, z = max(cuts[j], t0 - .2), min(cuts[j + 1], t1 + .3)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{z:.3f}", "-i", str(mp3), "-af",
            f"afade=t=in:d=0.03,afade=t=out:st={z - a - .05:.3f}:d=0.05", "-ar", "48000", "-ac", "2", str(out / f"b-{k}-{j}.wav")], check=True)
        row.append({"text": l, "dur": round(t1 - t0, 2), "head": round(t0 - a, 2), "file": f"build/vo/b-{k}-{j}.wav"})
        start = sc["beats"][j][0]; limit = sc["beats"][j + 1][0] if j + 1 < len(lines) else sc["dur"]
        if start + (t1 - t0) > limit - 0.15: late.append(f"  幕 {k + 1} 句 {j + 1}: {start:.1f} + {t1 - t0:.2f} s 超过 {limit:.1f}")
    vo.append(row)
(here / "vo.js").write_text("// 由 vo.py 生成：每句配音的实测时长（秒）、录音开头的静音与录音位置，顺序与 script.js 的节拍一致\nconst VO = "
                            + json.dumps({"speed": speed, "beats": vo}, ensure_ascii=False) + ";\n")
print(f"speed {speed}; {sum(len(g) for g in vo)} 句; 本次计费 {billed} 字符")
for k, g in enumerate(vo): print(f"  幕 {k + 1}: " + " ".join(f"{v['dur']:.2f}" for v in g) + "   节拍 " + " ".join(str(b[0]) for b in script[k]["beats"]) + f"  幕长 {script[k]['dur']}")
print("超出节拍的句子：\n" + "\n".join(late) if late else "每句都在下一个节拍之前说完")
