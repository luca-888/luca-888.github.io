"""（改自 videos/dspark/vo.py）python3 vo.py [幕数] [--speed S] [--force]  -> build/vo/full-<幕数>-<S>.mp3、每句一个 build/vo/b-<幕>-<句>.wav 与 vo.js。
MiniMax TTS（音色与模型见 AGENTS.md）。前 <幕数> 幕的配音稿（beats.js，每句一个节拍）一次合成，语气连贯；句与句之间插入停顿标记 <#0.5#>，
返回的字幕文件按标记分句，每句一组起止时间。再在句间静音的中点切开：画面按节拍排时间（句间空档由各幕的 pre 决定），每句的录音放到它的节拍开头。
语速用接口的 speed 参数（--speed S，重新合成），不对录音做变速；句间空档改各幕 beatTimes 的 pre。不带幕数时合成全部幕。
key 从仓库根目录 .env.local 的 MINIMAX_API_KEY 读取；稿子与语速没变时不重复合成。"""
import base64, json, re, subprocess, sys, urllib.request
from pathlib import Path

here = Path(__file__).resolve().parent
VOICE, MODEL, PAUSE = "ttv-voice-2026100701390026-S2EIdGRY", "speech-2.8-hd", "<#0.5#>"
SAY = {"Qwen3-1.7B": "Qwen3 1.7B", "cross-entropy": "cross entropy", "log-sum-exp": "log sum exp", "log_softmax": "log softmax", "lm_head": "LM head", "one-hot": "one hot", "dX": "d X", "dW": "d W"}   # 只改送去合成的写法（连字符、下划线会被读成停顿），字幕不变
args = sys.argv[1:]
speed = float(args[args.index("--speed") + 1]) if "--speed" in args else 1.0
nums = [a for i, a in enumerate(args) if a.isdigit() and (i == 0 or args[i - 1] != "--speed")]
s = (here / "beats.js").read_text(); beats = json.loads(s[s.index("["):s.rindex("]") + 1])
n = int(nums[0]) if nums else len(beats); beats = beats[:n]
out = here / "build/vo"; out.mkdir(parents=True, exist_ok=True)
flat = [b for g in beats for b in g]
def say(t):
    for k, v in SAY.items(): t = t.replace(k, v)
    return t
full = PAUSE.join(say(b["text"]) for b in flat)
meta, mp3 = out / f"full-{n}-{speed}.json", out / f"full-{n}-{speed}.mp3"
if not meta.exists() or json.loads(meta.read_text())["text"] != full or "--force" in args:
    key = next((l.split("=", 1)[1].strip() for l in (here / "../../.env.local").read_text().splitlines() if l.startswith("MINIMAX_API_KEY=")), "")
    if not key: sys.exit("MINIMAX_API_KEY missing in .env.local")
    payload = {"model": MODEL, "text": full, "stream": False, "language_boost": "Chinese", "subtitle_enable": True,
        "voice_setting": {"voice_id": VOICE, "speed": speed, "vol": 1.0, "pitch": 0}, "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3", "channel": 1}}
    req = urllib.request.Request("https://api.minimaxi.com/v1/t2a_v2", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: res = json.loads(r.read())
    if res["base_resp"]["status_code"]: sys.exit(f"MiniMax: {res['base_resp']}")
    mp3.write_bytes(bytes.fromhex(res["data"]["audio"]))
    with urllib.request.urlopen(res["data"]["subtitle_file"], timeout=60) as r: sub = json.loads(r.read())
    meta.write_text(json.dumps({"text": full, "sub": sub, "usage": res["extra_info"]["usage_characters"]}, ensure_ascii=False))
    print("characters billed this run:", res["extra_info"]["usage_characters"])
sub = json.loads(meta.read_text())["sub"]
assert len(sub) == len(flat), (len(sub), len(flat))
# 本机只有 ffmpeg（没有 ffprobe）：从 ffmpeg 的输出里读时长
_d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", subprocess.run(["ffmpeg", "-i", str(mp3)], capture_output=True, text=True).stderr)
total = int(_d.group(1)) * 3600 + int(_d.group(2)) * 60 + float(_d.group(3))
est = lambda t: len(re.findall(r"[\u4e00-\u9fff]", t)) / 4.2 + len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%\-]*", t)) * .4
bounds = [(x["time_begin"] / 1000, x["time_end"] / 1000) for x in sub]   # 每句发声的起止
cuts = [0.0] + [(bounds[i][1] + bounds[i + 1][0]) / 2 for i in range(len(bounds) - 1)] + [total]
vo, i = [], 0
for k, g in enumerate(beats):
    row = []
    for j, b in enumerate(g):
        t0, t1 = bounds[i]; a, z = max(cuts[i], t0 - .25), min(cuts[i + 1], t1 + .35)
        f = out / f"b-{k}-{j}.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{z:.3f}", "-i", str(mp3), "-af",
            f"afade=t=in:d=0.03,afade=t=out:st={z - a - .05:.3f}:d=0.05", "-ar", "48000", "-ac", "2", str(f)], check=True)
        # 字幕短句的起点：接口只给整句的起止，句内按估算时长（汉字 4.2 字/秒，英文词与数字各 0.4 秒）分配
        w = [est(c) for c in b["cues"]]; cs = [round(sum(w[:q]) / sum(w), 3) for q in range(len(w))]
        row.append({"text": b["text"], "dur": round(t1 - t0, 2), "head": round(t0 - a, 3), "file": f"build/vo/{f.name}", "cs": cs})
        i += 1
    vo.append(row)
(here / "vo.js").write_text("// 由 vo.py 生成：配音的实测时长（文字与 beats.js 一致时覆盖按字数估算的 dur）与每句录音的位置\nwindow.VO = " + json.dumps({"speed": speed, "beats": vo}, ensure_ascii=False)
    + ";\nwindow.VO.beats.forEach((g, k) => g.forEach((v, j) => { const b = (window.BEATS[k] || [])[j]; if (b && b.text === v.text) { b.dur = v.dur; b.cs = v.cs; b.vo = v; } }));\n")
sp = sum(b - a for a, b in bounds); han = len(re.findall(r"[\u4e00-\u9fff]", "".join(b["text"] for b in flat)))
print(f"voice {total:.1f}s, speech {sp:.1f}s at speed {speed}; {han} 汉字; 句间静音 {min(bounds[i + 1][0] - bounds[i][1] for i in range(len(bounds) - 1)):.2f}–{max(bounds[i + 1][0] - bounds[i][1] for i in range(len(bounds) - 1)):.2f}s")
# 切点处的电平（应为静音）
for c in cuts[1:-1]:
    r = subprocess.run(["ffmpeg", "-ss", f"{c - .05:.3f}", "-t", "0.1", "-i", str(mp3), "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = float(re.search(r"max_volume: (-?[\d.]+|-inf)", r).group(1).replace("-inf", "-99"))
    if m > -45: print(f"  cut at {c:.2f}s is not silent: {m} dB")
for k, g in enumerate(vo): print(f"  幕 {k}: " + " ".join(f"{v['dur']:.2f}" for v in g))
