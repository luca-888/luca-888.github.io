"""python3 data.py [results.json]  -> data.js：把 H100 实测（public/measurements/fused-linear-cross-entropy/<时间戳>/results.json）整理成画面用的数据。
不带参数时取最新一次测量。显存一律换算成 GiB（画面上写作 GB），时间为毫秒。"""
import glob, json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(str(here / "../../public/measurements/fused-linear-cross-entropy/*/results.json")))[-1]
r = json.loads(Path(src).read_text())
G = 2 ** 30
gb = lambda b: round(b / G, 3)


def case(d):
    if not isinstance(d, dict) or "peak_bytes" not in d:
        return {"oom": True} if isinstance(d, dict) and d.get("oom") else {"error": (d or {}).get("error", "missing")}
    return {"gb": gb(d["peak_bytes"]), "ms": round(d["ms_median"], 2)}


out = {"src": str(Path(src).relative_to(here.parent.parent)), "env": r.get("env"), "shape": r.get("shape")}
mic = r.get("micro", {})
out["micro"] = {k: case(v) for k, v in mic.items()} if isinstance(mic, dict) else {}
sw = r.get("sweep", {})
out["sweep"] = {k: case(v) for k, v in sw.items()} if isinstance(sw, dict) else {}
out["sm90"] = {k: case(v) for k, v in r.get("sm90_shape", {}).items()} if isinstance(r.get("sm90_shape"), dict) else {}
out["correctness"] = r.get("correctness")
# kernel 时间：按类别合计（GEMM、CE kernel、其余）
kern = {}
for m, rows in (r.get("kernels") or {}).items():
    if not isinstance(rows, list):
        continue
    cat = {}
    for name, n, ms in rows:
        low = name.lower()
        k = ("gemm" if any(s in low for s in ("gemm", "sm90_xmma", "cutlass", "nvjet", "ampere", "wgmma", "matmul"))
             else "ce" if "cross_entropy" in low else "softmax" if "softmax" in low or "nll" in low
             else "other")
        cat[k] = round(cat.get(k, 0) + ms, 3)
    kern[m] = {"by_cat": cat, "top": rows[:8]}
out["kernels"] = kern
e2e = r.get("e2e") or {}
if "runs" in e2e:
    out["e2e"] = {"params_gb": gb(e2e["params_bytes"]), "runs": {}}
    for k, v in e2e["runs"].items():
        out["e2e"]["runs"][k] = ({"gb": gb(v["peak_bytes"]), "base_gb": gb(v["base_bytes"]), "s": round(v["step_s"], 2)}
                                 if "peak_bytes" in v else {"oom": True} if v.get("oom") else {"skipped": True})
    out["timeline"] = {}
    for m, tl in (e2e.get("timeline") or {}).items():
        pts = [[round(t / 1e6, 4), gb(b)] for t, b in tl["points"]]
        out["timeline"][m] = {"start_gb": gb(tl["start_bytes"]), "points": pts,
                              "peak_gb": max(p[1] for p in pts), "end_s": pts[-1][0],
                              "big": [{"gb": gb(a["size"]), "t": round(a["t"] / 1e6, 4), "frame": a["frame"]} for a in tl["big_allocs"]]}
(here / "data.js").write_text("// 由 data.py 从 " + out["src"] + " 生成\nwindow.DATA = " + json.dumps(out, ensure_ascii=False) + ";\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("timeline",)}, ensure_ascii=False, indent=1)[:6000])
for m, tl in out.get("timeline", {}).items():
    print(m, "start", tl["start_gb"], "peak", tl["peak_gb"], "end", tl["end_s"], "s; big:", [(b["gb"], b["t"], b["frame"]) for b in tl["big"][:12]])
