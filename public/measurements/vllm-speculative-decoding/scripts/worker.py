"""One vLLM configuration per process. Run only through run_locked.sh."""
import argparse, hashlib, json, os, sys, time, platform, importlib.metadata as md
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--method", default="baseline")   # baseline | draft_model | ngram | eagle3
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--draft-sample", default=None)  # greedy | probabilistic (draft_model only)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--concurrency", default="1,4,16,32")
ap.add_argument("--tasks", default="chat,summ")
ap.add_argument("--temps", default="0,0.8")
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--mrv1", action="store_true", help="force the V1 model runner (matched control for methods that fall back to it)")
ap.add_argument("--no-async", action="store_true", help="disable async scheduling")
ap.add_argument("--deadline", type=float, default=0, help="unix time after which to stop between cells")
a = ap.parse_args()

HUB = "/root/.cache/huggingface/hub"
def snap(repo):
    d = Path(HUB) / ("models--" + repo.replace("/", "--")) / "snapshots"
    return str(next(d.iterdir()))
TARGET, DRAFT, EAGLE = snap("Qwen/Qwen3-8B"), snap("Qwen/Qwen3-0.6B"), snap("AngelSlim/Qwen3-8B_eagle3")

spec = None
if a.method == "draft_model":
    spec = {"method": "draft_model", "model": DRAFT, "num_speculative_tokens": a.k}
    if a.draft_sample:
        spec["draft_sample_method"] = a.draft_sample
elif a.method in ("ngram", "ngram_gpu"):
    spec = {"method": a.method, "num_speculative_tokens": a.k, "prompt_lookup_min": 2, "prompt_lookup_max": 5}
elif a.method == "eagle3":
    spec = {"method": "eagle3", "model": EAGLE, "num_speculative_tokens": a.k}

if a.mrv1:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
from vllm import LLM, SamplingParams
import vllm, torch

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
kw = dict(model=TARGET, dtype="bfloat16", max_model_len=2048, max_num_seqs=32,
          gpu_memory_utilization=0.92, enable_prefix_caching=False, seed=0, disable_log_stats=False)
if spec:
    kw["speculative_config"] = spec
if a.no_async:
    kw["async_scheduling"] = False
t_load = time.time()
llm = LLM(**kw)
load_s = time.time() - t_load
meta = {"name": a.name, "method": a.method, "k": a.k, "spec": spec, "engine_kwargs": {k: v for k, v in kw.items() if k != "speculative_config"},
        "vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(), "load_s": load_s, "max_tokens": a.max_tokens, "mrv1": a.mrv1, "no_async": a.no_async}
(out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

prompts = json.load(open(a.prompts))
SPEC_KEYS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens", "num_accepted_tokens_per_pos")
def spec_snapshot():
    snap_ = {}
    for m in llm.get_metrics():
        n = m.name.replace("vllm:spec_decode_", "")
        n = n[:-6] if n.endswith("_total") else n
        if n in SPEC_KEYS:
            snap_[n] = list(m.values) if hasattr(m, "values") else m.value
    return snap_
def diff(after, before):
    d = {}
    for k, v in after.items():
        b = before.get(k, [0] * len(v) if isinstance(v, list) else 0)
        d[k] = [x - y for x, y in zip(v, b)] if isinstance(v, list) else v - b
    return d

done = out / "cells.jsonl"
finished = set()
if done.exists():
    for line in done.read_text().splitlines():
        r = json.loads(line); finished.add((r["task"], r["temp"], r["c"]))

def run_wave(ps, sp):
    t0 = time.perf_counter()
    res = llm.generate(ps, sp, use_tqdm=False)
    return time.perf_counter() - t0, res

for task in a.tasks.split(","):
    for temp in [float(x) for x in a.temps.split(",")]:
        for c in [int(x) for x in a.concurrency.split(",")]:
            if (task, temp, c) in finished:
                continue
            if a.deadline and time.time() > a.deadline:
                print("deadline reached; stopping before", task, temp, c, flush=True); sys.exit(3)
            waves = 2 if a.smoke else (8 if c == 1 else 3)
            pool = prompts[task]
            sp = SamplingParams(temperature=temp, top_p=1.0, max_tokens=a.max_tokens, seed=0 if temp == 0 else None)
            run_wave(pool[:c], SamplingParams(temperature=temp, max_tokens=16))  # warmup (also graph capture paths)
            before = spec_snapshot()
            wall, ntok, outs = 0.0, 0, []
            for w in range(waves):
                ps = [pool[(w * c + i) % len(pool)] for i in range(c)]
                dt, res = run_wave(ps, sp)
                wall += dt
                for r in res:
                    o = r.outputs[0]; ntok += len(o.token_ids)
                    outs.append({"wave": w, "n_in": len(r.prompt_token_ids), "n_out": len(o.token_ids), "ids": list(o.token_ids) if temp == 0 else None,
                                 "finish": o.finish_reason})
            after = spec_snapshot()
            rec = {"task": task, "temp": temp, "c": c, "waves": waves, "wall_s": wall, "out_tokens": ntok,
                   "tok_per_s": ntok / wall, "per_req_tok_per_s": ntok / wall / c,
                   "spec": diff(after, before) if spec else None, "outputs": outs}
            with done.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{a.name}] {task} T={temp} c={c}: {ntok / wall:.1f} tok/s total, {ntok / wall / c:.1f} per req", flush=True)
(out / "complete.json").write_text(json.dumps({"complete": True, "t": time.time()}))
