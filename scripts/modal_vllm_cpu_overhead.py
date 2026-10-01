"""vLLM CPU 开销文章实测：Modal 单卡 H100 上跑 async / MRV2 / CUDA Graph 配置矩阵。

用法（modal 在 ~/miniforge3/bin）：
  modal run scripts/modal_vllm_cpu_overhead.py::prepare   # CPU 容器：构建镜像、下载权重，不占卡
  modal run scripts/modal_vllm_cpu_overhead.py::check     # 本地：只打印矩阵与命令，不启动任何容器
  modal run scripts/modal_vllm_cpu_overhead.py::measure   # 申请 H100，跑完整矩阵（须用户确认后）
计划与预算见 docs/vllm-cpu-overhead-notes.md。
"""
import json
import os
import subprocess
import time

import modal

VLLM_VERSION = "0.30.0"
MODELS = {"qwen3-8b": "Qwen/Qwen3-8B"}  # 2026-09-30 用户决定只测 8B；0.6B 权重已下载，需要时加回
CONCURRENCY = [1, 8, 64, 256]
INPUT_LEN, OUTPUT_LEN = 128, 256
PROFILE_CONCURRENCY = [8, 64]  # profile 单独跑，输出更短，避免 trace 过大
PROFILE_OUTPUT_LEN = 32
DEADLINE_S = 105 * 60  # 超过后不再启动新配置，保存已有结果（GPU 占用上限 2 h，留出收尾）
PORT = 8000

# name -> (额外 serve 参数, 环境变量)
CONFIGS = {
    "default": ([], {}),
    "no_async": (["--no-async-scheduling"], {}),
    "mrv1": ([], {"VLLM_USE_V2_MODEL_RUNNER": "0"}),
    "piecewise": (["--compilation-config", json.dumps({"cudagraph_mode": "PIECEWISE"})], {}),
    "graph_none": (["--compilation-config", json.dumps({"cudagraph_mode": "NONE"})], {}),
}

HF, RESULTS, CACHE = "/hf", "/results", "/vllm-cache"
app = modal.App("vllm-engine-measure")
hf_vol = modal.Volume.from_name("vllm-engine-hf", create_if_missing=True)
res_vol = modal.Volume.from_name("vllm-engine-results", create_if_missing=True)
cache_vol = modal.Volume.from_name("vllm-engine-cache", create_if_missing=True)
image = (
    # FlashInfer 采样 kernel 需要 JIT 编译，须带 nvcc；CUDA 版本与 torch 2.13.0+cu130 一致
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": HF, "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_CACHE_ROOT": CACHE,
          "FLASHINFER_WORKSPACE_BASE": CACHE, "FLASHINFER_CUDA_ARCH_LIST": "9.0a"})
)
VOLUMES = {HF: hf_vol, RESULTS: res_vol, CACHE: cache_vol}


@app.function(image=image, volumes={HF: hf_vol}, timeout=3600, cpu=4, memory=16384)
def download_weights():
    from huggingface_hub import snapshot_download

    for repo in MODELS.values():
        snapshot_download(repo)
    hf_vol.commit()
    return {"vllm": VLLM_VERSION, "models": list(MODELS.values())}


def _wait_ready(proc, timeout=900):
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return time.time() - t0
        except Exception:
            time.sleep(3)
    raise TimeoutError("server not ready")


def _bench(model, out_dir, name, conc, num_prompts, out_len, profile=False):
    cmd = [
        "vllm", "bench", "serve", "--model", model, "--port", str(PORT),
        "--dataset-name", "random", "--random-input-len", str(INPUT_LEN),
        "--random-output-len", str(out_len), "--num-prompts", str(num_prompts),
        "--max-concurrency", str(conc), "--ignore-eos", "--num-warmups", "8",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result",
        "--result-dir", out_dir, "--result-filename", name + ".json", "--disable-tqdm",
    ]
    if profile:
        cmd.append("--profile")
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(f"{out_dir}/{name}.log", "w").write(r.stdout + "\n" + r.stderr)
    return r.returncode


def _run_one(cfg_name, model_key, run_dir):
    extra, env = CONFIGS[cfg_name]
    model = MODELS[model_key]
    out = f"{run_dir}/{cfg_name}/{model_key}"
    os.makedirs(out, exist_ok=True)
    trace_dir = f"{out}/traces"
    prof = json.dumps({"profiler": "torch", "torch_profiler_dir": trace_dir, "torch_profiler_with_stack": False})
    cmd = ["vllm", "serve", model, "--port", str(PORT), "--max-model-len", "2048",
           "--profiler-config", prof, *extra]
    log = open(f"{out}/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
    meta = {"config": cfg_name, "model": model, "cmd": cmd, "env": env, "status": "ok"}
    try:
        meta["startup_s"] = _wait_ready(proc)
        for c in CONCURRENCY:
            _bench(model, out, f"c{c}", c, max(64, 4 * c), OUTPUT_LEN)
        for c in PROFILE_CONCURRENCY:
            _bench(model, out, f"profile_c{c}", c, 2 * c, PROFILE_OUTPUT_LEN, profile=True)
    except Exception as e:  # 记录后继续下一个配置，不重试
        meta["status"] = f"error: {e}"
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    # 回退检查：日志里 async / MRV2 是否被自动关闭，由本地分析时读取 server.log
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    res_vol.commit()
    return meta


@app.function(image=image, gpu="H100", volumes=VOLUMES, timeout=2 * 3600, cpu=8, memory=65536, retries=0, max_containers=1)
def run_matrix(configs: list | None = None):
    t0 = time.time()
    run_dir = f"{RESULTS}/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir)
    env_info = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True).stdout
    open(f"{run_dir}/nvidia-smi.xml", "w").write(env_info)
    open(f"{run_dir}/env.txt", "w").write(subprocess.run("pip freeze | grep -iE 'vllm|torch|triton'; lscpu | head -20", shell=True, capture_output=True, text=True).stdout)
    done = []
    for cfg in (configs or list(CONFIGS)):
        for m in MODELS:
            if time.time() - t0 > DEADLINE_S:
                done.append({"config": cfg, "model": m, "status": "skipped: deadline"})
                continue
            done.append(_run_one(cfg, m, run_dir))
    json.dump({"elapsed_s": time.time() - t0, "runs": done}, open(f"{run_dir}/summary.json", "w"), indent=1)
    res_vol.commit()
    return {"run_dir": run_dir, "elapsed_s": time.time() - t0, "statuses": [(d["config"], d["model"], d["status"]) for d in done]}


@app.local_entrypoint()
def prepare():
    print(download_weights.remote())


@app.local_entrypoint()
def check():
    for cfg, (extra, env) in CONFIGS.items():
        for key, model in MODELS.items():
            print(cfg, key, env, "vllm serve", model, *extra)
    print("concurrency", CONCURRENCY, "profile", PROFILE_CONCURRENCY, "deadline_s", DEADLINE_S)


@app.local_entrypoint()
def measure():
    print(run_matrix.remote())


@app.function(image=image, timeout=600, cpu=2, memory=8192)
def validate_args():
    """CPU 容器：只解析 serve / bench 参数，不启动引擎，不占卡。"""
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"  # 无 GPU 时仅为通过设备推断，不影响参数解析
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.benchmarks import serve as bench_serve

    sp = EngineArgs.add_cli_args(FlexibleArgumentParser())
    out = {}
    for cfg, (extra, _env) in CONFIGS.items():
        prof = json.dumps({"profiler": "torch", "torch_profiler_dir": "/x", "torch_profiler_with_stack": False})
        ns = sp.parse_args(["--model", MODELS["qwen3-8b"], "--max-model-len", "2048", "--profiler-config", prof, *extra])
        ea = EngineArgs.from_cli_args(ns)
        out[cfg] = {"async_scheduling": ea.async_scheduling, "compilation_config": str(ea.compilation_config)[:80], "profiler": str(ea.profiler_config)[:80]}
    bp = FlexibleArgumentParser()
    bench_serve.add_cli_args(bp)
    bp.parse_args(["--model", "m", "--port", "8000", "--dataset-name", "random", "--random-input-len", "128", "--random-output-len", "256",
                   "--num-prompts", "64", "--max-concurrency", "8", "--ignore-eos", "--num-warmups", "8",
                   "--percentile-metrics", "ttft,tpot,itl,e2el", "--save-result", "--result-dir", "/x", "--result-filename", "a.json",
                   "--disable-tqdm", "--profile"])
    out["bench"] = "ok"
    return out


@app.local_entrypoint()
def validate():
    print(json.dumps(validate_args.remote(), indent=1))


@app.function(image=image, volumes={CACHE: cache_vol}, timeout=1800, cpu=8, memory=16384)
def warm_flashinfer():
    """CPU 容器：确认 nvcc 可用，并预编译 FlashInfer 采样模块（H100 = sm90a）到缓存 Volume，减少占卡时的编译。"""
    nvcc = subprocess.run("nvcc --version | tail -2", shell=True, capture_output=True, text=True).stdout
    from flashinfer.jit.sampling import gen_sampling_module

    spec = gen_sampling_module()
    spec.build()  # 只编译，不加载（无 GPU）
    cache_vol.commit()
    return {"nvcc": nvcc, "built": str(spec.name)}


@app.local_entrypoint()
def warm():
    print(warm_flashinfer.remote())
