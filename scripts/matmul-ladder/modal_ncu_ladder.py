"""1.3 矩阵乘阶梯：用 ncu 看 cuBLAS 和手写 kernel 的差别（N=8192，未锁频：Modal 容器里必须 --clock-control none）。

用法（先 unset 掉 *_proxy）：
  modal run labs/1.3-matmul-flops/modal_ncu_ladder.py --line tc --runs "cublas:sm90|gemm,wgmma_cluster:mm_wgmma_ws"
每一项是 "kernel 名:ncu 的 kernel 名正则"，各抓第一次调用。输出原样写到 out/ncu-<line>-<时间戳>.txt。
"""
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("is-1-3-ncu-ladder")
vol = modal.Volume.from_name("inference-study-results", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("cuda-nsight-compute-12-8")
         .run_commands("mkdir -p /build /src")
         .add_local_file(HERE / "matmul.cu", "/src/matmul.cu", copy=True)
         .run_commands("nvcc -O2 -arch=sm_90 -o /build/matmul /src/matmul.cu -lcublas")
         .add_local_file(HERE / "matmul_tc.cu", "/src/matmul_tc.cu", copy=True)
         .run_commands("nvcc -O3 -arch=sm_90a -std=c++17 -o /build/matmul_tc /src/matmul_tc.cu -lcublas -L/usr/local/cuda/lib64/stubs -lnvidia-ml"))
SECTIONS = "--section SpeedOfLight --section LaunchStats --section Occupancy --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis"


@app.function(image=image, gpu="H100", timeout=900, max_containers=1, retries=0, volumes={"/out": vol})
def run(name: str, line: str, runs: str, n: int) -> str:
    import glob
    import subprocess
    t0 = time.time()
    q = subprocess.run("nvidia-smi --query-gpu=name,power.limit --format=csv,noheader", shell=True,
                       capture_output=True, text=True).stdout.strip()
    if "HBM3" not in q or not q.endswith("700.00 W"):
        raise SystemExit(f"不是 H100 SXM 700 W：{q}")
    ncu = (glob.glob("/opt/nvidia/nsight-compute/*/ncu") + glob.glob("/usr/local/cuda/bin/ncu"))[0]
    exe = "/build/matmul_tc" if line == "tc" else "/build/matmul"
    log = [f"$ nvidia-smi --query-gpu=name,power.limit --format=csv,noheader\n{q}\n"]
    for item in runs.split(","):
        kname, regex = item.split(":", 1)
        cmd = f'{ncu} --clock-control none {SECTIONS} -k "regex:{regex}" -c 1 {exe} {kname} {n}'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        log.append(f"$ {cmd}\n{r.stdout}{r.stderr}")
    log.append(f"# GPU 容器内耗时 {time.time() - t0:.1f} s")
    text = "\n".join(log)
    Path("/out/1.3").mkdir(parents=True, exist_ok=True)
    Path(f"/out/1.3/{name}").write_text(text)
    vol.commit()
    return text


@app.local_entrypoint()
def main(line: str = "tc", runs: str = "", n: int = 8192):
    name = f"ncu-{line}-{time.strftime('%Y%m%d-%H%M%S')}.txt"
    t0 = time.time()
    text = run.remote(name, line, runs, n)
    (HERE / "out" / name).write_text(text)
    print(text)
    print(f"[本地] 写入 out/{name}，remote 调用总耗时 {time.time() - t0:.1f} s")
