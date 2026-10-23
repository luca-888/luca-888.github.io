"""1.3 矩阵乘阶梯：FP32 线（matmul.cu）和 Tensor Core 线（matmul_tc.cu）一版一版实测。

编译放在镜像构建里（CPU，不占卡），ptxas 的寄存器/溢出报告存进镜像，运行时原样贴进输出。
用法（先 unset 掉 *_proxy）：
  modal run labs/1.3-matmul-flops/modal_ladder.py --line fp32 --kernels vec,warptile --ns 4096,8192
  modal run labs/1.3-matmul-flops/modal_ladder.py --line tc --kernels all
  modal run labs/1.3-matmul-flops/modal_ladder.py --line all --tag final --ptxas --soak "cublas,wgmma_l2,wgmma_cluster,wgmma_tstore 4096,8192"
      # 全量：两条线所有版本 + cuBLAS 基线，最后加一段持续负载
输出原样写到 labs/1.3-matmul-flops/out/ladder-<line|tag>-<时间戳>.txt，Volume inference-study-results 留一份。
"""
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("is-1-3-ladder")
vol = modal.Volume.from_name("inference-study-results", create_if_missing=True)

NVCC_FP32 = "nvcc -O2 -arch=sm_90 -Xptxas -v -o /build/matmul /src/matmul.cu -lcublas"
NVCC_TC = "nvcc -O3 -arch=sm_90a -std=c++17 -Xptxas -v -o /build/matmul_tc /src/matmul_tc.cu -lcublas -L/usr/local/cuda/lib64/stubs -lnvidia-ml"
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .run_commands("mkdir -p /build /src")
         .add_local_file(HERE / "matmul.cu", "/src/matmul.cu", copy=True)
         .run_commands(f"{NVCC_FP32} > /build/matmul.ptxas.txt 2>&1 || (cat /build/matmul.ptxas.txt; exit 1)")
         .add_local_file(HERE / "matmul_tc.cu", "/src/matmul_tc.cu", copy=True)
         .run_commands(f"{NVCC_TC} > /build/matmul_tc.ptxas.txt 2>&1 || (cat /build/matmul_tc.ptxas.txt; exit 1)"))


@app.function(image=image, gpu="H100", timeout=900, max_containers=1, retries=0, volumes={"/out": vol})
def run(stamp: str, name: str, line: str, kernels: str, ns: str, ptxas: bool, soak: str = "") -> str:
    import subprocess
    t0 = time.time()
    q = subprocess.run("nvidia-smi --query-gpu=name,power.limit --format=csv,noheader", shell=True,
                       capture_output=True, text=True).stdout.strip()
    if "HBM3" not in q or not q.endswith("700.00 W"):   # Modal 的 H100 可能是 NVL
        raise SystemExit(f"不是 H100 SXM 700 W：{q}")
    log = [f"$ nvidia-smi --query-gpu=name,power.limit --format=csv,noheader\n{q}\n"]
    jobs = []
    if line in ("fp32", "all"):
        jobs.append((NVCC_FP32, "/build/matmul.ptxas.txt", "/build/matmul"))
    if line in ("tc", "all"):
        jobs.append((NVCC_TC, "/build/matmul_tc.ptxas.txt", "/build/matmul_tc"))
    for nvcc, ptx, exe in jobs:
        if ptxas:
            log.append(f"$ {nvcc}   # 镜像构建时运行\n{Path(ptx).read_text()}")
        cmd = f"{exe} {kernels} {ns}".strip()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        log.append(f"$ {cmd}\n{r.stdout}{r.stderr}")
    if soak and line in ("tc", "all"):                    # 持续负载：每个 kernel 连跑 2 s，看功耗墙下的频率和吞吐
        cmd = f"/build/matmul_tc soak {soak}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        log.append(f"$ {cmd}\n{r.stdout}{r.stderr}")
    log.append(f"# GPU 容器内耗时 {time.time() - t0:.1f} s")
    text = "\n".join(log)
    Path("/out/1.3").mkdir(parents=True, exist_ok=True)
    Path(f"/out/1.3/{name}").write_text(text)
    vol.commit()
    return text


@app.local_entrypoint()
def main(line: str = "all", kernels: str = "all", ns: str = "", tag: str = "", ptxas: bool = False, soak: str = ""):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"ladder-{tag or line}-{stamp}.txt"
    t0 = time.time()
    text = run.remote(stamp, name, line, kernels, ns, ptxas, soak)
    (HERE / "out" / name).write_text(text)
    print(text)
    print(f"[本地] 写入 out/{name}，remote 调用总耗时 {time.time() - t0:.1f} s")
