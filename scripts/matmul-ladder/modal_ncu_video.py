"""matmul-ladder-tc 视频用的 profiler 数据：每一版 ncu --set full + 计时，wmma 版再录一段 nsys。

一次 H100 调用做完：计时（含 wmma_pad 对照）→ 每个 kernel 一份 ncu 报告（8192 阶，wgmma_ws 加 4096 阶）→ nsys。
--part article：文章用的计时（cuBLAS + 阶梯 256–8192 阶、对照与微调变体）与连续 2 s 的功耗，不跑 ncu。
--part extra：第 2 版的拆分对照（mma_sync_ld 搬运经寄存器、mma_noswz 不换位）、两种指令的峰值（tc_peak.cu）、wmma 的 SASS。
ncu 在 Modal 容器里只能 --clock-control none（不锁频）。结果先写 Volume 再回传。
用法（先 unset 掉 *_proxy）：
  modal run scripts/matmul-ladder/modal_ncu_video.py [--part ladder|extra|nsys_soak]
输出：public/measurements/matmul-ladder-tc/<时间戳>/
"""
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "public" / "measurements" / "matmul-ladder-tc"
app = modal.App("matmul-ladder-tc-video-ncu")
vol = modal.Volume.from_name("inference-study-results", create_if_missing=True)
NVCC_TC = "nvcc -O3 -arch=sm_90a -std=c++17 -lineinfo -o /build/matmul_tc /src/matmul_tc.cu -lcublas -L/usr/local/cuda/lib64/stubs -lnvidia-ml"
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("cuda-nsight-compute-12-8", "cuda-nsight-systems-12-8")
         .run_commands("mkdir -p /build /src")
         .add_local_file(HERE / "matmul_tc.cu", "/src/matmul_tc.cu", copy=True)
         .run_commands(NVCC_TC)
         .add_local_file(HERE / "tc_peak.cu", "/src/tc_peak.cu", copy=True)
         .run_commands("nvcc -O3 -arch=sm_90a -std=c++17 -lineinfo -o /build/tc_peak /src/tc_peak.cu"))

LADDER = ["wmma", "wmma_pad", "mma", "mma_async", "wgmma", "wgmma_ws", "wgmma_l2", "wgmma_pipe", "wgmma_persist",
          "wgmma_cluster", "wgmma_tstore"]
NCU_RUNS = [(k, 8192) for k in LADDER] + [("wgmma_ws", 4096), ("wgmma_l2", 4096)]
ART_LADDER = [k for k in LADDER if k != "wmma_pad"]
ART_MICRO = ["wgmma_hilbert", "wgmma_micro", "wgmma_tstore_c1", "wgmma_micro_c1"]
EXTRA = ["wmma", "wmma_pad", "mma_sync_ld", "mma_noswz", "mma", "mma_async"]


@app.function(image=image, gpu="H100", timeout=900, max_containers=1, retries=0, volumes={"/out": vol})
def run(stamp: str, part: str) -> dict:
    import glob
    import subprocess
    t0 = time.time()
    d = Path(f"/out/matmul-ladder-tc/{stamp}")
    d.mkdir(parents=True, exist_ok=True)
    files = {}

    def sh(cmd, name=None):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        text = f"$ {cmd}\n{r.stdout}{r.stderr}"
        if name:
            (d / name).write_text(text)
            files[name] = text
            vol.commit()
        return r

    q = subprocess.run("nvidia-smi --query-gpu=name,power.limit,driver_version --format=csv,noheader", shell=True,
                       capture_output=True, text=True).stdout.strip()
    if "HBM3" not in q or "700.00 W" not in q:
        raise SystemExit(f"不是 H100 SXM 700 W：{q}")
    ncu = (glob.glob("/opt/nvidia/nsight-compute/*/ncu") + glob.glob("/usr/local/cuda/bin/ncu"))[0]
    nsys = (glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys") + glob.glob("/usr/local/cuda/bin/nsys"))[0]
    sh(f"nvidia-smi; nvcc --version; {ncu} --version; {nsys} --version", "env.txt")
    if part == "nsys_soak":                                         # 和文章同口径：连续跑 2 s，看后段每次的耗时
        sh(f"{nsys} profile --trace=cuda --force-overwrite=true -o {d}/nsys-wmma-soak /build/matmul_tc soak wmma 8192", "nsys-wmma-soak.log")
        sh(f"{nsys} stats --report cuda_gpu_trace --format csv {d}/nsys-wmma-soak.nsys-rep", "nsys-wmma-soak-trace.csv")
        files["gpu-seconds.txt"] = f"{time.time() - t0:.1f}\n"
        (d / "gpu-seconds.txt").write_text(files["gpu-seconds.txt"])
        vol.commit()
        return files
    if part == "article":
        sh(f"/build/matmul_tc cublas,{','.join(ART_LADDER)}", "timing-ladder.txt")
        sh(f"/build/matmul_tc cublas,{','.join(EXTRA)} 8192", "timing-ablation-8192.txt")
        sh(f"/build/matmul_tc cublas,wgmma_tstore,{','.join(ART_MICRO)} 2048,4096,8192", "timing-micro.txt")
        sh("/build/matmul_tc soak cublas,wgmma_tstore,wgmma_micro_c1 4096,8192", "soak.txt")
        files["gpu-seconds.txt"] = f"{time.time() - t0:.1f}\n"
        (d / "gpu-seconds.txt").write_text(files["gpu-seconds.txt"])
        vol.commit()
        return files
    if part == "extra":
        sh(f"/build/matmul_tc {','.join(EXTRA)} 8192", "timing-extra-8192.txt")
        sh("/build/tc_peak", "tc-peak.txt")
        sh("cuobjdump -sass -fun mm_wmma /build/matmul_tc | grep -E 'LDS|LDSM|HMMA|STS|LDG|BAR' | sed -E 's/ +/ /g' | sort | uniq -c | sort -rn | head -40",
           "sass-wmma-summary.txt")
        for k in ["mma_sync_ld", "mma_noswz"]:
            rep = d / f"ncu-{k}-8192"
            sh(f'{ncu} --clock-control none --set full -k "regex:^mm_" -c 1 -f -o {rep} /build/matmul_tc {k} 8192', f"ncu-{k}-8192.log")
            sh(f"{ncu} -i {rep}.ncu-rep --page raw --csv", f"ncu-{k}-8192-raw.csv")
        for k in ["peak_mma", "peak_wgmma"]:
            rep = d / f"ncu-{k}"
            sh(f'{ncu} --clock-control none --set full -k "regex:{k}" -c 1 -f -o {rep} /build/tc_peak', f"ncu-{k}.log")
            sh(f"{ncu} -i {rep}.ncu-rep --page raw --csv", f"ncu-{k}-raw.csv")
        files["gpu-seconds.txt"] = f"{time.time() - t0:.1f}\n"
        (d / "gpu-seconds.txt").write_text(files["gpu-seconds.txt"])
        vol.commit()
        return files
    sh(f"/build/matmul_tc {','.join(LADDER)} 8192", "timing-8192.txt")
    sh(f"{ncu} --query-metrics --chip gh100 | grep -iE 'tensor|bank|ldsm|stall' ", "ncu-metrics-available.txt")
    for k, n in NCU_RUNS:
        rep = d / f"ncu-{k}-{n}"
        sh(f'{ncu} --clock-control none --set full --import-source yes -k "regex:^mm_" -c 1 -f -o {rep} /build/matmul_tc {k} {n}',
           f"ncu-{k}-{n}.log")
        sh(f"{ncu} -i {rep}.ncu-rep --page raw --csv", f"ncu-{k}-{n}-raw.csv")
        sh(f"{ncu} -i {rep}.ncu-rep --page details --print-details all", f"ncu-{k}-{n}-details.txt")
    sh(f"{nsys} profile --trace=cuda --force-overwrite=true -o {d}/nsys-wmma /build/matmul_tc wmma 8192", "nsys-wmma.log")
    sh(f"{nsys} stats --report cuda_gpu_trace --format csv {d}/nsys-wmma.nsys-rep", "nsys-wmma-trace.csv")
    files["gpu-seconds.txt"] = f"{time.time() - t0:.1f}\n"
    (d / "gpu-seconds.txt").write_text(files["gpu-seconds.txt"])
    vol.commit()
    return files


@app.local_entrypoint()
def main(part: str = "ladder"):
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("" if part == "ladder" else f"-{part}")
    t0 = time.time()
    files = run.remote(stamp, part)
    out = OUT / stamp
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    print(f"[本地] {len(files)} 个文件写入 {out}，remote 总耗时 {time.time() - t0:.1f} s；.ncu-rep / .nsys-rep 在 Volume inference-study-results:/matmul-ladder-tc/{stamp}/")
