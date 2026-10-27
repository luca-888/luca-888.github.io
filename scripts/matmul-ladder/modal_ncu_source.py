"""从 Volume 里的 ncu 报告导出逐行（CUDA 源码行）统计：每行的停顿采样、执行次数等，视频画面里“停在哪一行”用。

只读报告，不占 GPU。用法（先 unset 掉 *_proxy）：
  modal run scripts/matmul-ladder/modal_ncu_source.py --stamp 20261027-134747 --kernels wmma-8192,wmma_pad-8192,mma-8192
输出：public/measurements/matmul-ladder-tc/<stamp>/ncu-<kernel>-source.csv
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "public" / "measurements" / "matmul-ladder-tc"
app = modal.App("matmul-ladder-tc-ncu-source")
vol = modal.Volume.from_name("inference-study-results", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("cuda-nsight-compute-12-8")
         .run_commands("mkdir -p /src")
         .add_local_file(HERE / "matmul_tc.cu", "/src/matmul_tc.cu", copy=True))


@app.function(image=image, cpu=2, timeout=900, max_containers=1, retries=0, volumes={"/out": vol})
def export(stamp: str, kernels: str) -> dict:
    import glob
    import subprocess
    ncu = glob.glob("/opt/nvidia/nsight-compute/*/ncu")[0]
    d = Path(f"/out/matmul-ladder-tc/{stamp}")
    files = {}
    for k in kernels.split(","):
        r = subprocess.run(f"{ncu} -i {d}/ncu-{k}.ncu-rep --page source --print-source cuda,sass --csv", shell=True,
                           capture_output=True, text=True)
        files[f"ncu-{k}-source.csv"] = r.stdout + r.stderr
    return files


@app.local_entrypoint()
def main(stamp: str, kernels: str):
    files = export.remote(stamp, kernels)
    for name, text in files.items():
        (OUT / stamp / name).write_text(text)
        print(name, len(text))
