"""把 modal_ncu_video.py 的 ncu raw CSV 汇总成 summary.json（视频概念稿用的数字）。

用法：python3 scripts/matmul-ladder/video_ncu_summary.py public/measurements/matmul-ladder-tc/<时间戳>（目录里每个 ncu-*-raw.csv 一项）
单位统一：时间 ms，字节 GB，比例 %。停顿原因用 smsp__average_warps_issue_stalled_*_per_issue_active（不含 selected），
share 是该原因占全部停顿的百分比。
"""
import csv
import json
import re
import sys
from pathlib import Path

SCALE = {"ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1, "msecond": 1, "s": 1e3, "second": 1e3,
         "byte": 1e-9, "Kbyte": 1e-6, "Mbyte": 1e-3, "Gbyte": 1, "Tbyte": 1e3}


def load(path):
    lines = path.read_text().split("\n", 1)[1].splitlines()     # 第一行是命令
    h, u, v = list(csv.reader(lines[:3]))
    def get(name):
        i = h.index(name)
        x = v[i].replace(",", "")
        return float(x) * SCALE.get(u[i], 1) if x not in ("", "n/a") else None
    return h, get


def main(d):
    d = Path(d)
    out = {}
    for f in sorted(d.glob("ncu-*-raw.csv")):
        m = re.match(r"ncu-(.+?)(?:-(\d+))?-raw\.csv", f.name)
        k, n = m.group(1), int(m.group(2) or 0)
        h, get = load(f)
        st = {re.sub(r"smsp__average_warps_issue_stalled_|_per_issue_active\.ratio", "", x): get(x) or 0
              for x in h if x.startswith("smsp__average_warps_issue_stalled_") and x.endswith("_per_issue_active.ratio")}
        st.pop("selected", None)
        tot = sum(st.values())
        wf = get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")
        bc = get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
        out[f"{k}-{n}" if n else k] = {
            "kernel": k, "N": n,
            "time_ms": get("gpu__time_duration.sum"),
            "tflops": 2 * n ** 3 / (get("gpu__time_duration.sum") * 1e-3) / 1e12 if n else None,
            "tensor_pipe_active_pct": get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
            "issue_active_pct": get("smsp__issue_active.avg.pct_of_peak_sustained_active"),
            "sm_clock_ghz": get("smsp__cycles_elapsed.avg.per_second"),
            "smem_pipe_pct": get("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed"),
            "l1tex_throughput_pct": get("l1tex__throughput.avg.pct_of_peak_sustained_active"),
            "l2_overall_throughput_pct": get("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
            "dram_read_gb": get("dram__bytes_read.sum"),
            "l2_hit_pct": get("lts__t_sector_hit_rate.pct"),
            "l2_throughput_pct": get("lts__t_sectors.avg.pct_of_peak_sustained_elapsed"),
            "smem_ld_wavefronts_per_request": wf / (wf - bc) if wf and bc is not None and wf > bc else None,
            "smem_ld_bank_conflicts": bc,
            "stalls": {a: {"per_issue": round(b, 3), "share_pct": round(100 * b / tot, 1)}
                       for a, b in sorted(st.items(), key=lambda t: -t[1]) if b > 0},
        }
    (d / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for key, r in out.items():
        top = ", ".join(f"{a} {s['share_pct']}%" for a, s in list(r["stalls"].items())[:3])
        print(f"{key:20s} {r['time_ms']:7.3f} ms  TC {r['tensor_pipe_active_pct']:5.1f}%  DRAM {r['dram_read_gb']:.2f} GB  "
              f"L2 hit {r['l2_hit_pct']:.1f}%  smem {r['smem_pipe_pct']:.0f}%  L1 {r['l1tex_throughput_pct']:.0f}%  L2 {r['l2_overall_throughput_pct']:.0f}%  {top}")


if __name__ == "__main__":
    main(sys.argv[1])
