# 矩阵乘法阶梯视频（上集 · FP32）：画面素材出处

视频没有使用外部图片。画面上的一手材料均来自作者自己的实测仓库 `inference-study`（H100 80GB HBM3 SXM，700 W，CUDA 12.8.1，Modal，2026-10-20 至 21），原文照录。

## 叙事结构的参考（只借结构，内容与数字均为本片实测）

| 文章 | 作者 | 用于 | 查询日期 |
| --- | --- | --- | --- |
| [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) | Simon Boehm | 开头先给成绩表、纸面估算（最少 FLOPs 与访存）、Kernel 1 前的线程层级 primer、每个 kernel 以上一版的 profiler 证据开头 | 2026-10-21 |
| [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | Pranjal Shankhdhar | Hopper 段的顺序（TMA/wgmma → warp 特化 → 调度 → 集群多播 → 异步写回）、照实记录变慢的步骤、结尾按矩阵规模对比与功耗墙、Hilbert 曲线调度（下集末幕“未完成”一项，该文中约 +1%） | 2026-10-21 |

## 画面素材

| 画面 | 内容 | 出处 |
| --- | --- | --- |
| 开场、第 9 幕阶梯 | 各版本 8192 阶 TFLOPS 与 cuBLAS 基线；tuned 44.53 | `labs/1.3-matmul-flops/out/ladder.json`（原样输出 `ladder-final-20261021-121144.txt`） |
| 第 2 幕带宽 | 显存拷贝实测 3051 GB/s | `labs/1.2-hbm-bandwidth/out/copy-20261020-182319.txt` |
| 第 3 幕终端 | SM count 132、smem per SM 228 KiB、registers per SM 65536 | `labs/1.1-h100-anatomy/out/probe-20261020-172840.txt` |
| 第 3 幕线程上限 | 每 SM 2048 线程、每 block 1024 线程 | CUDA 计算能力 9.0 规格 |
| 第 4–8 幕代码 | 各版的改动行（超宽行截断为 …） | `labs/1.3-matmul-flops/matmul.cu` |
| 第 4–8 幕 SM 剖面 | 面板布局与各版状态的语义 | `tracker/pages/s1.3-matmul-flops.html` 的 `HW` 与 `hwState` |
| 第 6 幕寄存器数 | ptxas：matmul_reg 用 104 个寄存器 | `ladder-final-20261021-121144.txt` 的 ptxas 输出 |
| 第 7 幕 bank 冲突 | vec 读 B 为 2 路冲突（推算：相邻线程 rB 相隔 32 B，8 个线程跨 256 B） | 文章 `content/matmul-ladder-fp32.md` 的推算 |
| 第 8 幕 ncu | FMA 管线 68.3%（dbuf）、79.8%（cuBLAS SGEMM） | `labs/1.3-matmul-flops/out/ncu-fp32-20261021-120232.txt` |
| 第 2、9 幕规格 | FP32 峰值约 67、BF16 dense 989 TFLOPS；每 SM 每时钟 FP32 128、Tensor Core 2048 次乘加 | NVIDIA H100 白皮书 / 规格表 |
| “复杂度与下界”一幕的平衡点 | BF16 dense 989 TFLOPS ÷ HBM 3.35 TB/s ≈ 295 FLOP/B；强度上限 2N³ ÷ (3N² × 4 B) = N/6 FLOP/B（算术） | NVIDIA H100 SXM 规格 |

## 理论出处（“复杂度与下界”一幕）

| 内容 | 出处 | 核对情况 |
| --- | --- | --- |
| ω < 2.371339（2024） | Alman, Duan, Vassilevska Williams, Xu, Xu, Zhou, *More Asymmetry Yields Faster Matrix Multiplication*, arXiv 2404.16349（v1 2024-04-25） | 2026-10-21 用 WebFetch 读摘要页：当前 v3（2026-08-19 修订）摘要写的是 ω < 2.371177；2.371339 为 v1 的结果，经 2608.16884 摘要中“previous best of 2.371339”交叉核对 |
| ω < 2.371177（2026，AlphaEvolve 参与） | Dupont, Eisenberger, Kozlovskii, Mehrabian, Ruiz, See, Zhou, Alman, Vassilevska Williams, Balog, *Improving the matrix multiplication exponent with modern optimization and AlphaEvolve*, arXiv 2608.16884（2026-08-17） | 2026-10-21 用 WebFetch 读摘要页 |
| Strassen：7 次乘法，O(N^2.807) | V. Strassen, *Gaussian elimination is not optimal*, 1969 | 通行结论，未读原文 |
| I/O 下界 Ω(N³/√M)，分块可达到 | Jia-Wei Hong, H. T. Kung, *I/O complexity: The red-blue pebble game*, STOC 1981（doi 10.1145/800076.802486） | 2026-10-21 标题、作者、会议经 Semantic Scholar 核对；ACM 页面 403，**未读原文**，结论按通行引述 |


## 文章配图（视频静帧）

`content/matmul-ladder-fp32.md` 的 9 张 PNG 取自本视频定稿（2026-10-21 v5d）的画面：在 `videos/matmul-ladder-fp32/` 运行 `node render.mjs --stills --scale 2`（deviceScaleFactor 2，输出 `build/stills@2x/`），取节拍结束前 0.05 s 的帧，按下表的 CSS 像素区域裁剪（像素为其 2 倍），再用 Pillow 量化为 256 色 PNG。文章按 CSS 宽度显示（`matmul-ladder-fp32.blocks.tsx` 的 `CSS_WIDTH`），字号与视频原画面相同。2026-10-22 导出。

2026-10-22 修整：裁剪框改为“画面元素的外接框四周各加 24 CSS 像素”，保证文字与标签完整；框内属于相邻元素的像素（SM 剖面的指向箭头、旁边面板的边缘、下方文字）用底色 #f1ede4 遮去。`matrix.png` 两段之间留 28 CSS 像素。

| 文件 | 静帧 | 裁剪（CSS 像素，x0, y0, x1, y1） |
| --- | --- | --- |
| `matrix.png` | 第 1 幕 17.4 s（外积四层叠加）：矩阵在上、公式在下拼接 | 79, 245, 806, 843；975, 274, 1588, 412 |
| `grid.png` | 第 3 幕 4.8 s（grid） | 817, 253, 1683, 860 |
| `block-warp.png` | 第 3 幕 15.4 s（block 与 warp） | 732, 253, 1835, 774 |
| `naive.png` | 第 4 幕 7.8 s（相邻线程读相邻地址、5.6 倍） | 77, 290, 770, 1018 |
| `smem.png` | 第 5 幕 15.0 s（k0 循环、As / Bs） | 80, 264, 697, 957 |
| `bank.png` | 第 5 幕 51.3 s（32 个 bank） | 85, 285, 816, 751 |
| `eighth.png` | 第 5 幕 51.3 s（128 个 FP32 单元亮 16 个、1/8） | 961, 254, 1481, 770 |
| `reg-issue.png` | 第 6 幕 35.4 s（16 条读取 + 64 条乘加） | 953, 273, 1627, 618 |
| `ceiling.png` | 第 10 幕 24.3 s（FP32 128 对 Tensor Core 2048、阶梯、强度刻度） | 961, 218, 1910, 975 |
