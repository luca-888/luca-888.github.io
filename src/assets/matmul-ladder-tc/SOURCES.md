# 矩阵乘法阶梯视频（下集 · Tensor Core）：画面素材出处

视频没有使用外部图片。画面上的一手材料均来自作者自己的实测仓库 `inference-study`（H100 80GB HBM3 SXM，700 W，CUDA 12.8.1，Modal，2026-10-20 至 21），原文照录。

## 叙事结构的参考（只借结构，内容与数字均为本片实测）

| 文章 | 作者 | 用于 | 查询日期 |
| --- | --- | --- | --- |
| [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) | Simon Boehm | 开头先给成绩表、纸面估算（最少 FLOPs 与访存）、Kernel 1 前的线程层级 primer、每个 kernel 以上一版的 profiler 证据开头 | 2026-10-21 |
| [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) | Pranjal Shankhdhar | Hopper 段的顺序（TMA/wgmma → warp 特化 → 调度 → 集群多播 → 异步写回）、照实记录变慢的步骤、结尾按矩阵规模对比与功耗墙、Hilbert 曲线调度（下集末幕“未完成”一项，该文中约 +1%） | 2026-10-21 |

## 画面素材

| 画面 | 内容 | 出处 |
| --- | --- | --- |
| 第 1、11 幕阶梯 | 各版本 8192 阶 TFLOPS 与 cuBLAS 基线 | `labs/1.3-matmul-flops/out/ladder.json`（原样输出 `ladder-final-20261021-121144.txt`） |
| 第 2–9 幕代码 | 各版的改动行（超宽行截断为 …） | `labs/1.3-matmul-flops/matmul_tc.cu` |
| 第 2–9 幕 SM 剖面 | 面板布局与各版状态的语义 | `tracker/pages/s1.3-matmul-flops.html` 的 `HW` 与 `hwState` |
| 第 7 幕对照 | 集群版非常驻 686.18、常驻 721.96（同一轮） | `labs/1.3-matmul-flops/out/ladder-dev6-20261021-120951.txt` |
| 第 8 幕 L2 流量 | 2048 tile × 128 步 × 48 KB ÷ 1.485 ms ≈ 8.7 TB/s；集群后 48 KB → 32 KB，约 5.9 TB/s（均为推算） | 文章 `content/matmul-ladder-tc.md` 的推算 |
| 第 8 幕 ncu | Tensor 管线 92.9%（wgmma_cluster）、97.0%（cuBLAS） | `ncu-tc-20261021-120213.txt`、`ncu-tc-20261021-120328.txt` |
| 第 10 幕终端 | `matmul_tc soak` 两行（节选，未改动）与各规模数字 | `ladder-final-20261021-121144.txt` L329–408 |
| 第 1、2 幕规格 | BF16 dense 989 TFLOPS；每 SM 每时钟 2048 次乘加 | NVIDIA H100 白皮书 / 规格表 |

## 文章静帧（`content/matmul-ladder-tc.md` 引用，2026-10-22）

视频定稿后用 `node render.mjs --stills --scale 2`（在 `videos/matmul-ladder-tc/` 运行）导出各拍结束处的 2 倍静帧（3840 × 2160），只裁剪当前机制所在的区域，CSS 宽度不超过约 1000 px，未改动画面内容；256 色无抖动量化压缩。2026-10-22 修整：裁剪框改为画面元素外接框四周各加 24 CSS 像素，框内属于相邻元素的像素用底色 #f1ede4 遮去；`ws-l2.png`、`tile-order.png` 两段之间留 28 CSS 像素。`sizes.json` 记录每张图的 CSS 宽度，正文按这一宽度封顶显示。

| 文件 | 静帧（幕 / 秒，CSS 像素区域） | 用于 |
| --- | --- | --- |
| `tc-rate.png`、`tc-instr.png` | 1 / 20.6 s，左侧点阵、右侧指令面板 | 一、每时钟乘加次数；标量与矩阵块乘加 |
| `wmma.png` | 2 / 35.1 s，硬件塔与 bank 面板 | 二、寄存器中转、4 路 bank 冲突 |
| `swizzle.png` | 3 / 22.8 s，换位后的网格 | 三、8 轮 → 1 轮 |
| `stages.png` | 4 / 9.3 s，矩阵与槽位环 | 三、三级缓冲 |
| `wgmma-path.png`、`wgmma-parts.png` | 5 / 26.6 s，硬件塔；代码、mbarrier、warpgroup、指令体积 | 四 |
| `ws.png` | 6 / 21.9 s，生产者、槽位、消费者、寄存器柱 | 五 |
| `ws-l2.png` | 6 / 47.6 s，工作集条与 tile 图上下拼接 | 五、按行发放 136 MB |
| `tile-order.png` | 7 / 31.8 s，工作集条与 tile 图上下拼接 | 六、分组发放 65 MB |
| `l2-read.png` | 7 / 31.8 s，SM 与 L2 读出 | 八、8.7 TB/s |
| `cluster.png` | 8 / 35.4 s，集群与多播（右上代码框的边缘用底色遮去） | 八 |
| `tstore-path.png`、`tstore-time.png` | 9 / 17.3 s，硬件塔；时间线与共享内存预算 | 九 |
| `power.png`、`micro.png` | 10 / 37.9 s，左右两半 | 十、功耗墙；补充实测（`labs/1.3-matmul-flops/out/micro-20261022-105500.json`） |
