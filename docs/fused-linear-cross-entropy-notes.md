# Fused Linear Cross Entropy 笔记

状态：2026-10-07 用户确认方案：Modal H100、只做视频（`videos/fused-linear-cross-entropy/`）、暂无配套文章。配音由我决定用 MiniMax（机制讲解需要口播）。Modal 用本机登录的 workspace `zxc876163546`。

## 源码版本

| 仓库 | 位置 | 版本 |
| --- | --- | --- |
| Liger-Kernel | `~/Documents/ChatGPT/reference-repos/liger-kernel` | v0.8.4 之后的 main，`b5cdbf7`（2026-10-07） |

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | Liger `ops/fused_linear_cross_entropy.py`（chunk 公式、forward 内算梯度、`dW` 累加、`reduction="none"` 延迟到 backward） | LinkedIn Liger 团队 | 已读全文 | 核心 | 第三至五幕 |
| 2 | Liger `ops/cross_entropy.py`（online softmax 两遍扫描、梯度原地写回 logits、`dx_y` 单独用 FP32 重算） | LinkedIn Liger 团队 | 已读 kernel 主体 | 核心 | 第三幕 |
| 3 | [Liger-Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)，FLCE 一节 | Hsu, Dai, Kothapalli 等 | 已读 FLCE 与 CE 两节（tex） | 核心 | 动机、chunk 公式 |
| 4 | [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)（CCE，ICLR25） | Wijmans 等（Apple） | 已读 intro、相关工作、结果表、gradient filtering（tex） | 重要 | 相关系统对比与反对意见 |
| 5 | Liger `ops/cutedsl/ops/fused_linear_cross_entropy_sm90.py` docstring | Liger 团队 | 已读 docstring | 重要 | 现状：SM90 专用实现 |
| 6 | [mgmalek/efficient_cross_entropy](https://github.com/mgmalek/efficient_cross_entropy) | Malek | 未读，只知是 Liger 引用的原型 | 次要 | 出处 |
| 7 | torchtune chunked CE（`CEWithChunkedOutputLoss`） | PyTorch | 未读，只读 CCE 论文中的描述 | 次要 | 相关系统对比 |

## 阅读要点

- **问题**：logits 形状 BT × V，与 hidden states BT × H 相比放大 V/H 倍。Liger 论文例：Gemma 256k 词表、batch 8 × 4096，BF16 logits 16.8 GB。CCE 例：Gemma 2 2B，8192 token，loss 部分 24 GB、整个 classifier head 训练 28 GB。
- **CE kernel**：每个 token 一个 program；第一遍 online softmax 求 max 与 sum（得 lse），第二遍把 `softmax(x)/N` 写回 logits 所在显存；目标位 `dx_y` 用 FP32 单独重算后覆盖（避免 `softmax − 1` 在 BF16 下抵消）。loss = lse − x_y。
- **FLCE**：`inc_factor = ceil(V / (c·H))`，`chunk = nextpow2(ceil(BT / inc_factor))`，c 为 `chunk_mem_const`（默认 1），使每个 chunk 的 logits ≈ BT × H 个元素。每个 chunk：`mm` → CE kernel（原地得 dZ）→ `dX_chunk = dZ @ W` → `addmm(dW, dZᵀ, X_chunk)`。梯度在 forward 里算完并保存，backward 只乘 `grad_output`（为 1 时跳过）。
- **mean 的分母**：forward 前用 `target_mask.sum().item()` 取全局非 ignore token 数传给每个 chunk（一次同步），因此 chunk 之间不需要再缩放。

## 现状变化

- 论文 Remark 说按 chunk size / BT 缩放梯度；当前源码改为向 kernel 传全局 `n_non_ignore`，不再事后缩放。
- `reduction="none"` 时梯度需要逐 token 的 `grad_output`，当前版本改为 forward 只算 loss、backward 重算 logits（`defer_grads`）。
- 新增 `chunk_mem_const`（默认 1），用更多显存换更少的 chunk。
- `dW` 累加从“临时 FP32 张量再相加”改为 `torch.addmm(..., out=grad_weight)`，不再分配与参数同大的临时张量。
- 新增 CuTe DSL SM90 实现：logits 只写一次 HBM 并原地复用为 dZ，`dW` 推迟到 backward；docstring 自报 H200、M = H = 4096、V = 128256 时约 18 ms，Triton FLCE 约 141 ms（待实测核对）。

## 相关系统与反对意见

- **CCE**：logits 不写 HBM，分块在 SRAM 里算 log-sum-exp；backward 利用 softmax 稀疏性跳过可忽略的块（gradient filtering）。批评 Liger：chunk 多则省显存、chunk 少则快，两者不可兼得；Liger 在其测试中显存比 baseline 少 95%，时间超过两倍；loss 与梯度同时算，loss 上的任何变换必须写进 kernel。
- **torchtune**：按 chunk 切分 + `torch.compile`，logits 仍按 chunk 物化。

## 待核实

- Triton FLCE 在 H100 上相对 eager 的实际耗时比（CCE 测 A100 约 1.5 倍，SM90 docstring 测得更慢）。
- 端到端训练中 logits 尖峰占峰值显存的比例。

## 实验计划（Modal H100 × 1）

- **目标**：为视频提供真实数据：logits 显存尖峰、各实现的峰值显存与耗时、chunk 大小的取舍、kernel 时间构成。
- **途径**：Modal，单容器，`gpu="H100"`，脚本 `scripts/fused-linear-cross-entropy/modal_measure.py`；镜像 Python 3.12、最新 torch、`liger-kernel[cutedsl]`（commit `b5cdbf7`）、`cut-cross-entropy`、transformers。
- **矩阵**（一次远端调用内顺序执行）：
  1. 正确性：BT = 4096，loss、dX、dW 对 FP32 参考的误差（eager、eager BF16、Liger CE、FLCE、FLCE SM90、CCE）。
  2. 微基准：全片统一用 Qwen3-1.7B 的 lm_head（H = 2048、V = 151936，BF16）；BT ∈ {4K, 8K, 16K, 32K, 64K} × {eager（FP32 upcast）、Liger CE、FLCE、CCE}；指标为一步 forward + backward 的峰值显存增量和耗时中位数（warmup 3、计时 10）。OOM 记为 OOM。
  3. `chunk_mem_const` ∈ {1, 2, 4, 8, 16, 32, 128}，BT = 16384（chunk 从 256 到 16384 即不切分）。
  3b. CuTe SM90 实现要求 V 为 256 的倍数，151936 不满足；在其 docstring 的形状（M = H = 4096、V = 128256）上单独比较 eager、FLCE、FLCE SM90、CCE。
  4. kernel 构成：BT = 16384，eager 与 FLCE 各一次 profiler。
  5. 端到端：随机初始化 Qwen3-1.7B（只取 config），gradient checkpointing，AdamW（fused，纯 BF16），batch 1，seq ∈ {4K … 128K}，eager 与 FLCE 的峰值显存与单步时间；seq 16K 的一步显存时间线。
- **正确性检查**：第 1 项；各实现 loss 相对参考的误差应在 BF16 量级。
- **完成条件**：results.json 回传到 `public/measurements/fused-linear-cross-entropy/<时间戳>/`。

### 时间与费用（modal.com/pricing，2026-10-07 查询，USD）

| 项目 | 估计 |
| --- | --- |
| 准备（镜像构建、取 config） | 约 10 min，仅 CPU 容器 |
| 排队 | 不确定，不计入占卡 |
| GPU 占用 | 预计 10–20 min；函数超时 30 min，25 min 后不再启动新测试项 |
| 结果回传 | 秒级 |

费率：H100 $0.001097/s；CPU $0.0000131/核/s；内存 $0.00000222/GiB/s。容器 8 核、64 GiB：20 min ≈ $1.32 + $0.13 + $0.17 ≈ **$1.6**；30 min 上限 ≈ **$2.4**。未计 Starter 计划免费额度剩余与镜像构建的 CPU 费用（量级 < $0.1）。

### 执行约束

单次调用、容器上限 1、不重试；异常或触及上限时保存已有结果并停止，补测前更新计划。完成后确认容器已停止，记录实际占用。

## 实测记录（Modal H100 × 1，2026-10-07 23:58–00:01 CST）

- 一次调用完成全部矩阵，客户端墙钟 172 s（含冷启动），估算 GPU 费用约 $0.2；此前另有一次 CPU 容器的镜像构建与依赖检查。app 已停止。
- 环境：H100 80GB HBM3，驱动 610.57.04，torch 2.14.1+cu130，triton 3.8.0，transformers 5.19.0，liger-kernel 0.8.4（`b5cdbf7`），cut-cross-entropy 25.1.1，nvidia-cutlass-dsl 4.8.0。
- 原始数据：`public/measurements/fused-linear-cross-entropy/20261007-160054/results.json`；画面数据由 `videos/fused-linear-cross-entropy/data.py` 生成。

### 主要读数（H = 2048，V = 151936，BF16，一步 forward + backward）

| BT | eager | Liger CE | FLCE | CCE |
| --- | --- | --- | --- | --- |
| 4096 | 6.96 GB / 20.3 ms | 1.76 GB / 12.1 ms | 0.61 GB / 71.8 ms | 0.60 GB / 17.0 ms |
| 16384 | 27.8 GB / 82.7 ms | 5.28 GB / 50.0 ms | 0.72 GB / 79.4 ms | 0.64 GB / 67.5 ms |
| 65536 | OOM | 19.4 GB / 198 ms | 1.12 GB / 206 ms | 0.83 GB / 270 ms |

- `chunk_mem_const` 扫描（BT = 16384）：chunk 256（64 块）0.72 GB / 79.3 ms；512 0.79 / 58.6；1024 0.93 / 51.9；2048 1.22 / 49.8；4096 1.80 / 49.3；8192 2.96 / 49.2；16384（不切分）5.28 / 49.2。
- kernel 构成（BT = 16384，profiler）：FLCE 中 `addmm` 累加 dW 64 次共 39.4 ms，logits GEMM 17.3 ms，dX GEMM（splitK）15.6 ms，CE kernel 5.2 ms；eager 中 GEMM 38.8 ms、log_softmax 前后向 26.6 ms、类型转换等 15.8 ms。解释（我的分析）：FLCE 的块数约等于 V/H（≈ 64–75），与 BT 无关，每块都对完整 dW（0.58 GB）做一次读改写，块小时这一项受带宽限制。
- 正确性（BT = 4096，对 FP32 参考）：FLCE loss 误差 2.4e-5，dX / dW 相对误差 1.7e-3 / 8.1e-4，与 eager 相同；CCE 的 dX 相对误差 1.9e-2（gradient filtering）。
- SM90 形状（M = H = 4096，V = 128256）：eager 26.0 ms / 5.9 GB；Triton FLCE 60.8 ms / 1.0 GB；CuTe SM90 FLCE 19.5 ms / 2.0 GB；CCE 78.9 ms / 1.0 GB。docstring 自报的 Triton 141 ms（H200）未复现，本次为 61 ms。
- 端到端（Qwen3-1.7B，随机初始化，gradient checkpointing，fused AdamW，batch 1）：参数与优化器状态 9.7 GB。eager 峰值 4K/8K/16K/32K = 17.1/24.6/39.5/69.3 GB，64K 起 OOM；FLCE 13.5/14.0/15.1/18.2/26.2/42.1 GB（到 128K）。16K 单步 eager 0.935 s、FLCE 0.933 s。
- 显存时间线（16K）：eager 在 forward 末尾升高 27.4 GB（12.1 → 39.5 GB），FLCE 峰值 15.1 GB。

## 视频

源文件 `videos/fused-linear-cross-entropy/`：`script.md`（配音稿）→ `subs.py`（节拍）→ `vo.py`（MiniMax，speed 1.05，计费 2812 字符）→ `film.html` + `scenes-a.js`、`scenes-b.js`；渲染 `node render.mjs build/fused-linear-cross-entropy.mp4`。9 幕，约 4 分 49 秒，16:9。

自行决定的处理：
- 全片统一用 Qwen3-1.7B 的形状，SM90 实现因 V 不是 256 的倍数改在其 docstring 形状上单独比较，画面上标明形状。
- 第 2 幕的 logits 柱、第 7 幕 CCE 跳过的块为示意，画面已标注。
- 第 1 幕显存柱的第三段（log_softmax 输出与梯度）由实测峰值减去 BF16 logits 与 FP32 副本得到，未逐项测量。
