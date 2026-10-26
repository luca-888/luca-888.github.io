# LLM Roofline 视频资料

状态：方案已确认（2026-10-20，用户：先做不需要实测的部分；TTS 先不合成，有字幕即可，预留 TTS 的位置）。11 幕 12 fps 预览已渲染，配音未合成，第 6、9 幕为实测占位。先做视频，文章以后再写。分镜见 `videos/llm-roofline/storyboard.md`。

## 范围与硬件

- Slug `llm-roofline`。主题：LLM 推理的建模与量级估算——roofline、arithmetic intensity、prefill 受算力限制而 decode 受带宽限制的原因、显存账（权重 + KV cache + activation）、一次 decode step 的理论下限。目标是让观众能对“卡在哪、量级多少”做判断。
- 全片只跟一个例子：**Qwen3-8B（BF16）在一张 H100 SXM 80GB 上，用 vLLM 服务**。选它是因为仓库里已有这一配置的实测（`public/measurements/vllm-cpu-overhead/20260930-094235/default/qwen3-8b/`，vLLM 0.30.0，Modal H100）：启动日志里的显存账与 decode ITL 都能直接上画面，主线不需要新的 GPU 实验。
- 可选实测见文末，需用户对该次运行单独同意。

## 现状变化

| 早期说法 | 现在 | 出处 |
| --- | --- | --- |
| roofline 的横轴叫 arithmetic intensity | 原论文刻意用 operational intensity（按 cache 之后的 DRAM 流量算），并说明不用 arithmetic intensity 是因为后者常指处理器到 cache 的流量；LLM 推理圈（JAX scaling book、Yuan et al.、kipply）普遍写 arithmetic intensity，指的都是“FLOPs / HBM 字节” | Williams 2009 §2；JAX scaling book |
| “decode 一律受带宽限制” | 按算子看：MHA/GQA 的 decode attention 强度 ≈ group size，与 batch 无关；**MLA 的 decode attention 强度 ≈ 2·h_q，128 个 query head 时越过 H800 的拐点，受算力限制**（FlashMLA deep dive）；矩阵乘法在 batch 足够大时也越过拐点。DeepSeek-V3 报告：即使大规模 EP，decode 中每个 expert 的 batch 通常不超过 256，expert GEMM 仍受访存限制 | FlashMLA 2025-04-22；DeepSeek-V3 报告 |
| 单卡 HBM 是唯一瓶颈 | 大规模 EP 下 all-to-all 通信可以成为下限：DeepSeek ISCA'25 论文估算 InfiniBand 上 decode 每 token 下限 14.76 ms，NVL72 约 0.82 ms | arXiv 2505.09343 |
| 下一代 GPU 会让 decode 变成算力限制 | 拐点几乎不动：H100 BF16 为 989 / 3.35 = 295 FLOP/byte，B200 为 2250 / 8 = 281；低精度反而把拐点右移（H100 FP8 591，B200 FP4 1125） | NVIDIA 规格页；JAX scaling book 同值 |
| vLLM 默认 `gpu_memory_utilization=0.9` | 现为 **0.92**（`config/cache.py:102`）；v0.21 起默认把 CUDA graph 显存从 KV 预算里扣掉；预算按总显存 × 比例算，不按空闲显存 | vLLM c41b2639e2 |
| SGLang `mem_fraction_static` 固定默认值 | 默认 `None`，按 GPU 显存、chunked prefill 大小、max batch 等启发式计算（H100 约 0.83–0.84，为子任务按代码推算，未运行）；SGLang 不做空跑，vLLM 用真实的 profile run 加一次试捕获 CUDA graph | SGLang 6fc8d9da `arg_groups/memory_hook.py` |

## 相关系统与不同意见

- **vLLM 与 SGLang 的显存账**：两者都是“总显存 × 比例 − 权重 − activation − CUDA graph = KV pool”。vLLM 用一次 `max_num_batched_tokens` 规模的 dummy forward 实测 activation 峰值；SGLang 用公式预留。片中用 vLLM（有实测日志），SGLang 只进 notes。
- **两者都内置了 FLOPs / 字节估算**：vLLM `v1/metrics/perf.py`（`--enable-mfu-metrics` 打出 `MFU: %.1f TF/s/device %.1f GB/s/device`）；SGLang `metrics_reporter.py`，留有 speed-of-light 百分比的空钩子。都没有“给定 GPU 规格输出 decode 下限”的独立工具。
- **overhead-bound**（Horace He）：小 kernel、Python 与 launch 开销可以让 GPU 两种资源都闲着；判别法是 batch 翻倍时耗时只涨约 10%。片中第 3 幕的实测与下限之差就是这部分加上 kernel 达不到峰值带宽。
- **实测带宽利用率**：Databricks 定义 MBU =（参数字节 + KV 字节）/ TPOT / 峰值带宽，batch 1 时 2×H100 为 60%、4×A100 为 55%；kipply 的 13B 实测为理论延迟的 62–76%。本片的实测：4.52 ms 下限 / 6.84 ms 实测 = 66%。
- **按阶段贴标签太粗**：Yuan et al. 与 JAX scaling book 都按算子给强度；本片在第 7 幕把 attention 与权重矩阵乘法拆成两个点。

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| 本仓库实测：`public/measurements/vllm-cpu-overhead/20260930-094235/default/qwen3-8b/`（`server.log` 显存账与 KV token 数，`c1/c8/c64/c256.log` ITL 与吞吐） | 本站 | 已读 | 核心（实机材料） | 1、3、5、8 |
| [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)，CACM 52(4), 2009 | Williams, Waterman, Patterson | §1–5、§7 已读（子任务） | 核心 | 4（公式原文、Fig. 1 拐点） |
| [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) / [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) / [HGX](https://www.nvidia.com/en-us/data-center/hgx/) 规格页 | NVIDIA | 已读（子任务） | 核心 | 3、10（dense：H100 BF16 989 TF、FP8 1979 TF、3.35 TB/s、80 GB；B200 2.25 / 4.5 / 9 PF、8 TB/s、180 GB） |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)：Roofline 与 Transformer Inference 两章 | Austin et al.（Google DeepMind），2025 | Roofline 全文、Inference 大部分（子任务） | 核心 | 4–7、9（B_crit、decode 下限公式、attention 强度与 batch 无关） |
| vLLM 源码 c41b2639e2：`v1/worker/gpu_worker.py` `determine_available_memory`、`utils/mem_utils.py`、`v1/core/kv_cache_utils.py`、`v1/kv_cache_interface.py`、`config/cache.py`、`v1/metrics/perf.py` | vLLM 团队 | 已读（子任务，行号见共享笔记） | 核心 | 8 |
| [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)，2022 | kipply | 经摘要抓取读过 | 重点 | 2、7（2P FLOPs/token、KV 公式、A100 拐点 208、实测 62–76%） |
| [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)，2023 | Databricks | 已读（子任务） | 重点（实测对照） | 3（MBU） |
| [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)，2022 | Horace He | 已读（子任务） | 重点（不同意见） | 3（overhead） |
| [FlashMLA new kernel deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md)，2025 | DeepSeek | 经摘要抓取读过 | 重点（现状变化） | 10（MLA decode attention 受算力限制） |
| [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware](https://arxiv.org/abs/2505.09343)，ISCA'25 | DeepSeek | LaTeX 相关节已读（子任务） | 重点（现状变化） | 10（每 token KV：MLA 70 KB、Qwen-2.5-72B 327 KB、Llama-3.1-405B 516 KB；EP 通信下限） |
| [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)，2024 | Yuan et al. | roofline 一节 LaTeX 已读（子任务） | 补充 | 4、6（Llama-2-7B prefill 强度 1024–1215、decode ≈ 1） |
| [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)，2023 | NVIDIA | 已读（子任务） | 补充 | 8（KV 公式） |
| [DeepSeek V3/R1 inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)，2025 | DeepSeek | 经摘要抓取读过 | 补充（相关系统） | notes（大 EP 抬高 batch） |
| SGLang 源码 6fc8d9da：`arg_groups/memory_hook.py`、`kv_cache_configurator.py`、`pool_configurator.py`、`metrics_reporter.py` | SGLang 团队 | 已读（子任务） | 补充（相关系统） | notes |

子任务的完整摘录（含行号、原文公式与可上画面的原图 URL）在会话共享笔记 `llm-roofline-web-research.md`、`llm-roofline-code-research.md`；方向确认后把用到的部分并入本文件。

## 阅读要点与本片的数

Qwen3-8B 配置：36 层，hidden 4096，32 个 query head、8 个 KV head（GQA，group 4），head_dim 128，FFN 12288，词表 151936，不共享 embedding。

| 量 | 值 | 算法 |
| --- | --- | --- |
| 参数 | 8.19 B | 按配置逐项相加 |
| 权重字节（显存占用） | 16.4 GB（15.26 GiB；vLLM 日志 15.27 GiB） | × 2 byte |
| 每步读出的权重 | 15.1 GB | 36 层（每层 192.9 M 参数）+ lm_head 622 M，共 7.57 B 参数 × 2 byte；embedding 表 622 M 每步只取一行，不计 |
| 每 token 运算 | 15.1 GFLOP | 2 × 7.57 B |
| 每 token KV | 144 KiB | 36 × 2 × 8 × 128 × 2 byte |
| decode 下限（batch 1） | 带宽 4.52 ms，算力 0.015 ms | 15.1 GB / 3.35 TB/s；15.1 GFLOP / 989 TFLOPS |
| 实测 ITL（vLLM，输入 128、输出 256） | 并发 1：6.84 ms，145 tok/s；并发 8：7.03 ms，1118 tok/s | 实测，下限 / 实测 = 66% |
| 拐点 | 295 FLOP/byte | 989 / 3.35 |
| prefill 2048 token | 算力 28.8 ms ≫ 带宽 4.52 ms；强度约 1880 | 2 × 6.95 B × 2048 + lm_head 一次；估算，未实测 |
| decode attention 强度 | ≈ 4 FLOP/byte，与 batch 无关 | 每个 2 byte 的 K/V 元素被 4 个 query head 各做一次乘加 |
| KV 超过权重 | batch 内上下文合计 > 10.3 万 token；64 × 2048 时 KV 19.3 GB > 15.1 GB | 15.1 GB / 144 KiB |
| 显存账（vLLM 日志，利用率 0.92） | 可用 72.84 GiB = 权重与非 torch 16.17 + activation 峰值 1.68 + CUDA graph 0.45 + KV 54.99；KV 共 400,448 token；2048 token 的请求最多 195.53 个 | 实测日志 |
| 显存放满时一步的下限（195 × 2048） | 读取 15.1 + 58.9 = 74.0 GB → 22.1 ms；运算 2.95 TFLOP → 3.0 ms；吞吐上限约 8800 tok/s，为 batch 1 下限（221 tok/s）的 40 倍 | 计算估算 |
| MLA decode attention 强度（第 10 幕） | ≈ 240 FLOP/byte | 自行估算：DeepSeek-V3 每 token 每层 latent 576 维（BF16 1152 byte），128 个 query head 各做 QK（576）与 AV（512）两次乘加，≈ 2 × 128 × 1088 / 1152；与 FlashMLA 文中“越过 H800 拐点”一致 |

## 待核实

- 术语：已按确认加入 `docs/terminology.md`（arithmetic intensity / roofline / 拐点）；第 4 幕画面注明原论文称 operational intensity。
- Williams 2009 Fig. 1(a)：已从 Berkeley 预印本取出内嵌位图，放在 `src/assets/llm-roofline/`，出处见 SOURCES.md。
- 实测 ITL 的上下文很短（≤ 384 token），只用于 batch 1 与 8 的对照；长上下文与 prefill 的实测需要新跑（见下）。

## 可选实测（需用户对本次运行明确同意）

Modal 单卡 H100，vLLM 0.30 离线引擎，Qwen3-8B BF16，一次运行做两组：

1. **prefill 用时随 prompt 长度**（64 → 8192 token，batch 1，只生成 1 个 token，关 prefix caching）：预期前段平、约 300 token 之后线性上升，就是时间轴上的 roofline 拐点。用于第 6 幕。
2. **decode 每步用时随 batch**（1 → 256，固定上下文 2048）：预期 batch 小时几乎不变、之后随 KV 线性上升。用于第 7、9 幕的实测点。

预估：H100 约 $4/h，含镜像与权重下载约 20–25 分钟，**约 $1.5–2**；`timeout` 40 分钟兜底（上限约 $2.7）。草稿期不跑，画面留占位。
