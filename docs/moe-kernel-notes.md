# MoE Kernel 笔记

视频（暂无配套文章）：一个 MoE 层在 GPU 上怎样算。以 Qwen3-30B-A3B 的一层为例，从路由讲到 vLLM 的 Triton fused MoE kernel，再到 batch 大小怎样决定瓶颈。2026-10-20 用户要求从头重做（旧版半成品已删除），沿用 2026-10-08 在 Modal H100 上的实测数据，不做新的 GPU 实测。

## 参考清单（按重要程度）

2026-10-20 调研，经 subagent 打开原文与源码阅读。

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | vLLM `model_executor/layers/fused_moe/`：`fused_moe.py`、`moe_align_block_size.py`、`csrc/moe/moe_align_sum_kernels.cu`、`oracle/unquantized.py`、`modular_kernel.py`、`experts/trtllm_bf16_moe.py` | vLLM | 已读 | 核心 | 第 4–8 幕 |
| 2 | 本站实测：`public/measurements/moe-kernel/20261008-105322/`（H100，vLLM 0.31.0）与 `scripts/moe-kernel/modal_measure.py` | 本站 | 已读 | 核心 | 全片数字 |
| 3 | [MegaBlocks](https://arxiv.org/abs/2211.15841)（MLSys 2023）§2–6 | Gale、Narayanan、Young、Zaharia | 已读 tex | 高 | 第 9 幕 |
| 4 | SGLang `srt/layers/moe/`（`utils.py`、`quantization/unquant.py`、`fused_moe_triton/`、`model_overrides/qwen3_moe.py`） | SGLang | 已读 | 中 | 第 9 幕 |
| 5 | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) README、`csrc/jit_kernels/heuristics/runtime.hpp` | DeepSeek | 已读 | 中 | 第 9 幕 |
| 6 | transformers `integrations/moe.py`、`models/qwen3_moe/modeling_qwen3_moe.py` | Hugging Face | 已读 | 高 | 第 3 幕（逐个 expert 循环） |
| 7 | PyTorch `torch.nn.functional.grouped_mm`、`aten/src/ATen/native/cuda/GroupedBlas.cpp` | PyTorch | 已读 | 中 | 第 9 幕 |
| 8 | [ScatterMoE](https://arxiv.org/abs/2403.08245)（COLM 2024） | Tan、Shen、Panda、Courville | 已读 tex | 中 | 第 9 幕 |
| 9 | [Accelerating MoE's with a Triton Persistent Cache-Aware Grouped GEMM Kernel](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)（2025-08） | Less Wright、Adnan Hoque、Garrett Goon | 已读 | 中 | 对比 |
| 10 | [Accelerating MoE model inference with Locality-Aware Kernel Design](https://pytorch.org/blog/accelerating-moe-model/)（2024-11） | Adnan Hoque 等 | 已读 | 中 | 对比 |
| 11 | DeepEP README（V2.5） | DeepSeek | 已读 | 低 | 只记现状 |
| 12 | [Qwen3-30B-A3B config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | Qwen | 已读（2026-10-08） | 核心 | 模型数字 |

源码版本：vLLM HEAD `fce64bc5c2`（2026-10-20），对照实测所用 v0.31.0 `db9527a468`；SGLang `78185f4b`；DeepGEMM `5081f85`；DeepEP `b5a33ee`；PyTorch `a6b28b6`；transformers `536ecc0`；triton `85609ca`。

## 阅读要点

- **模型**：hidden 2048，`moe_intermediate_size` 768，128 个 expert，top-8，`norm_topk_prob` true，48 层。一个 expert 的 w1（gate 与 up 合并，1536 × 2048）与 w2（2048 × 768）共 9.44 MB（BF16），一层 128 个共 1.21 GB。
- **Triton 路径**（`fused_experts_impl`，`fused_moe.py:1652`）：对齐（`_prepare_expert_assignment`）→ GEMM1（w1，N = 1536）→ `silu_and_mul` → GEMM2（w2，乘 router 权重）→ `moe_sum`。v0.31.0 与 HEAD 的 `fused_moe.py`、`moe_align_block_size.py`、`experts/triton_moe.py` 逐字节相同，实测路径即当前路径。
- **kernel 内部**（`fused_moe.py:381-611`）：一个 program 算一个 `BLOCK_SIZE_M × BLOCK_SIZE_N` 块；`expert_ids[pid_m]` 决定读哪个 expert 的权重；A 的行用 `sorted_token_ids // top_k` 直接 gather，不搬动 token；`token_mask` 屏蔽补齐行；`expert_id == -1`（不在本卡）写零后返回；结果按原行号写回；`GROUP_SIZE_M` 排序提高 L2 复用。
- **对齐**（`moe_align_block_size`）：`sorted_token_ids` 存展平后的 (token, k) 下标 `t·top_k + k`，按 expert 分组，每组补齐到 `BLOCK_SIZE_M` 的倍数；补齐位填 `M·top_k`；`expert_ids` 每块一个，尾部空块为 -1。两个 CUDA kernel：`moe_align_block_size_kernel`（计数、前缀和、填 `expert_ids`）与 `count_and_sort_expert_tokens_kernel`（原子操作散放下标）。源码 docstring 的例子 expert 编号与 `num_experts=4` 不一致，上画面需改为 0 起编号。
- **跳过对齐**（`fused_moe.py:1541-1587`）：`num_tokens × top_k × 4 ≤ num_experts` 时每个 (token, k) 一块，不跑对齐 kernel；Qwen3-30B-A3B 下为 token 数 ≤ 4。
- **默认 tile**（BF16，`get_default_config`，`fused_moe.py:1370-1409`）：BLOCK_M 16（M ≤ 32）、32（≤ 96）、64（≤ 512）、128；BLOCK_N 64 / 128，BLOCK_K 128 / 64（以 M = 64 为界）；源码注释：小 batch 是 memory-bound，大 batch 是 compute-bound。H100 上 `E=128,N=768` 没有调优文件（H200、H20、B200 有），实测用的就是默认值。
- **后端选择**（`oracle/unquantized.py`）：CUDA 默认优先级 FlashInfer TRTLLM → FlashInfer CUTLASS → Triton；SM90 上两个 FlashInfer 被移到最后（注释：比 Triton 慢）。所以 H100 BF16 走 Triton；B200 上 Qwen3 走 FlashInfer TRT-LLM 的 Monolithic 版本（路由也在 kernel 内）。SGLang 同样：H100 默认 Triton（源自同一设计），SM100 上 Qwen3 改走 `flashinfer_trtllm`。
- **transformers**：`eager` 是逐个 expert 的 Python 循环，`nonzero`/`where` 每个 expert 触发一次设备到主机同步；`grouped_mm` 是 `torch.sort` + `histc` + `cumsum` 求偏移，再调 CUTLASS grouped GEMM，不补齐。
- **DeepGEMM**：只对 M 轴分组，N、K 固定。contiguous 布局用于训练与 prefill，每个 expert 段对齐到 128（SM100 理论值 256）；masked 布局用于开启 CUDA Graph 的 decode，CPU 不知道每个 expert 分到多少 token。
- **MegaBlocks**：以前的框架按 capacity 丢 token 或补齐到相同大小（Tutel 需要的 capacity factor 可达 11）；MegaBlocks 改写成 block-sparse 运算（128 × 128 块），不丢 token；block-sparse 矩阵乘达到 cuBLAS batched 的 98.6%。
- **ScatterMoE**：MegaBlocks 把分组后的 token 复制并补齐到 HBM；ScatterMoE 只补齐下标，与 vLLM 的 `sorted_token_ids` 思路相同；训练吞吐比 MegaBlocks 高 38.1%（1.5B，8 × A100）。
- **PyTorch 博客**：persistent grouped GEMM（grid = SM 数）在大形状下有利；2024 年那篇在 vLLM fused MoE kernel 上测得 decode 时 L2 命中率低，按列优先排 tile 后在 H100 上快 4.4 倍。

## 现状变化

- vLLM：旧的顶层 `fused_moe(hidden_states, w1, w2, gating_output, …)` 与 `VLLM_FUSED_MOE_CHUNK_SIZE` 分块循环已删除；路由移到 `router/`。v0.31.0 之后新增 DeepGEMM Mega MoE（SM100，把 EP 通信、两次 GEMM、SwiGLU 融成一个 kernel）与 `fused_globalize_align_block_size`（DeepEP V2 的 decode 元数据与对齐融成一次可被 CUDA Graph 捕获的 launch）。
- triton_kernels：`matmul_ogs` 已改名为 `matmul`，gather / scatter 融进 ragged-M 矩阵乘。
- DeepEP：V2.5 删除了 V1 API（包括 low-latency 的 masked 布局）与 NVSHMEM 后端；DeepGEMM README 里“接 DeepEP low-latency 输出”的说法相对 DeepEP 已过时。
- PyTorch：`torch.nn.functional.grouped_mm` 已公开（torch ≥ 2.10，SM80+）。

## 相关系统对比

| 系统 | H100 BF16 | B200 BF16 | 做法 |
| --- | --- | --- | --- |
| vLLM | Triton `fused_moe_kernel` | FlashInfer TRT-LLM（Monolithic） | Triton 版只补齐下标、gather 输入、非 persistent |
| SGLang | Triton（同源） | `flashinfer_trtllm`（Qwen3） | 同上 |
| transformers | sort + histc + cumsum + CUTLASS grouped GEMM | 同 | 精确偏移，不补齐 |
| DeepGEMM | contiguous / masked | 对齐 256，Mega MoE | 只按 M 分组 |
| MegaBlocks | block-sparse，128 × 128 | — | 训练为主，复制并补齐 |

反对意见与取舍：补齐浪费 MMA 行，排序要多两个 kernel；在 decode 中权重读取占主导，补齐行的浪费不是瓶颈（实测 16 个 token 时补齐率 88%，GEMM 仍跑到 H100 显存带宽规格的约 86%）。persistent 调度对大形状有利，小 M 的 decode 则取决于权重读取与 L2 复用。

## 实测（Modal H100 × 1，2026-10-08，vLLM 0.31.0）

原始数据在 `public/measurements/moe-kernel/20261008-105322/`。第 24 层，eager 模式（不用 CUDA Graph），墙钟时间含主机开销；kernel 级结论用 profiler 的 GPU 时间。

- **真实路由**：16 个 decode 请求各 1 个 token（第 20 步）：激活 65 个 expert，63 个空闲；最忙的 expert 分到 10 个 token，51 个只有 1–2 个；共 128 行。
- **激活 expert 数**（第 24 层）：decode 1、2、4、8、16、32、64、128 个 token 为 8、13.9、23.7、34.6、49.0、66.6、79.3、89.0；prefill 1024、4096 个 token 为 92.7、107.0。路由不均匀：均匀路由下 64 个 token 应激活约 126 个。
- **单层耗时**（ms，中位数）：

  | token 数 | 1 | 4 | 16 | 64 | 256 | 1024 | 4096 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 逐个 expert 循环 | 1.81 | 4.55 | 17.58 | 20.41 | 17.78 | 24.58 | 28.81 |
  | transformers `grouped_mm` | 0.47 | 0.46 | 0.56 | 0.60 | 0.61 | 1.02 | 2.19 |
  | vLLM `fused_experts` | 0.22 | 0.25 | 0.40 | 0.46 | 0.44 | 0.60 | 1.06 |

- **kernel 数**（16 个 token）：循环 821 个 kernel 与 326 次设备到主机拷贝；fused 7 个（copy 1.6 µs、align 3.1、count_sort 1.4、GEMM1 139.9、silu_and_mul 2.2、GEMM2 72.5、moe_sum 2.4），两次 GEMM 占 95%。
- **带宽估算**：16 个 token 时两次 GEMM 读 65 个 expert 共 613 MB，用时 212 µs，约 2.9 TB/s（H100 规格 3.35 TB/s）；1024 个 token 时约 195 TFLOPS，远低于 BF16 规格 989。
- **对齐结果**（16 个 token，BLOCK_M 16）：补齐后 1040 行（65 块），真实 128 行，缓冲按最坏情况开 2048 行。
- **tile 配置**：1–32 个 token 为 (16, 64, 128)，64 为 (32, 64, 128)，128–512 为 (64, 128, 64)，1024 以上为 (128, 128, 64)。
- **正确性**：fused 与循环的最大相对误差 3.9e-3 到 8.1e-3（BF16）。

## 确认范围

- 2026-10-20：用户要求重做，直接做视频不等确认；不做新实测，需要实测处留占位；不合成配音，字幕按估计时长，留好 TTS 接口。方案与分镜见 `videos/moe-kernel/storyboard.md`；11 幕 12 fps 预览已渲染（`videos/moe-kernel/build/`），占位 1 处（B200 实测）。
