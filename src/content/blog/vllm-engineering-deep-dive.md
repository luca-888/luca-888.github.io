---
title: "vLLM 工程拆解：从 API Server 到 PagedAttention"
description: "基于 vLLM main 分支，按入口、配置、引擎、调度、KV cache、Worker、模型适配、Attention backend 和底层 kernel 拆解这个 LLM serving 工程。"
pubDate: 2026-04-28
tags: ["vLLM", "LLM Serving", "Inference", "Systems", "AI Infra"]
---

这篇是对 `vllm-project/vllm` 的工程拆解。阅读对象不是“如何启动一个 vLLM 服务”，而是想搞清楚这个仓库为什么这么大、核心模块怎么分层、一次请求如何从 OpenAI API 进入 scheduler，再落到 GPU worker 和 attention kernel。

本文基于 `main` 分支的一次源码快照，解析到的 commit 是 `de3da0b97cd9db8b1d429312992a5759c89ef881`。vLLM 迭代很快，具体文件名和内部接口后续可能变化，但整体架构和关键抽象相对稳定。

## 1. 一句话定位

vLLM 不是一个普通的 `generate` wrapper，而是一个面向高并发 LLM serving 的推理系统。

它的核心能力来自四层组合：

| 层 | 解决的问题 |
| --- | --- |
| Scheduler | 多请求、多长度、多状态下，本轮该执行哪些 token |
| KV cache manager | 有限 GPU 显存里如何分配、复用、回收 KV cache |
| Worker / model runner | 如何把调度结果变成高效 GPU forward |
| Kernel backend | attention、GEMM、MoE、sampling、quantization 如何贴近硬件执行 |

所以 vLLM 的本质不是“把模型跑起来”，而是持续回答一个 serving 问题：

```text
在很多请求同时存在、prompt/output 长度不可预测、GPU 显存有限的情况下，
下一步应该把哪些 token 拼成 batch，KV cache 放在哪里，最后如何流式返回？
```

## 2. 工程规模

当前仓库大约有：

| 区域 | 规模 |
| --- | --- |
| 全仓库文件 | 约 4753 个 |
| `vllm/` Python 源码 | 约 1629 个 `.py` 文件 |
| `tests/` 测试文件 | 约 1317 个 |
| `vllm/model_executor/models/` 模型适配文件 | 约 282 个 `.py` 文件 |
| `csrc/` C++/CUDA/HIP 源码 | 约 267 个相关源码文件 |

这个规模说明 vLLM 的复杂度主要不在 API 层，而在模型适配、调度、KV cache、分布式执行和底层 kernel。

顶层目录可以粗略分成几类：

```text
vllm/              Python 主体源码
csrc/              C++ / CUDA / HIP 扩展
docs/design/       架构和设计文档
tests/             单元、集成、分布式、kernel、模型测试
benchmarks/        throughput / latency / serving benchmark
examples/          offline / online / multimodal / tool calling 示例
requirements/      不同硬件和开发场景的依赖
cmake/             原生扩展构建逻辑
```

## 3. 总体分层

把仓库按运行时责任拆开，可以得到下面这张逻辑图：

```text
用户入口
  LLM Python API / vllm serve / OpenAI API / gRPC / Anthropic / Batch

配置层
  EngineArgs -> VllmConfig
  ModelConfig / CacheConfig / ParallelConfig / SchedulerConfig / CompilationConfig

引擎层
  LLMEngine / AsyncLLM
  InputProcessor / OutputProcessor
  EngineCoreClient

核心调度层
  EngineCore
  Scheduler
  KVCacheManager
  StructuredOutputManager

执行层
  Executor
  UniProcExecutor / MultiprocExecutor / RayExecutor / external_launcher

Worker 层
  GPUWorker / XPUWorker / CPU path
  GPUModelRunner

模型层
  ModelRegistry
  ModelLoader
  model_executor/models/*
  layers/attention, linear, fused_moe, quantization, lora, mamba

Kernel 层
  csrc/*
  Triton kernels
  FlashAttention / FlashInfer / CUTLASS / FlashMLA / ROCm AITER / CPU kernels
```

这也是读 vLLM 时最重要的心智模型：API server 只是入口，真正的系统核心在 `EngineCore -> Scheduler -> Executor -> Worker -> ModelRunner` 这条链路。

## 4. 入口层：用户请求如何进入系统

vLLM 有多个入口，但主线可以分成离线和在线两类。

离线 Python API：

```text
from vllm import LLM
  -> vllm/entrypoints/llm.py::LLM
  -> EngineArgs
  -> vllm/v1/engine/llm_engine.py::LLMEngine
  -> EngineCore
  -> Scheduler + Executor + Worker
```

在线 OpenAI-compatible server：

```text
vllm serve
  -> vllm/entrypoints/cli/main.py
  -> vllm/entrypoints/openai/api_server.py
  -> AsyncEngineArgs.create_engine_config()
  -> AsyncLLM.from_vllm_config()
  -> EngineCoreClient
  -> EngineCore
```

重要文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/entrypoints/llm.py` | 离线 `LLM` Python API |
| `vllm/entrypoints/cli/main.py` | `vllm` 命令入口 |
| `vllm/entrypoints/openai/api_server.py` | FastAPI server 启动、路由注册、engine client 构建 |
| `vllm/entrypoints/openai/*/serving.py` | chat/completion/responses/realtime 等协议适配 |
| `vllm/entrypoints/openai/engine/serving.py` | OpenAI serving 公共基类、错误处理、beam search 等 |

OpenAI server 层负责的是协议和服务接口：参数校验、chat template、tool calling、reasoning parser、streaming response、错误响应、metrics。真正的推理不在这里，而是交给 `AsyncLLM` 和 engine core。

## 5. 配置层：EngineArgs 到 VllmConfig

vLLM 的 CLI 参数很多，因为它暴露的是一个完整 serving runtime 的配置面。入口参数先进入 `EngineArgs`，再生成统一的 `VllmConfig`。

核心文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/engine/arg_utils.py` | `EngineArgs` / `AsyncEngineArgs`，CLI 参数转配置 |
| `vllm/config/vllm.py` | `VllmConfig` 聚合所有子配置 |
| `vllm/config/model.py` | 模型、tokenizer、dtype、HF config |
| `vllm/config/cache.py` | KV cache、block size、prefix caching、KV dtype |
| `vllm/config/parallel.py` | TP / PP / DP / EP / DCP 等并行配置 |
| `vllm/config/scheduler.py` | token budget、max seqs、chunked prefill、policy |
| `vllm/config/compilation.py` | `torch.compile`、CUDA graph、优化等级 |
| `vllm/config/speculative.py` | speculative decoding |
| `vllm/config/kv_transfer.py` | disaggregated prefill / remote KV transfer |

`VllmConfig` 的设计目的很明确：所有核心对象都拿同一个总配置，避免每加一个功能就层层改 constructor。vLLM 的模型、worker、scheduler、executor 都会从这个总配置里取自己关心的部分。

这也解释了为什么 vLLM 配置对象很大。它不是简单“模型参数”，而是包含模型结构、内存策略、并行策略、调度策略、编译策略、服务策略的系统配置。

## 6. EngineCore：vLLM 的心跳

`vllm/v1/engine/core.py` 是 V1 引擎内核。它负责把调度、KV cache 和模型执行串起来。

初始化阶段大致做这些事：

```text
加载插件
创建 model_executor
profiling 可用 GPU memory
根据模型层规格生成 KV cache config
初始化 worker KV cache
warmup / compile / CUDA graph capture
创建 Scheduler
准备 structured output / multimodal receiver / KV connector
```

主循环最关键的是 `EngineCore.step()`：

```text
if scheduler.has_requests():
    scheduler_output = scheduler.schedule()
    future = model_executor.execute_model(scheduler_output)
    model_output = future.result()
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output,
        model_output,
    )
```

这就是 vLLM 的心跳：

```text
schedule -> execute -> sample -> update -> output
```

如果 pipeline parallel 或 async scheduling 打开，`step_with_batch_queue()` 会把调度和执行进一步重叠：先尽量填充 batch queue，再等待更早的 batch 返回。这样可以减少 pipeline bubble，也能让 CPU 准备下一步输入时 GPU 继续执行。

## 7. Scheduler：吞吐核心

`vllm/v1/core/sched/scheduler.py` 是 vLLM 最值得细读的文件之一。

它的一个关键设计是：调度器不把 prefill 和 decode 写死成两个完全独立阶段。每个 request 维护自己已经计算到哪里：

```text
num_computed_tokens
num_tokens_with_spec
num_prompt_tokens
num_output_tokens
spec_token_ids
```

每一轮调度的目标是让 `num_computed_tokens` 追上这个 request 当前需要处理的 token 数。这个统一模型可以覆盖：

| 能力 | 为什么能放进同一套调度 |
| --- | --- |
| chunked prefill | 长 prompt 可以只调度一段 token |
| continuous batching | running 和 waiting request 每步都可重新组合 |
| prefix caching | 已命中的 prefix 不需要重复计算 |
| speculative decoding | 一次调度多个 draft token，再根据接受情况修正 |
| preemption | KV 不够时回收低优先级 request |
| multimodal / encoder input | 额外受 encoder compute/cache budget 约束 |
| remote KV transfer | 等待远端 KV 或失败后 recompute |

调度流程可以简化成：

```text
new_step_starts()

先处理 running requests:
  计算本轮 num_new_tokens
  检查 token budget / max model len / encoder budget
  分配 KV slots
  如果 KV 不够，按策略 preempt 低优先级 request

再处理 waiting requests:
  检查 LoRA 限制
  查 local prefix cache
  查 remote KV connector
  检查 chunked prefill 是否允许
  分配 KV slots
  加入 running

返回 SchedulerOutput
```

`SchedulerOutput` 是 engine core 和 model runner 的桥。它包含本轮要执行的请求、每个请求要执行多少 token、新分配的 block、spec decode token、encoder input、structured output 信息等。

模型执行完后，`Scheduler.update_from_output()` 会：

```text
读取 sampled_token_ids / logprobs / pooler output
处理 speculative accepted/rejected tokens
更新 request token 状态
检查 stop condition
释放完成请求的 KV / encoder cache
生成 EngineCoreOutput
统计 perf / prefix cache / KV connector 指标
```

所以 scheduler 不只是“排队器”，它是 request 生命周期和 KV 生命周期的中心。

## 8. KV Cache：PagedAttention 的系统基础

vLLM 最著名的是 PagedAttention。现代代码已经比最初论文更复杂，但基本思想仍然是：

```text
请求 token 序列
  -> 切成固定大小 block
  -> block 映射到物理 KV cache page
  -> attention backend 根据 block table 找历史 KV
```

核心文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/v1/core/kv_cache_manager.py` | scheduler 侧 KV 分配、prefix cache 查询、释放 |
| `vllm/v1/core/block_pool.py` | 物理 block 池 |
| `vllm/v1/core/kv_cache_utils.py` | block hash、KV cache config 生成 |
| `vllm/v1/kv_cache_interface.py` | 不同 KV cache spec 抽象 |
| `csrc/attention/paged_attention_v1.cu` | CUDA paged attention kernel |
| `csrc/attention/paged_attention_v2.cu` | 另一版 paged attention kernel |

`KVCacheManager` 的职责包括：

| 职责 | 说明 |
| --- | --- |
| prefix cache lookup | 根据 block hash 找最长已计算 prefix |
| slot allocation | 为新 token 分配 KV block |
| preemption support | 显存不足时释放其他 request |
| block recycling | 请求结束或中止后回收 block |
| cache events / metrics | 输出 KV 使用率、命中率等 |
| multi-group cache | 支持不同层或不同结构的 KV cache group |

`vllm/v1/kv_cache_interface.py` 里有一组很重要的 spec：

| Spec | 场景 |
| --- | --- |
| `FullAttentionSpec` | 标准 full attention |
| `SlidingWindowSpec` | sliding window attention |
| `MLAAttentionSpec` | DeepSeek-style MLA |
| Mamba / hybrid spec | attention + state-space 混合模型 |
| quantized KV mode | FP8、INT8 per-token-head、NVFP4 等 KV cache |

这层抽象使 vLLM 不只服务标准 Llama 类 decoder，也能扩展到 MLA、Mamba、hybrid attention、encoder-decoder、多模态等更复杂结构。

## 9. Executor / Worker：控制面和数据面分离

`Executor` 是控制面抽象。它负责把 engine core 的命令发给实际 worker。

核心文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/v1/executor/abstract.py` | `Executor` 抽象基类 |
| `vllm/v1/executor/uniproc_executor.py` | 单进程执行 |
| `vllm/v1/executor/multiproc_executor.py` | 多进程，一般一 GPU 一个 worker |
| `vllm/v1/executor/ray_executor.py` | Ray 分布式执行 |
| `vllm/v1/worker/gpu_worker.py` | GPU worker，持有 model runner |
| `vllm/v1/worker/gpu_model_runner.py` | 真正准备 batch、执行 forward、采样 |

`Executor` 提供几个关键 RPC：

```text
get_kv_cache_spec()
determine_available_memory()
initialize_from_config()
compile_or_warm_up_model()
execute_model()
sample_tokens()
```

常见部署里，进程模型大概是：

```text
API server process
EngineCore process
GPU worker process per GPU
DP coordinator process, only when DP > 1
```

比如单机 4 GPU、`tp=4`：

```text
1 API server + 1 engine core + 4 GPU workers = 6 processes
```

`EngineCore` 做调度和状态管理，worker 做模型权重加载、forward、KV cache 写入和 kernel 执行。这个分离能让 serving control plane 和 GPU data plane 更清晰。

## 10. GPUModelRunner：GPU 执行主干

`vllm/v1/worker/gpu_model_runner.py` 是另一个核心大文件。它把 scheduler 的抽象输出转换成 GPU 可执行输入。

`execute_model()` 里大致做这些事：

```text
更新 persistent batch states
准备 num_scheduled_tokens / req_ids
构造 input_ids / positions / logits_indices
决定 CUDA graph / eager / padding / microbatch
生成 slot_mappings
构建 attention metadata
处理 Mamba state / KV transfer / encoder output
设置 forward context
执行 model forward
取 sample_hidden_states
compute_logits
保存 execute_model_state
返回 None，等待 sample_tokens()
```

为什么 `execute_model()` 有时返回 `None`？因为 vLLM 把 forward 和 sampling 拆开了：

```text
execute_model()
  -> forward 得到 logits
  -> 暂存 execute_model_state
  -> 返回 None

sample_tokens()
  -> 应用 grammar bitmask
  -> sampler 采样
  -> 更新 request state
  -> speculative draft proposal
  -> 返回 ModelRunnerOutput
```

这种拆分让 structured output、speculative decoding、pipeline parallel、async scheduling 更容易组合。

`GPUModelRunner` 里还有一个非常重要的概念：persistent batch。它不是每一步都从 Python list 重新构造所有输入 tensor，而是维护一组持久状态，然后对新增、完成、preempt 的请求做增量更新。这样可以降低 CPU overhead，并减少 CPU/GPU 同步。

## 11. Attention Backend：不是一种 attention

vLLM 的 attention 层不是固定调用一个 kernel，而是通过 backend registry 选择。

核心文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/model_executor/layers/attention/attention.py` | 模型 attention layer 统一入口 |
| `vllm/v1/attention/backends/registry.py` | backend 枚举和注册 |
| `vllm/v1/attention/selector.py` | 根据配置选择 backend |
| `vllm/v1/attention/backends/*` | 具体 backend 实现 |
| `vllm/v1/attention/ops/*` | Triton / wrapper ops |

常见 backend 包括：

| Backend | 场景 |
| --- | --- |
| `FLASH_ATTN` | NVIDIA 上常用的 FlashAttention 路径 |
| `FLASHINFER` | FlashInfer backend |
| `TRITON_ATTN` | Triton attention |
| `FLEX_ATTENTION` | PyTorch FlexAttention 相关路径 |
| `FLASHMLA` / `CUTLASS_MLA` / `TRITON_MLA` | MLA attention |
| `ROCM_ATTN` / `ROCM_AITER_*` | AMD ROCm |
| `CPU_ATTN` | CPU attention |
| `TURBOQUANT` | TurboQuant KV cache 场景 |

backend 自动选择会检查：

```text
硬件架构
model dtype
KV cache dtype
head size
block size
attention type
MHA / MQA / GQA / MLA
是否支持 sliding window / sink / multimodal prefix
是否支持 DCP / context parallel
```

这就是为什么 vLLM 能在不同 GPU 代际、不同模型结构、不同 KV dtype 下选择不同执行路径。

## 12. 模型适配：ModelRegistry 和 ModelLoader

vLLM 支持大量 Hugging Face 架构，核心不在动态调用 `AutoModelForCausalLM`，而是维护自己的模型实现和映射。

核心文件：

| 文件 | 角色 |
| --- | --- |
| `vllm/model_executor/models/registry.py` | HF architecture 到 vLLM model class 的映射 |
| `vllm/model_executor/model_loader/__init__.py` | `get_model()` 和 loader registry |
| `vllm/model_executor/model_loader/default_loader.py` | safetensors/bin/pt 等默认加载 |
| `vllm/model_executor/models/*.py` | 各模型架构实现 |
| `vllm/model_executor/layers/*` | attention、linear、MoE、norm、RoPE、quantization |

模型加载路径大概是：

```text
get_model()
  -> get_model_loader()
  -> loader.load_model()
  -> get_model_architecture()
  -> ModelRegistry resolve architecture
  -> 初始化 torch.nn.Module
  -> load_weights()
```

`registry.py` 里能看到大量映射，例如：

```text
LlamaForCausalLM      -> llama.py
Qwen3ForCausalLM      -> qwen3.py
DeepseekV4ForCausalLM -> deepseek_v4.py
MixtralForCausalLM    -> mixtral.py
```

现代 vLLM 统一模型 constructor：

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = "")
```

这样 model runner 不需要知道每个模型自己的初始化参数，所有模型都从 `VllmConfig` 中取配置。

## 13. Quantization / LoRA / MoE

这些能力不是外部插件式的小功能，而是深入执行路径。

Quantization 相关目录：

```text
vllm/model_executor/layers/quantization/
csrc/quantization/
csrc/libtorch_stable/quantization/
```

支持的路径包括 FP8、MXFP8、MXFP4、NVFP4、INT8、INT4、GPTQ、AWQ、GGUF、TorchAO、ModelOpt 等。量化不仅影响权重加载，也影响 linear layer、MoE、KV cache dtype 和 attention backend 选择。

LoRA 相关目录：

```text
vllm/lora/
vllm/plugins/lora_resolvers/
vllm/model_executor/layers/*lora*
```

vLLM 支持多 LoRA serving，调度器里也要限制本轮 batch 中同时出现的 LoRA 数量，避免超过 `max_loras`。

MoE 相关目录：

```text
vllm/model_executor/layers/fused_moe/
csrc/moe/
vllm/distributed/eplb/
vllm/distributed/elastic_ep/
```

MoE 的难点不只是 expert layer forward，还包括 expert parallel、all-to-all、专家负载均衡、路由统计、不同 quant dtype 下 fused MoE kernel。

## 14. 分布式：TP / PP / DP / EP / KV transfer

vLLM 的分布式能力是组合式的。

| 模式 | 含义 |
| --- | --- |
| TP | tensor parallel，切模型张量 |
| PP | pipeline parallel，切模型层 |
| DP | data parallel，多 engine core / 多副本 |
| EP | expert parallel，MoE expert 分布 |
| DCP / PCP | decode / prefill context parallel |
| DBO | dual batch overlap / microbatch 相关优化 |
| KV transfer | disaggregated prefill/decode 或远端 KV 复用 |
| EC transfer | encoder cache transfer |

相关目录：

```text
vllm/distributed/
vllm/distributed/kv_transfer/
vllm/distributed/ec_transfer/
vllm/distributed/eplb/
vllm/v1/executor/
vllm/v1/engine/coordinator.py
```

一个细节：vLLM 的 control RPC 和 data plane 通信是分开的。`Executor.collective_rpc()` 适合控制命令，真正的 tensor 通信会走 NCCL、shared memory、custom all-reduce、pipeline send/recv、KV connector 等路径。

## 15. 原生扩展和 Kernel 层

`csrc/` 是 vLLM 的性能地基。它覆盖：

| 目录 / 文件 | 作用 |
| --- | --- |
| `csrc/attention/` | paged attention、MLA、attention utils |
| `csrc/cache_kernels.cu` | KV cache reshape/copy/update |
| `csrc/sampler.cu` / `csrc/topk.cu` | sampling / top-k |
| `csrc/layernorm_kernels.cu` | norm kernel |
| `csrc/moe/` | fused MoE、top-k routing、permute/unpermute |
| `csrc/quantization/` | AWQ、GPTQ、GGUF、Marlin、W8A8 等 |
| `csrc/cpu/` | CPU attention、GEMM、MoE |
| `csrc/rocm/` | ROCm kernels |
| `csrc/custom_all_reduce.cu` | 自定义 all-reduce |

构建由 `setup.py + CMakeLists.txt` 驱动。`VLLM_TARGET_DEVICE` 决定 CUDA、ROCm、CPU、XPU 等构建路径。`pyproject.toml` 里当前构建依赖锁定到 PyTorch `2.11.0`。

这层也解释了为什么 vLLM 的安装和编译比普通 Python 包复杂：它实际包含大量和硬件、CUDA 架构、Torch ABI、CMake 相关的扩展。

## 16. 测试和 Benchmark

vLLM 的测试覆盖面很大，不只是 API 正确性。

典型测试目录：

```text
tests/v1/core/          scheduler / KV cache / prefix cache
tests/v1/attention/     backend selection / MLA / sparse / Triton / FlashAttention
tests/v1/worker/        GPU model runner / input batch / profiler
tests/v1/spec_decode/   EAGLE / ngram / MTP / rejection sampler
tests/distributed/      TP / PP / DP / distributed executor
tests/model_executor/   model layer and weight loading
tests/quantization/     quantization correctness
tests/entrypoints/      OpenAI API and server behavior
```

Benchmark 则覆盖：

```text
benchmarks/latency.py
benchmarks/throughput.py
benchmarks/serve.py
benchmarks/sweep/
benchmarks/kernels/
```

读测试是理解 scheduler 边界条件的好办法。尤其是 `tests/v1/core/test_scheduler.py`、`tests/v1/core/test_prefix_caching.py`、`tests/v1/worker/test_gpu_model_runner.py`。

## 17. 一次请求的细节路径

把上面所有层串起来，一次 chat completion 请求可以这样理解：

```text
HTTP request
  -> OpenAI protocol validation
  -> chat template / tokenizer / multimodal processor
  -> EngineCoreRequest
  -> EngineCoreClient
  -> Scheduler.add_request()
  -> waiting queue
  -> Scheduler.schedule()
      -> prefix cache lookup
      -> token budget decision
      -> KV block allocation
      -> SchedulerOutput
  -> Executor.execute_model()
  -> Worker.execute_model()
  -> GPUModelRunner.execute_model()
      -> update persistent batch
      -> prepare input_ids / positions
      -> build attention metadata
      -> model forward
      -> compute logits
  -> GPUModelRunner.sample_tokens()
      -> grammar bitmask if needed
      -> sampler
      -> speculative draft if enabled
  -> Scheduler.update_from_output()
      -> append token
      -> check stop
      -> free cache if finished
  -> OutputProcessor
  -> detokenizer
  -> streaming response
```

这条链路里，最容易低估的是 scheduler 和 KV cache manager。模型 forward 只是其中一段，serving 的复杂性主要来自“如何持续安排多个请求的下一步 token”。

## 18. 读代码路线

建议按下面顺序读：

1. `docs/design/arch_overview.md`
   先理解入口、进程模型和大组件关系。

2. `vllm/entrypoints/llm.py` 与 `vllm/entrypoints/openai/api_server.py`
   看离线和在线请求怎么进入系统。

3. `vllm/v1/engine/llm_engine.py` 与 `vllm/v1/engine/core.py`
   看 engine wrapper 和 engine core 主循环。

4. `vllm/v1/core/sched/scheduler.py` 与 `vllm/v1/core/kv_cache_manager.py`
   重点读 token budget、prefix cache、preemption、block allocation。

5. `vllm/v1/worker/gpu_model_runner.py`
   看 batch state、attention metadata、CUDA graph、forward、sample。

6. `vllm/model_executor/models/registry.py` 和一个具体模型，例如 `llama.py`、`qwen3.py`
   看模型如何被注册、构造和加载权重。

7. `vllm/v1/attention/backends/registry.py` 与 `csrc/attention/`
   看 attention backend 和 paged attention kernel。

## 19. 总结

vLLM 可以理解成一个 LLM 推理操作系统：

| 子系统 | 类比职责 |
| --- | --- |
| API server | 系统调用入口 |
| Scheduler | 任务调度器 |
| KVCacheManager | 内存管理器 |
| Executor / Worker | 进程和设备执行器 |
| ModelRunner | 运行时和 batch 编排器 |
| Attention backend / kernels | 硬件驱动和加速路径 |
| ModelRegistry / Loader | 应用二进制和插件加载器 |

真正值得重点研究的是三条主线：

| 主线 | 为什么重要 |
| --- | --- |
| `Scheduler + KVCacheManager` | 决定吞吐、显存效率、尾延迟 |
| `GPUModelRunner + attention metadata` | 决定 GPU 是否被高效喂满 |
| `ModelRegistry + kernels + quantization` | 决定支持多少模型、硬件和精度组合 |

用一句话收束：

```text
vLLM = API protocol + token scheduler + paged KV memory manager + distributed model runtime + hardware kernels
```

如果只看 `vllm serve`，它像一个模型服务器；如果沿着源码往下看，它更像一个专门为自回归生成设计的高性能运行时。

<div class="reference-list">
  <a class="reference-card" href="https://github.com/vllm-project/vllm">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM GitHub Repository</strong>
      <small>github.com</small>
    </span>
  </a>
  <a class="reference-card" href="https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM Architecture Overview</strong>
      <small>github.com</small>
    </span>
  </a>
  <a class="reference-card" href="https://github.com/vllm-project/vllm/blob/main/docs/design/paged_attention.md">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM Paged Attention Design Note</strong>
      <small>github.com</small>
    </span>
  </a>
  <a class="reference-card" href="https://github.com/vllm-project/vllm/blob/main/docs/design/attention_backends.md">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>vLLM Attention Backend Feature Support</strong>
      <small>github.com</small>
    </span>
  </a>
  <a class="reference-card" href="https://arxiv.org/abs/2309.06180">
    <span class="reference-icon"></span>
    <span class="reference-body">
      <strong>Efficient Memory Management for Large Language Model Serving with PagedAttention</strong>
      <small>arxiv.org</small>
    </span>
  </a>
</div>
