# Vllm · CPU Overhead 文章资料

状态：正文与 5 张图已写完（`content/vllm-cpu-overhead.md`、`src/VllmCpuOverheadFigures.tsx`，`status: draft`），已用 Modal H100 实测数据填充；图为直接精画未经草图确认。

## 范围与硬件

- 独立成篇。continuous batching、token budget、KV block 等背景只就地交代一句，不展开（调度与 prefix caching 各有独立文章）。
- Slug `vllm-cpu-overhead`；标题按词表：主标题“Vllm”，子标题“CPU Overhead：让 GPU 不等 CPU”（2026-10-01 用户指定英文）。2026-10-01 由 `vllm-engine` 改名：本文讲的是一步执行路径上的 CPU 开销，横跨 engine、scheduler、worker 三层，并非 vLLM engine 的全貌。Modal 上的 app / volume 仍叫 `vllm-engine-*`。
- 源码基准：vLLM v0.30.0（2026-09-22）与 main@e006d76（2026-09-29），本地浅克隆在 `~/Documents/ChatGPT/reference-repos/vllm`；SGLang main@8854857。
- 实验：需要。主线是 profile 时间线与每步 CPU 开销分解，必须实测；改用 Modal 单卡 H100（用户 2026-09-30 同意），计划见文末。

## 现状变化（以源码与 release notes 为准）

原计划的关键词“persistent batch、piecewise CUDA Graph + torch.compile”都来自 2025-01 的 V1 alpha blog，到 v0.30 已有明显变化：

| 时间 | 变化 | 出处 |
| --- | --- | --- |
| v0.11.0 | V0 引擎代码全部删除，V1 成为唯一引擎；CUDA Graph 默认模式从 `PIECEWISE` 改为 `FULL_AND_PIECEWISE` | release notes |
| v0.14.0 | async scheduling 默认开启（#27614），调度 step N+1 与 GPU 执行 step N 重叠 | release notes |
| 2026-03 | Model Runner V2 发布：persistent state 与每步输入解耦、GPU 端 Triton 准备输入、async-first、Triton sampler | MRV2 blog、`docs/design/model_runner_v2.md` |
| v0.25.0 | MRV2 成为 dense 模型默认；旧 PagedAttention kernel 删除 | release notes |
| v0.29.0 | MRV2 成为所有模型默认，MRV1 视为弃用，计划 v0.32 删除；遇到 MRV2 未支持的特性时回退 MRV1 | release notes、`config/vllm.py:use_v2_model_runner` |

结论：“persistent batch”要讲成 MRV1 的做法及其问题，再讲 MRV2 如何重做；“piecewise CUDA Graph”要讲成 `FULL_AND_PIECEWISE` 的双模式分派；async scheduling 是原计划漏掉、但对 CPU 开销最关键的一环。

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| vLLM 源码：`v1/engine/core.py`（`step`、`step_with_batch_queue`、`run_busy_loop`、IO 线程）、`async_llm.py`、`core_client.py`、`v1/executor/multiproc_executor.py`、`v1/worker/gpu/model_runner.py`（`execute_model`）、`v1/core/sched/async_scheduler.py`、`config/vllm.py`（`max_concurrent_batches`、async / MRV2 默认逻辑） | vLLM 团队 | 已读上述函数；`input_batch.py`、`cudagraph_utils.py`、`output_processor.py` 细节待读 | 核心 | 1–5 |
| [Model Runner V2 设计文档](https://docs.vllm.ai/en/latest/design/model_runner_v2/) | vLLM 团队（Woosuk Kwon 维护） | 全文已读 | 核心 | persistent batch 的 V1 问题与 MRV2 解法、async barrier、`StagedWriteTensor`、GPU 端输入准备 → 3、4 |
| [Model Runner V2 blog](https://vllm.ai/blog/2026-03-24-mrv2)（2026-03-24） | vLLM 团队 | 摘要级已读，原文待通读 | 核心 | 设计原则；Qwen3-0.6B / 1×GB200 吞吐 25K vs 16K tok/s（+56%），GLM-4.7-FP8 / 4×GB200 TPOT −6.3%（官方自报）→ 4 |
| [CUDA Graphs 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/) | vLLM 团队 | 全文已读 | 核心 | `CUDAGraphMode`、`BatchDescriptor`、dispatcher、嵌套 wrapper、uniform decode 定义 → 5 |
| [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | vLLM 团队 | 进程架构一节已读 | 核心 | API server / EngineCore / GPU worker / DP coordinator 的数量关系 → 1、2 |
| [vLLM V1 alpha blog](https://vllm.ai/blog/2025-01-27-v1-alpha-release)（2025-01-27） | vLLM 团队 | 已读 | 重点 | V0 的 CPU 瓶颈动机（GPU 每步 ~5 ms）、进程拆分、persistent batch、piecewise CUDA Graph 的初衷；“最高 1.7× 吞吐” → 引言、现状变化 |
| [torch.compile 集成文档](https://docs.vllm.ai/en/latest/design/torch_compile/) | vLLM 团队 | 目录已读，正文待读 | 重点 | 编译缓存、动态 shape、按 attention 切分 FX graph → 5 |
| [Optimization Levels](https://docs.vllm.ai/en/latest/design/optimization_levels/) | vLLM 团队 | 已读 | 补充 | `-O0`…`-O3` 与默认 `-O2` 启动时间取舍 → 5 |
| [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)（2025-09） | Aleksa Gordić | 摘要级已读 | 重点 | engine step 三阶段、进程与线程布局的第三方梳理，交叉核对 → 1、2 |
| [Can Scheduling Overhead Dominate LLM Inference?](https://mlsys.wuklab.io/posts/scheduling_overhead/)（2024-09） | Srivatsa、Li、Zhang、Abhyankar（UCSD WukLab） | 摘要级已读 | 重点（对比 / 动机） | vLLM v0.5.4 调度开销最高占端到端一半，主因是输入张量构建、detokenize、metadata，而非调度算法本身 → 引言 |
| [SGLang v0.4 blog：zero-overhead batch scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) | LMSYS / SGLang 团队 | 相关节已读 | 重点（相关系统） | overlap scheduler 先行一批、future token 解依赖；受 NanoFlow 启发 → 4 |
| SGLang 源码：`managers/scheduler.py`（`event_loop_normal` / `event_loop_overlap`）、`tokenizer_manager.py`、`detokenizer_manager.py` | SGLang 团队 | 仅定位 | 重点（相关系统） | 三进程拆分（tokenizer / scheduler / detokenizer）与 vLLM 对照 → 2、4 |
| [Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference](https://arxiv.org/abs/2603.22774)（IISWC'26） | Chung、Jia、Jezghani、Kim | 摘要已读 | 补充（反对意见 / 边界） | CPU 核数不足时 GPU 空转，补足 CPU 使 TTFT 降 1.47–7.11× → 6 |
| [Blink: CPU-Free LLM Inference](https://arxiv.org/abs/2604.07609)（2026-04） | Siavashi et al.（KTH） | 摘要已读 | 补充（反对意见） | 另一条路线：把调度与 KV 管理放进 GPU persistent kernel + SmartNIC，完全去掉 host CPU → 6 |
| [Python Multiprocessing 设计文档](https://docs.vllm.ai/en/latest/design/multiprocessing/) | vLLM 团队 | 已读 | 补充 | fork / spawn 取舍，最多一句 → 2 |
| TensorRT-LLM executor / overlap scheduler 文档 | NVIDIA | 未读 | 补充（可选） | C++ runtime 路线的对照 → 6 |
| NanoFlow（OSDI'25） | Zhu et al. | 未读 | 补充（可选） | 设备内重叠的上游思路 → 4 |

## 阅读要点

- **进程布局**：单卡时 EngineCore 进程内用 `UniProcExecutor` 直接驱动 worker，共 2 个进程（API server + EngineCore）；TP>1 用 `MultiprocExecutor`，每卡一个 worker 进程，EngineCore 通过共享内存 `MessageQueue` 广播 `SchedulerOutput`。API server 与 EngineCore 之间是 ZMQ（ROUTER / PULL）。DP>1 时多一个 coordinator，API server 数默认等于 DP。
- **EngineCore 内部三个线程**：输入线程（socket → queue）、主线程 busy loop、输出线程（queue → socket）；IO 线程释放 GIL，序列化 / 反序列化与 forward 重叠。
- **API server 负责的 CPU 工作**：`InputProcessor`（tokenize、多模态预处理）、`OutputProcessor`（增量 detokenize、stop string、组装 `RequestOutput`、流式返回）。这些都不在 EngineCore 的关键路径上。
- **同步 step**：`schedule → execute_model(non_block) → get_grammar_bitmask → future.result() → sample_tokens → update_from_output`。
- **async scheduling**：`max_concurrent_batches = 2`（MRV2 为 `pp_size + 1`），`step_with_batch_queue` 在 batch queue 未满时先调度下一批再等结果；`AsyncScheduler` 用 `num_output_placeholders` 占位还没回来的 token，下一步直接按“多一个 token”排程。v0.14 起默认开启；pooling 模型、部分 spec decode 方法、PP + MRV1 时自动关闭。
- **MRV2 execute_model 的顺序**：finish / free / add / update 请求状态 → `block_tables.apply_staged_writes()` → `gather_batch_req_state` → `dispatch_cg_and_sync_dp` 得到 `BatchDescriptor` 与 CUDA Graph 模式 → FULL 时 `cudagraph_manager.run_fullgraph`，PIECEWISE 时 `run_pw_graph`，否则 eager → `sample_tokens` 单独调用。
- **MRV1 persistent batch 的问题**（设计文档原话要点）：persistent state 直接当模型输入，有严格的顺序与布局要求，请求增删要整表重排；需要 `CachedRequestState` 备份；async 下要 async barrier，易漏。
- **MRV2 解法**：每个请求在 `max_num_reqs` 行的状态表里占固定一行，直到结束或被抢占（抢占当作结束）；每步按 attention backend 需要的顺序在 GPU 上 gather；大表（block table）用 `StagedWriteTensor` 只传 diff；`input_ids`、`positions`、`query_start_loc`、`seq_lens` 用 Triton kernel 在 GPU 上生成；非 pinned CPU 状态 + 临时 pinned 拷贝避免 CPU/GPU 竞争，不再需要 barrier。
- **CUDA Graph**：`FULL_AND_PIECEWISE` 默认——uniform decode 走整图（含 attention），prefill / 混合 batch 走 piecewise（attention 以外的段进图）；batch 先 pad 到最近的 capture size，没有对应 key 就 eager。cascade attention 强制 piecewise。
- **V0 时代的开销分布**（WukLab，v0.5.4）：瓶颈是 Python 构建输入张量、detokenize、metadata，调度算法本身占比小——正好对应 V1 的三刀：detokenize 挪进 API 进程、persistent batch、async 重叠。

## 文章主线

Decode 一步的 GPU 时间只有几毫秒，而每一步都要由 CPU 决定“这一步算谁、输入是什么、结果给谁”。引擎设计的核心问题是：怎样让这些 CPU 工作不让 GPU 等。vLLM 的答案分三层：**挪走**（tokenize / detokenize / 网络 IO 挪到别的进程与线程）、**藏住**（async scheduling 让下一步的调度与这一步的 GPU 执行重叠）、**做薄**（每步输入准备从 Python 重建变成增量 diff，再到 MRV2 的 GPU 端生成；CUDA Graph 把几百次 kernel launch 变成一次 replay）。全文以一个请求从 HTTP 进来到流式返回为示例路径，用实测 timeline 检验每一层的作用。

## 章节提纲

1. **一步 decode 里的 CPU 与 GPU**：GPU 时间随模型变小、batch 变小而缩短，CPU 开销基本不变，所以小模型 / 快 GPU 上 CPU 成为瓶颈（V1 blog 与 WukLab 数据）。给出一张“同步执行”的 timeline 作为起点。
2. **一个请求的路径**：API server（tokenize）→ ZMQ → EngineCore 输入线程 → busy loop（schedule → execute → update）→ worker / model runner → 输出线程 → API server（增量 detokenize、流式返回）。进程与线程为什么这么拆；单卡 2 进程、TP=4 时 6 进程。一张进程图，高亮这个请求的路径。
3. **把调度藏到 GPU 后面**：async scheduling 与 batch queue；placeholder 如何解开“下一步依赖这一步采样结果”的问题；与 SGLang overlap scheduler 的同与不同；关闭时的 timeline 对比（实测）。
4. **把每步输入做薄**：MRV1 persistent batch（只传 diff）→ 其耦合问题 → MRV2 的状态表 + gather + GPU 端 Triton 准备输入；官方 MRV2 数据（注明自报）与本文实测。
5. **把 kernel launch 合成一次**：torch.compile 按 attention 切图；`FULL_AND_PIECEWISE` 的双模式分派、padding 到 capture size、启动时间与显存代价（`-O` 级别）。CUDA Graph 原理不展开，一句话交代。
6. **CPU 仍然会成为瓶颈的地方**：CPU 核数不足（IISWC'26 数据）、tokenize 长上下文 / 多模态预处理、structured output 的 grammar bitmask、DP 多 API server；另一条路线（Blink 把服务栈放上 GPU / SmartNIC）作为对照，一段即可。
7. **实测**（若确认硬件）：见下。

启动阶段（加载权重 → profile 显存 → 定 KV cache 大小 → 编译 → capture）只在第 5 节一句带过，不单独成节；DP coordinator 留给分布式一文。

## 核心图（草图阶段，待确认方向）

1. **一个请求穿过的进程与线程**：回答“一个请求从 HTTP 到流式返回经过哪些进程 / 线程，哪些 CPU 工作不在 GPU 的关键路径上”。横向泳道：API server、EngineCore（输入线程 / 主循环 / 输出线程）、Worker、GPU；一条彩色路径标出请求与其第一个 token。
2. **同步 vs async 的 step timeline**（实测数据绘制，ECharts 或 SVG 甘特）：回答“GPU 空隙从哪里来、async 之后还剩什么”。上下两组：CPU 行（schedule / prepare / update / output）与 GPU 行（forward / sample），空隙用醒目色块。
3. **MRV1 与 MRV2 的 persistent batch**：回答“为什么请求增删在 V1 要重排整表、MRV2 只需写一行再 gather”。左：状态张量行序 = 输入行序，删除 req B 后整表下移；右：固定行状态表 → gather 索引 → 本步输入。
4. **CUDA Graph 分派**：回答“一个 batch 最终走整图、分段图还是 eager”。一个 batch 描述（13 个 token、3 个 decode + 1 个 prefill chunk）→ pad → dispatcher → 三种路径，piecewise 图中 attention 段留空。
5. **实测图**（ECharts）：每步 CPU 时间分解的堆叠条（同步 / async、MRV1 / MRV2、CUDA Graph 开关），以及 ITL / 吞吐随配置变化。

## 待核实

- MRV2 下 `prepare_inputs` 与 MRV1 `_prepare_inputs` 的实际 CPU 耗时差异（需实测，不照搬官方 +56%）。
- `OutputProcessor` 的增量 detokenize 实现与 `stream_interval` 的作用。
- async scheduling 下 structured output 为何要 defer 采样（`pending_structured_output_tokens`）；是否写进正文。
- SGLang overlap scheduler 的当前实现（`event_loop_overlap`、future token map）与默认是否开启。
- V1 alpha blog 1.7× 数据的测试条件；是否引用。
- `FULL_AND_PIECEWISE` 默认的 capture sizes 列表与 padding 规则（`cudagraph_capture_sizes`）。
- MRV1 是否已在 main 删除（计划 v0.32），发布前按最新 release 复核。

## 实验计划（Modal H100，2026-09-30 起草，待用户确认后才申请 GPU）

- **目标**：测出 decode 每步 GPU 之间的空隙，以及 async scheduling、Model Runner V2、CUDA Graph 各自压掉多少；不引用官方 +56% 等自报数字作结论。
- **硬件与途径**：Modal 单卡 H100 SXM5，1 个容器，无并发，无多卡。用户 2026-09-30 同意用 Modal。
- **软件**：vLLM v0.30.0；镜像与依赖在无 GPU 环境构建，模型权重提前下载到 Modal Volume（CPU 容器完成，不占卡）。
- **模型**：Qwen3-0.6B（CPU 占比高）、Qwen3-8B（对照）。
- **测试矩阵**：decode 为主，固定输入 128 / 输出 256，并发 1 / 8 / 64 / 256。
  1. 基线：默认（async 开、MRV2、`FULL_AND_PIECEWISE`）。
  2. `--no-async-scheduling`。
  3. `VLLM_USE_V2_MODEL_RUNNER=0`（MRV1，源码确认可切换，v0.30 仍支持）。
  4. `cudagraph_mode` 取 `PIECEWISE` 与 `NONE`。
  共 5 种配置 × 2 模型 × 4 档并发；每种配置在同一次服务启动内扫完 4 档并发。
- **采集**：`vllm bench serve` 测吞吐与 ITL；torch profiler（`--profiler-config`，稳态 20–30 step）导出 trace，回传后在本地统计每步 CPU 各段与 GPU 空闲时间。Nsight Systems 不使用，避免容器权限不确定性。
- **正确性检查**：各配置下同一 prompt 的输出 token 数与基线一致；无 OOM、无回退告警（日志检查 async / MRV2 是否被自动关闭，若被关闭则该配置作废并记录）。
- **预期输出**：`public/measurements/vllm-cpu-overhead/` 下的原始 JSON、trace 摘要、脚本、环境记录（驱动、CUDA、vLLM 版本、CPU 型号）。
- **完成条件**：矩阵跑完或触及上限。

### 时间与费用预估（价格取自 modal.com/pricing，2026-09-30 查询，USD）

| 项目 | 估计 |
| --- | --- |
| 准备（镜像构建、权重下载） | 约 20 min，仅 CPU 容器，不占卡 |
| 排队 | 不确定，不计入占卡时间 |
| GPU 占用 | 预计 1–1.5 h；**上限 2 h**（10 个配置 × 约 6–9 min，含冷启动与编译） |
| 结果回传 | 数分钟，在 GPU 容器结束后进行 |

费率：H100 $0.001097/s（≈$3.95/h）；CPU $0.0000131/核/s；内存 $0.00000222/GiB/s。容器按 8 核、64 GiB 估：

- 1.5 h：GPU ≈ $5.9，CPU ≈ $0.57，内存 ≈ $0.77，合计约 **$7.3**。
- 2 h（上限）：GPU ≈ $7.9，CPU ≈ $0.75，内存 ≈ $1.02，合计约 **$9.7**。
- 不确定项：Volume 存储与出口流量未计（预计很小）；冷启动与编译耗时未实测，最坏可能使占用逼近上限；Starter 计划每月 $30 免费额度是否已被其他用途消耗，需以账户为准。这些不当作零，但量级远小于上面的合计。

### 执行约束

- 单次远端调用完成整个矩阵，函数超时 2 h，容器上限 1，不自动重试；触及超时或预算上限时停止并保存已有结果，补测前更新剩余矩阵、时间与费用再问用户。
- 设备探测与试运行计入本计划，不额外占卡；脚本先在本地 / CPU 容器做语法与参数检查。
- 完成或中止后立即确认容器已停止，记录实际占用时长与账单（估算与实际分开写）；写作与等待期间不保持 GPU。

## 实测记录（Modal H100，Qwen3-8B）

- 第一次（旧号）：镜像缺 nvcc，FlashInfer 采样 JIT 失败，10 组全部启动失败，无数据；约 17 min GPU。
- 第二次（旧号）：workspace 因支出上限被停用，未产出完整数据。
- 第三次（新号 `luca2`，2026-09-30 17:29）：客户端约 2 min 后取消，无数据；约 3 min GPU。
- 第四次（新号，17:42–18:12 CST，约 31 min GPU）：在 `piecewise` 配置的启动阶段被取消（Modal 日志 `Received a cancellation signal`，原因未明）。已完成并有数据：`default`、`no_async`、`mrv1`（各含并发 1/8/64/256 的吞吐与 ITL，以及并发 8/64 的 torch profiler trace）。未完成：`piecewise`、`graph_none`。数据在 `public/measurements/vllm-cpu-overhead/20260930-094235/`。
- 初步数据（output tok/s，并发 1/8/64/256）：default 145/1118/6834/12149；no_async 117/875/5170/11114；mrv1 145/1115/6840/12989。仅为原始读数，未分析显著性，未做重复测量。
- 第五次（新号，2026-09-30 18:25 CST，`modal deploy` + `.spawn()`，约 17.7 min GPU，估算约 $1.3）：补测 `piecewise`、`graph_none`，均成功，数据在 `public/measurements/vllm-cpu-overhead/20260930-102519/`。之后已停止 deployed app。
- 补测读数（output tok/s，并发 1/8/64/256）：piecewise 132/1029/6634/12302；graph_none 71/569/4098/12421。至此 5 种配置均已采集，仍是单次读数、未重复；trace 尚未分析。

## Trace 解析（scripts/analyze_cpu_overhead_traces.py → public/measurements/vllm-cpu-overhead/step_breakdown.json）

方法：取 rank0 worker 的 torch profiler trace，每个 pure-decode step 有一对 `execute_context_0(0)_generation_N(N)` 注释（CPU 线程 + GPU）。period = 相邻两步 GPU 注释起点之差；gpu_busy = 本步周期（本步注释起点到下一步注释起点）内**全部** kernel 时间的并集；idle = period − gpu_busy，即 GPU 上没有任何 kernel 在跑的时间；去掉首尾各 20% step 后取中位数。profile 先跑并发 8、后跑并发 64（按 trace 时间戳配对）。Qwen3-8B / H100，µs：

| 配置 | 并发 | period | gpu_busy | idle | cpu_exec | launches | kernels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default | 8 / 64 | 7066 / 8164 | 6775 / 7900 | 296 / 264 | 2152 / 2210 | 6 | 494 / 459 |
| no_async | 8 / 64 | 9690 / 12085 | 6797 / 7891 | 2886 / 4193 | 2056 / 2346 | 6 | 494 / 459 |
| mrv1 | 8 / 64 | 7079 / 8194 | 6763 / 7910 | 313 / 283 | 2519 / 2842 | 12 | 489 / 453 |
| piecewise | 8 / 64 | 12481 / 12523 | 6811 / 7871 | 5659 / 4679 | 10577 / 9928 | 149 | 492 / 457 |
| graph_none | 8 / 64 | 42193 / 50594 | 6811 / 7862 | 35401 / 42735 | 40565 / 47669 | 148–184 | 492 / 456 |

**口径更正（2026-10-01）**：初版只计 GPU 注释区间内的 kernel（gpu_busy c8 约 6.2 ms），把注释之外的 lm_head 收尾、sampling 与下一步输入准备 kernel（每步约 0.6 ms）算成了空转，得出“空转 3.5 → 0.9 ms”和“MRV1 的 GPU 时间高约 7%”。后者是口径假象：MRV1 把 sampling 放在注释区间内。按全部 kernel 计，三种配置的 gpu_busy 一致（c8 约 6.8 ms），空转为 2.9 → 0.3 ms（c8）、4.2 → 0.3 ms（c64）。正文、图 1 与视频均已按新口径更新。

一步的 kernel 结构（no_async c8，以视频提取脚本 `videos/vllm-cpu-overhead/extract.py` 核对）：输入准备小 kernel（`_prepare_pos_seq_lens`、`_gather_block_tables`、`_compute_slot_mappings` 等）→ embedding + 36 层（每层一个 `reshape_and_cache`，注释内约 480 个 kernel）→ lm_head GEMM（约 0.44 ms）→ sampling（约 0.17 ms，到 `_post_update` 结束）。

注意（写正文必须交代）：
- **profiler 有开销。** 与无 profiler 的 benchmark ITL（c8：default 7.03、no_async 9.02、mrv1 7.03、piecewise 7.57、graph_none 13.95 ms）对照，default / mrv1 / no_async 的 period 与 ITL 基本吻合，但 piecewise（12.5 vs 7.6 ms）与 graph_none（42 vs 14 ms）被 profiler 放大 1.6–3 倍——profiler 拦截每次 cudaLaunchKernel。所以这两种配置的 period / idle / cpu_exec 绝对值不可用，只能用 benchmark 的 ITL；gpu_busy 与 kernel 数不受影响。
- async scheduling 把空转从 2.9 ms（c8）/ 4.2 ms（c64）压到约 0.3 ms；benchmark 里 ITL 差 2.0 ms（c8）。
- MRV2 的 cpu_exec 比 MRV1 短 0.37 ms（c8）/ 0.63 ms（c64），但在 async 下与上一步 GPU 重叠，ITL 相同。
- 单次 profile，无重复；step 数 35–40。

## CPU 耗时拆解（2026-10-01，正文第一节）

同一批 trace，主线程上 `execute_model` 注释与其中的 `cudaGraphLaunch`、GPU 上 `_post_update`（sampling 结束）与下一步 embedding kernel（forward 开始）配对，去首尾 20% 取中位数，µs：

| 配置 | 并发 | 处理输出 + 调度（sampling 结束 → execute_model 开始） | 准备输入（execute_model 开始 → cudaGraphLaunch） | cudaGraphLaunch 调用 | execute_model 总长 |
| --- | --- | --- | --- | --- | --- |
| no_async | 8 / 64 | 655 / 1632 | 1546 / 1803 | 443 / 455 | 2048 / 2342 |
| default (MRV2) | 8 / 64 | —（与上一步 GPU 重叠） | 1586 / 1687 | 491 / 466 | 2150 / 2206 |
| mrv1 | 8 / 64 | — | 1780 / 2053 | 463 / 466 | 2523 / 2821 |

- no_async c8：三项合计约 2.65 ms，对上每步空转 2.9 ms（其余为步内小间隙）；GPU 在 cudaGraphLaunch 返回时开始执行（launch 开始到 embedding kernel 约 445 µs）。
- 准备输入这段 trace 里只有上百个 aten 小 op（slice / copy_ / fill_ / FA3 `get_scheduler_metadata` 等），op 本身合计很短，其间空白即 Python 开销（未开 Python 栈采样，未进一步细分）。
- profiler 下读数偏大：no_async c8 步长 9.69 ms，对应无 profiler ITL 9.02 ms（约 +7%）。
- 一步 GPU 结构：36 层跨度约 6.4 ms，lm_head 0.44 ms，sampling 约 0.18 ms，kernel 忙碌时间合计约 6.8 ms。

## Launch 次数与准备输入细分（2026-10-01）

- **每步 launch 次数**（主线程 cudaLaunchKernel / cuLaunchKernelEx / cudaGraphLaunch，按相邻两次 execute_model 起点分窗，中位数，c8）：graph_none 492；piecewise 168（37 次 graph）；default / no_async 25（1 次 graph）；mrv1 21（1 次 graph）。前表 `launches` 列（6 / 12 / 149 / 148–184）只统计 execute_model 注释之内且窗口不同，**不要再引用**；视频初版写的“约 6 次 launch”据此是错的，已更正为 25。
- **default 的 25 次构成**：execute_model 内 10 次 = 1 次 cudaGraphLaunch + 4 个 MRV2 Triton 输入 kernel（`_prepare_pos_seq_lens`、`_combine_sampled_and_draft_tokens`、`_gather_block_tables`、`_compute_slot_mappings`）+ 5 个小 kernel（elementwise 填充 / 拷贝、FA3 `prepare_varlen_num_blocks`）；execute_model 之后 15 次 = lm_head（gather + GEMM）与 sampling（temperature、top-k/p、softmax、`_post_update` 等）。即 **lm_head 与 sampling 不在 CUDA Graph 里**。
- **piecewise 的 168 次**：execute_model 内 153 = 37 次 graph + 4 个 Triton 输入 kernel + 112 次 cudaLaunchKernel（其中每层 attention 附近约 3 个：`reshape_and_cache_flash`、FA3 attention（cutlass device_kernel）、一个 Triton 融合 kernel）；其后 15 次同 default。
- **准备输入细分**（no_async c8，execute_model 开始 → 第一个 Triton 输入 kernel → 最后一个 → cudaGraphLaunch）：0.94 / 0.21 / 0.30 ms，graph launch 0.44 ms。
- **处理输出、调度**：sampling 结束到下一次 execute_model 之间，trace 中主线程只有两次 event record，其余为未被 profiler 记录的 Python；内容按 `EngineCore.step()` 源码归纳（schedule → execute_model → sample_tokens → update_from_output：追加 token、`check_stop`、释放结束请求的 block；`allocate_slots` 分配 block）。
- **MRV1 准备输入**：`gpu_model_runner._prepare_inputs` 用 numpy 计算 positions、token 下标、cumsum 等，再 `copy_to_gpu`（源码核对）。

## 封面样稿（2026-10-01）

- 已读用户提供的全文；以 CPU 准备工作与 GPU 执行重叠、压缩等待间隙为抽象主线，避免教学式编号、图例及虚构硬件。
- 用户要求八种风格，并反馈之前 3:4 上下留白大；确认尝试 9:16 长竖版，放大标题和主体、收紧内部留白。本次比例不自动覆盖其他文章的默认 3:4 约定。
- 八版：蓝紫透明、珊瑚透明、复古印刷、瑞士排版、纸雕、深蓝铜色、紫底节奏、烟灰叠层。文件与提示词在 `covers/vllm-cpu-overhead/portrait-9x16-20261001/`，待用户选定。
- 已请求更高原生分辨率，但应以 `prompts.json` 中逐图记录的工具实际返回尺寸为准；未插值放大。

- 用户选中烟灰叠层版，要求文字下移避免平台遮挡；修订版 `08-monochrome-safe-v2.png` 顶部预留约 12% 安全区，左右留白增加，保留长竖版与原风格。
