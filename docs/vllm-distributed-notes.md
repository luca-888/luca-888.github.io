# 分布式推理 文章资料

状态：正文与 4 张图已起草（`content/vllm-distributed.md`、`src/articles/vllm-distributed/VllmDistributedFigures.tsx`，`status: draft`）。未做硬件测试，正文“实测”一节为占位，测试方案与成本估算见文末。参考方案与图未经逐步确认：用户 2026-10-01 授权直接写到可发布水平。

## 范围与硬件

- 独立成篇。prefill / decode、KV cache、prefix caching、chunked prefill 等背景只就地交代一句。
- Slug `vllm-distributed`；标题：主标题“分布式推理”，子标题“TP、DP + EP 与 PD 分离”（用户 2026-10-02 要求去掉“Vllm”：正文以 DeepSeek-V4-Pro 与 SGLang 配置为主线，与 vLLM 的关联不再紧密；slug 未改）。
- 源码基准：vLLM v0.30.0（2026-09-22）。实际阅读的是本地浅克隆 main@e006d76（2026-09-29），关键结论用 `git show v0.30.0:<path>` 回查过（all2all 后端列表、group 布局、`num_kv_head_replicas`、EP / DCP / PD 文档原文）。两者在 `distributed/` 下的差异限于 all2all、EPLB 内部实现，不影响正文表述。SGLang main@8854857，TensorRT-LLM main@111687f9，Dynamo main@2de120f。
- 实验：需要多卡，本次不做（用户 2026-10-01 决定）。

## 现状变化（以源码与文档为准）

发起时的印象是“TP、PP、PD 分离”三件事，v0.30 的实际范围更大：

| 项 | 现状 | 出处 |
| --- | --- | --- |
| 并行维度 | `parallel_state.py` 的 rank 布局为 `ExternalDP × DP × PP × PCP × TP`，另有 EP、EPLB、DCP、ETP 等 group；EP group 仅在 MoE 模型上创建 | `initialize_model_parallel` |
| DP | `--data-parallel-size`，每个 DP rank 一个 EngineCore 进程；内部 LB 只看 running / waiting 队列，另有 hybrid 与 external LB 两种模式 | `docs/serving/data_parallel_deployment.md` |
| MoE 默认行为 | 不加 `--enable-expert-parallel` 时 expert 层是规模 `DP × TP` 的 TP；加上后 EP 规模自动取 `TP × DP` | 同上、`expert_parallel_deployment.md`、`fused_moe/config.py` |
| all2all 后端 | 默认 `allgather_reducescatter`；另有 `deepep_high_throughput`、`deepep_low_latency`、`deepep_v2`、`nixl_ep`、`flashinfer_nvlink_*`、`mori_*`；`naive` 与 `pplx` 已被映射为默认后端 | `config/parallel.py` |
| EPLB | `--enable-eplb`，默认 `window_size=1000`、`step_interval=3000`、`use_async=true`；算法改写自 DeepSeek EPLB | 文档、`eplb/policy/default.py` |
| DBO | `--enable-dbo`，只支持 DP + EP，文档称目前需要 DeepEP | `docs/design/dbo.md` |
| Context parallel | 新增 `-dcp`：KV cache 按 token 位置交错分片，消除 `tp_size / H` 倍的重复；prefill CP 仍在开发 | `context_parallel_deployment.md` |
| PD 分离 | 仍标 experimental；9 种 connector，主力 `NixlConnector`（默认 pull，另有 push 模式与租约续期）；文档明确“DOES NOT improve throughput” | `docs/features/disagg_prefill.md`、`docs/design/nixl_kv_*.md` |
| all-reduce | `CudaCommunicator` 依次尝试 FlashInfer、NCCL symmetric memory、custom all-reduce、PyNccl；custom all-reduce 上限 8 MB，world size 2 或 NVLink 全连接时启用 | `device_communicators/cuda_communicator.py`、`custom_all_reduce.py` |

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| vLLM 源码：`distributed/parallel_state.py`（`initialize_model_parallel`、`GroupCoordinator.dispatch/combine`）、`model_executor/layers/linear.py`（`QKVParallelLinear`、`RowParallelLinear`）、`device_communicators/all2all.py`（`AgRsAll2AllManager`）、`cuda_communicator.py`、`custom_all_reduce.py`（`should_custom_ar`）、`eplb/eplb_state.py`（术语与文件头）、`eplb/policy/default.py`、`fused_moe/expert_map_manager.py`、`fused_moe/config.py`（并行配置）、`v1/worker/gpu/dp_utils.py`、`pp_utils.py`（前 140 行）、`kv_connector.py` | vLLM 团队 | 已读上述函数；`eplb/rebalance_execute.py`、`nixl/*.py`、`shm_broadcast.py`、DeepEP 各 manager 只看了结构 | 核心 | 一、三、四 |
| vLLM 文档：[Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)、[Data Parallel](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)、[Expert Parallel](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)、[Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | vLLM 团队 | 全文已读（本地 `docs/`） | 核心 | 全文 |
| vLLM 设计文档：`dbo.md`（前 60 行）、`nixl_kv_push_connector.md`（前 140 行）、`nixl_kv_cache_lease.md`（前 40 行）、`nixl_connector_usage.md`（前 80 行）、`nixl_connector_compatibility.md`（前 50 行）、`kv_transfer/README.md` | vLLM 团队 | 部分已读 | 重点 | 三、四 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053)（arXiv 1909.08053） | Shoeybi et al.，NVIDIA | Model Parallel Transformers 一节全文已读（arXiv 源文件 `modelpar.tex`） | 核心 | 一 |
| [DistServe](https://arxiv.org/abs/2401.09670)（arXiv 2401.09670，OSDI'24） | Zhong et al. | abstract、intro、tradeoff analysis、discussion 全文与 evaluation 的 latency breakdown 已读；placement 算法未读 | 核心 | 一、四 |
| [Splitwise](https://arxiv.org/abs/2311.18677)（arXiv 2311.18677，ISCA'24） | Patel et al.，UW / Microsoft | abstract、7 条 insight 标题、KV-cache transfer 设计与评测已读；集群设计与模拟器未读 | 重点 | 四 |
| [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) | LMSYS / SGLang 团队 | cache-aware LB 与 DP attention 两节已读 | 重点 | 二、三 |
| [Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang 团队 | 到 TBO 消融一节为止已读，之后（toolkits、limitations）未读 | 核心（相关系统） | 三、四 |
| SGLang 源码：`srt/layers/dp_attention.py`、`srt/disaggregation/prefill.py`、DeepSeek-V3 cookbook | SGLang 团队 | 仅定位与文件头 | 补充 | 三、四 |
| TensorRT-LLM 文档：`features/parallel-strategy.md`（全文）、`features/disagg-serving.md`（前 60 行）、DWDP tech blog（到 roofline 一节开头） | NVIDIA | 如左 | 重点（相关系统 / 反对意见） | 三、四 |
| [Beyond the Buzz: A Pragmatic Take on Inference Disaggregation](https://arxiv.org/abs/2506.05508) | Mitra et al.，NVIDIA | 仅 abstract | 重点（反对意见） | 四 |
| [TaiChi](https://arxiv.org/abs/2508.01989) | Wang et al. | 仅 abstract | 重点（反对意见） | 四 |
| Dynamo `lib/kv-router/src/conditional_disagg.rs` | NVIDIA | 前 60 行（策略定义与默认阈值） | 补充（相关系统） | 四 |
| [Helix Parallelism](https://arxiv.org/abs/2507.07120) | Bhatia et al.，NVIDIA | 仅 abstract；正文未引用，只在 notes 中作为 DCP 的出处 | 补充 | 三 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 与 HF `config.json` | DeepSeek-AI | 报告仅 abstract；config 全部相关字段已读 | 补充 | 三、四的参数计算 |
| DeepSeek 官方推理系统说明（open-infra-index）、DeepEP / EPLB 仓库 | DeepSeek-AI | 未读，只通过 SGLang blog 与 vLLM 源码注释间接了解 | 补充 | — |
| NIXL 仓库与文档 | NVIDIA | 未读，只通过 vLLM connector 文档了解 | 补充 | — |

## 发起时的要点逐条核实

| 要点 | 结论 | 依据 |
| --- | --- | --- |
| TP 切每层的权重矩阵，每层两次 all-reduce | 成立 | Megatron-LM 原文“only two all-reduces in the forward path”；vLLM `RowParallelLinear.forward` |
| TP 能装下大模型也能降延迟，但依赖卡间带宽 | 成立 | DistServe 的 intra-op 加速系数 $K$（$1<K<2$）；vLLM 文档建议无 NVLink 时改用 PP |
| PP 按层分段，通信少，只解决装不下，不降延迟 | 成立 | DistServe：inter-op 的请求延迟 $D_s \approx D$，吞吐可线性扩展 |
| DP 有两层意思 | 成立 | vLLM DP 文档：dense 模型各 rank 独立；MoE 下 rank 之间要同步 |
| MLA 的 KV 无法按 head 切，TP 下每张卡存一份完整的 | 成立 | vLLM context parallel 文档（“1 kv-head … `-tp 8` causes 8x KV cache duplication”）、SGLang v0.4、TensorRT-LLM parallel-strategy |
| 单个 expert 很小，切开后通信比计算贵 | **部分修改** | 一手来源没有“通信比计算贵”的说法。能核实的是：TP 把每个 expert 切片且每张卡处理全部 token，EP 让每张卡只处理路由到本地 expert 的 token（TensorRT-LLM）；高 TP 度下切片过窄影响 GEMM 效率（SGLang 针对 dense FFN 的说明）。EP 的通信并不比 TP 少。正文改为“expert 又多又小，不用再切”，不写通信贵于计算 |
| EP 把整个 expert 放在一张卡上，token 按路由结果在卡间交换 | 成立，但有补充 | vLLM 默认后端是 all-gather + reduce-scatter，所有 token 发给所有卡；只有 DeepEP 等后端才是按路由结果稀疏发送 |
| EP 带来 expert 负载不均，对应 `distributed/eplb` | 成立 | vLLM EP 文档、`eplb_state.py` |
| PD 分离：prefill 吃算力、decode 吃带宽且互相干扰 | 成立 | DistServe、Splitwise insight 5 |
| 权衡是 KV 传输成本与消除干扰的收益 | 成立，但有补充 | 还有 P/D 配比（rate matching）与资源闲置的成本；vLLM 文档明确不提升吞吐 |

## 正文中自己计算的数字

均按 HF `config.json` 计算，正文标注为“按配置计算”：

- Llama-3.3-70B（hidden 8192，80 层，8 个 KV head，head_dim 128，BF16；config 取自 `unsloth/Llama-3.3-70B-Instruct` 镜像）：all-reduce 每 token 8192 × 2 B = 16 KB；64 个 token 1.0 MB，160 次合计 0.17 GB；8192 个 token 134 MB，合计 21.5 GB。KV 每 token 80 × 8 × 128 × 2 × 2 B = 0.31 MB，4096 个 token 1.34 GB。
- DeepSeek-V3（hidden 7168，`moe_intermediate_size` 2048，256 个 routed expert，61 层其中前 3 层为 dense，`kv_lora_rank` 512 + `qk_rope_head_dim` 64）：每个 expert 3 × 7168 × 2048 = 4404 万参数；256 × 58 层 = 654B，占 671B 的 97%。KV 每 token 61 × 576 × 2 B = 70 KB，4096 个 token 0.29 GB。
- 100 Gbps = 12.5 GB/s，1.34 GB 约 0.11 s。

## 未能核实或只有间接依据

- **“SGLang 自报 5 倍”“DP attention 1.9 倍”“cache-aware LB 1.9 倍”**：均为厂商自报，未复现。
- **DeepEP 的稀疏 dispatch 细节、EPLB 原始算法**：未读 DeepSeek 仓库，依据是 vLLM 的改写实现与 SGLang 的转述。
- **NIXL 底层走 RDMA**：依据 vLLM 文档（UCX 为默认 backend，另有 GDS、LIBFABRIC）；同机时实际走什么传输未核实，正文写“通常走 RDMA”。
- **“DBO 来自 DeepSeek 的线上系统”**：依据 SGLang blog 的转述。
- **Beyond the Buzz、TaiChi、Helix**：只读了 abstract，正文只引用 abstract 中的结论。
- **DWDP 的 10% 同步开销**：TensorRT-LLM blog 自报，条件是 DeepSeek-R1、GB200、ISL/OSL = 8K/1。
- **vLLM 的 PP 实现细节**（stage 间 hidden state 的发送与采样结果回传）：只读了 `pp_utils.py` 前 140 行，正文只写了与实现无关的一般性结论。
- **vLLM 外部路由方案**（production-stack、llm-d 的 prefix-aware 路由）：本次未读，正文未展开。

## 文章主线

一张卡不够时有三种情形：装不下、请求太多、prefill 与 decode 互相拖慢。对应切模型、复制模型、拆阶段。切模型的方式由模型结构决定：稠密模型用 TP（PP 一句带过）；MoE 模型的 KV cache 无法按 head 切、expert 又多又小，所以 attention 用 DP、expert 用 EP；实例之间再做 PD 分离。每种并行回答三个问题：切的是什么、通信发生在哪、适用于什么模型。

## 章节提纲

1. **TP**：Megatron-LM 的切法、每层两次 all-reduce、通信量与 token 数成正比、KV 按 head 切及 head 不够分时的重复；PP 一段带过。
2. **DP**：多副本与路由（prefix cache 亲和）。
3. **MoE：DP attention + EP**：MLA 的 KV 切不开 → DP attention（及 vLLM 的 DCP 路线）；expert 整个放 → dispatch / combine、两类后端；同步与负载不均 → DP coordinator、EPLB、DBO；DWDP 作为反对意见。
4. **PD 分离**：为什么分（DistServe、Splitwise）、vLLM 的实现、要传多少、什么时候不值得。
5. **怎么选**：三个问题。
6. **实测**（占位）。
7. **小结**。

## 核心图

| 图 | 回答的问题 | 做法 |
| --- | --- | --- |
| `DistTpLayer` | TP 下一层里切了什么、哪里通信 | 两张卡两条 `Band`，权重块按一半标注，两根跨卡的 amber 条是 all-reduce |
| `DistKvTpDp` | 只有 1 个 KV head 时，同样 4 张卡能装多少请求 | 上下两行同一组 GPU，只改并行方式；重复的副本用红色，面积即浪费 |
| `DistEpPath` | 一个 token 在 MoE 层里怎么跨卡 | 4 张卡 8 个 expert 选 2 个，紫色描出 router → dispatch → expert → combine 的完整路径 |
| `DistPdFlow` | PD 分离下一个请求经过哪些实例、KV 何时传 | router / P / D 三个 `Band`，六步编号路径 |

语义色：权重与计算 blue，expert teal，KV cache green，重复的 KV red，通信 amber，示例路径 purple。四张图都只用 `figure-kit` 现有图元，未改 `figure-kit`。未做小红书窄版布局。

## 实测方案（未执行，留待用户决定）

按 `docs/gpu-experiments.md`：多卡配置，申请前一次性做完整计划。以下为计划草案，执行前需用户确认硬件与预算，并重新查询费率。

### 目标

为正文“实测”一节提供三组对照，每组在正文中只取一个代表性数字：

- **A. TP**：TP 对单步延迟的加速比，以及 all-reduce 占一步的比例。
- **B. DP attention + EP**：MLA 模型在 TP 与 DP + EP 下的 KV cache 容量与吞吐；EPLB 对负载均衡度的影响。
- **C. PD 分离**：同样两张卡，合并部署与 1P + 1D 的 P99 ITL、TTFT 与 KV 传输耗时。

### 硬件与获取途径

- Modal，单容器 4 × H100（80 GB），一次申请、一次远端调用完成 A、B、C（C 只用其中两张卡）。
- 需在容器内先记录 `nvidia-smi topo -m` 与 `nvidia-smi nvlink -s`：Modal 的多卡 H100 是否有 NVLink 直接决定 A 组结论的口径，未知。
- 备选：用户的 RTX PRO 6000 多卡机（无 NVLink，可对照 PCIe 下的 TP 与 PP）。本计划不含。

### 测试矩阵

| 组 | 模型 | 配置 | 负载 | 采集 |
| --- | --- | --- | --- | --- |
| A | Qwen3-32B（BF16 约 65 GB，单卡可装） | TP = 1、TP = 2、TP = 4、PP = 2 | 输入 128 / 输出 256，并发 1、8、64；另测输入 8192 / 输出 1 的 TTFT | ITL、TTFT、吞吐；并发 8 时 torch profiler 一段 trace，统计 all-reduce 耗时占比 |
| B1 | DeepSeek-V2-Lite（MLA，约 31 GB） | TP = 4；DP = 4 + EP | 启动日志；输入 512 / 输出 256，并发 64、256 | 启动日志中的 `GPU KV cache size`、吞吐、ITL |
| B2 | Qwen3-30B-A3B（128 个 expert，约 61 GB） | DP = 4 + EP；DP = 4 + EP + EPLB（`log_balancedness`） | 输入 512 / 输出 256，并发 256，持续到 EPLB 至少重排一次（默认 3000 步） | balancedness 日志、吞吐 |
| C | Qwen3-8B | 两个完整实例轮询；1P + 1D（`NixlConnector`，同机） | 输入 4096 / 输出 256，泊松到达，3 档请求率 | P50 / P99 ITL、TTFT、connector 统计的传输耗时 |

共 10 个服务配置。all2all 后端只用默认的 `allgather_reducescatter`：DeepEP 需要额外编译与驱动配置，不在本计划内，因此 B 组得不到 DeepEP 与 DBO 的数据。

### 依赖与准备（不占用 GPU）

- 镜像：vLLM v0.30.0（与其他几篇一致）、`nixl`、benchmark 客户端；在无 GPU 容器中验证 import 与 CLI 参数。
- 模型下载到 Modal Volume（CPU 容器）：约 65 + 31 + 61 + 16 = 173 GB。
- 负载脚本与 P/D 代理脚本（参照 vLLM `examples/disaggregated` 与 EP 文档里的 client orchestration）先在本地用 mock server 跑通。

### 正确性检查

- 每个配置启动后先发 1 个固定 prompt（temperature 0），核对各配置输出一致（PD 分离与合并部署、TP 与 DP + EP）。
- 检查日志中的 rank 分配行（`rank … is assigned as DP rank …`）与 all-reduce / all2all 后端选择。

### 时间与费用预估

费率为 2026-10-01 在 modal.com/pricing 查到的标价，执行前需重查：H100 $0.001097/s（$3.95/h），CPU $0.0000131/core/s，内存 $0.00000222/GiB/s。

| 阶段 | 时长 | 费用 |
| --- | --- | --- |
| 镜像构建与验证（无 GPU） | 约 20 min | < $1 |
| 模型下载（CPU 容器，8 core / 32 GiB） | 约 30–40 min | < $1 |
| GPU 占用：10 个配置 ×（启动约 4 min + 压测约 8 min） | 约 2 h | 4 × H100 $15.8/h + 32 core $1.5/h + 256 GiB $2.0/h ≈ $19.3/h，合计约 $39 |
| 结果回传与释放 | 约 5 min | 计入上项 |

- GPU 占用上限：3 h（容器 timeout 设为 3 h），预算上限 $60（美元）。
- 不确定项：4 × H100 的排队时间（不计费，但可能很长）；B2 等 EPLB 重排的时间可能超过预估；Volume 在 1 TiB 免费额度内，网络流量未计。
- 触及时间或预算上限时停止并保存已有结果，不自动重跑；补测前更新剩余矩阵与费用。

### 不在本计划内

- DeepSeek-V3 全尺寸（需 8 × H200 级别，Modal 标价约 $36/h）与多节点 EP。
- 跨节点 PD 分离与 RDMA 网络。
- DeepEP、DBO。

### 完成条件

10 个配置各有一份原始结果（JSON）与启动日志，放入 `public/measurements/vllm-distributed/`；资源已释放并核查；记录实际占用时间与账单。

## 补充：DeepSeek-V4-Pro 每个 token 的 KV 大小（2026-10-02，自行计算）

依据：HF `config.json` 的 `compress_ratios`（61 层中 30 层为 4、31 层为 128，末位 0 属于 MTP 层）；vLLM `vllm/models/deepseek_v4/attention.py`、`compressor.py`（fp8_ds_mla 布局：每个 state 584 B = 448 NoPE + 128 RoPE + 8 scale；只有 ratio = 4 的层带 indexer，indexer cache 每个 state 132 B）。

- 随长度增长的部分：30 × (584 + 132) / 4 + 31 × 584 / 128 ≈ 5.5 KB/token。1M token（1,048,576）≈ 5.8 GB。
- 每个请求固定的部分：各层 128 token 的滑动窗口（未压缩）与 compressor 状态，约 23 MB。
- 按视频里的显存估算：TP = 8 余 72 GB ≈ 1300 万 token（约 12 个满 1M 的请求）；DP attention + EP 每卡 47 GB ≈ 850 万 token（每卡约 8 个），8 卡 372 GB ≈ 6700 万 token（约 64 个）。
- 未扣：对齐 padding、prefill 的中间结果与缓冲区。未实测。

## 补充：正文的 MoE 示例改为 DeepSeek-V4-Pro（2026-10-02）

- 第三章的示例模型由 DeepSeek-V3 换成 V4-Pro，与视频一致：61 层、每层 384 个 expert + 1 个 shared、每个 token 选 6 个；权重 865 GB（expert 836 GB，其余 29 GB）。
- 新增 8 × B200 的显存账本表（TP = 8 对 DP attention + EP）与 KV 大小的换算；图 `DistKvTpDp` 的标题与图注去掉“MLA latent”的说法。
- 第四章“要传多少”的表里，DeepSeek-V3 一行换成 V4-Pro（每个 token 约 5.5 KB）。
- 第五章补了 SGLang cookbook 给 V4-Pro 的两档配置，以及 vLLM 里对应的通用参数。vLLM 没有针对 V4-Pro 的现成配方，未实测。
- 保留未改：EPLB 的 DeepSeek-R1 例子（出自 vLLM 文档）、SGLang 在 96 张 H100 上部署 DeepSeek-V3 的数字（原始来源就是 V3）。
- 新增来源：DeepSeek-V4-Pro 的 model card 与 config（已读）、SGLang DeepSeek-V4 cookbook（已读）、vLLM `vllm/models/deepseek_v4/`（已读，v0.30.0 中存在）。

## 补充：正文改为以 DeepSeek-V4-Pro 开篇，主线与视频一致（2026-10-02，用户要求）

- 新主线：865 GB 对 180 GB 装不下 → V4-Pro 里面有什么（权重与 KV 两样东西）→ TP → KV 切不开 → DP attention + EP（attention 为什么每张卡一份、expert 为什么分开放、两个代价）→ PD 分离 → 怎么选 → 实测 → 小结。
- 章节由七章变八章。原“二、DP：复制模型，分发请求”并入“六、怎么选”；稠密模型的 Llama-3.3-70B 只在“KV 切不开”一章作对照。
- TP 一章的 all-reduce 数据量表改按 V4-Pro（hidden size 7168，BF16，122 次）自行计算：decode 64 个请求每次 0.9 MB、合计 0.11 GB；prefill 8192 token 每次 117 MB、合计 14 GB。“每层 2 次”沿用 Megatron-LM 的结构，未在 vLLM 的 V4 实现里逐处核对。
- 新增两张图：`DistV4Anatomy`（模型结构）、`DistLedger`（两种切法下一张卡的显存），显存账本表相应删去。共 6 张图。
- 改写前的正文备份在 `content/vllm-distributed.bak-v1.md`。

## 审核记录（2026-10-02）

对照 vLLM v0.30.0 `vllm/models/deepseek_v4/attention.py` 与 config 复核正文后改了四处：
- TP 下 V4 的 attention：`fused_wqa_wkv`（Q 的降维与 KV 投影）是 `disable_tp=True` 的复制层，只有 `wq_b`、`wo_a`（按列）与 `wo_b`（按行，末尾 all-reduce）被切。正文原写“Q、K、V 的投影按 head 切”，对 V4 不成立，已改。
- 开头“能同时服务的请求多约 5 倍”改为“能装下的 KV cache 是前者的约 5 倍”（372 ÷ 72 ≈ 5.2）。
- “要传多少”表中 V4-Pro 一行补上每个请求固定的约 23 MB（滑动窗口与 compressor 状态），4096 token 合计约 0.05 GB。compressor 状态是否随 PD 传输未核对。
- attention “读前面所有 token”改为“读前面的 token”（压缩比 4 的层用 indexer 只取 top-k）。
复核无误：all-reduce 数据量表、Llama 的 0.31 MB/token、836/29/865 GB 的拆分、30 层与 31 层的压缩比计数、SGLang 两档配置的参数。
未核对：MoE 子层在 V4 的 NVIDIA 路径下是否恰好一次 all-reduce（“每层 2 次”沿用 Megatron-LM 结构）。
视频同样的问题：“KV 各存各的”一幕标题写“多了约 5 倍”，应为“约 5 倍”；未改成片。

## 事实核查（2026-10-02，第二轮）

对照 vLLM main@e006d76（关键路径在 v0.30.0 中存在）、HF config 与 model card、SGLang cookbook、DistServe / Splitwise 源文件、SGLang 两篇 blog、TensorRT-LLM DWDP、Dynamo `conditional_disagg.rs`、两篇 arXiv 摘要逐条核对。改了三处：
- PD 分离：D 不是“从第二个 token 开始生成”。KV 读完后 scheduler 把 `num_computed_tokens` 回退一位，重算最后一个 prompt token 并采样第一个输出 token（`vllm/v1/core/sched/scheduler.py` 的 `_update_waiting_for_remote_kv` 一段），P 的 `max_tokens=1` 输出不被使用。
- “8 × 47 = 372 GB”改为“8 × 46.5 = 372 GB”（正文与 `DistLedger` 图）。
- 参考文献中 SGLang v0.4 的标题改为原文标题。
未改、留作已知：custom all-reduce 的 8 MB 是默认上限，NCCL symmetric memory 默认开启时按架构与卡数收紧（8 卡 H100 为 256 KB，8 卡 sm_100 为 1 MB，`all_reduce_utils.py`）；865 GB 为十进制而 B200 的 180 GB 为标称值，口径未统一；V4 的 attention 代码里未见 DCP 分支；EPLB 的 DeepSeek-R1 例子出自 `eplb_state.py` 文件头而非文档。
视频：成片第 7 幕已是“约为 TP 的 5 倍”；“官方配置”实为 SGLang cookbook 的参数，未改。

## 精简（2026-10-02，用户要求）

正文由约 7,900 字减到约 6,400 字（不含参考）。删去：“实测”占位一章与“小结”一章；第二章 vLLM 多套 all-reduce 实现的两句；第四章 `--all2all-backend` 后端列表、`sync_cudagraph_and_dp_padding`、DP coordinator、EPLB 的 288 个 expert 与 3000 步、`--enable-dbo` 参数名；第五章 NIXL 租约与心跳；第六章多副本路由与 prefix caching 一段。去掉“自报”“按模型配置计算”等限定语。章节现为六章，以“怎么选”收尾。测试方案仍保留在本文件。

## 去 AI 腔（2026-10-02，用户要求）

去掉设问与预告句（“问题是…”“答案取决于…”“问题出在…”“每种并行要回答三个问题”“按顺序问四个问题”）、“换来的是 A，不是 B”式对比、“代价一 / 代价二”编号小标题与“X 为什么…”式小标题、列表里的加粗标签。“怎么选”由加粗问句列表改为一段正文。内容与数字未变。

## 小红书正文图（2026-10-02）

已在 `src/xhs/articles.ts` 登记并导出 14 页（`xhs/vllm-distributed/`）。六张图都补了紧凑布局（`useCompact`，520 宽）：TP 图改为两张卡左右并排、数据自上而下；KV 图每张卡 2 × 2；EP 图只收窄各列；PD 图改为 router / P / D 三条横带，路径沿两侧绕行；结构图 attention 在上、MoE 在下；账本图收窄色条。TP 图的格内小字改为“只接本卡的 head / 算出中间结果的一半 / 只接本卡那一半”，图例注明 all-reduce 是相加不是拼接。

## 封面设计（2026-10-02）

- 已读全文。用户确认“分布式雕塑”概念，并要求副标题 DeepSeek V4。通过一个整体拆成若干模块、少量跨间隙连接表现拆分与协同，不画具体硬件或通信流程。
- 3:4 竖版；冷白底、烟灰模块、钴蓝连接；标题 Distributed Inference，副标题 DeepSeek V4，小字 TP · DP + EP · PD。标题顶部保留安全空间。
- 内置 imagegen 生成样稿 `covers/vllm-distributed/sculpture-20261002/cover-v1.png`；同目录保存完整提示词与实际尺寸，待用户评估。

- 用户纠正：中英文都要有，中文主体；默认长竖版 9:16，不再 3:4。已同步 AGENTS.md 与 docs/xiaohongshu.md。重做 `cover-v2-zh-portrait.png`：中文“分布式 / 推理”为最大标题，英文 Distributed Inference 为辅，保留 DeepSeek V4 与并行方式小字；顶部约 12% 安全区。此条替代上面的旧比例与纯英文标题方案。

- 用户否定烟灰雕塑：太厚重、意义弱。确认新方向“同一整体，两种分法”：蓝紫薄片表达 TP 切片，珊瑚完整小模块分组表达 EP 按 expert 分配。已生成 `covers/vllm-distributed/slicing-grouping-20261002/cover-v1.png`，保留中文主体、英文辅助、DeepSeek V4 副标题和顶部安全区；是抽象对照，不是 token 流程图。

- 用户认为切片/分组版本仍不合适，要求更抽象。新稿去掉具体模块与切片，用蓝紫、珊瑚半透明曲面交叠表现协同氛围；保存至 `covers/vllm-distributed/abstract-20261002/cover-v1.png`，待评估。

- 用户要求瑞士风格、与视频一致。已读取 `videos/vllm-distributed/film.html` 并查看视频帧 s-16.png，沿用暖米白 #f1ede4、黑字 #141414、青绿 #0fa391、钴蓝 #2350d8 的平面视觉。生成 `covers/vllm-distributed/swiss-20261002/cover-v1.png`：中文主体、中英双语、DeepSeek V4、顶部安全区、抽象错位色带，待用户评估。

- 按用户“多几个版本”追加同系列瑞士风格四稿：错位网格、斜向色带、圆形分割、大色面；连同首稿打包 `covers/vllm-distributed/swiss-20261002/distributed-swiss-five-versions.zip`。均保持中英标题、DeepSeek V4、长竖版及视频配色，待选择。

- 2026-10-02 追加代码绘制版（canvas，非图像模型生成），与量化一篇的封面同一做法：`covers/vllm-distributed/swiss-code-20261002/cover.html`，`node render.mjs` 导出。图形取自视频账目：上面一条是 865 GB 的模型切成 8 段（expert 青绿、其余钴蓝），每段正对下面一张 180 GB 的卡，卡内按 TP = 8 的比例填充（expert 104、其余 4、空 72）。中文“分布式 / 推理”为主标题，英文 Distributed Inference 为辅，保留 DeepSeek V4 与 TP · DP + EP · PD；顶部约 12% 不放文字。实际尺寸：`portrait-1080x1920.png` 为 1080 × 1920（9:16），`landscape-1920x1080.png` 为 1920 × 1080（16:9）。待用户评估。

- 用户认为代码绘制版与量化一篇的封面过于雷同（都是标题在上、图形在下、同色系的竖条）。重做构图并错开主色：8 张卡顶到画面边缘，竖版图形在上、标题在下，横版图形在右、标题在左；以青绿大色块为主，DeepSeek V4 改为青绿。量化一篇保持标题在上、细刻度在下、钴蓝为主。上一版移到 `swiss-code-20261002/v1/`；`compare-portrait.png`、`compare-landscape.png` 是两篇封面的并排对照。待用户评估。

## 视频去掉官方 logo（2026-10-02）

- 背景：小红书上这条视频审核明显变慢，用户决定去掉第一页的 DeepSeek 官方 logo。是否因 logo 触发人工审核只是推测，未证实。
- 做法：第一页的 logo 图换成纯文字“DeepSeek V4-Pro”（黑色、同一字号与位置），其余不变。只重渲染了第一个 5 秒分段，其余 47 段沿用原缓存；音轨直接从原成片复制，未重新混音。
- 成片：`videos/vllm-distributed/build/vllm-distributed-nologo-1080p60-music.mp4`（带配乐）与 `vllm-distributed-nologo-1080p60.mp4`（只有音效），均 236.2 秒。
- 源文件：发布版（v3）的源文件原本只在另一个会话的临时目录里，已把去掉 logo 后的 `film3.html` 与 `render.mjs` 存到 `videos/vllm-distributed/v3/`。配乐的 wav 与渲染缓存没有搬进项目。
- 2026-10-07 整理：只保留带配乐的成片，改名为 `videos/vllm-distributed/build/vllm-distributed.mp4`；源文件移到 `videos/vllm-distributed/film.html` 与 `render.mjs`。配乐不在项目里，重渲只能得到无配乐版。
- 封面：`covers/vllm-distributed/video-cover/first-page-1920x1080.png`（16:9）与 `feed-1440x1080.png`（4:3）已换成无 logo 版；带 logo 的旧图改名为 `*-logo-*` 保留。
