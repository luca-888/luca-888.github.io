# EP 通信与 PP：调研笔记

视频 `videos/ep-pp-communication/`（先视频，文章以后写）。状态：方案已确认，12 fps 预览成片已做（2026-10-20）；配音未合成。

## 全片的问题与边界

DeepSeek-V3 的 FP8 权重 689 GB，一台 8 × H800 共 640 GB，放不下，必须跨机。跨机有两种切法：按 expert 切（EP），或按层切（PP）。两种切法在线上各传什么、传多少？怎样传得起？推理里各自用在哪？

- 运行例子统一用 DeepSeek-V3 / H800：hidden 7168，256 个 routed expert + 1 个 shared expert，top-8，61 层中 58 个 MoE 层，每个 token 最多去 4 个节点；节点内 NVLink 实测约 160 GB/s，节点间每卡一张 CX7 400 Gb/s（约 50 GB/s）。原因：DeepSeek 公开了这套配置的真实 profile、DeepEP 的实测表和部署配置，画面可以全部用一手数据。
- 与 MoE kernel 视频的分工：本片只讲 token 在卡间的搬运；expert 本身的计算（grouped GEMM 等）在本片里只是一个方块。
- 与已发的分布式推理文章（content/vllm-distributed.md）的关系：本片独立成篇，dispatch / combine 的定义在本片里重新交代。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 版本 / 日期 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [DeepEP](https://github.com/deepseek-ai/DeepEP) README（V2.5）、v1.2.1 README、首个 commit 的 README、EPv2 发布 README；`figures/normal.png`、`low-latency.png`；v1.2.1 `deep_ep/buffer.py`、`csrc/kernels/internode.cu` | DeepSeek | 当前 `b5a33ee`（2026-10-15）；V1 `ebfe47e46`、`v1.2.1`（`9af0e0d0e`）；EPv2 `b306af06a` | 已读（V2 kernel 源码未读） | 核心 | 第 4–8、10 幕 |
| 2 | [profile-data](https://github.com/deepseek-ai/profile-data)：`decode.json`、`prefill.json` 与两张重叠示意图 | DeepSeek | `4496024`（2025-03-21） | 已读并解析（`videos/ep-pp-communication/data.py`） | 核心（画面上的真实 trace） | 第 3、7、8 幕 |
| 3 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，§2.1 node-limited routing、§3.2.2 跨节点 all-to-all、§3.4 推理部署、§3.5 硬件建议 | DeepSeek-AI | arXiv 源文件 v2 | 已读相关章节 | 核心 | 第 2–4 幕 |
| 4 | [Day 6：DeepSeek-V3/R1 推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | DeepSeek | 2025-03-01 | 已读 | 高 | 第 7–9 幕 |
| 5 | [Insights into DeepSeek-V3（ISCA '25）](https://arxiv.org/abs/2505.09343)，通信上限的估算、scale-up / scale-out 融合 | DeepSeek-AI | arXiv 源文件 | 已读相关段落 | 高 | 第 10 幕 |
| 6 | [Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs](https://lmsys.org/blog/2025-05-05-large-scale-ep/) | SGLang 团队 | 2025-05-05 | 已读 | 高 | 第 6–9 幕 |
| 7 | SGLang 源码：`layers/moe/token_dispatcher/deepep.py`、`layers/moe/utils.py`（`DeepEPMode`）、`batch_overlap/`、`eplb/`、`managers/scheduler_pp_mixin.py` | SGLang | `78185f4b`（2026-10-20） | 已读 | 高 | 第 6–9、12 幕 |
| 8 | vLLM 源码：`config/parallel.py`（all2all backend）、`distributed/device_communicators/all2all.py`、`model_executor/layers/fused_moe/modular_kernel.py`（DBO）、`distributed/eplb/`、`config/vllm.py`（`max_concurrent_batches`）、`v1/engine/core.py`、`distributed/utils.py`（`get_pp_indices`）；`docs/design/dbo.md`、`docs/serving/parallelism_scaling.md` | vLLM | `fce64bc5c2`（2026-10-20） | 已读 | 高 | 第 8、10、12、13 幕 |
| 9 | [Pipeline Parallelism in SGLang: Scaling to Million-Token Contexts and Beyond](https://lmsys.org/blog/2026-01-15-chunked-pipeline/) | Shangming Cai 等 | 2026-01-15 | 已读 | 高 | 第 11–13 幕 |
| 10 | [EPLB](https://github.com/deepseek-ai/EPLB)：README 与 `eplb.py` | DeepSeek | `d52c72d` | 已读，本地运行过示例 | 中 | 第 9 幕 |
| 11 | SGLang GB200 NVL72 两篇（[Part 1](https://lmsys.org/blog/2025-06-16-gb200-part-1/)、[Part 2](https://lmsys.org/blog/2025-09-25-gb200-part-2/)）、[Ant Group H20 实践](https://lmsys.org/blog/2025-09-26-sglang-ant-group/) | SGLang、Ant Group | 2025 | 已读 | 中 | 第 8、10 幕 |
| 12 | vLLM 博客 [Large Scale Serving: 2.2k tok/s/H200 with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving)、[GB200 Part I](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) | vLLM 团队 | 2025-12、2026-02 | 已读 | 中 | 对比与现状 |
| 13 | NCCL 文档：[Point-to-point（all-to-all 示例）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)、[Collectives API（ncclAlltoAll）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html)、2.27–2.31 release notes | NVIDIA | 2026-10 查阅 | 已读 | 中 | 第 5 幕（为什么不用通用 all-to-all） |
| 14 | TensorRT-LLM 博客：[Scaling Expert Parallelism Part 1](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog04_Scaling_Expert_Parallelism_in_TensorRT-LLM.html) 及 Part 2/3、blog18（NVLinkOneSided AlltoAll）、blog19（DWDP） | NVIDIA | 2025–2026 | 已读 | 中 | 对比与反对意见 |
| 15 | Perplexity：[Efficient and portable MoE communication](https://research.perplexity.ai/articles/efficient-and-portable-mixture-of-experts-communication)、[Enabling Trillion-Parameter Models on AWS EFA](https://research.perplexity.ai/articles/enabling-trillion-parameter-models-on-aws-efa)；pplx-kernels README（`f70d1eb`） | Perplexity | 2025 | 已读 | 低 | 对比 |
| 16 | GPipe（1811.06965）、Megatron-LM 2021（2104.04473）的 bubble 公式 | Google、NVIDIA | — | 已读相关段落 | 低 | 第 12 幕（只用于对照训练） |
| 17 | NCCL EP（2603.13606）、GIN（2511.15076）、UCCL-EP（2512.19849）、TD-Pipe（2506.10470）、Seesaw（2503.06433） | — | — | 只读摘要 | 低 | 对比，不上画面 |
| 18 | Mooncake EP 设计文档、NIXL EP README、DeepEP hybrid-ep 分支说明 | — | Mooncake `dcddb56`、nixl `60dd8b7` | 已读 | 低 | 对比 |

## 阅读要点

### Dispatch / combine 的数据量

- 每个 token 每发一份：FP8 7168 B + 每 128 个数一个 FP32 scale（224 B）= 7392 B；combine 用 BF16，14336 B，约为 dispatch 的 2 倍。DeepEP 文档要求 combine 的输入为 BF16。
- top-8 不去重时，一个 token 一层 dispatch 约 57 KB、combine 约 115 KB。DeepEP low-latency 的表可以反算出它正是按（token, expert）每对发一份：128 token × 8 × 7392 B ≈ 7.6 MB，与 39 GB/s × 192 µs 相符。
- 链路：H800 NVLink 标称每方向 200 GB/s，实测约 160 GB/s；每卡一张 CX7，400 Gb/s ≈ 50 GB/s（DeepSeek-V3 报告：“160 GB/s, roughly 3.2 times that of IB (50 GB/s)”）。H800 把 NVLink 从 H100 的 900 GB/s（双向合计）砍到 400 GB/s（ISCA '25）。
- 通用 NCCL 不合适：NCCL 的 all-to-all 是一组 `ncclSend`/`ncclRecv`，每对的数据量要在 host 上给出；2.28.3 才有 `ncclAlltoAll`，而且每个 peer 数量相同。MoE 每层的数量由 router 决定，要先把计数同步回 CPU，破坏 CUDA graph（TensorRT-LLM Part 1 的原话）。

### DeepEP V1：normal 与 low-latency（2025 年 DeepSeek 线上的做法）

- **normal（高吞吐，训练与 prefill）**：配合 node-limited routing（每个 token 最多去 4 个节点）。token 先经 IB 发到目标节点上**同一编号**的 GPU，再经 NVLink 转发给持有 expert 的 GPU；同一节点的多个 expert 只走一次 IB。NVLink 是 IB 的 3.2 倍，所以每个节点平均 3.2 个 expert 的转发不增加耗时，最多可选 13 个 expert（4 × 3.2）而通信代价不变。kernel 内按 warp 分工：IB 发送、IB→NVLink 转发、NVLink 接收；报告写 20 个 SM 足够打满 IB 与 NVLink，profile 里通信 kernel 的 grid 为 24，计算 GEMM 的 grid 为 108（= 132 − 24）。
  - 实测（H800，4096 token/batch）：节点内 EP8 dispatch 153 GB/s、combine 158 GB/s（NVLink）；跨节点 EP16/32/64 为 43–58 GB/s（RDMA，PR #130 之后）。
  - 代价：发之前要先交换计数（`get_dispatch_layout` → `notify_dispatch`），接收方要知道收多少才能分配 tensor，CPU 要等 GPU 的计数信号（DeepEP README：“an implicit CPU wait for GPU received count signal”），形状每步变化，不能进 CUDA graph。`figures/normal.png` 画的就是 CPU 的 notify → 等待 → 分配 → launch。
- **low-latency（decode）**：纯 RDMA（IBGDA），每个 token 直接发到每个目标 expert 所在的 GPU，不经转发；接收 buffer 预先按 `[本地 expert 数, num_max_dispatch_tokens_per_rank × 卡数, 7168]` 分配，计数留在 GPU 上，形状固定，可以进 CUDA graph；建议每卡 token 数 < 256。`return_recv_hook=True` 时 kernel 只发出 RDMA 请求就返回，SM 全部让出，之后调用 hook 才接收。
  - 实测（H800 + CX7，128 token/batch，top-8）：EP128 dispatch 192 µs、combine 369 µs；EP8 在 PR #173 让 low-latency 尽量走 NVLink 之后为 77 / 114 µs。
- SGLang 的 `--deepep-mode auto`：batch 里有 prefill（extend）就用 normal，否则 low_latency（`layers/moe/utils.py` 的 `DeepEPMode.resolve`）；normal 模式会关掉 CUDA graph。两种模式在同一个通信组里难以混用，这是 MoE 大规模部署要做 PD 分离的原因之一（SGLang 2025-05 博客）。

### 两个 micro-batch 的重叠（真实 trace，`data.py` 提取）

- prefill（EP32，每卡 16K token 分两个 micro-batch）：通信在单独的 stream（24 个 SM），计算在另一条（108 个 SM）。每层每个 micro-batch：dispatch 中位数 4.56 ms、combine 8.62 ms、notify 约 0.9 ms；attention 约 3.0 ms，routed expert 的两个 GEMM 约 3.8 + 2.6 ms。一层（两个 micro-batch）约 34 ms（相邻两层同一 micro-batch 的 dispatch 间隔，中位数 34.0 ms）；每层计算 stream 的 kernel 时长之和 31.9 ms、通信 stream 30.8 ms，即忙 92% 与 89%，不重叠时约 63 ms。
- decode（EP128，每卡 128 个请求）：只有一条 stream，132 个 SM 全用于计算。`dispatch_ll` / `combine_ll` 是很短的发出与接收 kernel（发出约 17–22 µs，接收约 17–76 µs）；数据在另一个 micro-batch 的 attention（FlashMLA 约 243 µs）期间在网络上传。一层两个 micro-batch 约 1.42 ms，stream 忙 96.8%。DeepSeek 称之为 5 级流水（attention 拆成两段）。
- 框架里的名字：SGLang 叫 two-batch overlap（TBO，`batch_overlap/`），decode 时第二个 micro-batch 落后两个 stage；vLLM 叫 DBO（`--enable-dbo`，固定 2 个 micro-batch，decode 阈值 32 token）。代价：batch 拆半，GEMM 变小。SGLang 实测 decode 每卡 32 token 时 −27%，256 token 时 +25.5%；prefill 同 token 数下 +27–35%。
- SBO（single-batch overlap，Ant Group 与 SGLang）：不拆 batch，down GEMM 每算完 64 个 token 就原子加一个信号，combine 轮询信号、算完一块发一块；H20 上 batch 20–56 时 +8–10%。GB200 上 SGLang 改用这种细粒度重叠而不是 TBO。

### 负载不均与 EPLB

- 所有卡每层同步，一步的时间由最忙的卡决定。SGLang 的 expert 分布热图（DeepSeek-V3，prefill 与 decode）显示少数 expert 明显更热。
- DeepSeek 的做法：统计每个 expert 的负载，给最热的 expert 复制副本，放进冗余位置，再均衡地装到各卡。prefill 用 32 个冗余 expert：256 + 32 = 288 个位置，EP32 下每卡 9 个；约每 10 分钟按统计重排一次（V3 报告）。`eplb.py` 有两种策略：expert group 能整除节点数时分层（先按 group 装节点，节点内复制、再装卡），否则全局。
- 效果：SGLang 在 96 × H100 上 EPLB 让 prefill 吞吐 1.49 倍、decode 2.54 倍；均衡度定义为各卡平均负载 / 最大负载。vLLM 默认每 3000 步重排一次、异步搬权重。
- 反对意见：TensorRT-LLM 的 DWDP 认为逐层同步本身是问题（各卡序列长度波动 20% 时同步开销约 10%），改为 token 不动、按需拉取 expert 权重。

### PP

- 按层切成 stage，每步每个 stage 边界只传一次 hidden state：每 token 7168 × 2 B ≈ 14 KB，点对点只发给下一个 stage。对比 EP：每个 MoE 层两次 all-to-all，每 token 每层上限约 172 KB，58 层约 10 MB。SGLang 博客的估算：PP 总通信量 B·S·H·(P−1)，TP 约 4·B·S·H·L，“nearly order-of-magnitude reduction”。
- 气泡：只有一个 batch 时，任一时刻只有一个 stage 在算。推理没有 backward，不同 batch 互相独立，同时放 P 个 batch 就能填满：vLLM 源码 `max_concurrent_batches` 的注释 “PP requires PP-size concurrent batches to fill the pipeline”，`EngineCore` 的 batch queue 用来“eliminate pipeline bubbles”。代价：每个 batch 只有 1/P 的请求，GEMM 变小；单个 token 仍要串行走完 P 个 stage 和 P−1 次传输，延迟不降。
- 层的划分：vLLM `get_pp_indices` 把多出的层分给非最后的 stage（最后一个还要做 norm 与 lm_head），可用 `VLLM_PP_LAYER_PARTITION` 覆盖；SGLang 博客建议 DeepSeek-V3.1 用 `15,15,15,16`。
- 推理里的用处：
  - 装不下、又没有快速互联：vLLM 文档建议 TP = 每节点卡数，PP = 节点数；没有 NVLink 的卡（L40S）用 PP 代替 TP。
  - 超长 prompt 的 prefill：chunked PP 把 prompt 切成 chunk，在各 stage 间流水。SGLang 在 H20 上：DeepSeek-V3.1、128K 输入，TTFT 从 TP8 的 48.5 s 降到 PP4 TP8 的 15.5 s；prefill 吞吐 PP4 TP8 为 TP8 的 3.31 倍，TP32 为 2.54 倍。SGLang 不建议纯高阶 PP，节点内仍用 TP 或 CP。
  - 大 MoE 的 decode 基本不用 PP：DeepSeek、SGLang、Kimi K2 的部署都是 DP attention + EP；SGLang 把 decode 的 PP 列在 2026 H1 路线图上。DeepSeek-V3 训练用 PP16 + EP64（DualPipe），推理不用 PP。

## 现状变化（以源码与 release 为准）

- **DeepEP**：2026-04 的 V2 把 normal 与 low-latency 合并成同一个 `dispatch / combine` 接口，改用 NCCL GIN 后端，按流量解析式地选 SM 数，宣称比 V1 峰值高至多 1.3 倍、SM 少至多 4 倍；仍保留 hybrid（RDMA + NVLink 分层，相当于 V1 normal）与 direct（点对点，相当于 V1 low-latency）两种拓扑。不再支持 0 SM 的 RDMA EP（README：“zero-SM RDMA EP is not supported”），重叠改用 `defer_epilogue`。2026-09 的 V2.5 完全删除 V1 与 NVSHMEM 依赖，并加入实验性的 `PPBuffer`（相邻 PP rank 间 RDMA 收发）与动态冗余 expert 的接口。GitHub release 只有 v1.2.1。
- **vLLM**：`--all2all-backend` 默认 `allgather_reducescatter`；`pplx`（2026-02，#33724）与 `naive`（2026-04，#33728）已删除；新增 `deepep_v2`、`nixl_ep`、`moonep`、`flashinfer_nvlink_one_sided/two_sided`。环境变量 `VLLM_ALL2ALL_BACKEND` 已去掉。EPLB 默认异步。
- **SGLang**：`--moe-a2a-backend` 新增 `deepep_v2`（`--deepep-v2-mode direct|hybrid`，不能与 TBO/SBO 同开）与 `megamoe`（dispatch、GEMM、combine 融合）；TBO 代码移到 `batch_overlap/`。
- **硬件**：GB200 NVL72 把 72 张卡放进一个 NVLink 域，每卡每方向 900 GB/s；SGLang 在其上 EP 只走 NVLink，可以关掉 TBO。DeepSeek ISCA '25 的理论估算：每层两次 all-to-all，IB 50 GB/s 时 decode 每 token 至少 14.76 ms，NVL72 上 0.82 ms。
- **NCCL**：2.28 起有 device API（LSA / Multimem / GIN），2.30 起 `ncclPutSignal`/`ncclWaitSignal` 可进 CUDA graph；NVIDIA 的 NCCL EP（2026-03）把 LL / HT 两种模式做进 NCCL。

## 相关系统对比

| 系统 | 与 DeepEP V1 的不同 |
| --- | --- |
| pplx-kernels（已弃用）→ pplx-garden | NVSHMEM 只用三个原语，发送与接收拆成两个 kernel；后续改为 GPU kernel + CPU 代理驱动网卡，同一套代码跑 CX7 与 AWS EFA |
| TensorRT-LLM NVLinkOneSided | 只走 NVLink（GB200 NVL72）；对称内存本身就是接收 buffer，按 rank 排布，每个 token 只发一次；batch 1 时 dispatch 18.9 µs |
| Hybrid-EP（NVIDIA，DeepEP 分支） | NVLink 用 TMA、RDMA 用 IBGDA，少占 SM，支持更大的 NVLink 域与 PCIe |
| Mooncake EP / NIXL EP | 沿用 low-latency 的编程模型，分别加容错（超时、`active_ranks`）与运行时增减 rank |
| UCCL-EP | GPU 只发紧凑命令，CPU 代理驱动 EFA、AMD 等网卡 |
| DWDP（TensorRT-LLM） | 反对逐层同步：token 不动，拉 expert 权重 |

## 分歧与待核实

- DeepSeek 的 decode 配置三处不一致：V3 报告为 40 节点 EP320（attention TP4），Day 6 为 18 节点 EP144（DP attention），profile-data 为 EP128。prefill 的 attention：报告 TP4+SP 与 DP8，Day 6 为 DP32，profile 为 TP1。本片画面用 profile-data 的配置（EP32 prefill、EP128 decode），因为 trace 来自它。
- profile-data 的路由是模拟的完全均衡（README 说明），trace 里看不到负载不均；负载不均一幕用 SGLang 的实测热图。
- decode trace 中各 kernel 归到哪个 stage，是按 GEMM 形状推断的（2112 × 7168 为 q/kv down，24576 × 1536 为 q up，7168 × 16384 为 o_proj，4096 × 7168 / 7168 × 2048 为 expert）。
- SGLang 96 × H100：prefill（4 节点）与 decode（9 节点）是分开测的，不能说成“12 节点上 4P + 9D”。
- SGLang 博客与代码里 `ep_size == tp_size` 的约束、TBO 的阈值随版本变化，画面不出现具体 flag。

## 实测

草稿期不测。现有的实机材料：DeepSeek 的 H800 trace（profile-data）、DeepEP 官方实测表、SGLang 与 vLLM 的公开数字。可选的补充见分镜表末尾的占位，跑之前逐项征求同意。

## 确认范围

- 2026-10-20 用户确认：做成一支片子（约 5.6 分钟，14 幕），slug `ep-pp-communication`；先做不需要实测的部分，中途不再请示。
- M1（DeepEP V2 双机实测）、M2（真实路由统计）都不跑；第 9 幕的负载柱按 SGLang 热图的偏斜构造并标“示意”。
- 配音这次不合成：文案写进 `script.js`，字幕按估计时长排；`vo.py` 备好，合成后生成 `vo.js`，字幕与音轨自动改用实测时长。
