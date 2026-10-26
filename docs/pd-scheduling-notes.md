# PD 分离的调度：调研笔记

## 范围与状态

- 2026-10-09 用户确认方向：不重复 `vllm-distributed` 第五节（为什么要分、vLLM 的实现、传输量），改写分离之后才出现的调度问题。主线为“一个请求在分离部署下要经过四次决策”：选 P、要不要远端 prefill、D 的准入、P:D 配比。
- 同日用户要求“先写文章，先不做试验，预留”：正文已写，实测一节为占位，不填数字。
- 自绘图：2026-10-09 用户说“可视化有点干”，改为用图形编码数量（缓存命中、排队、D 显存都画成格子），并新增两张数量图：
  - `PdPickP`：三个 P 上的预计 TTFT。演示数据：prompt 6000 token，每个未命中 token 0.1 ms，队列中每个请求 200 ms。
  - `PdRatio`：P、D 能力之比与 goodput。用 DistServe 的单卡数字（5.6 / 10 / 1.6 请求/秒）；2P+1D 是论文中的例子，1P+1D、2P+2D、合并 ×3 是按同样方法推算的。

## 源码版本

| 仓库 | commit | 日期 |
| --- | --- | --- |
| vLLM | `4611c2c3` | 2026-10-08 |
| SGLang（含 sgl-model-gateway） | `51686cf8` | 2026-10-09 |
| Dynamo | `a83f479` | 2026-10-08 |
| llm-d-inference-scheduler | `ce60ef7` | 2026-10-08（本次新浅克隆到 reference-repos，未加入 sync.sh） |
| Mooncake | `dcddb56` | 2026-10-08（只看了 README，论文按 arXiv 源文件） |

## 参考清单（按重要程度）

| 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| [Mooncake](https://arxiv.org/abs/2407.00079)，FAST 2025 | Qin et al.（Moonshot AI） | 已读 TeX：§KVCache-centric Scheduling、§Overload-oriented Scheduling、§Experiment | 核心 | 选 P（算法 1）、early rejection、负载反相振荡、3P+1D 与 2P+2D |
| vLLM 源码：`v1/core/sched/scheduler.py`、`kv_connector/v1/nixl/*`；文档 `disagg_prefill.md`、`nixl_connector_usage.md` | vLLM | 已读相关段落 | 核心 | D 侧准入、`WAITING_FOR_REMOTE_KVS`、KV lease、push/pull、双向传输、示例 proxy |
| SGLang 源码：`srt/disaggregation/{prefill,decode}.py`、`arg_groups/fields/disagg.py`；gateway `policies/cache_aware.rs`、`routers/http/pd_router.rs` | SGLang | 已读相关段落 | 核心 | 先预留后 prefill（bootstrap）、`num_reserved_decode_tokens`、cache-aware 路由、P/D 同时下发 |
| Dynamo 文档：router `routing-concepts.md`、`conditional-disaggregation.md`、planner `planner-design.md` | NVIDIA | 已读 | 核心 | 代价函数、conditional disagg 阈值、planner 两层循环 |
| llm-d-inference-scheduler `docs/disaggregation.md` | llm-d | 已读 | 核心 | 先选 D、`prefix-based-pd-decider` |
| [Beyond the Buzz](https://arxiv.org/abs/2506.05508) | Mitra et al.（NVIDIA） | 已读 TeX：abstract、design principles、disaggregation in practice | 重要 | 固定比例 3.5 / 0.5 与动态 rate matching（原图） |
| [DistServe](https://arxiv.org/abs/2401.09670)，OSDI 2024 | Zhong et al. | 已读 intro、evaluation 段落 | 重要 | 干扰的数字：1.6 vs 5.6 / 10 rps，2P1D 3.3 rps/GPU |
| [TaiChi](https://arxiv.org/abs/2508.01989) | Wang et al. | 已读 abstract | 补充 | 合与分取决于 SLO，最多 +77% goodput |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310)，OSDI 2024 | Agrawal et al. | 此前 scheduler 文章已读 | 补充 | chunked prefill 作为不拆实例的替代 |

## 阅读要点

### vLLM

- 示例 proxy（`examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`）对 P、D 各用 `itertools.cycle` 轮询；生产路由交给 llm-d、Dynamo 等外部组件。请求顺序：先发 P（`max_tokens=1`，`do_remote_decode`），拿到 `kv_transfer_params` 再发 D（`do_remote_prefill`）。
- D 侧：`get_num_new_matched_tokens` 返回远端 token 数与 `load_kv_async=True`；scheduler 先 `allocate_slots`，成功后请求进入 `WAITING_FOR_REMOTE_KVS`，不参与本步计算；分配失败则 `break`，留在 waiting 队列。源码注释：异步加载在传输期间占着 block、不可抢占，只有在“空闲 block 减去其他在途预留”放得下时才接纳，以免死锁和可预见的抢占。
- 传输完成后 `_update_waiting_for_remote_kv` 缓存这些 block，并把 `num_computed_tokens` 回退 1，重算最后一个 prompt token 以采样第一个输出 token。加载失败按 `kv_load_failure_policy`（默认 recompute）在本地重算。
- P 侧：prefill 结束后 KV block 保留，等 D 来读。`kv_lease_duration` 控制保留时长，D 排队期间发心跳续租；过期则释放，指标 `vllm:nixl_num_kv_expired_reqs`，文档称它是 autoscaler / lease 调参信号。
- NIXL connector 现有 pull（D 读 P，默认）与 push（D 先注册 block，P 写入 D）两种实现。
- 双向传输：多轮对话中，D 上已有前几轮的 KV，P 可以先从 D 拉回已有 KV 再只算新 token；`kv_recompute_threshold` 默认 64，远端 token 少于它就在 P 本地重算。需要有状态 proxy 记录上一轮 D 返回的 `kv_transfer_params`。
- 文档仍标 experimental，并写明 “Disaggregated prefill DOES NOT improve throughput”。

### SGLang

- P 侧生命周期：Bootstrap Queue（握手与 D 侧预分配）→ Waiting Queue（运行 prefill）→ Inflight Queue（等待传输完成）。即默认在 D 预留好 KV 之后 P 才开始算。
- `disaggregation_decode_allocation_policy`：`early`（默认，先在 D 预留）与 `prefill_complete`（KV 先留在 P，prefill 结束后再在 D 接纳，与 vLLM pull 相近）。
- D 侧准入：每个活跃请求按 `num_reserved_decode_tokens`（默认 512）预留后续 decode 的空间；必要时 retract 正在运行的请求，`disaggregation_decode_retraction_backup` 决定被撤回请求的 KV 备份到 CPU 还是直接 503。
- gateway：PD 模式同时选一对 (P, D)，`tokio::join!` 并发把请求发给两者，用 `bootstrap_room` 配对；P 默认 `cache_aware`，D 可设 `power_of_two`。
- `cache_aware`：每个 worker 维护一棵近似 radix tree，存请求原文字符而非 token；匹配率 > `cache_threshold`（默认 0.5）选匹配最长的 worker，否则选树最小的；当 (max − min) > 32 且 max > 1.1 × min 时判为不均衡，改为最短队列。
- 新功能：`enable_pd_role_switch` 运行时 P/D 角色切换（2026-10-09 提交，文档未列），`optimistic_prefill_attempts`。正文未展开。

### Dynamo

- KV router 代价：`cost = prefill_load_scale × max(0, active_prefill_blocks + incoming_prompt_blocks − overlap_credit_blocks) + active_decode_blocks + incoming_active_blocks (+ 可选的每请求项)`，选最小者。缓存命中通过 overlap credit 抵扣 prefill 成本。
- Conditional disaggregation（experimental）：先看所选 D 上的缓存，effective ISL = prompt 长度 − D 上已缓存的前缀。默认 `isl_bounding`：effective ISL < 2048 且 effective / raw < 0.7 时在 D 本地 prefill+decode；另有 `prefill_load`（P 忙时本地做）与 `isl_or_load`。需要 D 发布 KV events，SGLang 暂不支持。
- Planner：throughput-based（预测请求数、ISL、OSL，按 perf model 定 P、D 下限，周期长）与 load-based（按实时指标 ±1 调整，周期短）两层循环；吞吐循环给下限，负载循环在其上调整。

### llm-d

- EPP 的 `disagg-profile-handler` 先选 D，再由 PD decider 判断是否需要 P。`prefix-based-pd-decider`：D 上未缓存的后缀 ≥ `nonCachedTokens` 才走远端 prefill，示例配置为 8；`promptTokens` 以下全部本地。
- 执行由 D 旁的 sidecar 完成：sidecar 先向 P 发 `max_tokens=1` 的 prefill，再把带 `kv_transfer_params` 的请求交给本地 vLLM。

### Mooncake

- 算法 1：对每个 P 估计 TTFT = T_queue + T_prefill(prompt, prefix_len)；若最佳匹配远长于本地（比值超过 `kvcache_balancing_threshold`），改为算 T_transfer + T_queue + T_prefill(prompt, best_prefix_len)，并把热点 KV 复制过去（hot-spot migration）。D 按负载选。TTFT 或 TBT 预计超 SLO 直接 429。
- 过载：先只看 P 的负载接收请求，会在 prefill 完成后被 D 拒绝，prefill 白算。Early rejection 在请求到达时按 P、D 两池负载的较大者决定是否接收。但 D 的负载滞后于 P，导致 P、D 负载反相振荡（Figure load_example，20 台机器、20 分钟）。改为预测 prefill 完成后 D 的负载（system-level，假设每个请求 decode 时长相同）。
- 8P+8D、23,000 条真实请求、2 倍回放速度：拒绝数 baseline 4183，early rejection 3771，基于预测 3589。
- 公开数据集：vLLM 4 实例 vs Mooncake 3P+1D 与 2P+2D；3P+1D 在 ArXiv Summarization 与 L-Eval 上吞吐分别高 20% 与 40%；2P+2D ITL 更低但 TTFT 不达标，原因是 P、D 负载不平衡。论文称真实集群中 P、D 需求在一段时间内稳定，比例可以预设。
- 测试用 dummy model（LLaMA2-70B 结构），A800 节点，800 Gbps RDMA。

### Mitra et al.

- 结论：分离对 prefill 占比高的流量与大模型最有效；动态 rate matching 与弹性扩缩是达到 Pareto 前沿的关键。
- DeepSeek-R1，ISL 16384、OSL 2048，Blackwell FP4 仿真：固定 Ctx:Gen = 3.5 在宽松延迟下好、收紧后落到合并部署之下；固定 0.5 在宽松延迟下只有约 0.42 的归一化吞吐。动态比例全程在上方。
- 数据来自 NVIDIA 内部仿真器，不是实机。

## 分歧与取舍

- **先选 P 还是先选 D。** Mooncake、SGLang 按 P 的缓存选 P、D 按负载；llm-d、Dynamo conditional disagg 先选 D，再按 D 上的缓存决定要不要 P。后者对应多轮 / agent 负载：上一轮的 KV 留在 D 上。vLLM 的双向传输解决的是同一个问题的另一面（P 从 D 拉回）。
- **先预留还是先计算。** SGLang 默认先在 D 预留，代价是 D 的显存在 P 排队和计算期间被占着；vLLM pull 默认先算，代价是 P 的显存被占着（lease），D 迟迟无法接纳时 KV 过期，prefill 白算。
- **是否提升吞吐。** vLLM 文档称不提升；DistServe、Mooncake 报告在 SLO 约束下的 goodput 提升。口径不同：前者是单纯吞吐，后者是满足 SLO 的请求率。正文用 goodput 的口径并说明一次。

## 实验占位

- 计划：Modal 2 × H100，同一模型，对比合并部署（2 个实例、chunked prefill、DP 分发）与 1P1D（`NixlConnector`），扫 prompt 长度，看 ITL p99 与满足 SLO 的请求率。
- 草稿期不测，正文只留占位，不填数字。正式测试前按 `docs/gpu-experiments.md` 写完整计划与费用。

## 待核实

- vLLM push 模式是否已成为某些部署的默认（当前文档未说明默认值，正文只写 pull）。
- SGLang role switch 的稳定性与是否有官方文档；正文未写。
- Dynamo planner 在 SGLang 后端上的支持范围。
