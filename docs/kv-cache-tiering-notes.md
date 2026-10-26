# KV cache 分层存储与传输：调研笔记

视频先行，文章以后写。slug `kv-cache-tiering`。状态（2026-10-20）：方案已确认，10 幕无声画面与字幕完成，12 fps 预览待审；配音未合成，实测未做（占位见 `videos/kv-cache-tiering/storyboard.md`）。

## 所读版本

| 仓库 | commit | 日期 |
| --- | --- | --- |
| vLLM | `fce64bc5c2`（部分阅读在 `c41b2639e2`，两者相差 10 天，所引代码一致） | 2026-10-20 |
| SGLang | `6fc8d9da`（`78185f4b` 复核 Mooncake 部分） | 2026-10-10 |
| Mooncake | `dcddb56c0c` | 2026-10-08 |
| LMCache | `688f735581` | 2026-09-30 |
| NIXL | `60dd8b7cd0`（v1.6.0） | 2026-10-20 |
| Dynamo | `a83f4796b6` | 2026-10-08 |

论文读 arXiv 源文件（Mooncake 2407.00079、LMCache 2510.09665、CachedAttention 2403.19708、CacheGen 2310.07240）；Mooncake FAST'25 版只有 PDF（仓库内 `FAST25-release/Mooncake-FAST25.pdf`），用 PyMuPDF 提取文字。CacheBlend 只读了摘要。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [Mooncake: Trading More Storage for Less Computation](https://www.usenix.org/conference/fast25/presentation/qin)（FAST'25）与 [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) | Qin et al.，Moonshot AI / 清华 | 已读（FAST §2–5；arXiv 全文） | 核心 | 第 3、5、7、8 幕：存算权衡公式、layer-wise prefill、全局池、Transfer Engine 带宽 |
| 2 | Mooncake 源码与文档（Transfer Engine、TENT、Mooncake Store、Conductor、性能文档） | kvcache-ai | 已读 | 核心 | 第 7、8 幕：vLLM PD 实测表、Store 淘汰与 SSD、Conductor 现状 |
| 3 | vLLM KV connector V1、`OffloadingConnector`、`NixlConnector` 源码与 `kv_offloading_usage.md`、`nixl_connector_usage.md` | vLLM | 已读 | 核心 | 第 4、5、8、9 幕：写入与读回时机、PD 拉取流程、现状 |
| 4 | [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665) 与源码 | Liu et al.，TensorMesh / UChicago | 已读 | 核心 | 第 3、6 幕：消息大小与吞吐、chunk 256、加载与重算的交叉点（B200） |
| 5 | [CachedAttention / AttentionStore](https://arxiv.org/abs/2403.19708)（ATC'24） | Gao et al.，华为 | 已读（设计、评测） | 重要 | 第 1、5 幕：多轮对话动机、layer-wise 预加载、调度感知的预取 |
| 6 | SGLang HiCache 源码与文档、[LMSYS HiCache blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/) | SGLang | 已读 | 重要 | 第 4、5、6 幕：写入策略、逐层加载事件、host 布局 |
| 7 | NIXL README 与插件列表 | NVIDIA | 已读 | 重要 | 第 8 幕：一侧 READ/WRITE、后端按内存类型选择 |
| 8 | Dynamo KV router 设计文档、KVBM 文档与 `offload/filter.rs` | NVIDIA | 已读 | 重要 | 第 4、7 幕：分层命中代价函数、SSD 写入过滤、KVBM 已弃用 |
| 9 | [CacheGen](https://arxiv.org/abs/2310.07240) | Liu et al. | 已读（摘要、评测） | 参考 | 第 10 幕边界：压缩换带宽 |
| 10 | NVIDIA 官方规格：H100、NVLink、GB200 NVL72、DGX H100 用户指南；PCI-SIG 5.0；Micron 9550 | 厂商 | 已读 | 参考 | 第 2、3 幕：各层容量与带宽 |
| 11 | llm-d tiered prefix cache 指南、VAST、NetApp 的 offload 对比 | 第三方 | 只看到搜索摘要，原页 404 或未逐页核对 | 参考（不上画面） | 反对意见的线索 |

## 阅读要点

### 一次传输的量级（全片的运行例子）

- Llama-3.3-70B：80 层、8 个 KV head、head dim 128，BF16。每个 token 的 KV 为 80 × 8 × 128 × 2 × 2 B = 320 KiB；Mooncake FAST §5.3.1 同为 320 KB/token。32K token 的会话为 10 GiB。TP = 8 时每张卡存 1 个 KV head，即 1.25 GiB。
- 一台 DGX H100：8 × 80 GB HBM3（3.35 TB/s）、2 TB DDR5、8 × 3.84 TB U.2 NVMe、8 × ConnectX-7（每块 400 Gb/s）。PCIe Gen5 x16 单向约 64 GB/s（官网写 128 GB/s 是双向）；NVLink 4 每卡 900 GB/s 也是双向合计。
- 容量（自行估算）：HBM 按 0.9 利用率扣除 131 GiB 权重与约 20 GiB activation，约 425 GiB，放 42 个 10 GiB 会话；DRAM 留 1.5 TiB 给 KV，约 150 个；NVMe 约 28 TiB，约 2,800 个。
- 搬 10 GiB（整机 8 卡并行，自行估算）：HBM 内 0.4 ms；DRAM → HBM 按每卡 PCIe 50 GB/s 计约 27 ms；远端 DRAM 经 RDMA 按 Mooncake 实测的 190 GB/s（8 × 400 Gb/s）约 57 ms；本地 NVMe 按 25 GB/s 约 0.43 s（Mooncake SSD 测试中 5 块 NVMe RAID0 约 27 GB/s）；S3 Express 约 1 GB/s（LMCache §8）约 11 s。
- 重算（自行估算）：Mooncake 的 prefill 计算量 flops(n) = l(4n²d + 22nd²)，l = 80、d = 8192、n = 32K，得 6.7 PFLOP；8 × H100 BF16 稠密峰值 989 TFLOPS、按 50% 利用率，约 1.7 s。交叉点带宽 10 GiB / 1.7 s ≈ 6.3 GB/s。Mooncake 论文给的同类结论：LLaMA3-70B、8K 前缀，8 × A800 需 6 GB/s，8 × H800 需 19 GB/s。

### 写入时机（offload）

- vLLM `OffloadingConnector`：每一步在 `build_connector_meta` 里把新写满的 chunk 全部存到 CPU（write-through），默认只存 prompt（`offload_prompt_only`）；拷贝在下一步的 side stream 上执行。文档提示 CPU 层要大于 GPU 的 KV cache，否则只是镜像。`SimpleCPUOffloadConnector` 的 `lazy_offload` 只存即将被淘汰的 block。
- SGLang HiCache 默认 `write_through`；`write_through_selective` 在第二次命中时才写；`write_back` 在被 GPU 淘汰时写。
- LMCache MP 模式的 lazy offload（`EVICTION_AWARE`）只存靠近 vLLM 淘汰端的 block；论文的 dynamic offloading 是同一思路。
- SSD 寿命：Dynamo `offload.rs:763` 警告不过滤会“accelerated SSD degradation”，`FrequencyFilter` 只把多次使用的 block 写到磁盘。Dynamo 文档：CPU 层小于 GPU KV cache 时每次前向都在搬（churn）。

### 读回时机（reload）

- vLLM：GPU prefix cache 未命中的部分交给 connector 查 CPU 层；命中则请求进入 `WAITING_FOR_REMOTE_KVS`，加载完成后才被调度（`scheduler.py:1386`）。
- 排队期间预取：CachedAttention 按调度队列的前瞻窗口把 SSD 中的 KV 提前搬到 DRAM，99.6% 的命中由 DRAM 提供；SGLang 的 L3 预取（`timeout` 策略：2 s + 每 1024 token 0.1 s，上限 30 s）；LMCache 的异步预取。
- 逐层加载：SGLang 每层记录一个 CUDA event，attention 计算第 N 层前 `layer_transfer_counter.wait_until(layer_id)`（`memory_pool.py:1684`）；CachedAttention 的 layer-wise pre-loading；LMCache `use_layerwise`（默认关闭）。重叠后耗时约为 max(加载, 计算)；新 token 很少时计算太短，加载仍露在外面（CachedAttention 的 buffer 公式）。

### 粒度与布局

- 一个 vLLM block（16 token）在一层、一张卡上的 KV 只有几 KB 到几十 KB。LMCache 论文引用的 RCCL 数据：64 KB 消息约 4 GB/s，1 MB 约 30 GB/s，16 MB 以上饱和在 49 GB/s。LMCache 因此按 256 token 的 chunk 跨层收集到连续 buffer 再传。
- SGLang host 布局（`pool_host/mha.py:304`）：`layer_first` 为 `(2, L, tokens, H, D)`，与 GPU 相同，一页散在 L 段；`page_first` 为 `(2, tokens, L, H, D)`，一页连续，可整块写入 L3；`page_first_direct` 为 `(2, pages, L, page_size, H, D)`。NixlConnector 对非 MLA 模型强制 `LBHNC` 布局以提高传输效率。

### 跨实例复用

- SGLang：L1、L2 属于单个实例，只有 L3 跨实例共享（同一台机器上的两个实例也不共享 L2）。
- Mooncake Store：master 管元数据，client 之间直接传数据；近似 LRU，90% 水位触发、淘汰 5%；soft pin 用于 system prompt。全局池对比本地缓存命中率最高 2.36 倍（FAST §5.3.2）；vLLM + Mooncake Store 在 GB200 上命中率从 1.7% 到 92.2%（Mooncake 性能文档）。
- Mooncake trace：缓存 5000 万 token 才接近理论最高命中率，至少要 20 个节点的 DRAM（FAST）；按 70B 的 320 KiB/token 约 15 TiB。
- 路由：Dynamo 的代价 = prefill_blocks − device_overlap − 0.75·host_overlap − 0.25·disk_overlap（默认权重），加上 decode 负载。Mooncake 开源的 Conductor 现在只是 KV 索引（订阅 vLLM 的 `BlockStored` / `BlockRemoved` 事件，回答最长匹配），路由由外部 router 决定。

### PD 分离的搬运

- vLLM `NixlConnector` 默认为 pull：P 做完整个 prefill，`request_finished` 返回 `delay_free=True` 与 `remote_block_ids` 等参数，block 保留（租期默认 30 s）；D 在 `start_load_kv` 中先经 ZMQ side channel 握手，再对每个远端 rank 发起 NIXL `READ`；完成后通知 P 释放。另有 `NixlPushConnector`（P 端 WRITE）。
- NIXL：一侧 READ/WRITE、按内存类型选后端（UCX 默认；GDS、Mooncake、libfabric、S3 等插件）；GPUDirect RDMA 下数据从一张卡的 HBM 经网卡直达另一张卡的 HBM。
- 实测（Mooncake 文档，H800、8 × CX-7 RoCE、Qwen3-8B、1P1D）：32K token 的 KV 4.50 GB，传输 31.65 ms、142 GB/s，占 TTFT 749.6 ms 的 4.2%。
- NVLink：GB200 NVL72 每卡 NVLink 5 单向约 900 GB/s，10 GiB 约 12 ms；SGLang 用 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` 走 MNNVL。

## 现状变化

- **逐层流式传输（PD）：论文有，主流 connector 现在不做。** Mooncake 论文与 Splitwise 都是每算完一层就传这一层；当前 vLLM 的 `NixlConnector`、`MooncakeConnector` 的 `save_kv_layer` 为空，docstring 写明 “does not do layerwise saving”，在 prefill 结束后整块传；SGLang 每个 prefill chunk 结束后传一次（`send_kv_chunk`），按层的 future 只在 NVLINK / BAREX 内存池下使用。connector API 保留了 `wait_for_layer_load` / `save_kv_layer`，LMCache 的 layerwise 模式（默认关闭）在用。我的分析：RDMA 下传输只占 TTFT 的几个百分点，逐层传输的消息小、效率低，chunk 级重叠已经够用。
- **Mooncake 的 KV 进的是 D 的 HBM，不是 DRAM。** 论文写 KV 流向 decode 节点的 CPU DRAM；vLLM 的 MooncakeConnector 直接 GPU → GPU 写入。
- **Mooncake Conductor**：论文中的全局调度器（估算 TTFT、热点迁移、提前拒绝）在开源代码中是 KV 索引服务。
- **Mooncake Transfer Engine**：分片从论文的 16 KB 变为默认 64 KB；TENT 按遥测动态分片，替代静态条带。
- **Dynamo KVBM 在 v1.5.0（2026-09-18）弃用**，计划 v1.6.0 删除，官方建议改用引擎自带的 offload（vLLM `OffloadingConnector`、SGLang HiCache）。只作设计参考，不作推荐。
- **vLLM V0 的传输栈已删除**（`kv_pipe`、`kv_lookup_buffer`、PyNcclConnector、P2pNcclConnector 均不在 factory 中），但 `docs/features/disagg_prefill.md` 的 Development 一节仍描述 V0 的抽象，不引用。V1 没有 swap 式抢占，CPU 复用改由 `OffloadingConnector` 提供；`--cpu-offload-gb` 是权重 offload，与 KV 无关。
- **vLLM offload 多层化**：`TieringOffloadingSpec` 在 CPU 之下接 `fs`、`obj`（S3）、`p2p`（经 NIXL 的实例间 RDMA）层，CPU 不再是最后一层；只有 CPU 层直接与 GPU 交换。
- **LMCache**：默认改为独立进程（`LMCacheMPConnector`，`--kv-offloading-backend lmcache` 即指向它）；PD 配置键从 `nixl_*` 改为 `pd_*`；layerwise 与异步加载默认关闭。
- **SGLang HiCache**：`hiradix_cache.py` 已并入 `UnifiedRadixCache`；默认布局 `page_first`、io 后端 `kernel`；新增 `page_first_kv_split`、`page_head` 布局和 `--hicache-host-memory-mode buffer_only`。文档说 page_first + direct 会退回 layer_first，代码实际改为 page_first_direct，以代码为准。

## 相关系统对比与反对意见

- **分层由谁管**：vLLM 与 SGLang 把 offload 做进引擎；LMCache、Mooncake Store 做成独立的缓存层；Dynamo 曾做 KVBM，现已退回引擎自带。
- **写入策略**：vLLM 默认立即写（write-through），SGLang 默认 write_through 并提供 selective 与 write_back，LMCache MP 与 vLLM 的 Simple connector 提供 lazy。立即写让淘汰不需要等拷贝，代价是占用 PCIe 与 DRAM。
- **重算可能更快**：LMCache 论文 §9.6（B200）：远端链路 32 Gbps 时，上下文超过 256K token 加载才比重算快；64 / 128 Gbps 时总是加载快；作者认为应当动态选择。CachedAttention：历史很长、新输入很短时，逐层加载无法被计算掩盖。
- **缓存放得下 HBM 时，offload 是净开销**（llm-d 指南的对比，页面已 404，未核实，不上画面）。
- **pinned memory 有代价**：SGLang 用 `--hicache-host-memory-fraction` 限制 host 池；大块 `cudaHostRegister` 会让进程退出卡住几十秒（`pool_host/base.py:274`）。
- **低带宽下收益消失**：Mooncake 的带宽扫描中，总带宽低于 100 Gbps 时 TTFT 急剧上升。
- **压缩换带宽是有损的**：CacheGen 比量化后的 KV 再小 3.5–4.3 倍，LMCache 的默认 serde 仍是不压缩。
- **基准偏差**：LMCache 论文对比的是 vLLM 0.10/0.11 的原生 offload，之后已改进；Mooncake 论文的模型是假的 LLaMA-70B 加回放 trace。

## 画面素材候选

- Mooncake arXiv 源文件图（矢量 PDF）：`architecturev2.pdf`、`overlap_latency.pdf`、`cache_unit_v2.pdf`；FAST 版 Fig. 4（拓扑矩阵）、Fig. 12（Transfer Engine 与 TCP / Gloo）、Fig. 13（带宽扫描）。
- Mooncake 文档：vLLM PD 实测表（`docs/source/performance/vllm/vllm-v1-pd-performance.md:27-37`）、`ssd_offload_per_turn.png`。
- LMCache 论文：消息大小与吞吐表、`fig_diff_lengths.pdf`（加载与重算的交叉点）。
- SGLang：`pool_host/mha.py` 的布局维度代码；`layer_transfer_counter.wait_until`；LMSYS blog 的 `hicache_layout.png`。
- vLLM：`--kv-offloading-size` 命令、`OffloadingConnector` 配置、`NixlConnector` 启动命令与 “does not do layerwise saving” 原文、NIXL 握手日志格式。
- Dynamo：router 代价函数、`FrequencyFilter` 注释。

取用时记入 `src/assets/kv-cache-tiering/SOURCES.md`。

## 待核实

- 各层带宽的实际可达值（PCIe 拷贝效率、NVMe RAID 读带宽）采用估算或他人实测；可选的 4090 实测见分镜表。
- 1.7 s 的重算时间按 50% 利用率估算，未实测。
