# kv-cache-tiering 画面素材出处

视频 `videos/kv-cache-tiering/` 目前没有使用原图文件，画面全部自绘；下表列出画面上直接引用的代码、表格与数字的来源。版本与阅读要点见 `docs/kv-cache-tiering-notes.md`。

| 幕 | 画面上的内容 | 来源 |
| --- | --- | --- |
| 2 | DGX H100 配置（8 × 80 GB HBM3、2 TB 内存、8 × 3.84 TB NVMe、8 × ConnectX-7） | NVIDIA DGX H100 用户指南 |
| 2 | 缓存 5000 万 token 才接近最高命中率 | Mooncake，FAST'25 |
| 3 | 远端内存经 RDMA 190 GB/s（8 × 400 Gb/s） | Mooncake，FAST'25 §5.4.1 |
| 3 | 交叉点公式与 A800 6 GB/s、H800 19 GB/s | Mooncake，FAST'25 §2.2 |
| 3 | S3 Express 约 1 GB/s | LMCache 论文 §8 |
| 4 | `vllm serve … --kv-offloading-size 1536 --kv-offloading-backend native` | vLLM `fce64bc5c2`，`vllm/config/cache.py` 的参数；命令为按本片例子改写 |
| 4 | 内存层要大于显存；SSD 写入过滤 | vLLM `docs/features/kv_offloading_usage.md`；Dynamo `lib/llm/src/block_manager/offload.rs:763`、`offload/filter.rs` |
| 5 | `request.status = RequestStatus.WAITING_FOR_REMOTE_KVS` | vLLM `fce64bc5c2`，`vllm/v1/core/sched/scheduler.py:1386` |
| 5 | `layer_transfer_counter.wait_until(layer_id)` | SGLang `6fc8d9da`，`python/sglang/srt/mem_cache/memory_pool.py:1684` |
| 6 | 消息大小与吞吐（64 KB 4 GB/s、1 MB 30 GB/s、≥16 MB 49 GB/s） | LMCache 论文 §3.1 引用的 RCCL 数据 |
| 6 | `(2, L, tokens, H, D)` 与 `(2, tokens, L, H, D)` | SGLang `6fc8d9da`，`python/sglang/srt/mem_cache/pool_host/mha.py:304` |
| 7 | 路由代价 `prefill_blocks − device_overlap − 0.75·host_overlap − 0.25·disk_overlap` | Dynamo `a83f4796b6`，router 设计文档 |
| 7 | 全局池命中率最高 2.36 倍 | Mooncake，FAST'25 §5.3.2 |
| 8 | Qwen3-8B、32K token、4.50 GB、31.65 ms、142 GB/s、TTFT 的 4.2% | Mooncake `dcddb56c0c`，`docs/source/performance/vllm/vllm-v1-pd-performance.md:27-37` |
| 9 | `"""NixlConnector does not do layerwise saving."""` | vLLM `fce64bc5c2`，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:300` |
| 10 | 32 Gbps 下 256K token 以上读回才比重算快 | LMCache 论文 §9.6 |
