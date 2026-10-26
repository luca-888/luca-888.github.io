# EP 通信与 PP：素材来源

原图原样使用并在画面上注明出处；自绘图与 trace 时间线的数据来源也记在这里。

| 文件 | 内容 | 来源 | 许可 / 说明 |
| --- | --- | --- | --- |
| `deepep-normal.png` | normal kernel 的 CPU / GPU 时序：notify → 等待 → 分配 tensor → dispatch → 计算 → combine | [DeepEP `figures/normal.png`](https://github.com/deepseek-ai/DeepEP/blob/b5a33ee/figures/normal.png)，commit `b5a33ee` | MIT |
| `deepep-low-latency.png` | 两条 stream + 通信 SM 的重叠 vs 一条 stream + 后台 RDMA 的 hook 式重叠 | [DeepEP `figures/low-latency.png`](https://github.com/deepseek-ai/DeepEP/blob/b5a33ee/figures/low-latency.png) | MIT |
| `profile-decode.jpg` | decode 两个 micro-batch 的重叠：Computation 132 SMs / Communication 0 SMs | [profile-data `assets/decode.jpg`](https://github.com/deepseek-ai/profile-data/blob/4496024/assets/decode.jpg)，commit `4496024` | 仓库未附许可证，按引用注明出处 |
| `profile-prefill.jpg` | prefill 两个 micro-batch 的重叠：108 SMs 计算 / 24 SMs 通信 | [profile-data `assets/prefill.jpg`](https://github.com/deepseek-ai/profile-data/blob/4496024/assets/prefill.jpg) | 同上 |
| `eplb-example.png` | EPLB 示例：2 节点 × 4 GPU 上的 expert 副本摆放 | [EPLB `example.png`](https://github.com/deepseek-ai/EPLB/blob/d52c72d/example.png)，commit `d52c72d` | MIT；示例数据，备用 |
| `lmsys-eplb-stat.png` | DeepSeek-V3 各层 256 个 expert 的 token 分布热图（prefill / decode） | [SGLang 博客 2025-05-05](https://lmsys.org/blog/2025-05-05-large-scale-ep/)，`/images/blog/large_scale_ep/eplb-stat.png` | 引用，注明出处 |
| `lmsys-eplb-balancedness.png` | 节点数增加时均衡度下降、EPLB 的改善 | 同上，`eplb-balancedness.png` | 引用，备用 |
| `lmsys-eplb-throughput.png` | EPLB 对 prefill / decode 吞吐的影响 | 同上，`eplb-throughput.png` | 引用，备用 |
| `lmsys-pp_bubbles_before.jpg` | chunked PP：固定 4K chunk 时各 PP rank 的时序与气泡 | [SGLang 博客 2026-01-15](https://lmsys.org/blog/2026-01-15-chunked-pipeline/)，`/images/blog/chunked_pipeline/pp_bubbles_before.jpg` | 引用，备用 |
| `lmsys-ds_ttft.png` | DeepSeek-V3.1 在 H20 上不同并行配置的 TTFT | 同上，`ds_ttft.png` | 引用，备用 |

## 数据

| 数据 | 来源 | 处理 |
| --- | --- | --- |
| `videos/ep-pp-communication/data.js` 中的 decode / prefill kernel 时间线 | profile-data `decode.json`、`prefill.json`（H800，PyTorch Profiler，均衡路由的模拟） | `videos/ep-pp-communication/data.py` 从 trace 中段各取一层，按 kernel 名分类 |
| DeepEP 带宽与延迟表 | DeepEP [v1.2.1 README](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/README.md) 与首个 commit `ebfe47e46` 的 README | 原样引用 |
| DeepSeek-V3 权重大小 689 GB | Hugging Face API `deepseek-ai/DeepSeek-V3` 的 `usedStorage`（2026-10-20 查询） | 原样引用 |
