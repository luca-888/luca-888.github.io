# FlashAttention 官方技术图

以下为 Tri Dao 提交的论文源文件中的原始 PNG，下载于 2026-09-18。共 3 幅图、4 个图像文件；Figure 3 的两个子图分别保留。没有截图、重绘、改色、裁剪或放大插值。

## 共同来源

- 作者：Tri Dao。
- 论文：*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*，arXiv:2307.08691v1，2023。
- 论文页面：https://arxiv.org/abs/2307.08691v1
- 正文：https://arxiv.org/html/2307.08691v1
- 原始 TeX 源文件包：https://arxiv.org/src/2307.08691v1
- 实际下载入口：https://arxiv.org/src/2307.08691 （下载时仅有 v1，与上面的固定版本对应）。
- 作者博客：https://tridao.me/blog/2023/flash2/
- 第一代算法原论文：https://arxiv.org/abs/2205.14135

## 图像清单

| 本地文件 | 原始文件路径 | 尺寸 | 论文位置 |
| --- | --- | --- | --- |
| `fa1-tiling-online-softmax.png` | `figs/flash_attention_diagram.png` | 5844 × 3018 | Figure 1，§2.3.1 |
| `fa2-thread-block-parallelism.png` | `figs/flashattention_fwd_bwd_parallel.png` | 2298 × 1268 | Figure 2，§3.2 |
| `fa1-warp-partitioning.png` | `figs/flash_partitioning.png` | 3710 × 2194 | Figure 3(a)，§3.3 |
| `fa2-warp-partitioning.png` | `figs/flash2_partitioning.png` | 3100 × 2186 | Figure 3(b)，§3.3 |

Figure 1 是 FA2 论文对第一代 FlashAttention 分块计算与 online softmax 的官方回顾图，**不是 FA1 论文中的 Figure 1**。它为了说明分块关系省略了 softmax 减去行最大值的步骤；正文应另外给出稳定形式，并在图注中保留这一说明。原图用 `A` 表示未归一化指数结果。

Figure 2 以 causal attention 展示 thread block 的分工：前向负责行块，反向负责列块。它表达逻辑工作划分，不是实测调度时间线。

Figure 3 的两张原图应作为一组展示，分别注明 FlashAttention 与 FlashAttention-2；蓝色表示各 warp 共同访问，橙色表示在 warp 间划分。请保留原图图例。

建议页面图注使用“来源：Tri Dao，FlashAttention-2，Figure 1 / Figure 2 / Figure 3”，链接至对应论文，避免标成本站自绘或实测结果。

## 权利信息

arXiv 页面标注的发布许可为 **arXiv.org perpetual non-exclusive license**：https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html 。本目录保留原作者署名与来源；这些论文图像不因存入本仓库而转换为本站代码许可，也不应标为 CC BY 或公共领域素材。

## 文件校验

原始 PNG 与本地副本逐字节一致。SHA-256：

```text
2e9aa7f189189139a18092915b8b354aaff0586ecf5f7efad4db21edf4425c7e  fa1-tiling-online-softmax.png
3a46bd503b19419b8cd8954e6fe0be6ad2aef0a3d885339616ee865839f533ad  fa1-warp-partitioning.png
ef1ddefa53f21572378b139c7bb1059c04fb0528438cfd01d9e2b023890d2269  fa2-thread-block-parallelism.png
b444cda9f8e0907da0e20c29b4c3e992ec86c070a14746382599213fc2ae6412  fa2-warp-partitioning.png
```
