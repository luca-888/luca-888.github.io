# 官方图来源

以下图片均直接下载自 PyTorch 官方博客的原始 PNG 地址，保留标签、色彩和连接；不是网页截图或缩略图。

- 页面：https://pytorch.org/blog/activation-checkpointing-techniques/
- pytorch-checkpoint.png：https://pytorch.org/wp-content/uploads/2025/03/fg5.png
- pytorch-selective.png：https://pytorch.org/wp-content/uploads/2025/03/fg8.png
- 获取：2026-09-22。正文分别引用于普通 AC 与 SAC 原理说明。

## 官方 benchmark

- 文件：`pytorch-memory-budget-benchmark.png`（1600 × 1045）
- 原图：https://pytorch.org/wp-content/uploads/2025/03/fg13.png
- 来源：上述 PyTorch 博客的 `(compile-only) Memory Budget API` 章节，原文明确称为 Transformer 的 real results。
- 获取：2026-09-22。保留官方原图，无重绘、裁切或改色。
- 图中横轴为 Memory Budget，纵轴为 Speed；原文没有给出速度单位、完整模型配置、硬件与原始数据。只解读趋势，不换算 ms、tokens/s 或加速倍数，不当作 eager SAC 与普通 GC 的对照实验。
- 正文约 50% 的结论来自图后官方说明，指 activation 保存开销，不扩展为整卡峰值下降 50%。
- 用户决定停止追加实验，正文改用官方结果；原有本地实验代码与数据保留归档。
