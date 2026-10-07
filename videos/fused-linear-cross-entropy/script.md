# Fused Linear Cross Entropy 视频脚本

横版 1920 × 1080，有配音（MiniMax，见 AGENTS.md），版式与配色沿用 DSpark 视频：纸色底，每幕一个结论标题，底部一行字幕。参考与实测见 `docs/fused-linear-cross-entropy-notes.md`。

## 主线

全片只问一个问题：**训练一步时，forward 末尾那段显存尖峰是什么，能不能不要？**

尖峰是 logits → logits 比 hidden state 大 74 倍 → 每个 token 的 loss 只要两个数，梯度用完就丢 → 先让梯度原地覆盖 logits（CE kernel）→ 再按 token 分块，logits 一次只存一块（FLCE）→ 结果拆开看：显存、时间、块大小 → 同类做法 → 回到开头的曲线。

## 贯穿全片的东西

- **一个模型**：Qwen3-1.7B，hidden size 2048，词表 151936。
- **一个 batch**：一步 16384 个 token（一条 16384 token 的序列）。
- **一个 token**：第 4 幕跟着它的一行 logits 走完 CE kernel 的两遍扫描。
- **第一块**：第 5 幕跟着 64 块里的第一块走完一次分块计算。
- **颜色**：hidden state 与 dX blue，lm_head 与 dW ink，logits yel，FP32 副本 yel 斜纹，梯度 red，loss teal，FLCE purple。

## 配音稿

数字均为 H100 实测（`public/measurements/fused-linear-cross-entropy/20261007-160054/`）或由形状推出。

| 幕 | 画面 | 配音 |
| --- | --- | --- |
| 0 开场 | 显存时间线（eager），forward 末尾的尖峰 | 这是 Qwen3-1.7B 训练一步的显存曲线，这一步有 16384 个 token。 |
| | 尖峰高度标出，与参数加优化器状态的虚线对照 | forward 结束时，显存突然升高 27 GB，接近参数和优化器状态的 3 倍。 |
| | 尖峰区间标为最后一层 | 这一段来自模型的最后一层。 |
| 1 logits | 一个 token：hidden state × lm_head → logits | 最后一层叫 lm_head，把每个 token 的 hidden state 变成词表上每个词的分数，称为 logits。 |
| | 按真实比例：74 份 hidden state 铺满一行 logits | hidden state 有 2048 个数，词表有 151936 个词，logits 比 hidden state 长 74 倍。 |
| | 16384 行：hidden state 64 MB，logits 4.6 GB | 16384 个 token 的 hidden state 只有 64 MB，logits 有 4.6 GB。 |
| | 显存柱：BF16 logits、FP32 副本、log_softmax 输出与梯度，合计 27.8 GB | 算 loss 之前，logits 还要转成 FP32，backward 时还有同样大小的梯度，这一层的峰值达到 27.8 GB。 |
| 2 loss 需要什么 | 一行 logits 的柱状示意，正确词高亮 | 回到一个 token。cross-entropy 的 loss，等于全部 logits 的 log-sum-exp 减去正确词的 logit。 |
| | 两个数从整行里取出 | 算 loss 只需要这两个数。 |
| | 柱子变为 softmax 减 one-hot | 梯度是 softmax 减去正确词的 one-hot，和 logits 一样长。 |
| | 梯度经两次乘法变成 dX 与 dW，然后丢弃 | 这个梯度只用两次：乘 lm_head 得到 hidden state 的梯度，与 hidden state 相乘累加到 lm_head 的梯度，之后就不再需要。 |
| 3 原地写回 | 一行 logits 分成 5 段，第一遍扫描更新 m 与 d | Liger 的 cross-entropy kernel 让每一行 logits 只读两遍。第一遍边读边更新最大值和指数和，这是 online softmax，读完就得到 loss。 |
| | 第二遍逐段把黄色改写成红色梯度 | 第二遍算出每个位置的梯度，直接写回 logits 所在的显存。 |
| | 实测峰值：eager 27.8 GB，Liger CE 5.3 GB | 这一层的峰值从 27.8 GB 降到 5.3 GB，但 logits 本身仍有 4.6 GB。 |
| 4 分块 | 16384 行切成 64 块 | Fused Linear Cross Entropy 把 lm_head 与 cross-entropy 合在一起，按 token 分块计算：16384 个 token 切成 64 块，每块 256 个。 |
| | 第一块：256 个 hidden state 乘 lm_head → 74 MB 的 logits | 跟着第一块走：256 个 hidden state 乘 lm_head，得到这一块的 logits，只有 74 MB。 |
| | 同一块原地变红 → 乘 lm_head 得 dX 的 256 行 | kernel 把它原地变成梯度，再乘 lm_head，得到这 256 个 token 的 hidden state 梯度。 |
| | 与 256 行 hidden state 相乘 → 加进 dW | 梯度再与这 256 个 hidden state 相乘，累加进 lm_head 的梯度。这块显存随后留给下一块。 |
| | 其余 63 块快速走完；块大小的算式 | 块的大小按词表与 hidden size 之比来定，让每块 logits 和全部 hidden state 差不多大。 |
| | backward 只做一次乘法 | 梯度在 forward 里就已算完。loss 是最后一层，backward 只需把它乘上传回来的系数。 |
| 5 显存与时间 | 峰值显存随 token 数：eager 斜线到 OOM，FLCE 贴着底 | 在 H100 上单独测这一层。eager 的峰值显存随 token 数线性增长，65536 个 token 时超出 80 GB；FLCE 到 65536 个 token 也只用 1.1 GB。 |
| | 时间随 token 数：两条线 | 代价是时间。16384 个 token 时两者接近，eager 83 毫秒，FLCE 79 毫秒；4096 个 token 时 FLCE 仍要 72 毫秒，是 eager 的 3.5 倍。 |
| | 16384 个 token 的时间拆成 kernel：FLCE 几乎全是矩阵乘，dW 累加一项 39 毫秒 | FLCE 的时间几乎都花在矩阵乘上。每一块都要把整个 lm_head 的梯度读出来、加上这一块的贡献再写回去，64 块就是 64 次，这一项用了 39 毫秒，eager 一次算完约 13 毫秒。 |
| | 块大小扫描：显存与时间 | 块越大，读写次数越少。每块 2048 个 token、共 8 块时，时间降到 50 毫秒，比 eager 快，显存只有 1.2 GB。 |
| 6 整步训练 | 两组柱：各序列长度的峰值显存，eager 在 65536 处 OOM | 放回完整训练：同一张 H100，eager 最长只能训练 32768 个 token 的序列，FLCE 到 131072 个 token 也只用 42 GB；16384 个 token 时，两者每步都是 0.93 秒。 |
| 7 同类做法 | CCE：logits 分块只在 SRAM，backward 跳过稀疏块 | Apple 的 Cut Cross Entropy 更进一步：logits 只在片上 SRAM 里分块计算，从不写入显存，backward 还跳过 softmax 小到可以忽略的块。 |
| | 16384 个 token 的四种实现：显存与时间 | 同样 16384 个 token，它用 0.64 GB、68 毫秒。 |
| | 两条批评 | 它对 Liger 的批评有两点：块多省显存，块少才快，两者难以兼得；loss 和梯度一起在 forward 里算，loss 上的任何变换都要写进 kernel。 |
| | Liger 的 Hopper 实现 | Liger 也在加入 Hopper 专用实现，logits 只写一次显存，原地变成梯度；在 4096 个 token、128256 的词表上，它用 19.5 毫秒，Triton 版本用 61 毫秒。 |
| 8 结尾 | 两条显存时间线：尖峰消失 | 回到开头的曲线。换成 FLCE 后，forward 末尾的尖峰消失，这一步的峰值从 39.5 GB 降到 15.1 GB。 |
