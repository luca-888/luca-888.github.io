# KV Cache 压缩视频脚本

横版 1920 × 1080，有配音（MiniMax，见 AGENTS.md），底部一行字幕。版式与配色沿用系列：纸色底，每幕左上一个结论标题。概念、画面设计与已知限制见 `keyframes.md`，出处见 `docs/attention-kv-compression-notes.md`。

## 配音稿

第一列非空的行开始新的一幕；`subs.py` 从这张表生成 `beats.js`。

| 幕 | 画面 | 配音 |
| --- | --- | --- |
| 0 封面 | 标题 | 大模型每生成一个 token，都要把之前所有 token 的 key 和 value 读一遍。这份缓存称为 KV cache。 |
| | 三根显存柱对着 80 GB 线 | 按最原始的 attention，一条 128K token 的请求要占 320 GiB。现在的模型只需要 40 GiB，甚至 11 GiB。 |
| 1 KV cache 从哪来 | 4 个历史 token 与当前 token | 先看缓存里存的是什么。模型生成第 5 个 token 时，attention 要从前 4 个 token 取信息。 |
| | query、key、点积、权重 | 当前 token 算出一个 query，每个历史 token 各有一个 key。query 与每个 key 做点积，经过 softmax 得到权重。 |
| | value、加权求和、输出 | 每个历史 token 还有一个 value。按权重把 value 加起来，就是这一步的输出。 |
| | 底部说明 | key 和 value 只由历史 token 决定，算一次就可以存下来，之后每一步都要读。这就是 KV cache。 |
| 2 有多大 | 一行变成 64 行 | 刚才只是一个 head。原始的 Multi-Head Attention，即 MHA，有很多个 head，每个 head 都有自己的 key 和 value。 |
| | 16384 个数、32 KiB | 以 64 个 head、每个 128 维为例，一个 token 在一层里要存 16384 个数，BF16 下是 32 KiB。 |
| | 乘层数与 token 数，柱子长高 | 乘上 80 层，再乘上 128K 个 token，就是 320 GiB。 |
| | 80 GB 线与红色斜线 | 一张 H100 只有 80 GB，这一条请求的缓存要 4 张卡才装得下。 |
| 3 decode 的瓶颈 | 显存 → 一次乘加 | 缓存大，还会让 decode 变慢。每一步只有一个新 token，从显存读入的每个数，只和 query 做一次乘加。 |
| | 两根条：1 对 295 | 折算下来，每读 1 字节只做 1 FLOP 的运算，而 H100 每读 1 字节可以做 295 FLOPs。 |
| | 底部说明 | 算力大部分时间在等数据，decode 的耗时取决于读了多少字节。缓存越小，这一步越快。 |
| 4 MQA | 64 个 query head 连到同一组 key、value | 最直接的办法：64 个 query head 全部保留，但只存一组 key 和 value，所有 head 都读这一组。 |
| | 名称与 1/64 | 这种做法称为 Multi-Query Attention，即 MQA。每层的缓存从 16384 个数缩小到 256 个，是原来的 64 分之一。 |
| | 说明 | 各个 head 的 query 不同，算出的权重也不同，但取到的内容都来自同一个 value。 |
| | 46 µs 与 3.8 µs | 提出 MQA 的论文在翻译模型上测过：decoder 每个 token 的耗时，MHA 是 46 微秒，MQA 是 3.8 微秒。 |
| 5 GQA | 64 个 query head 分成 8 组 | 折中的做法是分组。64 个 query head 每 8 个一组，每组共享一组 key 和 value，称为一个 KV head。 |
| | 名称与 Llama | 这就是 Grouped-Query Attention，即 GQA。8 个 KV head，每层 2048 个数，是 MHA 的 8 分之一。Llama-3.1-70B 用的就是这个配置。 |
| | 三种形式的缓存与 MMLU | 共享对质量的影响，DeepSeek 用三个只有 attention 不同的 7B 模型比过。MMLU 上，MHA 是 45.2，GQA 是 41.2，MQA 是 37.9。 |
| | 底部说明 | 另一组 876M 模型的实验里，几种形式相差不到 1 个点。 |
| 6 MLA | 标题 | DeepSeek 在 V2 中换了一个思路：head 之间不共享，改变缓存的内容。 |
| | MHA 一行 | MHA 里，key 和 value 都是 hidden state 的线性投影，缓存的是投影的结果。 |
| | MLA：下投影与 latent | Multi-head Latent Attention，即 MLA，把这次投影拆成两步。第一步是所有 head 共用的下投影，把 hidden state 压缩成一个 512 维的向量，称为 latent。 |
| | 上投影与还原出的 key、value | 第二步是上投影，每个 head 用自己的矩阵，从 latent 还原出 key 和 value。 |
| | 缓存框与底部说明 | 缓存里只放 latent。64 个 head 的 key 和 value 共 16384 维，全部由这 512 维决定。上投影矩阵是模型权重，不占缓存。 |
| 7 矩阵吸收 | 展开一行 | 但这样一来，decode 每一步都要把所有历史 token 的 latent 还原成 key，历史越长，计算越多。 |
| | 等式 | attention 用到的只是 query 与 key 的点积，并不需要 key 本身。key 等于上投影乘 latent，点积里相乘的顺序可以调换。 |
| | 吸收一行 | 先把上投影乘到 query 上，得到一个 512 维的向量，再直接与缓存的 latent 做点积，分数完全相同。这一步称为矩阵吸收。 |
| | 两行对照 | 这像查外文资料：与其把每一页都翻译过来，不如把问题翻译一次。不同的是，这里两种算法的结果严格相等。 |
| | 底部说明 | value 一侧同样处理：先对 latent 加权求和，最后只乘一次上投影。 |
| 8 Decoupled RoPE | 上半：四个块相乘 | 还剩位置编码。RoPE 把位置表示成旋转，旋转的角度取决于两个 token 的距离。 |
| | 红色的旋转块与说明 | 如果直接加 RoPE，这个旋转夹在 query 与上投影之间，每对 token 都不同，上投影就无法提前乘到 query 上。 |
| | 下半：两段对齐做点积 | MLA 把位置信息单独放在 64 维上。latent 不带位置，照常吸收；另外缓存一个 64 维、带 RoPE 的 key，所有 head 共享。分数是两段点积之和。 |
| | 512 + 64 = 576 | 于是每个 token 每层的缓存是 512 加 64，共 576 个数。 |
| 9 吸收后的形态 | MLA 的瓦片 | 吸收之后，64 个 head 读的是同一份 576 维的缓存，MLA 在 decode 时就是只有一个 KV head 的 MQA。 |
| | 四种形式并排 | 把四种形式放在一起，面积是每个 token 每层的缓存：MHA 16384 个数，GQA 2048，MLA 576，MQA 256。 |
| | 128K 请求一行 | 同一条 128K token 的请求，MHA 是 320 GiB，GQA 是 40，MLA 是 11.3。 |
| | MMLU | 质量上，DeepSeek 在约 250B 参数的模型上比较过：MLA 的 MMLU 是 59.0，MHA 是 57.5。 |
| 10 用计算换读取 | 标题 | MLA 的代价首先是计算。共享和吸收只减少读取，不减少运算，每读 1 字节做的运算随之增加。 |
| | 四根条与 295 线 | MHA 是 1，GQA 是 8，MQA 是 64，MLA 是 128，H100 的上限是 295。 |
| | 下半说明 | 吸收之后，每个 head 都要与 576 维的缓存做点积，head 越多，算得越多。 |
| | 64 与 128 个 head | 按 Kimi-K2 的测算，head 数从 64 加到 128，推理的 FLOPs 增加 83%，validation loss 只降低 0.5% 到 1.2%，所以 Kimi-K2 只用 64 个 head。 |
| 11 TP 下的重复 | GQA 一行 | 第二个代价出现在多卡上。tensor parallel 按 head 把 attention 分给多张卡。GQA 有 8 个 KV head，8 张卡每张存一个，没有重复。 |
| | MLA 一行，7 份红色斜线 | MLA 的每个 head 都要读完整的 latent，latent 不能按 head 切开，8 张卡各存一份，总占用是单卡的 8 倍。 |
| | 按请求分的一行 | SGLang 的 DP attention 改为按请求分：每张卡只存自己那部分请求的 latent。 |
| | 底部说明 | 在 8 张 H100 上，decode 吞吐提高到原来的 1.9 倍。 |
| 12 DeepSeek-V4 | MLA 一行，上投影划掉 | 到了 V4，DeepSeek 去掉了上投影。 |
| | V4 的一个向量 | 每个 token 只有一个 512 维的向量，所有 head 共用，它既是 key，也是 value，RoPE 只作用在最后 64 维。这种形式称为 Shared-KV MQA，相当于直接训练吸收之后的形态。 |
| | 16 个 token 合成 4 个 entry | V4 还沿序列方向压缩：每 4 个或每 128 个 token 合成一个 KV entry。 |
| | 底部说明 | 两者叠加，1M token 的上下文下，V4 的 KV cache 约为 GQA 的 2%。 |
| 13 各模型 | GQA 三行 | 这是几个公开模型每个 token 的 KV cache。Llama、Qwen、MiniMax 用 GQA，仍然是多数。 |
| | MLA 三行 | GLM-5、DeepSeek-V3 和 Kimi-K2 用 MLA；Kimi-K3 只在 24 层里用 MLA，其余是线性 attention。 |
| | V4 一行 | DeepSeek-V4-Pro 最小，每个 token 只有 7.7 KiB。 |
| 14 结尾 | 三根柱子，第一句小结 | 回到开头那条 128K token 的请求。MQA 和 GQA 让多个 query head 共享 KV head，320 GiB 降到 40。 |
| | 第二句小结 | MLA 只缓存 latent，借矩阵吸收在 decode 时直接读它，降到 11.3 GiB，代价是更多的计算和多卡下的重复。 |
| | 第三句小结 | DeepSeek-V4 则直接训练吸收之后的形态。 |
