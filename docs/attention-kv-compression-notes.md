# KV Cache 压缩：MQA、GQA 与 MLA 文章资料

状态：用户 2026-10-08 授权“直接做，你来定”，跳过方案确认与草图确认，直接写正文与精画；`status: 'draft'`，不 commit。决定：以 DeepSeek V4 收尾；核心图 3 张，TP 切分只在正文说明，不单独画图。2026-10-08 正文与 3 张图已完成（`content/attention-kv-compression.md`、`src/articles/attention-kv-compression/`），已做成稿自查与 de-ai-edit（subagent，改 3 处），dev server 上检查过图与公式。三篇 attention 变体文章的第一篇（2026-10-08 用户确定拆分与顺序：本篇 → 稀疏 attention → 线性 attention），各篇独立成篇，不写“上一篇 / 下一篇”。不做实测，数字来自论文与官方 config 计算。

- 源码基准：vLLM e006d76（v0.30.0）；SGLang main@8854857。
- 论文全文均读 arXiv 源文件包。

## 范围

- 从 MHA 到 MQA、GQA：减少 KV head 数。
- MLA：每 token 只缓存低秩 latent；矩阵吸收、decoupled RoPE、decode 与 prefill 的两种算法。
- 代价与对比：head 数与 decode FLOPs、TP 下 latent 复制、GLA、GQA→MLA 转换。
- 现状：GQA 仍是多数；MLA 被 Kimi、GLM 沿用；DeepSeek V4 改为 shared-KV MQA。
- 不展开：token 压缩与稀疏选择（CSA/HCA、DSA、indexer）留给稀疏 attention 篇；KDA / Gated DeltaNet 留给线性 attention 篇；gated attention、QK-norm 一句带过。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434)（§2.1 MLA，附录 MHA/GQA/MQA 与 MLA/MHA 消融） | DeepSeek-AI | 已读相关章节 | 核心 | 3、4、开头 |
| 2 | [vLLM `mla_attention.py`](https://github.com/vllm-project/vllm/blob/e006d76/vllm/model_executor/layers/attention/mla_attention.py) 文件头注释（forward_mha / forward_mqa / chunked prefill） | vLLM | 已读 | 核心 | 5 |
| 3 | [Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487)（GTA / GLA） | Zadouri, Strauss, Dao | 已读 §2–3 | 核心 | 1、3、6 |
| 4 | [DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348)（§2.3 hybrid attention、Other Details、Efficiency、KV cache 管理、Model Setups） | DeepSeek-AI | 已读相关章节 | 重要 | 7 |
| 5 | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)（MQA） | Shazeer | 已读 | 重要 | 1、2 |
| 6 | [GQA](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 已读 | 重要 | 2 |
| 7 | [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)（Number of Attention Heads 段） | Moonshot AI | 已读该段 | 重要 | 6 |
| 8 | [SGLang v0.4 blog：DP attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) | LMSYS | 已读该节 | 一般 | 6 |
| 9 | [TransMLA](https://arxiv.org/abs/2502.07864) | Meng et al. | 已读摘要与表达力附录开头 | 一般 | 6 |
| 10 | 官方 config.json：Llama-3.1-70B（unsloth 镜像，官方仓库需授权）、Qwen3-235B-A22B、gpt-oss-120b、MiniMax-M3、DeepSeek-V3、Kimi-K2、Kimi-K3、GLM-5 / 5.3、DeepSeek-V4-Pro / V4.1-Flash | 各厂商 | 已读（2026-10-08） | 重要 | 图 3、7 |
| 11 | [FlashMLA README](https://github.com/deepseek-ai/FlashMLA) | DeepSeek | 已读（WebFetch 摘要） | 一般 | 备查；2026-09-30 起去掉 Hopper，只支持 V4.1 |
| 13 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 已知（未重读） | 背景 | 1：MHA 的出处（2026-10-08 用户要求交代） |
| 12 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | DeepSeek-AI | 未读（只用 config） | 备查 | — |

## 阅读要点

- **MQA**：decode 时 memory/compute 比为 Θ(n/d + 1/b)，MQA 把 n/d 项缩小到 1/h。WMT 实验中 decoder 每 token 46 µs → 3.8 µs（TPUv2，batch 1024），BLEU 略降。
- **GQA**：mean pool KV 投影 + 5% 预训练算力 uptrain；T5-XXL 上 GQA-8 推理 0.28 s、均分 47.1，MHA 1.51 s、47.2，MQA 0.24 s、46.6。MQA 从头训练有 loss spike。指出 TP 分片时单个 KV head 会被每张卡复制，GQA 去掉这部分浪费。
- **DeepSeek-V2 MLA**：
  - 每 token 每层缓存 $d_c + d_h^R = 512 + 64 = 576$ 个元素，相当于 2.25 组 GQA；MHA 为 $2 n_h d_h = 32768$（128 head × 128）。
  - 附录：7B dense 模型 MMLU，MHA 45.2、GQA-8 41.2、MQA 37.9；~250B MoE，MHA 每 token 860.2K 元素、MLA 34.6K（4%），MMLU 57.5 vs 59.0。
  - RoPE 夹在 $W^Q$ 与 $W^{UK}$ 之间，使吸收失效，因此加一个所有 head 共享的 64 维 RoPE key。
  - 部署：FP8 权重 + 平均 6 bit 的 KV cache 量化，8×H800 生成吞吐 > 50K tok/s，是 DeepSeek 67B 的 5.76 倍（这一数字混合了 MoE、量化等因素，不能全归 MLA）。
- **vLLM 实现**：
  - decode 走 forward_mqa：把 $W_{UK}$ 乘进 q，变成 head dim 576（QK）/ 512（V）的 MQA，输出再乘 $W_{UV}$。
  - prefill 走 forward_mha：把 latent 展开成每 head 的 K/V，做 QK dim 192、V dim 128 的普通 MHA；context 很长时按 workspace 分块展开，用 LSE 合并。
  - 选择依据是 scheduler 标记的 prefill / decode。`get_num_kv_heads` 对 MLA 返回 1（“When using MLA during decode it becomes MQA”）。
- **GTA/GLA**：
  - decode 的算术强度：MHA ≈ 1，GQA ≈ g，MQA ≈ h，MLA ≈ 2h（latent 同时当 K 和 V）。h=128 时 MLA ≈ 256，接近 H100 的 ~295 FLOPs/byte。
  - TP 按 head 切分时，MLA 的 latent 每个 rank 都要存一份；TP=4 时 MLA 与 GQA-8 每卡 KV 大小相当。
  - GLA 把 latent 拆成 $h_c$ 个头，可以按卡分片；GLA kernel 在 query 长度 > 1（如 speculative decoding）时比 FlashMLA 最多快 2 倍。
- **GLA 论文的四形式对照（2026-10-08 用户要求质量表加 MLA）**：876M dense，FineWeb-Edu 50B token，加宽 FFN 对齐 MHA 参数量。FineWeb-Edu perplexity：MHA 11.501、GQA-4 11.340、MQA 11.413、MLA 11.363（$d_R=32$）；7 项下游平均：MHA 56.0、GQA-4 56.9、MQA 56.4、MLA 56.7（$d_R=48$ 时 57.8）。小规模下 MHA 不是最好，与 DeepSeek 7B 结论不一致，正文如实并列。
- **Kimi K2**：MLA，heads 从 128 降到 64。128K 序列下 heads 64→128 推理 FLOPs 增加 83%，而 validation loss 只改善 0.5–1.2%。原因：吸收后每个 query head 都要和 576 维 latent 做点积，head 数直接决定 decode FLOPs。
- **SGLang DP attention**：MLA 只有一个 KV head，TP8 时 KV cache 在每卡重复；改为 attention 部分按 DP 切请求，MoE 前后 all-gather / 重分发。8×H100、DeepSeek-Coder-V2 FP8 上 decode 吞吐为 v0.3 的 1.9 倍。
- **TransMLA**：同样 KV cache 与 query head 数下，表达力 GQA < MLA（分解形式）< MQA；提供 GQA→MLA 转换，6B token 微调恢复。
- **DeepSeek V4（现状变化）**：
  - CSA / HCA 的核心 attention 是 “Shared Key-Value MQA”：每个 KV entry 512 维，同时作为 key 与 value；没有 MLA 的 $W^{UK}/W^{UV}$ 上投影（config 无 `kv_lora_rank`，`num_key_value_heads=1`，`head_dim=512`）。
  - RoPE 只作用在最后 64 维；输出再乘位置 $-i$ 的 RoPE，抵消 value 带进来的绝对位置。
  - 输出投影分组（V4-Pro 16 组、每组 1024 维，即 `o_lora_rank`）。
  - KV 存储 RoPE 维 BF16、其余 FP8。1M 上下文时 KV cache 约为 BF16 GQA8（head dim 128）的 2%、V3.2 的 10%。报告没有解释为什么去掉低秩上投影。

## 现状（2026-10-08，官方 config 与 vLLM 源码）

| 模型 | attention 形式 | 每 token 每层缓存（元素） |
| --- | --- | --- |
| Llama-3.1-70B | GQA，64 q / 8 KV，d=128，80 层 | 2048 |
| Qwen3-235B-A22B | GQA，64 q / 4 KV，d=128，94 层 | 1024 |
| gpt-oss-120b | GQA，64 q / 8 KV，d=64，36 层（一半为 128 窗口 sliding） | 1024 |
| MiniMax-M3 | GQA，64 q / 4 KV，d=128，60 层，加稀疏选择 | 1024 |
| DeepSeek-V3 / Kimi-K2 | MLA，128 / 64 q，512+64，61 层 | 576 |
| GLM-5 / 5.3 | MLA（qk 256、v 256），64 q，512+64，78 层，加 DSA indexer | 576 |
| Kimi-K3 | 混合：KDA 线性层 + 24 层 MLA（93 层，每 4 层一层 MLA） | 576（仅 MLA 层） |
| DeepSeek-V4-Pro / V4.1 | shared-KV MQA，512 维（RoPE 64），加 token 压缩、稀疏选择与 128 窗口 | 512（每个压缩后的 entry） |

图 3 的整模型数字（BF16，每 token，只计 full attention 层，不计 DSA indexer 的 key 与线性层状态；128K 请求 = KiB ÷ 8 GiB）：Llama-3.1-70B 320 KiB（40 GiB）；Qwen3-235B-A22B 188（23.5）；MiniMax-M3 120（15）；GLM-5 87.75（11.0）；DeepSeek-V3、Kimi-K2 68.625（8.6）；Kimi-K3 27（3.4，24 层 MLA）；DeepSeek-V4-Pro 7.7 KiB（0.97 GiB）：config 的 `compress_ratios` 前 61 项中 30 层为 4、31 层为 128，每个 entry 512 维，BF16 平均每 token 7928 B；按实际的 FP8 + BF16 存储（每 entry 576 B）为 4460 B，加 FP4 indexer key 480 B，合计为 61 层 BF16 GQA-8（head dim 128）的 1.98%，与报告的“约 2%”一致。2026-10-08 用户要求去掉假想的 MHA 对照行，并在图中加入 V4。

前言表（2026-10-08 用户要求：做等价对比，不混入层数等其他因素；不写真实模型，直接用虚构配置）：固定 80 层、64 个 query head、head dim 128，只改 attention 形式。MHA 每层 16384 个元素，2560 KiB，128K 为 320 GiB；GQA-8 每层 2048，320 KiB，40 GiB；MLA（512 + 64）每层 576，90 KiB，11.25 GiB。

第二章直觉（2026-10-08 用户要求，后要求去重、按图讲）：新增“MHA 的信息传递”一节（query / key / value 的作用与 head 输出公式）；“共享方式”改为先放图 1，再按图中四行各用一条说明缓存了什么，MLA 也在此处给出定义；不再另设对照表。均为自己的分析，无新增来源。

精简（2026-10-08 用户确认方案）：8 章改为 7 章。质量差异表只留 DeepSeek 7B 与 876M 各一行，T5-XXL 并成一句；原第五章（vLLM 两种算法）并入“MLA 的代价”首节，去掉代码与分块展开；GLA 缩为一句，TransMLA 从正文与参考中删去；现状一章删去模型配置表，只留图 3；V4 只留位置编码与序列压缩，删去输出投影分组与存储精度。下方章节提纲为精简前的版本。

第三、四章重写与新图（2026-10-08 用户要求：MLA 与 decoupled RoPE 难懂，用图呈现）：低秩联合压缩补“32768 维由 512 维决定”；矩阵吸收改为“展开 / 吸收”两条对照；新增图 `::decoupled-rope::`（上：query × R × W × latent 的冲突；下：query 与缓存各分 512 + 64 两段、对齐做点积），核心图由 3 张变为 4 张。随后用户仍觉第三章难懂：低秩联合压缩改为“MHA 缓存投影结果，MLA 把投影拆成两步、缓存中间结果”的讲法，新增图 `::mla-latent::`（MHA 与 MLA 的投影路径，描边块为缓存）；矩阵吸收先讲按定义计算的问题，再讲“只需要点积、不需要 key 本身”。核心图共 5 张。

章节重排（2026-10-08 用户确认）：参考 Raschka《A Visual Guide to Attention Variants in Modern LLMs》（https://magazine.sebastianraschka.com/p/visual-attention-variants ，已读全文，只借用“一种形式一章、章首给定义与代表模型”的分法，未引用其内容）。现为：一 MHA；二 MQA 与 GQA；三 MLA（低秩联合压缩、矩阵吸收、Decoupled RoPE、吸收后的形态、质量）；四 MLA 的代价；五 Shared-KV MQA；六 各模型的 KV cache；七 小结。原第三、四章合并，576 的结论移到“吸收后的形态”。

MHA 配图（2026-10-08 用户要求）：`::mha-flow::`，3 个 head × 4 个历史 token 的 K/V 网格，示例路径为 head 1 的 query → 点积 → softmax → 加权求和 → 输出。核心图共 6 张。

## 文章主线

decode 每步都要把整个 KV cache 从显存读一遍，计算量却很少（MHA 的算术强度约为 1），所以每 token 缓存的字节数决定了 decode 速度与并发上限。MQA/GQA 让多个 query head 共享一组 K/V。MLA 只缓存一个低秩 latent，再借矩阵吸收让 decode 直接在 latent 上做 MQA，同一份数据既当 K 又当 V。代价转移到计算（head 数决定 FLOPs）和并行（latent 无法按 head 切分）。DeepSeek V4 直接训练“吸收后”的形态：一个 512 维、K=V 的共享 KV head。

## 章节提纲

0. 开头：定义（每 token KV cache = 层数 × 每层缓存）、目的、一个代表结果（DeepSeek-V2 附录：~250B MoE 上 MLA 每 token 34.6K 元素，MHA 860.2K，MMLU 59.0 vs 57.5）。
1. **decode 的瓶颈**：每步读一遍 KV cache；算术强度 ≈ 1；MQA 论文 46 µs → 3.8 µs。
2. **共享 KV head：MQA 与 GQA**：缩小到 $h_{kv}/h$；质量差异（DeepSeek 7B 消融）；GQA 成为默认配置的原因（uptrain、TP 切分）。
3. **MLA：低秩联合压缩与矩阵吸收**：$c^{KV}$；$W^{UK}$ 并入 query、$W^{UV}$ 并入输出；吸收后等价于 K=V 的 MQA，算术强度 ≈ 2h。
4. **Decoupled RoPE**：RoPE 为何阻止吸收；共享 64 维 RoPE key；576 的由来。
5. **Prefill 与 decode 的两种算法**：vLLM forward_mha / forward_mqa；长 context 分块展开。
6. **代价**：head 数与 decode FLOPs（Kimi K2）；TP 下 latent 复制 → DP attention（SGLang）、GLA 分组 latent；GQA→MLA 转换（TransMLA）。
7. **现状**：GQA 仍是多数；MLA 在 Kimi、GLM 中与 DSA、KDA 组合；DeepSeek V4 去掉上投影，改为 512 维 shared-KV MQA。
8. 小结。

## 核心图（待草图确认）

| 图 | 回答的问题 | 粗稿 |
| --- | --- | --- |
| 1 | 一个 decode token 每层要读多少缓存 | 128 个 query head 排成一行，下方是各方案缓存的 K/V 块，面积按元素数：MHA 32768、GQA-8 2048、MQA 256、MLA 576（512 latent + 64 RoPE）；连线表示哪些 query head 共享哪一块 |
| 2 | 矩阵吸收把计算移到了哪里 | 一个 head 的两条路径并排：展开路径 latent → $W^{UK}$ → k → 与 q 点积；吸收路径 q → $W^{UK\top}$ → 与 latent 点积；`Link focus` 标出一个 token 的路径，标注两边的维度（128 vs 512） |
| 3 | 真实模型每 token 的 KV cache 有多大 | 横向条形图（ECharts），BF16、只计 full attention 层：Llama-3.1-70B、Qwen3-235B、DeepSeek-V3、Kimi-K2、GLM-5 等，加 DeepSeek-V4-Pro（按压缩后的 entry 平均到每 token）；不画假想模型 |
| 可选 4 | TP 下每卡存多少 | TP=8 时 GQA-8 每卡 1 个 KV head，MLA 每卡完整 latent；DP attention 每卡只存自己的请求 |

## 待核实问题

1. DeepSeek V4 去掉低秩上投影的原因：报告未说明，正文只陈述结构变化与 TransMLA 的表达力结论，不推测动机。
2. Llama-3.1-70B 的 config 来自 unsloth 镜像，正式成稿前尝试读取 meta-llama 官方仓库。
3. GLM-5 每 head 的 qk 为 256（nope 192 + rope 64），需要确认其吸收后 latent 仍为 512+64，以 vLLM `glm5next` 或 GLM 的 MLA 实现为准。
4. V4 的 shared-KV MQA 是否在 vLLM 中也走 MLA 后端（`MLAAttentionSpec`、`fp8_ds_mla` 576B slot），用于说明“吸收后的 MLA 与 V4 共用同一类 kernel”。
5. 是否需要实测：初定不做；如需代表数字，可在 4090 上测 GQA 与 MLA decode kernel 在长 context 下的耗时。

## 视频

2026-10-09 用户确认分镜后制作，源文件在 `videos/attention-kv-compression/`，概念、分镜与已知限制见该目录的 `keyframes.md`，配音稿见 `script.md`。横版 1920 × 1080，MiniMax 配音，15 幕，约 8 分 47 秒。全片用一个配置（80 层、64 个 query head、head dim 128），因此算术强度一幕写 MLA 约 128（2h 代入 h = 64），文章里的 256 对应 h = 128。视频不讲 prefill 与 decode 的两套算法、GQA 由 MHA checkpoint 转换、GLA 与 V4 的 −t 旋转。RoPE key 在视频里用 teal（系列里 red 表示装不下与重复），与文章图中的红色不同。
