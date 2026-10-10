# 稀疏 Attention：Sliding Window、Attention Sink、DSA 与 CSA 文章资料

状态：2026-10-09 用户要求“正式完成，多参考优秀 blog 表达、图片和官方图”“完成文章再通知我”，据此跳过方案确认与草图确认，直接完成调研、正文与图；`status: 'draft'`，未 commit。正文 `content/sparse-attention.md`，页面 `src/articles/sparse-attention/`，原图 `src/assets/sparse-attention/`。已做成稿自查与 de-ai-edit（subagent，改 7 处，另按其疑点改 5 处），dev server 上检查过 9 张图、公式与 console。三篇 attention 文章的第二篇，独立成篇，不写“上一篇 / 下一篇”。不做实测，数字来自论文、官方 config 与自己的计算。

系列框架（2026-10-09 用户确认）：KV 压缩（每个 token 存得更小）→ 稀疏（每个 query 只读一部分）→ 线性（只存固定大小的状态）。attention sink 不单独成篇，作为本篇第三章。

- 源码基准：vLLM `4611c2c`（2026-10-08）。
- 论文全文均读 arXiv 源文件包的 `.tex`；DSA 论文只有 GitHub 上的 PDF，用 `scripts/pdf_tools.py text` 提取。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [DeepSeek-V4](https://arxiv.org/abs/2606.19348)（§2.3 hybrid attention、Other Details、Efficiency、KV cache 管理、on-disk cache、Model Setups、MRCR） | DeepSeek-AI | 已读相关章节 | 核心 | 3、6 |
| 2 | [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf)（DSA） | DeepSeek-AI | 已读全文（附录未读） | 核心 | 5 |
| 3 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) | Yuan et al. | 已读摘要、方法、实验、效率、讨论 | 核心 | 1、4、5 |
| 4 | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | Xiao et al. | 已读摘要、§3 与相关图表 | 核心 | 3 |
| 5 | [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)（local/global 段与消融） | Gemma Team | 已读相关段落与两张消融图 | 重要 | 2 |
| 6 | [Mistral 7B](https://arxiv.org/abs/2310.06825)（SWA、rolling buffer） | Jiang et al. | 已读相关段落 | 重要 | 2 |
| 7 | [gpt-oss model card](https://arxiv.org/abs/2508.10925)（Attention 段） | OpenAI | 已读该段 | 重要 | 2、3 |
| 8 | vLLM `kv_cache_interface.py`、`single_type_kv_cache_manager.py`、`models/deepseek_v4/attention.py`、`models/minimax_m3/common/` | vLLM | 已读相关类与文件头 | 重要 | 2、5、6 |
| 9 | [MoBA](https://arxiv.org/abs/2502.13189) | Lu et al.（Moonshot） | 已读摘要、方法、实验设置 | 一般 | 5 |
| 10 | [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732) | Barbero et al. | 已读摘要与引言 | 一般 | 3 |
| 11 | 官方 config.json：Mistral-7B-v0.1 / Instruct-v0.2、Gemma 3 27B（unsloth 镜像）、Gemma 4 31B、gpt-oss-120b、DeepSeek-V3.2、DeepSeek-V4-Pro、GLM-5、MiniMax-M3 | 各厂商 | 已读（2026-10-09） | 重要 | 2、5、6 |
| 12 | [How Attention Sinks Keep Language Models Stable](https://hanlab.mit.edu/blog/streamingllm) | Guangxuan Xiao | 已读（WebFetch 摘要） | 表达参考 | 3：sink token 与 sink 标量的对照 |
| 13 | [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants) | Raschka | 已读 SWA、DSA 两节（WebFetch 摘要） | 表达参考 | 章节分法：一种形式一章，章首给定义与代表模型 |
| 14 | [Longformer](https://arxiv.org/abs/2004.05150) | Beltagy et al. | 未读（已下载源文件） | 备查 | 未引用 |

## 阅读要点

- **NSA**：
  - decode 是 memory-bound，训练与 prefill 是 compute-bound，两个阶段优化目标不同。
  - 三个分支：compression（block 长 32、步长 16，MLP 压缩）、selection（block 64，top-16，含固定的首 block 与 2 个 local block）、sliding window（512）。三分支各有独立 K/V，gate 加权相加。
  - selection 的 block 重要性来自 compression 分支的 attention score；GQA 组内 head 求和后共享选择。
  - 读取量（等效 token）：8192 → 2048，16384 → 2560，32768 → 3584，65536 → 5632，预期 decode 加速 4× / 6.4× / 9.1× / 11.6×。64K 下 Triton 实现 forward 9×、backward 6×（对 Triton FlashAttention-2，8×A100）。
  - 27B MoE（3B active，GQA-4，64 head），270B token 预训练。LongBench 平均 0.469，full attention 0.437，Exact-Top 0.423。
  - 讨论：3B 模型上 auxiliary loss 打分（类似 SeerAttention）与 Quest 式启发式打分的 loss 都高于 NSA；ClusterKV 式聚类不适合训练。
- **DSA（V3.2-Exp）**：
  - indexer：$I_{t,s} = \sum_j w^I_{t,j}\,\mathrm{ReLU}(q^I_{t,j}\cdot k^I_s)$，ReLU 为吞吐考虑，可用 FP8。config：`index_n_heads=64`、`index_head_dim=128`、`index_topk=2048`。
  - 基于 MLA 的 MQA 模式实现，原因是“kernel 层面每个 KV entry 必须被多个 query 共用”。
  - 训练：dense warm-up 1000 步（16 × 128K，2.1B token，lr 1e-3，只训 indexer，KL 对齐 head 求和后 L1 归一的主 attention 分布）；稀疏训练 15000 步（480 × 128K，943.7B token，lr 7.3e-6），indexer 输入 detach。
  - 评测与 V3.1-Terminus 接近（MMLU-Pro 85.0 / 85.0，SWE Verified 67.8 / 68.4，BrowseComp 40.1 / 38.5）。主 attention $O(L^2)\to O(Lk)$，indexer 仍 $O(L^2)$。短序列 prefill 用 masked MHA 模拟 DSA。
  - 成本图（H800，2 USD / GPU 小时）只有曲线，正文未引用数值。
- **DeepSeek V4**：
  - CSA：$m=4$，相邻 block 重叠压缩（每个 entry 来自 $2m$ 个 token），之后 DSA top-k（Pro 1024，Flash 512）；indexer key 同样压缩，indexer 用 FP4。
  - HCA：$m'=128$，不重叠，不选择。
  - 每层另有 $n_\text{win}=128$ 的未压缩窗口 KV，与压缩 entry 拼接进同一次 core attention；原因是因果性（读不到本 block 内的其他 token）与近处 token 更相关。
  - sink：每个 head 一个可学习 logit，加在 softmax 分母，引 StreamingLLM 与 gpt-oss。
  - Pro：61 层，前 2 层 HCA，其后 CSA / HCA 交替（config `compress_ratios` 前 61 项中 30 个 4、31 个 128）；Flash：43 层，前 2 层纯 sliding window。
  - KV cache：1M 上下文约为 BF16 GQA-8（head dim 128）的 2%。缓存分为按 block 分页的压缩 entry（每 block 覆盖 lcm(m, m′) 个原始 token）与每请求固定大小的 state cache（窗口 KV 与未凑满 block 的尾部 token）。
  - on-disk prefix cache：窗口 KV 约为压缩 entry 的 8 倍，三种策略（全存 / 周期 checkpoint / 不存，重算最后 $n_\text{win}\cdot L$ 个 token）。
  - MRCR：128K 内稳定，之后下降。2026-10-10 读 arXiv 源文件 `figures/mrcr.pdf`（MRCR 8-needle，Average MMR）：V4-Pro-Max 8K–1M 依次为 0.90 / 0.85 / 0.94 / 0.90 / 0.92 / 0.82 / 0.66 / 0.59，V4-Flash-Max 为 0.91 / 0.84 / 0.87 / 0.85 / 0.87 / 0.76 / 0.60 / 0.49。正文取 128K 0.92、1M 0.59。注意 `tables/large_eval.tex` 的“MRCR 1M (MMR)”V4-Pro-Max 为 83.5，口径与图不同，正文不用。
- **StreamingLLM**：
  - Llama-2-13B、PG19 第一本书（65K token）：0+1024 PPL 5158.07，4+1020 PPL 5.40，4 个换行符 + 1020 PPL 5.60；dense 5641，重算窗口 5.43。
  - 成因：softmax 归一；首 token 对所有后续 token 可见。Llama-2-7B 除最底两层外各层各 head 都大量关注首 token。
  - 160M 预训练对照（cache 0+1024 / 1+1023 / 2+1022 / 4+1020）：vanilla 27.87 / 18.49 / 18.05 / 18.05；zero sink 29214 / 19.90 / 18.27 / 18.01；learnable sink 1235 / 18.01 / 18.01 / 18.02。
- **gpt-oss**：banded window 与 dense 交替，bandwidth 128；64 q head、head dim 64、GQA 8 KV head；dense 层用 YaRN 扩到 131072；每个 head 在 softmax 分母中有一个 learned bias，“enables the attention mechanism to pay no attention to any tokens”。
- **Gemma 3**：local : global = 5 : 1，窗口 1024，从 local 层开始；Gemma 2 为 1 : 1、4096。消融图：比例 1:1 到 7:1 的 Δperplexity 在 ±0.03 内，窗口 512 到 4096 在 ±0.02 内。2B 模型 32K 上下文下 KV cache 相对权重的额外内存：global only 60%，1:3 加 sw=1024 低于 15%。global 层 RoPE base 1M，local 层 10k。
- **Mistral 7B**：v0.1 全部层 SWA，W=4096，32 层理论覆盖约 131K；rolling buffer cache，32K 序列缓存为 1/8。v0.2 model card：“32k context window”“No Sliding-Window Attention”，config `sliding_window: null`。
- **MoBA**：block 为单位，打分为 query 与 block 内 key 均值的点积，无新增参数；当前 block 固定选中并加 causal mask。1M 模型 block 4096、top-12（稀疏度 95.31%），最后 3 层保留 full attention，其余 29 层 MoBA。摘要称已用于 Kimi 的长上下文请求。
- **Barbero et al.**：sink 让 head 不活跃，减缓 over-mixing / 表示坍缩；Llama 405B 典型 prompt 上近 80% 的 attention 在 BOS。
- **vLLM（`4611c2c`）**：
  - `SlidingWindowSpec` / `SlidingWindowManager`：`remove_skipped_blocks` 在每次分配前释放窗口外的 block，单请求 block 上限为 `cdiv(sliding_window - 1 + in_flight, block_size) + 1`；prefix cache 命中从右向左找 `cdiv(window - 1, block_size)` 个连续命中的 block。
  - V4：`SlidingWindowMLASpec`、`CircularBufferSpec`（每请求一个 block，存仍在压缩中的 token 组的原始 key）、`tokens_per_state=compress_ratio`；`attn_sink` 默认 `-inf`、从权重加载；候选数不超过 top-k 时 `_fill_short_context_topk_indices` 直接全选。
  - 有 `sinks` 的模型：gpt-oss、MiMo-V2、Granite 新版、Laguna、DeepSeek V4 / V4.1、HY-V4。有 `sliding_attention` 层的模型：Gemma 2/3/3n/4、gpt-oss、MiMo-V2、Step-3.5、Cohere2、EXAONE 4、Granite 新版、AFMoE、Laguna。无 sink、无窗口：DeepSeek V3.2、MiniMax-M3、Kimi-K3。
  - MiniMax-M3：block-sparse GQA，indexer 选 top-k block 加固定的 init / local block。

## 现状变化与相关系统对比

- Mistral 7B 的全层 SWA 在 v0.2 被去掉；现行模型（Gemma、gpt-oss）都是 local / global 交替。
- DeepSeek 的三代：NSA（三分支、block 选择、复用 compression 分数、从头训练，未见于已发布模型）→ DSA（单分支、逐 token、独立 indexer 加 KL loss、由 dense checkpoint 继续训练）→ V4 CSA / HCA（压缩、选择、窗口三者都回来，但拼接进一次 attention；top-k 从 2048 降到 1024 个压缩 entry）。
- NSA 论文认为 auxiliary loss 打分不如复用 compression 分数；DSA 采用了 KL loss。正文并列两者的条件，不下结论。
- 同类：GLM-5（DSA，`index_n_heads=32`，top-k 2048）、MiniMax-M3（block 128、top 16 block、`sparse_local_block=1`、4 个 index head、前 3 层不稀疏）、MoBA。

## 自己的计算

- 开头表（128K = 131072 token，一层一步）：full 131072；Gemma 3 local 层 1024；DSA 2048；V4-Pro CSA 层 1024 个压缩 entry 加 128 个窗口 entry。
- Gemma 3 27B：62 层，`sliding_window_pattern=6`，global 层 10 层、local 层 52 层。128K 下 entry 总数 10 × 131072 + 52 × 1024 = 1363968，全 global 为 8126464，比值 16.8%。
- V4-Pro 1M（1048576）：CSA 层 262144 个 entry，HCA 层 8192 个；top-1024 对应 4096 个原始 token。
- NSA 64K：4096（压缩）+ 1024（16 × 64）+ 512（窗口）= 5632。

## 图

| 图 | 类型 | 回答的问题 |
| --- | --- | --- |
| `::sparse-read-set::` | 自绘 | full / sliding window / 学习式选择一步各读哪些 token，其余 token 能否释放 |
| `::mistral-swa::` | 原图（Mistral 7B Fig. 1） | 窗口 mask 与跨层传递 |
| `::sink-attention::` | 原图（StreamingLLM Fig. 2） | attention 落在哪里 |
| `::sink-scheme::` | 原图（StreamingLLM Fig. 1） | 丢掉开头 token 的后果与保留 sink 的效果 |
| `::sink-softmax::` | 自绘（示意数值） | sink 滑出后其余权重如何被放大，sink logit 如何补上 |
| `::nsa-read-set::` | 自绘 | NSA 三个分支在 decode 一步各读什么，三路由 gate 合并 |
| `::dsa-two-stage::` | 自绘 | indexer 读全部小 key，主 attention 只读 top-k |
| `::v4-read-set::` | 自绘 | CSA 与 HCA 如何压缩、各读哪些 entry |

自绘图语义色：blue 为被读取的未压缩 KV，purple 为被读取的 entry（DSA 的 latent、V4 的压缩 entry），teal 为 indexer key，gray 为留在缓存但不读，虚线空格为已释放，red 为 sink。原图出处见 `src/assets/sparse-attention/SOURCES.md`。

未采用的原图：V4 的 `HCA.pdf`（是 CSA 图的子集，正文用一句话说明）、`kv_cache.pdf`，DSA 论文 Fig. 1（MLA 细节过多），MoBA `running_example.pdf`，Gemma 3 `mem_lc.pdf`。

## 2026-10-09 改版（用户反馈：文字太多、视觉层次不够、读着累）

- 正文从约 21.7K 字符减到约 14.7K：删去 rolling buffer、换行符实验、over-mixing 解释、sink token 的 perplexity、DSA 第二个公式与训练步数、NSA 的 auxiliary loss 对照、压缩公式、MoBA 段落、vLLM 对 V4 的实现、prefix cache 的三种策略；这些要点仍在上文“阅读要点”中。
- 开头表格改为通用形式（每步读取量写成 $L$ 的函数，128K 作为代入列）。
- 新增自绘图 `::read-vs-context::`（读取量随上下文的变化，双对数；NSA 用 $L/16 + 1536$，HCA 用 $L/128 + 128$，均为按论文配置的近似）与 `::local-global-layers::`（Gemma 3 27B 的 62 层排列与 128K 下的 KV cache，17%）。
- 第二、四、五、六章章首加规格卡 `::spec-*::`（读取哪些、每步读取、缓存、代表模型），样式在 `src/articles/sparse-attention/sparse-attention.css`；卡片文字在 `sparse-attention.blocks.tsx`。
- 三种显式 sink 的做法由列表改为表格。
- 第一章新增论点“收益随上下文增长，短上下文等价于 full attention”。

## 2026-10-09 二次压缩（用户要求：前言压缩，整体只留关键逻辑）

正文减到约 10.7K 字符。前言只留定义、KV entry 与表格；各章去掉三级标题，每个论点一到两句。删去：local / global 配比表（并入一句）、vLLM 的 block 回收与 prefix caching、Llama 405B 的 80%、index score 公式、DSA 两阶段训练的分步、V4 的 prefix cache 成本；参考中相应去掉 Barbero 与 vLLM 两条。图与规格卡未动。

## 待核实

1. Gemma 3 27B 的 config 取自 unsloth 镜像（官方仓库需授权），正式发布前尝试读取 google 官方仓库。
2. MiniMax-M3 的稀疏 attention 只读了 config 与 vLLM 文件头，未找到官方技术报告；`sparse_local_block=1` 解释为“固定选中最近 1 个 block”是按字段名与 vLLM 注释的理解。
3. “NSA 的 3B 对照实验从头训练”是根据论文比较 training loss 曲线推断的，论文未逐字说明。
4. DSA 论文的成本曲线没有可引用的数值；如需代表数字，可在 4090 上测 top-k 稀疏读取与全量读取的 decode 耗时。
5. 不加 sink、从头用 SWA 训练的模型（Mistral 7B v0.1）把注意力放在哪里，未查到资料，正文未写。
6. MoBA 在 Kimi 线上的部署只有论文摘要一句话，正文未写。

## 前言复杂度表（2026-10-10 加入）

- 每个 entry 的元素数（K、V 合计，每层）：Llama-2-7B MHA 32 head × 128 × 2 = 8192；Llama-3.1-70B GQA 8 KV head × 128 × 2 = 2048；DeepSeek-V3 MLA 512 + 64 = 576；DSA 另加 128 维 indexer key。V4 压缩 entry 的维度未核实，表中只写“压缩 entry”。
- 线性 attention 行：Qwen3-Next Gated DeltaNet，transformers `configuration_qwen3_next.py`（commit f339035）默认值 linear_num_value_heads 32、key/value head dim 128 → state 32 × 128 × 128 = 524288 元素/层 = 256 个 GQA entry；full_attention_interval 4（每 4 层 3 层线性）。
- prefill 计算按 query–entry 对数的量级记，DSA / CSA 第二项为 indexer。

## 2026-10-10 核对

- 重新读取 Hugging Face config：V4-Pro（61 层，`compress_ratios` 前 61 项 30 个 4、31 个 128，`sliding_window` 128，`index_topk` 1024，`index_n_heads` 64，`index_head_dim` 128，`head_dim` 512）、V3.2、GLM-5、MiniMax-M3，与正文一致。
- V4 压缩 entry 的维度 $d'$ = `head_dim` = 512：vLLM `origin/main`（`2d69083c`）`vllm/models/deepseek_v4/attention.py` 中压缩 entry 与窗口 KV 都按 `head_dim` 存。V4 与 M3 的代码路径现为 `vllm/models/…`。
- 按元素数，V4-Pro 1M 下 (30 × 262144 + 31 × 8192 + 61 × 128) × 512 ≈ 4.16e9，GQA-8 为 2048 × 1048576 × 61 ≈ 1.31e11，比值 3.2%；论文的 2% 是另一口径（推测含低精度存储），正文只引论文数字。
- 前言表补上 indexer 项：DSA 缓存 $(d_c + d_I) L$、读取 $d_c k + d_I L$，CSA 同理（$L/4$ 个压缩 indexer key）；128K 下 DSA 的 indexer 读 128 × 131072 ≈ 16.8M 元素，主 attention 读 576 × 2048 ≈ 1.2M。首段不再写“与 $L$ 无关的上限”。
- NSA selection 正文写明首 block 与最近 2 个 block 固定选中；posts.ts 去掉 `vLLM` tag；SOURCES.md 改为三张图。

## Sink 的 value 范数（2026-10-10 补充，已读 arXiv 2410.10781 源文件 v 2025-03）

- Gu et al., When Attention Sink Emerges in Language Models: An Empirical View, ICLR 2025。第 4 节 / Figure 2：LLaMA3-8B Base 从第 2 个 block 起，第一个 token 的 hidden state 范数很大（massive activations），但 key 与 value 的 ℓ2 范数显著小于其他 token；sink 源于 query 与 k_1 的夹角小，而非范数乘积大。
- 结论：sink 相当于 key bias，存放多余的 attention，不参与 value 计算；只有第一个 token 能被所有 token 看到，所以 sink 落在第一个 token。K bias（v* = 0）设置下 sink 转移到 bias 上，性能相当。
- 正文用法：第三节“Sink 让 head 在无需读取时输出接近零”一段中“value 向量范数远小于其他 token”。

## 压缩方式（2026-10-10 补充，已读 DeepSeek-V4 arXiv 源文件 main.tex §2.3）

- NSA：每 l=32 个 token 的 key（value 另用一个）连同块内位置编码送入可学习 MLP φ，步长 d=16；selection 的 block 分数由重叠的 compression 分数求和（NSA 式 9）。
- V4 HCA：C = H·W^KV，Z = H·W^Z；块内 S = Softmax_row(Z + B)（B 为可学习位置偏置，按通道），C^Comp = Σ S_j ⊙ C_j。权重由 token 的 hidden state 决定，逐通道。
- V4 CSA：两组投影 C^a/Z^a（当前 m 个 token）与 C^b/Z^b（前 m 个 token），2m 个位置一起 softmax 后加权求和，相邻 entry 共用 m 个 token；i=0 时 b 部分填 −inf / 0。压缩 entry 同时作 key 与 value（MQA）。
- 正文：NSA 列表写“key 与 value 分别经一个 MLP 压成”；第六节写“权重由每个 token 的 hidden state 算出，在块内做 softmax”。

## 小红书封面与正文页（2026-10-10）

- 封面：代码绘制（HTML + SVG，非图像模型生成），源文件 `covers/sparse-attention/cover.html`，`node covers/sparse-attention/render.mjs` 导出 `portrait-1080x1920.png`，实际尺寸 1080 × 1920（9:16）。第一稿是带 sink / 窗口 / 选中三种颜色的 token 网格，并写了 DeepSeek V4 与四个关键词；用户反馈“过于具体”“不要带 DeepSeek V4”。第二稿改为抽象点阵：淡色小点中散布 13 个实心大圆（12 个钴蓝、1 个珊瑚），文字只留“稀疏 / Attention / Sparse Attention”；暖米白底、黑字，顶部 12% 无文字。用户随后要求“配一个论文的图”并给出标题句式“稀疏 Attention：如何……”。第三稿（当前）：上方白底卡片放 Mistral 7B Figure 1 原图（`src/assets/sparse-attention/mistral-swa.svg`，未改动，卡片下注明出处），下方“稀疏 / Attention”，横线下一行“如何只读一部分 KV，而不掉效果”，英文 Sparse Attention；顶部 12% 无文字，文字在 y 774–1490。用户再给出影片结尾一幕的截图，要求“参考这个，竖版”。第四稿（当前，canvas 绘制）：把该幕（`videos/sparse-attention/scenes.js` 的 s11）改排成竖版，“稀疏”黑、“Attention”钴蓝两行大字，副标题“每一步，只读其中一小部分”，128K 个 token 的一页（512 × 256，紫色为选中的 1024 个压缩 entry，蓝色为最近 128 个，选点种子与影片相同），下方两条“每步读取”（全部读 128K 个、只读一部分 1024 + 128 个）与“上下文再长，这一条也不变”；影片里的“开场 / 现在”在封面上改为“全部读 / 只读一部分”。不放论文图与 DeepSeek V4。用户反馈黄色的 query 看不见，要求加放大小窗：页缩为 896 × 448（一格 1.75 px），query 方块放在页外右下；页右下角 32 列 × 14 行连同 query 用黑框圈出，虚线引到右下的放大小窗（一格 13 px），小窗下标“query：正在生成的 token”“最近 128 个 token”；两条“每步读取”移到左下、宽 424。文字在 y 260–1660，顶部 12% 无文字。第四稿为定稿。
- 小红书标题：用户要求写稀疏 attention 要解决的问题与难点，不写具体模型与数字，句式“稀疏 Attention：如何……”。
- 正文页：`src/xhs/articles.ts` 登记后 `npm run xhs -- sparse-attention` 导出 14 页。为让前言表在 1080 宽下不被截断，表头“decode 每步读取”改为“每步读取”，CSA 的代表模型写 DeepSeek-V4，DSA / CSA 两行公式去掉 `\cdot`。

## 定稿（2026-10-10）

- 文章、视频与小红书已发布，`src/posts.ts` 改为 published，日期 2026-10-10。
- 封面定稿留在 `covers/sparse-attention/`（竖版 1080 × 1920，横版 1920 × 1440）；`videos/sparse-attention/build/` 只保留成片与配音缓存，逐幕片段、无声版、预览与 storyboard 已删除。
