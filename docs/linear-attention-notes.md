# 线性 Attention 与混合架构文章资料

状态：2026-10-10 用户回复“直接文章”，据此确认方案并跳过草图确认，直接完成正文与图；`status: 'draft'`，未 commit。正文 `content/linear-attention.md`，页面 `src/articles/linear-attention/`，原图 `src/assets/linear-attention/`。已做成稿自查与 de-ai-edit（subagent）。2026-10-10 用户反馈“视觉层次不够丰富，文本太多”：正文从约 18K 字压到约 12K 字，新增 state 一步写入/读取图、S-NIAH 条形图、各模型层排布色带图，模型表、混合比例消融、位置编码分歧、MiniMax 三代结构改为表格，“每层缓存”图移到第七章开头。三篇 attention 文章的第三篇，独立成篇，不写“上一篇 / 下一篇”。2026-10-09 用户确认系列按三条路线组织（KV 压缩 → 稀疏 → 线性）；本篇代表“只存固定大小的 state”这条路线。不做实测，数字来自论文、官方 config 与自己的计算。

- 源码基准：vLLM `ac18817e`（2026-10-10）。
- 论文全文优先读 arXiv 源文件包的 `.tex`；Kimi K3 技术报告只有 GitHub 上的 PDF，用 `scripts/pdf_tools.py text` 提取。
- 站内已有：LLM 量化一文提到 Qwen3.8-27B 64 层中 48 层是 Gated DeltaNet、FP8 KV cache 收益较小；稀疏 attention 一文的开销表有“线性 attention”一行。本篇不重复量化内容。

## 参考清单（按重要程度）

| # | 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- | --- |
| 1 | [Gated Delta Networks](https://arxiv.org/abs/2412.06464)（ICLR 2025） | Yang, Kautz, Hatamizadeh | 已读摘要、§3 gated delta rule、S-NIAH case study、recall 表、hybrid 段 | 核心 | 3、5 |
| 2 | [Kimi Linear](https://arxiv.org/abs/2510.26692) | Kimi Team | 已读 §1–4、§5 消融与长上下文、§6 讨论 | 核心 | 2、3、4、5 |
| 3 | [Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) | Kimi Team | 已读摘要、§2.1、§3.4 长上下文、§5.1、§5.4.1–5.4.2 KDA 部分 | 核心 | 4、5、6 |
| 4 | [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320) | Qwen Team | 已读摘要、引言、§GDN Hybrid | 核心 | 3、5 |
| 5 | [Why Did M2 End Up as a Full Attention Model?](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)（2025-10-30） | MiniMax | 已读（WebFetch 摘要） | 核心（反方） | 7 |
| 6 | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | Katharopoulos et al. | 已读 §3.2 linearized attention | 重要 | 1 |
| 7 | [MiniMax-01](https://arxiv.org/abs/2501.08313) | MiniMax | 已读 §2.2 lightning attention 与 hybrid 实验 | 重要 | 4、5、7 |
| 8 | vLLM `kv_cache_interface.py`（`MambaSpec`）、`config/cache.py`（`mamba_cache_mode`、ReplaySSM）、`layers/mamba/mamba_utils.py`、`single_type_kv_cache_manager.py`（`MambaManager`） | vLLM | 已读相关类与函数 | 重要 | 6 |
| 9 | [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)（2026-07-22） | vLLM Team | 已读（WebFetch 摘要） | 重要 | 6 |
| 10 | 官方 config.json：Qwen3-Next-80B-A3B、Qwen3.8-27B、Qwen3.8-Flash-Next、Qwen3.8-2.4T-A95B、Kimi-Linear-48B-A3B、Kimi-K3、MiniMax-Text-01、MiniMax-M3 | 各厂商 | 已读（2026-10-10） | 重要 | 5、6 |
| 11 | [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) | Songlin Yang | 已读（WebFetch 摘要） | 表达参考 | 1、3：key-value 记忆、正交性与容量、“先读旧值再写差值” |
| 12 | [Transformers are SSMs (Mamba-2)](https://arxiv.org/abs/2405.21060) | Dao, Gu | 已读摘要 | 一般 | 2：decay 与 mask 的对偶 |
| 13 | [Repeat After Me](https://arxiv.org/abs/2402.01032) | Jelassi et al. | 已读摘要 | 一般 | 5：固定 state 的复制与检索上限 |
| 14 | [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) | Yang et al. | 已读 §2 背景（并行、递推与 chunkwise 形式）、§3 WY 表示的开头 | 重要 | 4：chunkwise / WY 表示 |
| 15 | [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)、[Mamba-3](https://arxiv.org/abs/2603.15569) | Hatamizadeh et al.；Lahoti et al. | 只读搜索摘要 | 备查 | 现状：研究前沿，尚未进入已发布的大模型 |

## 阅读要点

- **Linear attention（Katharopoulos）**：softmax 换成非负核 $\phi(q)^\top\phi(k)$ 后，按结合律先算 $\sum_j \phi(k_j)v_j^\top$，复杂度从 $O(N^2)$ 到 $O(N)$，可以写成 RNN。指数核的特征映射是无限维的，因此无法精确线性化 softmax。
- **容量问题（Songlin Yang 的讲解）**：state 是 key-value 联想记忆，用 $k_j$ 读取时得到 $v_j$ 加上其他 key 的串扰；$d$ 维空间最多 $d$ 个互相正交的 key。原始 linear attention 只加不删，串扰随长度累积。
- **Gated DeltaNet**：$S_t = \alpha_t(I-\beta_t k_t k_t^\top)S_{t-1} + \beta_t k_t v_t^\top$；decay 负责整体清除，delta rule 先读出 $k_t$ 已关联的值，只写入差值。S-NIAH（1.3B，100B token）：
  - S-NIAH-1（pass-key）8K：DeltaNet 98.8，Mamba2 30.4，GDN 91.8 —— decay 过快会丢失。
  - S-NIAH-2（essay 中的数字）8K：DeltaNet 14.4，Mamba2 17.0，GDN 29.6；4K：18.6 / 56.2 / 92.2 —— 没有遗忘时记忆互相覆盖。
  - S-NIAH-3（UUID）2K：47.0 / 47.6 / 84.2。
  - 真实 recall 任务平均：Transformer++ 37.0，GDN 30.6，Mamba2 29.8；混合 GDN-H1（GDN + SWA）39.0，H2 40.1。
- **KDA（Kimi Linear）**：把 GDN 的标量 $\alpha_t$ 改为逐 channel 的 $\mathrm{Diag}(\alpha_t)$；是 DPLR 的受限形式（$a=\beta k$，$b=k\odot\alpha$），kernel 约为通用 DPLR 的 2 倍速度。chunkwise：chunk 内用矩阵乘（Tensor Core），chunk 间串行传 state。
  - 48B-A3B，1.4T token，3:1 KDA:MLA，MLA 层 NoPE。
  - 混合比例消融（validation PPL）：3:1 5.65，1:1 5.66，7:1 5.70，15:1 5.82，0:1（纯 MLA）5.77。
  - 128K 长上下文平均：Kimi Linear 54.5，MLA 52.2，GDN-H 51.2；RULER 84.3 / 81.3 / 80.5。
  - KV cache 最多减少 75%；1M 上下文 decode 比 MLA 快 6×，prefill 1M 快 2.9×。
- **Kimi K3**：2.8T-A104B，1M 上下文；93 层 = 69 KDA + 24 Gated MLA（每 3 层 KDA 配 1 层 MLA，最后一层再加一层 MLA）；KDA 96 head，head dim 128，short conv 4；MLA 全部 NoPE，位置信息交给 KDA，直接外推到 1M。
  - 下界化的 decay：$g = g_\min\,\mathrm{Sigmoid}(e^A z)$，$g_\min=-5$，16-token tile 的累计 log-decay 在 $(-80, 0)$，倒数缩放在 BF16 范围内，所有 causal tile 都能走 Tensor Core。
  - KDA context parallel：每个 rank 算局部转移矩阵与局部 state，一次固定大小的 all-gather 后前缀扫描恢复入口 state。
  - Prefix cache：KDA state 与 MLA KV 放进同一个分页池；state 很大，只能在稀疏边界存 checkpoint，共享 block size 被迫取 1024–6144 token。解法是分离物理块（6144）与 hash 块（512）：在 512 边界存 checkpoint，命中可以落在物理块内部（例：前 2800 token 相同，命中 B = 2560）。
  - 投机解码：state 原地更新，拒绝 draft 后无法回滚；只缓存 draft token 的投影输入，在片上重放（与 ReplaySSM 同思路）。
- **Qwen3.8-Next（Flash-Next）**：125B-A6B；3 GDN : 1 global attention；continued pretraining 时把 global attention 层换成 QSA（压缩 indexer + micro-block 选择）——线性与稀疏两条路线叠加。
  - 消融（25B-A3B，28 层）：九项平均 full attention 49.87，SWA（窗口 128）混合 51.15，GDN 混合 53.81。
  - full attention 层保留 RoPE：NoPE 在预训练中看不出差别，但 post-training 后“无法终止生成”的比例明显更高 —— 与 Kimi 的 NoPE 选择相反。
  - 输出 gate 改为 sigmoid（原 GDN 为 SiLU）；FlashQLA kernel 相对 FLA Triton 前向 2–3×、反向约 2×。
- **MiniMax**：Text-01 为 7 层 lightning attention + 1 层 softmax，共 80 层（config 中 70 : 10）；论文称纯 lightning 在 NIAH 上明显不足，混合后超过 softmax。M2 博客：混合模型在更大规模上出现“复杂多跳推理”的明显缺陷；线性 attention 理论上在“几千 token”处才比 full attention 便宜；对低精度 state 敏感，prefix cache 与投机解码尚未解决。M3 config 为 GQA（64 q head / 4 KV head）+ 稀疏 block 选择，没有线性层。
- **vLLM**：GDN / KDA 的 state 以 `MambaSpec` 表示；`mamba_cache_mode="align"` 时只在 block_size 整数倍且为 step 末尾的位置保存 state 用于 prefix caching；KDA recurrent state 默认 FP32（可选 BF16）。

## 自己的计算（正文第七章与图 3、图 4）

- 口径：state 按 vLLM 默认的 FP32（`mamba_ssm_dtype: float32`；KDA `kda_state_dtype` auto 为 FP32），KV 按 BF16；conv state 很小（Qwen3.8-27B 共约 2.8 MiB），不计。
- **交叉点的比较对象**：线性层的 state 与“同样这些层改用 full attention 时的 KV”比较。初稿曾拿 48 层 state 与其余 16 层的 KV 比（得到 2.3K / 15.7K），口径不对，已改。
- **Qwen3.8-27B**（64 层 = 48 GDN + 16 GQA；GDN 48 value head、16 key head、head dim 128；GQA 4 KV head、head dim 256）：
  - state：48 × 128 × 128 × 4 B = 3 MiB / 层，48 层 144 MiB。
  - 16 层 GQA 的 KV：16 × 2 × 4 × 256 × 2 B = 64 KiB / token，256K 时 16 GiB；64 层全部 GQA 时 64 GiB。
  - 48 层若用 GQA：192 KiB / token，交叉点 144 MiB / 192 KiB = 768；按 decode 读写量（2 × 144 MiB）为 1536。
- **Kimi K3**（93 层 = 69 KDA + 24 MLA；KDA 96 head、head dim 128；MLA 576 维）：
  - state：96 × 128 × 128 × 4 B = 6 MiB / 层，69 层 414 MiB。
  - 69 层若用 MLA：69 × 576 × 2 B = 77.625 KiB / token，交叉点约 5461；按读写量约 10923。1M 时 77.6 GiB。
- 结论：MLA 每 token 的 KV 已经很小，线性层在更长的上下文下才省显存；与 MiniMax M2 说的“几千 token”同一量级。

## 图

| 占位 | 章 | 内容 | 回答的问题 |
| --- | --- | --- | --- |
| `state-step` | 一 | 一个 head 的写入（S + v kᵀ）与读取（S q），128 × 128，64 KiB / head，Qwen3.8-27B 一层 3 MiB | 线性 attention 的一步是什么 |
| `delta-write` | 三 | 直接相加 / 先 decay / delta rule 三种写入后用 k、k′ 读出的内容（示意值 α = 0.5，β = 1） | decay 与 delta rule 各解决什么 |
| `niah-bars` | 三 | S-NIAH 三个任务、三种模型的条形图，右侧注明缺少的机制 | 两种机制为何互补 |
| `qwen-gdn-module` | 三 | Qwen3.8-Next 报告原图 | 实际 GDN 层的部件 |
| `chunkwise` | 四 | 4K token、C = 64 的 64 个 chunk，放大两个 | 训练与 prefill 为什么能并行 |
| `layer-strips` | 六 | 6 个混合模型按 config 的层排布 | 各模型怎么混合 |
| `hybrid-layers` | 七 | Qwen3.8-27B 64 层在 256K 下每层的缓存 | 混合模型每层缓存什么、多大 |
| `state-vs-kv` | 七 | 双对数：Qwen3.8-27B 与 K3 的 state 对 KV，交叉点 768 / 5.5K | 线性层从多长开始省显存 |
| `k3-prefix-cache` | 七 | K3 报告 Figure 12（裁出图本身） | prefix cache 命中边界 |

## 未写入正文的内容

- K3 KDA context parallel：每个 rank 算本段的转移矩阵与局部 state，一次固定大小的 all-gather 后前缀扫描恢复入口 state（篇幅原因删去）。
- KDA 相对通用 DPLR：少两次二级分块、约三次矩阵乘，kernel 约 2 倍速度。
- Kimi Linear：KV cache 最多减少 75%，1M decode TPOT 为 MLA 的 1/6（batch size 1），prefill 1M 快 2.9×。
- Mamba-3、Gated DeltaNet-2：研究前沿，尚未进入已发布的大模型。

## 分歧与待核实

- full attention 层是否用位置编码：Kimi（NoPE）与 Qwen（RoPE）结论相反。
- 混合比例：Qwen / Kimi 3:1，MiniMax-01 7:1，Kimi 消融中 7:1 验证集变差。
- MiniMax M2 的反对意见与 Kimi、Qwen 的正面结果规模与评测不同，正文只陈述各自的证据。
- K3 serving 时 KV cache 与 state 的实际 dtype 未核实（以上按 BF16 KV / FP32 state 计算）。
- Kimi Linear 的 6× decode 加速为 batch size 1 的 TPOT（论文图注），正文未用。
