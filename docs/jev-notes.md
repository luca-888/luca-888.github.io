# Jev 文章资料

状态：用户 2026-10-03 确认方案并提供 Jev API key。正文与 4 张图已写完（`content/jev.md`、`src/articles/jev/JevFigures.tsx`，`status: draft`）。用户 2026-10-03 决定：不做 Kev 的硬件实测，只研究 Jev；原第九节“实测”已删除。图未单独走草图确认，随初稿一并交用户看。

## 范围

- 用户 2026-10-03 指定方向：机制拆解或逆向，不做“是什么 + 教程”。
- 暂定 slug `jev`；暂定标题“Jev：不生成文本的模型怎么做决定”。
- Jev 本体闭源（无论文、无权重）。文章把三类信息分开写：TypeSafe 官方说法、外部黑盒观测（Archer Hume 的逆向）、开源复刻 Kev 的源码。机制细节以 Kev 源码为准，并注明“这是 Kev 的做法，Jev 是否相同未公开”。
- 源码基准：Kev `84847f0`（2026-10-01，Kev 1.0）；system-one-adapter-python `e1d4cc9`（2026-09-22）。本地浅克隆在 `~/Documents/ChatGPT/reference-repos/{kev,system-one-adapter-python}`。

## 文章主线

LLM 做一次分类要串行生成一段 JSON；Jev 把问题改写成“在给定选项上读一个分布”，于是不需要 decode。用一个小例子（一条工单 + 三个问题）贯穿：先看 token 怎么排成一条序列，再看 attention mask 怎么让多个问题共享 state 又互不可见，再看 pointer head 怎么从 hidden state 读出概率，然后用黑盒观测对照 Jev 的行为，最后是 calibration 与能力边界。

## 章节提纲

1. 输入与输出：state、三类问题（Choice / Score / Noul）、返回的概率。
2. 用 LLM 做同一件事：生成 JSON 的串行 decode 与 output token 成本；logprobs 读法的限制。
3. 一条序列装下多个问题：`<state> … <q> … <opt> … </opt> <decide>` 布局、block-causal mask、每个问题的 position ID 从 state 末尾重新开始（图 1）。
4. 读出概率：`<decide>` 与各 `</opt>` 的 hidden state 经两个线性层做点积，softmax 得分布；option 之间互相可见，所以顺序会影响结果（图 2）。
5. Jev 的黑盒证据：token 计数可加、问题互相隔离、延迟随问题数的变化、option 顺序敏感、fake delimiter 无效（图 3）。
6. state 只算一次：prefix KV cache；Qwen3.5 的 Gated DeltaNet 层不认 mask，改成“每个问题一行、共享 state cache”。
7. Calibration：temperature；RLCD 官方只给了名字，Kev 用 cross-entropy。
8. 边界：算术与日期、知识量由底座决定、“不会 hallucinate”的实际含义、与 zero-shot classifier（NLI、GLiNER）的关系。
9. 实测（RTX 4090）。

## 核心图（待出草图）

| 图 | 回答的问题 |
| --- | --- |
| 图 2 attention mask 矩阵 | 多个问题在一条序列里，谁能看见谁 |
| 图 3 pointer head | 概率是从哪些位置的 hidden state 读出来的 |
| 图 4 耗时曲线 | state 变长、问题变多时，Jev 服务端耗时怎么变（Hume 的数据） |
| 图 1 两条路径对比 | 同一份答案，生成 JSON 与直接读出各要多少次 forward（按步数画，不是实测耗时） |

## 实验

- 不做 Kev 的硬件实测（用户 2026-10-03 决定）。Kev 只作为解释机制的源码参照。
- Jev API 黑盒探测已完成，见“Jev API 探测”一节。key 存于 `.env.local` 的 `TYPESAFE_API_KEY`（已被 `.gitignore` 忽略）。

## 现状变化

| 项 | 早期 / 文档描述 | 当前 |
| --- | --- | --- |
| Kev 的 packed mask | README 的“一条序列 + block-causal mask”只适用于 attention-only 底座（Qwen3） | 现行 Kev 全部基于 Qwen3.5 / 3.8（含 Gated DeltaNet），实际走 row 形式：每个问题一行，state 经 cache 复用（`model.py: rows_form`、`_branch_rows_from_prefix`） |
| Kev-27B | 首版为 LoRA adapter | v2（2026-09-30）为全量权重 fine-tune，再与 adapter 版按 0.85 : 0.15 平均 |
| Jev 版本 | 发布时 | 文档当前为 `jev-1.13`（jaggedness 页 2026-10-02 更新） |

## 相关系统与反对意见

- “只是 classifier”：r/singularity 高赞评论、KDnuggets、Swapnil Talekar 的 Substack。对照对象：NLI zero-shot、GLiNER、LLM logprobs。
- “不会 hallucinate”：官方自己注明“not empirical”，指 schema 必然匹配，不是答案必然正确。
- 先例之争：HN 上 nandakishor_ml 称 2025-03 已开源同类架构（arXiv 2503.23303）；回复指出那是单任务 classifier，不是任意问题的 zero-shot。
- 官方 benchmark：workflow 由自家团队设计，官方承认偏向高端；Kev 维护者以“gold 是两个闭源模型答案的平均”为由移除了 TypeSafe 公开 evals。

## 参考清单（按重要程度）

| 来源 | 作者 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| Kev 源码 `kev/model.py`、`kev/shared_prefix.py`，README “How It Works / Training / Benchmarks / Serving Performance / Limitations” | Jared Palmer | `model.py`、`shared_prefix.py` 全读；README 200–419 行已读；`serve.py`、`cuda_graphs.py`、`train.py`、`calibrate.py` 未读 | 核心 | 3、4、6、7、9 |
| [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)（2026-09-17） | Archer Hume | 原文全文已读（抓取 HTML 转文本）；evidence.json 未读 | 核心 | 5、6 |
| [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | TypeSafe AI | 经 WebFetch 摘要阅读 | 核心 | 1、7、8 |
| [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) | TypeSafe AI | 经 WebFetch 摘要阅读 | 重点 | 7 |
| TypeSafe 文档：[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)、[Models](https://docs.typesafe.ai/models)、[API reference](https://docs.typesafe.ai/api)、[System One](https://docs.typesafe.ai/concepts/system-one) | TypeSafe AI | 四页 `.md` 原文全读；confidence、primitives、cookbooks 已下载未读 | 核心 | 1、3、6 |
| [Laya](https://github.com/NandhaKishorM/laya) README、[AnyJev](https://github.com/nokia-applied-research/AnyJev) README | Nandha Kishor M；Nokia Applied Research | 各读了架构、checkpoint 表与结果表相关段落，未通读 | 重点 | 8 |
| [Jev Architecture Explained](https://agentunicorn.ai/research/jev-architecture-open-source)、[Jev's Open Rivals](https://trilogyai.substack.com/p/jev-open-decision-models) | — | 经 WebFetch 摘要阅读；只作开源生态的线索，正文未引用其数字 | 一般 | 8 |
| Kev `PLAN.md` “What we have learned”（180–273 行） | Jared Palmer | 该段已读，其余 3000 行未读 | 重点 | 7、8 |
| [Jev introduces a new shape of LLM](https://simonwillison.net/2026/Sep/21/jev/) | Simon Willison | 经 WebFetch 摘要阅读 | 重点 | 1、8 |
| system-one-adapter-python README | TypeSafe AI | README 前 80 行已读；源码未读 | 一般 | 2 |
| [HN 讨论](https://news.ycombinator.com/item?id=49736660)、[Jev and the Return of the Classifiers](https://swapniltalekar.substack.com/p/jev-and-the-return-of-the-classifiers)、[KDnuggets](https://www.kdnuggets.com/what-everyone-is-getting-wrong-about-typesafe-ais-jev) | 多人 | 经 WebFetch 摘要阅读 | 一般 | 8 |
| [Wikipedia: Jev (AI model)](https://en.wikipedia.org/wiki/Jev_(AI_model)) | — | 经 WebFetch 摘要阅读；只作线索 | 一般 | — |
| GLiNER、NLI zero-shot classification 论文 | — | 未读 | 一般 | 8 |
| Qwen3.5 架构（Gated DeltaNet） | Qwen 团队 | 未读 | 一般 | 6 |

## 阅读要点

- **布局与 mask（Kev `model.py`）**：`encode` 生成 `[<state> …]` 后接每个问题的 `[<q> instr <opt> o </opt> … <decide>]`；`seg` 标记 state 为 0、问题 k 为 k；`pos` 在每个问题处从 `len(state)` 重新开始。`branch_mask_batch`：`attend(i,j)` 当且仅当 `j<=i` 且（`seg[j]==0` 或 `seg[j]==seg[i]`）。delimiter 复用 Qwen 的 5 个冷门 special token，不新增 embedding；用户文本里的 `<|name|>` 在 tokenize 前被改写，所以 option 边界伪造不了。
- **Pointer head**：`q = Linear(d, 256)` 作用在 `<decide>`，`k = Linear(d, 256)` 作用在各 `</opt>`，logit = `k·q / sqrt(256)`；eval 时除以 temperature。backbone 不带 vocab head。
- **Row 形式**：hybrid 底座无法遵守 block-causal mask，每个问题成为一行 `state + branch`；serving 时 state 先单独跑一遍得到 cache，各 branch 在 cache 上续跑。一次 forward 的 token 预算 16,384。packed 与 row 在 fp32 下相差 4e-6 以内。
- **训练**：cross-entropy；rank-16 LoRA + head 一起训；`decision-v7` 共 12,576 条。未使用 Jev 输出。PLAN 第 13 条：calibration loss（label smoothing、CE + Brier、focal）相对 CE 没有收益；RL 未尝试，理由是对 proper score 做 REINFORCE 与 log loss 最优解相同。
- **Calibration**：单一 temperature（Kev-4B 2.41、Kev-9B 2.19、Kev-27B 1.32）。Kev-9B 在 new sources 上 ECE 0.103 → 0.041。temperature 不改变排序，所以 5% 错误预算下可自动化的比例仍低于 Jev（0.52–0.69 对 0.70）。
- **知识由底座决定**：MMLU-Pro，未训练的 Qwen3.8-27B 0.635，Kev-27B 0.665，Jev 0.840。
- **Kev serving（README 表）**：Kev-4B 在 H100 上 6 个问题 + 短文本 18.1 ms，文本命中 cache 时 12.9 ms。
- **逆向文章（经摘要）**：1 个问题 268 tokens、2 个问题 276 tokens；state 相同时 1 / 100 / 1,500 个问题的服务端耗时中位数 86.5 / 82 / 610 ms；单 branch 上限约 32,768、整个请求约 65,536 tokens；secret 放在兄弟问题里检出概率 0.00，放在 state 里 0.90–0.92；加入无关 option 后已有两项的 log-odds 由 +0.38 变为 +0.11；tokenizer 与 Qwen 在 415 个探针中 348 个一致，与 192 个公开 tokenizer 均不完全相同；MMLU 1,200 题 ECE 0.0313；约 30k tokens 用时约 160 ms，作者据此推测约 10B active 参数的 MoE（推测）。
- **官方（经摘要）**：parallel sampler、RLCD、端到端 70–500 ms、input $0.042/MTok、output 免费、cardinality 上限 255、训练数据来源未说明。

## 待核实问题

1. 逆向文章的全部数字回原文与 evidence.json 核对；作者 repo 里是否有可复用的探测脚本。
2. Jev 的 output token 计数公式（4 + 15/答案 + id 长度）与官方 adapter 是否一致。
3. Qwen3.5 Gated DeltaNet 为什么不认 attention mask，Kev 的 cache 里除 KV 外还存了什么（conv window、recurrent state）。
5. RLCD 是否有任何进一步的公开信息（访谈、播客）。
6. GLiNER / NLI zero-shot 与 pointer head 的具体差别。
7. 已核实：“router 按请求选 checkpoint”是 Laya 的功能，不是 Jev 的；Jev 官方文档写明同一套权重服务所有账号。

## Jev API 探测（2026-10-03，jev-1.13.0）

脚本 `scripts/jev/probe.py`，原始结果 `public/measurements/jev/jev-probes.json`，正文用到的子集在 `src/data/jev.json`。响应头已不再带 `x-envoy-upstream-service-time`，本机只能测往返耗时（经代理，约 600–700 ms 起步），所以延迟曲线引用 Hume 的服务端数据。

| 探测 | 结果 |
| --- | --- |
| 示例请求（工单 + 3 个问题） | `payments` 1.00；`urgent` 0.87；`frustration` 1.09（0.00 / 0.91 / 0.09）；input 410、output 70 tokens |
| token 计数 | 同一 Noul 重复 1 / 2 / 3 / 5 / 10 遍：298 / 311 / 324 / 350 / 415；state ×20：792 / 805 / 909。每个问题 +13，与 state 长度无关 |
| 问题隔离 | 暗号写在兄弟问题里：0.00（5/5）；写在 state 里：1.00（5/5） |
| 选项数与耗时 | 2 / 20 / 100 / 200 个选项：output_tokens 33 / 204 / 1,004 / 2,103；往返中位数 665 / 744 / 692 / 677 ms |
| 选项顺序 | 归属不明确的工单、三个选项六种排列：`payments` 0.72–0.86。第一版用的工单结果饱和（全 1.00），已弃用 |
| 否定 | “生气吗” 0.83，“没有生气吗” 0.24，合计 1.07 |
| 往返耗时（仅存档） | 问题数 1 → 1,500：693 → 1,624 ms；state 300 → 26,550 tokens：720 → 1,339 ms。含网络与上传，不进正文 |

## 与提纲的出入

- 原第 2 节“logprobs 读法的限制”并入第八节的路线对比。
- 原第 6 节 Gated DeltaNet 与 row 形式未写进正文：属于 Kev 适配 Qwen3.5 的实现细节，与 Jev 的机制无关。
- 原“生成 JSON 对比读出概率的实测耗时”一图改为按步数画的示意（图 1），不再做实测对比。

## 视频（2026-10-03）

- 成片 `videos/jev/build/jev.mp4`（2026-10-07 整理后只保留 1.1 倍速版，原速版 `jev-1080x1920.mp4` 已删）：竖版 9:16，60 fps，约 59 秒，配音 + 字幕 + 音效 + 8-bit 背景音乐。源文件 `videos/jev/film.html`，渲染 `film.mjs`，配音与时间轴 `vo.py`（稿子在 `narration.json`），背景音乐 `bgm.py`。
- 形式：一局连续的回合制对战（原创像素角色，不使用任何现成游戏的素材），机制讲解穿插在回合之间。用户反馈定下的方向：不要纯方块、少放测试数据、竖版、顶部保留 Jev 主标题、口播最后加。
- 画面里的概率全部是 jev-1.13.0 对各回合局面的实际返回（`public/measurements/jev/jev-probes.json` 的 `battle` 字段）；伤害数值与对手出招是脚本设定的游戏规则。
- “不会算”一幕：先试了四组“选哪一招能打倒”的选择题，Jev 全部选对，未采用；改问是非题“这一击能打倒吗”（实际 28 < 30），返回 0.64，同一问法在 20 / 25 / 30 / 40 血时为 0.76 / 0.68 / 0.64 / 0.54。
- 结尾“它算 AI 还是分类器”：选项顺序为（AI，分类器）时返回 0.33 / 0.67，对调后为 0.45 / 0.55；片中用前者。
- 旧稿：`film-v1.html`（横版纯方块）、`film-h.html`（横版对战）、`film-v3.html`（竖版，配音前）。
