# DSpark 文章资料

状态：用户 2026-10-03 提出“做一篇 dspark 的文章”，随后说明“没问题就开始做……先做关键帧和对应的文章”，据此未单独走参考方案确认，直接写了初稿（`content/dspark.md`、`src/articles/dspark/DsparkFigures.tsx`，`status: draft`）与视频关键帧（`videos/dspark/`）。图与关键帧未单独走草图确认，随初稿一并交用户看。

## 范围

- Slug `dspark`；标题“DSpark”，子标题“DeepSeek V4 的 Speculative Decoding：并行生成 draft，按把握验证”。
- 独立成篇：拒绝采样只用一段带过，不依赖站内的 speculative decoding 一文。
- 未做实验。全部数字来自 DSpark 论文，在论述附近标注出处，不写全局说明。可补的实测：4090 上用 vLLM 跑 Qwen3-8B 加 `deepseek-ai/dspark_qwen3_8b_block7`，与已有的 EAGLE-3、draft model 数据对比（需要含 DSpark 的 vLLM 版本，现有 v0.30 环境是否包含未核实）。
- 源码基准：vLLM `e006d76`（2026-09-29，浅克隆，最近的 tag 为 v0.30.0；v0.30.0 是否已含 DSpark 未核实，正文按 commit 引用）；SGLang `8854857`（2026-09-29）。

## 文章主线

MTP-1 每轮只猜 1 个，因为多猜有两个问题：并行起草后面的 token 接不上，固定长度的验证在高并发下浪费算力。DSpark 用 Markov head 解决前者，用 confidence head 加调度器解决后者。按“MTP-1 原理与只猜 1 个的原因 → 多猜的代价（起草与验证） → 并行起草的拼接错误 → Markov head → 每个 token 被收下的概率 → 按负载决定验几个 → 工程实现 → 线上结果 → 支持情况与局限”展开。

## 参考清单

| 资料 | 作者 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| [DSpark](https://arxiv.org/abs/2607.05147) | Cheng et al.，DeepSeek-AI 与北京大学 | arXiv 源文件包的 `.tex` 全文已读（含附录的反例）；图 2、7、8 已看 | 核心 | 全文 |
| vLLM `dspark/speculator.py`、`adaptive_verification.py` | vLLM 团队 | 两个文件全文已读；`dspark/utils.py` 已读；`models/deepseek_v4/nvidia/dspark.py` 只读了文件头；`config/speculative.py` 只 grep 了相关字段 | 核心（现状） | 六、八 |
| vLLM 文档 `adaptive_verification.md`、speculative decoding README 的方法表 | vLLM 团队 | 前者全文已读，后者只读了相关行 | 核心（现状） | 六、八 |
| SGLang `kernels/ops/speculative/dspark/`、`benchmark/dspark_sps_profiler.py`、cookbook（Kimi-K3、Qwen3.8-27B、MiniCPM5-2B） | SGLang 团队 | 只读了 `dspark_schedule.py` 开头；cookbook 已核对 Kimi-K3 与 Qwen3.8-27B 的 drafter 均为 RadixArk 训练（`RadixArk/Kimi-K3-DSpark`、`RadixArk/Qwen3.8-27B-DSpark`），Kimi-K3 页面的 DSPARK 速度数字用模拟接受长度得到 | 补充（相关系统） | 八 |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec) README | DeepSeek-AI | 通过 WebFetch 的摘要间接读，未读原文 | 补充 | 八 |
| [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) model card | DeepSeek-AI | 未读（抓取失败）；只引用其存在，链接取自论文 | 补充 | 八 |
| [DFlash](https://arxiv.org/abs/2602.06036)、[EAGLE-3](https://arxiv.org/abs/2503.01840) | Chen et al.；Li et al. | 未读原文，只通过 DSpark 论文的 §2.2 了解 | 背景 | 二 |

## 现状变化（论文与源码的差别）

- 论文算法 1 在吞吐第一次下降时停止；线上与 vLLM 都改为“用旧 confidence 定预算、用当前存活概率做 top-k”。vLLM 保留两份拷回 CPU 的 confidence（`_stale_confidences`），与论文“两步之前”一致。
- vLLM 的目标函数是“期望接受数 ÷（draft 耗时 + verify 耗时）”，比论文的 $\tau\cdot\mathrm{SPS}(B)$ 多算了 drafter 的耗时。代价表在 CUDA graph 捕获的尺寸之间是阶梯函数，超过捕获上限后线性插值。
- vLLM 的 DSpark 还有论文没提的 `dspark_draft_topk`（只对 base logits 的 top-k 加 Markov 偏置）和缩小的 draft 词表，正文未写。
- adaptive verification 目前只支持带 confidence head 的 DSpark；要求完整 CUDA graph，不支持 LoRA 与 PP（vLLM 文档）。

## 相关系统与不同做法

- vLLM 另有 dynamic speculative decoding（按 batch size 配 `num_speculative_tokens`），不依赖逐 token 的 confidence。只读了文档表格中的一行，正文未展开。
- 论文 related work 列出的 TurboSpec、SpecDec++、Tetris 等负载感知或阈值方案均未读，正文只用“静态阈值”一句概括其共同局限，这是论文的说法。
- 未找到公开的反对意见或第三方复现；论文自己承认的局限是起草成本固定（§5.4 Limitations），已写入第八节。

## 待核实

- 第三节“DSpark 聊天各位置在 0.73 到 0.77 之间”、DFlash 第 3 位 0.64 是按论文图读数（2026-10-06 渲染 `figs/position_cond_accept.pdf` 重读），论文正文无数字；K07、K12 的曲线同源。
- 正文把 MTP-1 的验证预算写作“每轮验 2 个 token，其中 1 个是 draft”，依据是论文“MTP-1's static 2 tokens”与图 8 下排，含 target 自己的输出位。
- 术语表规定平均接受长度记为 $L$；本文沿用论文的 $\tau$，因为 $L$ 在论文公式里是延迟。是否改术语表待用户定。
- 自绘图未做小红书的 compact 布局。

## 核心图

| 图 | 回答的问题 | 实现 |
| --- | --- | --- |
| 图 0 `::dspark-mtp::` | MTP-1 省在哪：同样 4 次 forward，正常 decode 出 4 个、MTP-1 出 7 个 | 自绘，示意（2026-10-06 加，粗稿待确认） |
| 图 1 `::dspark-cycle::` | 一轮里四个部件各做什么 | 自绘，γ = 4 的示例路径 |
| 图 2 `::dspark-markov::` | 并行起草为什么会拼错，Markov head 改了什么 | 自绘，示意概率 |
| 图 3 `::dspark-position::` | 三种 drafter 各位置的接受率 | 论文原图 |
| 图 4 `::dspark-schedule::` | 同样的请求在空闲与繁忙时各验几个 | 自绘，示例由代码按贪心算出 |
| 图 5 `::dspark-online::` | 线上的吞吐与每用户速度 | 论文原图 |
| 图 6 `::dspark-budget::` | 验证长度随并发怎样变化 | 论文原图（`figs/online_service_tradeoff.pdf`） |

## 2026-10-06 修订

- 开头：不再说 MTP-1 只猜 1 个是“因为”两个问题；表格第一行补上自回归起草耗时随长度增长。
- 第三节：补层数比较的条件（论文 §Drafter Depth：Qwen3-4B、block size 7、三类任务都超过）；主表中 DFlash、DSpark 为 5 层，EAGLE-3 为 1 层（§exp setup）。
- 第六节“不能偷看”重写因果链。
- 第七节：并发阈值按论文分别写 200（V4-Flash）与 150（V4-Pro）；加入验证长度随并发变化的原图，V4-Flash 在 200 并发约 3.5 个为读图数。
- de-ai-edit：subagent 审读命中 9 处（两个章节标题、“不能偷看”等口语，一处回指第二节的路标句），用户确认后全部改入。
- 开头重写为两段（MTP-1 的原理与只猜 1 个的原因 → DSpark 的改动与线上结果），删去问题表格；加图 0 解释 decode 读权重为主、多验 1 个 token 几乎免费。
- de-ai-edit 第二遍：命中 3 处口语（第二节两处、第六节一处），用户确认后改入。
- 2026-10-06 按用户要求整体重写行文：每节先说要解决的问题再给做法；第一节把起草与验证两项代价并列，作为后文两条线的起点。数字、公式、图与代码未变。改前版本未入库。
- 2026-10-06 第二次理顺行文逻辑（用户要求“优化行文逻辑”）：开头补上“多验几乎免费只在 batch 小时成立”，连接 MTP-1 原理与只起草 1 个的原因；摘要数字注明为 V4-Flash。第一节改为“第 k 个只有前 k−1 个都被接受才被接受”的因果，验证代价写明 batch 小与接近算力上限两种情形。第二节就地说明 EAGLE-3 是自回归 drafter；第三节以“兼取两者”承接第二节的取舍，RNN 对照移到 Markov head 定义旁。第四节改写开头的论证（原句用聊天与数学的平均接受长度差来说明“靠后易被拒”，因果不成立），补 0.75⁵ ≈ 0.24 的自算示例，并说明需要 confidence head 是因为起草时 target 分布未知。第五节补“加一个 token 时 τ 与 SPS 的变化”；第六节写明逐个加入为何无偏（a_k 只依赖前 k−1 个 token）、两步之前的 confidence 是为了不让 GPU 等 CPU 拷回。全文“猜 / 收下”统一为“起草 / 接受”；第四、五节标题改为“Confidence head”“按负载调度验证长度”。数字、公式、图与代码未变；图 0 的图注仍用“猜”。改前版本存于会话 scratchpad，未入库。
- de-ai-edit（第二次理顺后）：subagent 命中 3 处口语（第六节两处、第八节一处），用户确认后改入。
- 2026-10-06 用户认为“起草”不自然：全文动词改为“生成 draft / draft token”，与 vLLM speculative decoding 一文的“猜”“draft token”保持一致。
- 2026-10-07 按视频 v7 配音稿（`videos/dspark/script.md`）调整文章逻辑：开头先给定义与结果，MTP-1 的原理与图 0 移入第一节；第一节由“两项代价”改为“每个 token 的耗时 → 三种途径（draft 更快、接受更多、减少无效验证）→ 四个部件”；第二节按“自回归 → 并行（mask、一次 forward）→ 独立采样的代价”展开；第三节按“缺的信息 → 偏置表 → 低秩分解 → 效果”；原第四节拆为“验证的代价与存活概率”与“Confidence head”，顺序与视频第 9–11 幕一致；线上结果末尾回到三种途径。新增内容：mask 的说明、Qwen3 词表 151,936、存活概率示例 0.80/0.70/0.65/0.60（示例数值，取自视频第 10 幕，替换原 0.75⁵ ≈ 0.24）。视频中的“参数为完整矩阵的 0.40%”“第 2 位 0.65”未写入正文（前者未核对，后者正文沿用第 3 位 0.64 的读图数）。未重跑 de-ai-edit。
- 2026-10-07 自绘图改用视频的视觉语言（用户认为视频的图更好；粗稿，待确认）：上文 ink、drafter 与 draft token purple、target 与送去验证的 token blue、接受 teal、拒绝 red、不验证或丢弃为斜纹；示例由 of course / no problem 改为视频的“当然可以 / 没问题”，正文同步。图 0 改为“一轮的时间条 + 接受与拒绝两种结果”（原为 4 次 forward 的对比）；图 1 改为 5 个 draft token、验 3 个、接受 2 个；新增 `::dspark-draft::`（自回归 1 层 × 5 次与并行 5 层 × 1 次的时间条，第 1 位 0.53 对 0.72）；Markov 图加偏置表的一行；调度图加排序队列与两条吞吐曲线（曲线各自缩放到自身范围）。`figure-kit` 新增 `ink`、`void` 两种填充。视频第 10、11 幕（存活概率柱、分布重叠）未单独成图。词表大小改为 DeepSeek-V4-Flash 的 129,280（视频取自 checkpoint config，本次未重新核对；此前误写为 Qwen3 的 151,936）。改前的图存于会话 scratchpad，未入库。
