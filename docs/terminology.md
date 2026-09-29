# 术语与缩写规范

本文件是博客术语的统一维护入口，记录希望固定使用的英文缩写与专业名称。正文保持自然中文，已收录的概念采用下表首选写法。词表随文章和用户指定的用词逐步补充，不追求收录所有专业词汇。

## 使用规则

- 默认读者熟悉领域内的常见术语，直接使用首选英文或缩写，不附中文译名，也不强制首次展开英文全称。只有非常难懂、不常见或容易误解的概念，才就近补充一次简短中文解释。文章核心术语需要建立简称对应时，可使用“英文全称（缩写）”。
- 表格、图例与自绘图沿用统一术语，优先保证视觉清晰与扫读效率，不堆叠括注或双语标签；大小写、连字符与产品名称保持一致。少见术语的必要解释放在就近正文或图注中。
- 只有通行且含义明确的缩写才进入词表，不为缩短文字临时创造缩写。没有稳定缩写的术语保留专业英文或自然中文。
- 耗时、显存、吞吐、序列长度、梯度累积、计算图和算子等表达使用自然中文；英文主要用于固定术语、缩写及与代码直接对应的概念。
- 同一概念不随意交替使用中文译名、英文全称和多个简称。解释概念时可以使用中文，但指代既定技术名称时以首选写法为准。
- 缩写存在歧义时，就近说明语境。不同概念即使相关也不能通过文字替换合并，例如 gradient checkpointing 与保存训练 checkpoint。
- 源码标识符、API 名称、命令、直接引文和官方原图保留原文。介绍外部资料的命名差异时可以使用其原名，并说明对应关系。
- 修改文章时统一本次涉及的内容；整站批量统一作为单独任务处理，避免误改引用和代码。

## 术语表

以下条目用于正文、表格、图例与文章摘要。“全称或含义”列供内部维护参考，不自动作为正文括注。

| 首选写法 | 全称或含义 | 首次引入与适用范围 |
| --- | --- | --- |
| Vllm / vLLM | vLLM 推理框架 | 用户指定：十篇 vLLM 主题文章的主标题统一写作“Vllm”，方向单列为子标题；正文、技术图、源码标识符及外部资料名称继续使用官方写法 vLLM。 |
| MLE | Maximum Likelihood Estimation | 用户指定：首次引入写作“Maximum Likelihood Estimation（MLE）”，后文与标题使用 MLE；不再使用“最大似然”作为正文主称。 |
| NLL | Negative Log-Likelihood | 基础介绍首次写作“negative log-likelihood（NLL）”，后文使用 NLL；明确使用求和或平均。 |
| CE | Cross-Entropy | 基础介绍首次写作“cross-entropy（CE）”，标题可用“交叉熵”；正文数学记法 H(p,q) 的第一项为平均权重分布。 |
| KL / KL 散度 | Kullback–Leibler divergence | 正文可简称 KL，明确写出分布顺序；不把 forward / reverse 的名称与固定参数角色脱离使用。 |
| policy | RL 中给定状态的动作概率分布；在 LLM 中，由模型参数决定的上下文到 next token 概率分布的函数 | 用户指定：RL 语境统一使用 policy，不与“策略”交替使用。首次就近说明其与模型参数、next token probability 和 log-prob 的关系；标题与句首可写 Policy。普通语境中的“策略”（如自动重计算策略）不受此约定影响。 |
| policy gradient | 通过 reward 加权的 log-prob 梯度优化 policy 的方法 | 直接使用 policy gradient，不写“策略梯度”；涉及 advantage、概率比等具体形式时按对应公式解释。 |
| reward | 对模型生成结果的评分或反馈信号 | 用户指定：RL 语境统一使用 reward，不与“奖励”交替使用；标题与句首可写 Reward。GRPO 中区分原始 reward 与由组内比较得到的 advantage，保留“组内比较”“组内均值”等自然中文。 |
| advantage | 相对于 baseline 的表现，用作 policy 更新的权重 | 直接使用 advantage；GRPO 的 outcome reward 语境中按文中公式说明组内中心化与标准差归一化，不与原始 reward 混称。 |
| GC | Gradient Checkpointing | 首次写作“Gradient Checkpointing（GC）”；用于训练中的 activation 重计算。在垃圾回收语境中重新定义，不能沿用此含义。 |
| AC | Activation Checkpointing | 首次写作“Activation Checkpointing（AC）”；GC 文章以 GC 为主称，介绍 PyTorch 等资料中的 AC 命名时说明两者在该语境下指同类技术，不来回换称。 |
| SAC | Selective Activation Checkpointing | 首次写作“Selective Activation Checkpointing（SAC）”；后文固定使用 SAC。Selective 表示选择性，不将 SAC 当作自动选择策略的简称。 |
| LoRA | Low-Rank Adaptation | 直接使用 LoRA，无需展开或附中文；保留大小写，不写作 LORA 或 lora，源码标识符除外。 |
| RMSNorm | Root Mean Square Normalization | 直接使用 RMSNorm，无需展开或附中文。 |
| SDPA | Scaled Dot-Product Attention | 直接使用 SDPA，无需展开或附中文；讨论具体接口时保留 `scaled_dot_product_attention`。 |
| BF16 | bfloat16 | 无需展开或附中文；正文、表格与图中统一使用 BF16，代码保留 `torch.bfloat16` 等实际标识符。 |
| FP32 | 32-bit floating point | 无需展开或附中文；正文、表格与图中统一使用 FP32，代码保留 `torch.float32` 等实际标识符。 |
| reentrant / non-reentrant | PyTorch checkpoint 的两种实现方式 | 正文保留英文，不附“可重入 / 非重入”等翻译；需要时直接解释机制，不另造缩写。标题或图例首字母可大写，参数保持 `use_reentrant`。 |
| forward / backward | 前向计算 / 反向传播 | 描述模型执行阶段与 autograd 行为时优先使用英文，不另造缩写；解释含义时可用中文。 |
| activation | 激活，即模型计算中的中间结果 | 训练显存与 GC 语境中优先使用 activation；讨论 activation function 时明确写出完整名称或“激活函数”。 |
| tensor | 张量 | 讨论 PyTorch 对象、保存与恢复行为时优先使用 tensor；数学概念的解释可用“张量”，代码保留 `Tensor` 等实际标识符。 |
| checkpoint / training checkpoint | 重计算边界 / 用于恢复训练的状态存档 | GC 语境保留 checkpoint；需要区分磁盘存档时使用 training checkpoint，不笼统混称“检查点”。 |
| rematerialization | 通过重新计算恢复中间结果 | 保留英文，不使用“重物化”等直译；解释具体动作时使用“重计算”或“重算”。 |
| 自动重计算策略 | 由编译器等自动决定保存与重算的策略 | 使用这一中文描述，不另造英文缩写，不与 SAC 混称。 |
| memory budget | 自动重计算策略的相对显存预算 | 讨论 `activation_memory_budget` 时使用 memory budget；后续可简称“预算”。标题或图例首字母可大写，不将其写成全局显存硬上限。 |
| saved tensors | autograd 为 backward 保存的 tensor | 讨论保存与恢复机制时使用 saved tensors；单个对象用 saved tensor，API 与源码名称保持原样。 |
| microbatch / microbatch size | 一次 forward + backward 处理的样本批次 / 该批次的样本数 | 指批次时使用 microbatch，给出大小时使用 microbatch size，例如“microbatch size 为 4”；序列长度直接用中文，不混写 sequence。 |
| kernel | GPU 上执行的计算程序 | GPU 执行语境中保留 kernel；ATen 算子仍称“算子”，不将一次算子调用与一次 kernel 执行混为一谈。 |
| early stop | 所需 saved tensors 恢复齐全后提前停止重算 | 描述 non-reentrant 的该机制时使用 early stop；解释动作可以说“提前停止重算”。 |
| params | parameters，模型参数 | 在训练显存组成、表格与图例中直接使用 params，不附中文。描述“冻结参数”“可训练参数”等动作或属性时可保留中文。 |
| grads | gradients，梯度 | 在训练显存组成中直接使用 grads，默认由上下文明确指参数梯度，不附中文；描述梯度传播、输入梯度与数学推导时保留准确的限定，不将所有梯度统称为参数梯度。 |
| optimizer states | 优化器状态 | 讨论训练显存与状态存储时直接使用完整英文，不附中文；不自造 OS 等缩写。 |

## 维护方式

新增条目时记录首选写法、全称或含义，以及需要说明的语境。用户明确指定的写法优先；确定新的写法后更新原条目，不并列维护相互冲突的规则。对尚未确定的候选词，在讨论中确认后再纳入固定词表。
