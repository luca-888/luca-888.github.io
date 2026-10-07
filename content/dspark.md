DSpark 是 DeepSeek-V4 线上使用的 speculative decoding 方案，替换原来的 MTP-1。主模型（target）与输出分布不变，改动只在 drafter：每轮生成的 draft token 由 1 个增加到最多 5 个，并按负载决定每个请求验证其中几个。V4-Flash 上线后，吞吐相同时每个用户的生成速度快 60% 到 85%。[DSpark, 2026](https://arxiv.org/abs/2607.05147)

## 一、每个 token 的耗时

MTP-1 的一轮有两步：target 上的一层 MTP 模块作为 drafter，用 target 的 hidden state 生成 1 个 draft token；target 在下一次 forward 中验证。这次 forward 同时给出再下一位的分布，所以 draft 被接受时一轮产出 2 个 token，被拒绝时只有 target 在该位补的 1 个。decode 的耗时主要在从显存读权重，多验 1 个 token 几乎不增加耗时。

::dspark-mtp::

drafter 生成 $\gamma$ 个 draft token 时，target 用一次 forward 从左到右逐位做拒绝采样，遇到第一个被拒的位置就停，并在该位补一个自己的 token，输出分布与 target 单独生成相同。第 $k$ 个 token 只有在前 $k-1$ 个都被接受时才会被接受，所以越靠后的 token 越难被接受。

一轮产出的 token 数（含 target 补的一个）的平均值称为平均接受长度 $\tau$，每个 token 的平均耗时为

$$
\frac{T_\text{draft} + T_\text{verify}}{\tau}
$$

提速有三种途径，对应式中的三项：

- **draft 更快**：降低 $T_\text{draft}$。
- **接受更多**：提高 $\tau$。
- **减少无效验证**：每个 draft token 在 target 的 batch 中占一个位置，被拒的 token 占用的 $T_\text{verify}$ 没有产出。

第三项决定了 MTP-1 的 $\gamma$。多验 token 几乎不增加耗时，只在 batch 较小时成立；并发高时 target 的算力接近饱和，固定多验的 token 一旦被拒，占用的是其他请求的算力，总吞吐下降，所以线上每轮只生成 1 个 draft token。

DSpark 的一轮有四个部件：并行的 backbone 一次 forward 给出各位置的预测，使 $T_\text{draft}$ 不随 $\gamma$ 增长；Markov head 逐位采样，提高 $\tau$；confidence head 估计每个 draft token 被接受的概率，调度器按负载选出送去验证的 token，减少无效验证。

::dspark-cycle::

## 二、自回归与并行生成

前面各位都被接受时第 $k$ 位被接受的比例，称为第 $k$ 位的条件接受率。

自回归 drafter（如 MTP、EAGLE-3）逐个生成 draft token，生成 $\gamma$ 个要运行 $\gamma$ 次，$T_\text{draft}$ 随 $\gamma$ 线性增长。为了控制这部分耗时，drafter 只能做得很小，预测不够准：聊天任务上，EAGLE-3 第 1 位的条件接受率为 0.53（target 为 Qwen3-4B）。

并行 drafter（[DFlash](https://arxiv.org/abs/2602.06036)）在上一个 token 之后接 $\gamma$ 个 mask，mask 是占位 token，表示该位置等待预测；所有位置一起经过 backbone，一次 forward 给出全部 $\gamma$ 个位置。$T_\text{draft}$ 几乎不随 $\gamma$ 增长，省下的时间可以用于更深的网络，DFlash 第 1 位的条件接受率为 0.72。

::dspark-draft::

并行的代价是各位置相互独立采样，第 $k$ 位看不到前面各位实际采到的 token：target 会回答“当然可以”或“没问题”时，两个位置各自的分布都正确，独立采样却可能得到“当然问题”。验证在第一个被拒的位置停止，之后的 draft token 全部丢弃。

::dspark-markov::

三种 drafter 各位置的条件接受率如下：

::dspark-position::

EAGLE-3 第 1 位为 0.53，之后逐位升高；DFlash 第 1 位为 0.72，第 3 位降到 0.64。第 1 位取决于网络容量，层数多的 DFlash 占优；后面各位取决于能否看到已采样的 token，自回归的 EAGLE-3 占优。遇到拒绝即停，第 1 位对平均接受长度影响最大，因此 DFlash 的平均接受长度仍高于 EAGLE-3。

## 三、Markov head

并行生成时，第 $k$ 位缺少的信息是前一位实际采到的 token。DSpark 保留 DFlash 的并行 backbone 来保证第 1 位，再加一张训练得到的表 $B$：每个 token 对应一行，是加在后一位 logits 上的偏置。backbone 一次 forward 给出各位置的 base logits $U_1,\dots,U_\gamma$；之后从左到右采样，每一位加上前一个 token 对应的那一行：

$$
p_k(v \mid x_{k-1}) = \mathrm{softmax}\big(U_k + B(x_{k-1},\cdot)\big)_v
$$

第 1 位采到“当然”后，第 2 位“可以”的 logit 升高、“问题”降低。偏置只取决于前一个 token，与更早的 token 无关，这个模块因此称为 **Markov head**。论文改用 RNN 记住块内全部前缀作对照，提升很小。

完整的 $B$ 是 $V \times V$ 的表（$V$ 为词表大小，DeepSeek-V4-Flash 为 129,280），无法存储。DSpark 将其分解为秩 256 的 $W_1 W_2$：$W_1$ 按前一个 token 查出向量，$W_2$ 投影为词表上的偏置。串行部分只有查表和投影，一轮耗时比 DFlash 多 0.2% 到 1.3%。偏置加在 logits 上，draft 概率仍可精确计算，拒绝采样照常无损。

聊天任务上，DSpark 第 1 到 7 位的条件接受率都在 0.73 到 0.77：第 1 位与 DFlash 相当，后面各位不再下降。target 为 Qwen3-8B、$\gamma = 7$ 时的平均接受长度：

| drafter | 数学（GSM8K） | 代码（HumanEval） | 聊天（MT-Bench） |
| --- | --- | --- | --- |
| EAGLE-3 | 5.30 | 4.33 | 2.66 |
| DFlash | 5.33 | 4.64 | 3.11 |
| DSpark | 6.17 | 5.52 | 3.72 |

表中 DFlash 与 DSpark 为 5 层，EAGLE-3 为 1 层。以 Qwen3-4B 为 target、$\gamma = 7$ 时，2 层的 DSpark 在三类任务上也都超过 5 层的 DFlash。

## 四、验证的代价与存活概率

draft token 增加到 5 个以后，是否全部验证取决于负载。batch 小时多验几个几乎不增加耗时；并发高时每个 token 都占用算力，固定验证 5 个，被拒的部分就是无效计算。高并发时只应验证最可能被接受的 token，这需要逐个估计接受概率。

第 $j$ 个 token 只有在前面各位都被接受时才有效。记第 $k$ 位的条件接受率为 $c_k$，第 $j$ 个 token 被接受的概率称为**存活概率**：

$$
a_j = c_1 c_2 \cdots c_j
$$

例如条件接受率依次为 0.80、0.70、0.65、0.60 时，存活概率为 0.80、0.56、0.36、0.22，越靠后的 token 越不值得验证。接受情况也随任务变化，$\gamma = 7$ 时聊天的平均接受长度为 3.72，数学为 6.17，所以验证几个不能取固定值。

## 五、Confidence head

第 $k$ 位的真实条件接受率为 $1 - \tfrac12\lVert p^d_k - p^t_k\rVert_1$，即 drafter 与 target 在第 $k$ 位的分布 $p^d_k$、$p^t_k$ 重叠的部分。生成 draft 时 target 尚未运行，$p^t_k$ 未知，只能由 drafter 自己估计。

DSpark 为此加了 **confidence head**：线性层加 sigmoid，输入为 backbone 的 hidden state 与前一个 token 的 Markov embedding，以真实条件接受率为训练标签，输出 $c_k$ 的估计值。调度器用存活概率计算期望产出，偏差会使它多验或少验，因此需要校准：原始 $c_k$ 的误差为 3% 到 8%，逐位调 temperature 后约 1%。

## 六、调度器

设有 $R$ 个请求，第 $r$ 个验 $\ell_r$ 个 draft token，送入 target 的 token 数为 $N = \sum_r (1 + \ell_r)$，期望产出为

$$
\tau = \sum_r \Big(1 + \sum_{j \le \ell_r} a_{r,j}\Big)
$$

$\mathrm{SPS}(N)$ 为 batch 含 $N$ 个 token 时 target 每秒的步数，启动时测得；吞吐 $\Theta = \tau \cdot \mathrm{SPS}(N)$。加入一个 draft token 使 $\tau$ 增加它的存活概率，同时使 $N$ 加 1、$\mathrm{SPS}$ 下降。调度器把所有请求的 draft token 按存活概率从高到低逐个加入，直到 $\Theta$ 不再上升。存活概率在请求内递减，选中的总是每个请求开头的几个。

::dspark-schedule::

空闲时 $\mathrm{SPS}$ 随 $N$ 下降慢，12 个里验 10 个；繁忙时下降快，只验 5 个：A 4 个，B 1 个，C 0 个。同一配置适应两种负载，不必按并发调 $\gamma$；静态阈值（如存活概率低于 0.5 不验）不看负载，做不到这一点。

## 七、工程实现

**选择不能依赖 token 本身。** 无损要求是否验证第 $k$ 个 token 与该 token 无关。$c_{k+1}$ 及之后的 confidence 由第 $k$ 个 token 算出；若看完全部候选再全局选截断位置，第 $k$ 个 token 是否入选会受 $c_{k+1}$ 以后的值影响，即间接取决于它自己，输出分布会产生偏差：论文附录的反例中，target 分布 (0.7, 0.3) 变为 (0.85, 0.15)。因此论文的算法逐个加入、在 $\Theta$ 首次下降时即停，决定第 $k$ 个 token 时只用到由前 $k-1$ 个 token 算出的 $a_k$。

**预算用两步之前的 confidence。** 调度在 CPU 上进行，若等当前一步的 confidence 拷回 CPU 再定 batch 大小，GPU 会空等。线上因此用两步之前、已拷回 CPU 的 confidence 定出验证预算 $K$，再在 GPU 上用当前存活概率取 top-$K$。预算与当前 token 无关，可以全局搜索。vLLM 的 `AdaptiveVerificationManager` 采用同样的做法，代价表由 CUDA graph 实测，并计入 drafter 耗时：

```python
survival_probability = np.cumprod(stale_confidences, axis=1)
scores = np.sort(survival_probability[valid])[::-1]
accepted = np.concatenate(([num_reqs], num_reqs + np.cumsum(scores)))
costs = draft_cost_ms[num_reqs] + verify_cost_ms[base : base + max_budget + 1]
draft_budget = int(np.argmax(accepted / costs))
```

**各请求验证长度不同。** 普通 decode kernel 假设各请求的 query 长度相同，按请求调度验证长度后这一假设不再成立。DeepSeek 把所有 token 展平为一维，用一个标记 tensor 记录每个 token 所属的请求与位置，供 sparse attention 使用；V4 上只改了 index-attention 和 compress 两个 kernel。

## 八、线上结果

线上的 DSpark 以 3 层 MoE 为 backbone，最多生成 5 个 draft token。对照的 MTP-1 每轮验 2 个 token：1 个 draft，1 个 target 自己的输出位。

::dspark-online::

V4-Flash 上，每用户 80 tok/s 时，DSpark 每张卡的吞吐比 MTP-1 高 51%；吞吐相同时，每用户快 60% 到 85%。V4-Pro 为 57% 到 78%。

并发在 200（V4-Flash）或 150（V4-Pro）以内时，每个请求平均验 4 到 6 个 token，并发越高越少，V4-Flash 在 200 并发时约 3.5 个。

::dspark-budget::

这一提升由三项改动共同得到：并行 backbone 一次 forward 生成 5 个 draft token，Markov head 补上前一个 token 的信息使每轮接受的 token 更多，confidence head 与调度器按负载决定验证几个。

## 九、支持情况与局限

DeepSeek 开源了 V4-Flash 与 V4-Pro 的 DSpark checkpoint 和训练代码 [DeepSpec](https://github.com/deepseek-ai/DeepSpec)（MIT）。vLLM 用 `method: dspark`，调度器开关为 `enable_adaptive_verification`，需要完整 CUDA graph，不支持 LoRA 与 PP。SGLang 用 `--speculative-algorithm DSPARK`，cookbook 中 Kimi-K3、Qwen3.8-27B 等已有第三方训练的 DSpark drafter。

- **draft 成本无法节省。** 调度器只能少验，backbone 每轮都要跑，对接受率低的请求是净损失。
- **需要富余算力。** 论文的线上 batch 受 KV cache 容量与流量限制，未到算力上限，多验的代价才低；负载达到算力上限时，调度器退化为接近不做 speculative decoding。

## 参考

- Cheng et al. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147). 2026.
- DeepSeek-AI. [DeepSpec](https://github.com/deepseek-ai/DeepSpec) · [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark).
- Chen et al. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036). 2026.
- Li et al. [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840). 2025.
- Leviathan et al. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). ICML 2023.
- vLLM `e006d76` 源码：`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`、`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`、`vllm/models/deepseek_v4/nvidia/dspark.py`；文档 [Adaptive Verification](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/)。
- SGLang `8854857` 源码：`python/sglang/kernels/ops/speculative/dspark/`；cookbook 的 Kimi-K3 与 Qwen3.8-27B 页面。
