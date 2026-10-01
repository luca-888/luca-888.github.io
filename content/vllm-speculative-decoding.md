Decode 每步只产出一个 token，却要把整个模型的权重从显存读一遍。**Speculative decoding** 让一个便宜的 drafter 先猜后面几个 token，target 模型用一次 forward 并行验证：猜对的直接收下，猜错的位置换成 target 自己该给的 token。

关键在验证规则。用**拒绝采样**接受 draft token、拒绝时按残差分布补采，输出分布与 target 单独逐 token 采样完全相同，快慢只取决于猜得准不准、猜得贵不贵。本文的实测使用 vLLM v0.30.0，单卡 RTX 4090，目标模型 Qwen3-8B（BF16）。

## 一、验证为什么几乎免费

Decode 的计算强度很低：每步只处理一个 token，权重却要整体读一遍。Qwen3-8B 的 BF16 权重约 16 GB，4090 的显存带宽约 1 TB/s，读一遍要 16 ms 左右，单请求 decode 的上限约 60 tok/s；实测为 59 tok/s，每步 17 ms。

同一次 forward 里多放几个 token，权重仍然只读一次，多出来的只是算力。实测中并发从 1 增加到 32，每步同时处理 32 个 token，每个请求的速度只从 59 tok/s 降到 41 tok/s，即每步耗时从 17 ms 增到 24 ms。所以 target 一次验证 $\gamma+1$ 个位置，代价只比普通的一步 decode 略高。这是 [Leviathan et al.](https://arxiv.org/abs/2211.17192) 与 [Chen et al.](https://arxiv.org/abs/2302.01318) 共同的前提：显存带宽受限、算力有富余。富余算力被别的请求用掉以后，收益随之消失（第六节）。

## 二、拒绝采样：分布为什么不变

记 target 在当前位置的分布为 $p$，drafter 的分布为 $q$。一轮 speculative decoding 如下：

1. drafter 自回归地采样 $\gamma$ 个 token $x_1,\dots,x_\gamma$，其中 $x_i \sim q_i$。
2. target 对“已有前缀 + 这 $\gamma$ 个 token”做一次 forward，得到 $\gamma+1$ 个位置的分布 $p_1,\dots,p_{\gamma+1}$。
3. 从左到右逐位判定：以概率 $\min(1,\ p_i(x_i)/q_i(x_i))$ 接受 $x_i$。
4. 遇到第一个被拒绝的位置，从残差分布 $\mathrm{norm}(\max(p_i-q_i,\,0))$ 采样一个 token 代替它，本轮结束，后面的 draft token 全部丢弃。
5. 如果 $\gamma$ 个 token 全被接受，再从 $p_{\gamma+1}$ 采一个 **bonus token**。第 2 步的那次 forward 已经算出了这个位置的分布，所以这个 token 不额外花时间。

因此每轮至少产出 1 个 token，至多 $\gamma+1$ 个。

::spec-verify-path::

对单个 draft token $x$，这套规则只有三条路：

- $p(x)\ge q(x)$：target 比 draft 更想要它，直接收下。
- $p(x)<q(x)$：draft 猜多了，只以 $p(x)/q(x)$ 的概率收下。
- 没收下：拒绝，从 target 还缺的那部分（残差分布）里补一个。

三条路加起来，每个位置的输出都服从 $p$。输出 $x$ 有两个来源：draft 采到 $x$ 且被收下，或者 draft 被拒绝后补采到 $x$。

$$
\begin{aligned}
P(\text{输出}=x) &= q(x)\min\!\Big(1,\frac{p(x)}{q(x)}\Big) + P(\text{拒绝})\cdot\frac{\max(p(x)-q(x),\,0)}{\sum_{x'}\max(p(x')-q(x'),\,0)} \\
&= \min(p(x),q(x)) + \max(p(x)-q(x),\,0) = p(x).
\end{aligned}
$$

第二步用到 $P(\text{拒绝})=1-\sum_x\min(p,q)=\sum_x\max(p-q,0)$：被拒绝的概率恰好等于残差分布的归一化常数，两者约掉。

::spec-overlap::

图 2 把这个等式画了出来：每个 token 的 $p$ 与 $q$ 各有一部分重叠，重叠部分 $\min(p,q)$ 直接被接受；$q$ 多出的部分被拒绝，$p$ 多出的部分正好由残差采样补上。判定沿前缀逐位进行，第 $i$ 位用到的上下文正是前面已被接受的 token，与 target 自回归时看到的一致，所以整段输出与 target 单独采样同分布。

单位置的接受概率是重叠面积：

$$
\alpha=\sum_x\min(p(x),q(x))=1-\tfrac12\|p-q\|_1
$$

即 $1$ 减去 $p$、$q$ 的总变差距离。$p$ 与 $q$ 越接近，$\alpha$ 越高。

### 两个特例

- **Greedy（temperature 为 0）**：$p$ 是 argmax 处的 one-hot，draft token 被接受当且仅当它等于 target 的 argmax；被拒绝时残差分布也只剩 argmax。输出与 target 贪心解码逐 token 相同。
- **top-p、top-k 与 temperature**：先按采样设置改写 $p$ 与 $q$，再套用同一规则，Chen et al. 也是这样处理的。

### vLLM 的实现

`RejectionSampler` 把判定和补采写成两个 Triton kernel，全部留在 GPU 上。判定位于 [`rejection_random_sample_kernel`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/sample/rejection_sampler.py)，就是上面的接受规则，用预先生成的均匀随机数 $u$ 做比较：

```python
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

补采位于 `sample_recovered_tokens_kernel`，从 $\max(p-q,0)$ 中采样，但不做归一化，也不做前缀和：

```python
prob = tl.maximum(target_prob - draft_prob, 0.0)
score = prob * inv_q        # inv_q = 1 / Exponential(1) 噪声
```

对每个 token 取 $\arg\max(\text{prob}_i / E_i)$，$E_i$ 为独立的 $\mathrm{Exp}(1)$ 噪声，等价于指数竞赛：$E_i/\text{prob}_i \sim \mathrm{Exp}(\text{prob}_i)$，其中最小者胜出的概率正比于 $\text{prob}_i$，因此结果服从归一化后的残差分布。

`draft_sample_method` 默认是 `greedy`：draft 始终取 argmax，此时 $q$ 是点质量，kernel 用 `NO_DRAFT_PROBS` 分支，接受概率退化为 $p(x)$，补采是“把 draft token 从 $p$ 中去掉再采样”。点质量也是合法分布，因此结果仍然精确，代价是 $\alpha$ 只等于 target 恰好采到 draft 那一个 token 的概率。设为 `probabilistic` 则从 $q$ 采样并保存完整 draft logits，多占显存。

实现与证明对得上：把图 1 位置 3 的 $p$、$q$ 交给 vLLM 的 `rejection_sample` 采样 800 万次，输出的经验分布与 $p$ 的总变差距离只有 $1.7\times10^{-4}$，就是采样噪声的大小。

## 三、加速比模型

一轮的收益由三个量决定：单位置接受率 $\alpha$、每轮猜的 token 数 $\gamma$，以及 drafter 一步相对 target 一步的耗时比 $c$。假设各位置的接受相互独立、$\alpha$ 相同（Leviathan et al. 的简化假设），每轮产出的 token 数是截尾几何分布，期望为

$$
\mathbb{E}[\text{tokens}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

每轮耗时是 $\gamma$ 步 draft 加一步 target，即 $(\gamma c+1)T$，所以相对普通 decode 的加速比为

$$
\text{speedup}=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}
$$

::spec-gamma::

图 3 是这个公式的两种情形。每条曲线都先升后降：$\gamma$ 越大，后面的 token 越难全部猜中，每轮产出趋于上限 $1/(1-\alpha)$，draft 的开销却随 $\gamma$ 线性增长，所以存在最优 $\gamma$。$\alpha=0.7$、$c=0.2$ 时最优是 $\gamma=3$，加速 1.58×；$\alpha$ 越高、$c$ 越小，值得猜得越长。

只要 $\alpha>c$ 就有某个 $\gamma$ 能加速。代价也在公式里：验证的位置越多，总算术运算量越大，节省的只是权重与 KV 的读取，因此它只在显存带宽受限时有用。

## 四、三类 drafter

三种 drafter 都能接入同一个验证器，区别在于 $q$ 从哪里来，以及它把 $\alpha$ 与 $c$ 推向哪里。

| drafter | 谁来猜 | 开销 $c$ | $\alpha$ 取决于 | vLLM 配置 |
| --- | --- | --- | --- | --- |
| draft model | 同 tokenizer 的小模型，自回归跑 $\gamma$ 次 | 与模型大小成正比，另占权重与 KV | 两个模型的输出分布是否接近 | `method: draft_model` |
| n-gram | 在 prompt 与已生成内容里匹配当前末尾的 n-gram，取其后 $\gamma$ 个 token | 一次查找，无模型 | 文本是否重复 | `method: ngram` |
| EAGLE-3 | 接在 target 上的轻量 draft head，输入 target 的 hidden states | 很小 | head 的训练数据与 target 是否匹配 | `method: eagle3` |

**Draft model** 最直接：Leviathan et al. 的 T5 实验中，比 target 小约两个数量级的模型收益最好。本文用 Qwen3-0.6B 为 Qwen3-8B 做 draft，两者同系列、同 tokenizer，输出天然接近。

**n-gram** 完全不用模型，匹配长度在 vLLM 里取 2 到 5 个 token。Leviathan et al. 已经指出即使是 bigram 也有 $\alpha\approx0.2$，并设想过“从上下文拷贝”这类 draft；摘要、代码改写这类会复述输入的任务最合适。它给的是确定的 token，$q$ 是点质量。

**EAGLE-3** 不用独立的小模型：draft head 接在 target 上，融合它的多层 hidden states 直接预测 token（[EAGLE](https://arxiv.org/abs/2401.15077) 最初预测的是特征，[EAGLE-3](https://arxiv.org/abs/2503.01840) 改为预测 token）。论文里的 draft 是树形的，用多分支提高命中；vLLM v0.30 走链式 draft，源码里树形还是待办，所以论文的加速比不能直接套到 vLLM 上。本文使用第三方 AngelSlim 为 Qwen3-8B 训练的 EAGLE-3 权重。

## 五、实测

负载是开放问答（聊天）和段落摘要两类，加速比相对不开 speculative decoding 的 baseline；下面的数字默认取单请求、$\gamma=3$、聊天。

draft model 最快，加速 1.56×。Qwen3-0.6B 猜得准，第 1 位的接受率 0.67，平均每次 target forward 产出 2.4 个 token。

EAGLE-3 猜得少但便宜，加速 1.41×。它平均每次只产出 1.7 个 token，但 draft head 很轻，一轮只比普通的一步 decode 多花约两成时间，所以已经接近 draft model。

n-gram 取决于文本是否重复。聊天里没有可以复述的内容，不加速（0.97×）；摘要会复述原文，加速 1.2×。

$\gamma$ 并非越大越好。draft model 取 $\gamma=5$ 时，平均产出只从 2.4 涨到 2.7 个 token，一轮的耗时却涨到普通一步的 2.8 倍，结果比不开还慢（0.95×）。

::spec-position::

图 4 画出 $\gamma=5$ 时各位置的累计接受概率。draft model 与 EAGLE-3 的实测几乎贴着 $\alpha_1^i$ 的虚线，第三节的独立假设够用：只用第 1 位的接受率就能预测平均接受长度，误差在 4% 以内。n-gram 一旦匹配上，后面的 token 往往继续匹配，所以实测高于虚线。

::spec-speedup::

并发升高，收益缩小。并发从 1 增加到 32，draft model 的加速从 1.56× 降到 1.20×：target 的一步越来越接近算力瓶颈，多验证的 token 不再免费。摘要在并发 4 处还略有上升，之后同样回落。

temperature 0 下，输出与 baseline 并不逐位相同，完全一致的请求只有四成左右。这不是算法的问题：baseline 自己换一个并发也只有 25% 到 50% 一致。BF16 下 batch 形状不同，logits 有微小差别，greedy 在两个接近的 token 间换了选择，后面的序列就整个不同。证明成立于精确算术，数值上不保证逐位一致，Chen et al. 的论文也写明了这一点。

## 六、边界

**并发。** 前文的加速来自“算力有富余”。[vLLM 2024 年的 blog](https://vllm.ai/blog/2024-10-17-spec-decode) 在 Llama3-70B、4×H100、高 QPS 下测到 draft model（ShareGPT）与 prompt lookup（CNN/DailyMail）分别慢 1.4× 与 1.8×，官方文档也把它定位为中低 QPS 的延迟优化。[MagicDec](https://arxiv.org/abs/2408.11049) 则报告在中长序列、batch 32 到 256 时，用稀疏 KV 的 draft 仍能加速（摘要中最高 2.51×）：瓶颈转到 KV 读取后，验证同样不占额外带宽。收益取决于当下的瓶颈是什么。

**显存。** draft model 要占权重和 KV。本文配置下 KV 池比 baseline 小四成以上（约 1.5 万对 2.7 万 token），摘要并发 32 时装不下，图 5 因此没有画这个点。

**不是所有“推测”都无损。** 拒绝采样是精确的，放宽接受条件的方案不是。[Medusa](https://arxiv.org/abs/2401.10774) 的 typical acceptance 提高了接受率，但按 EAGLE 论文的说明，它的非 greedy 生成不保证与 target 同分布。
