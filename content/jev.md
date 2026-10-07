用 LLM 做判断，通常是让它生成一段 JSON：工单该分给哪个团队、要不要升级、客户有多不满。模型真正要给出的只是几个选项上的概率，却要一个 token 一个 token 地把 JSON 写出来。**Jev** 是 TypeSafe AI 在 2026 年 9 月发布的 decision model：输入一段文本和一组带类型的问题，返回每个问题的概率分布，不生成任何文本。

Jev 没有公开论文和权重。本文的机制部分以开源复刻 [Kev](https://github.com/jaredpalmer/kev) 的源码为准，它用同样的接口、在 Qwen 上实现了同一种计算；Jev 自己的行为用 API 探测来对照，探测的版本是 `jev-1.13.0`。两者对得上的地方，说明这种设计能解释 Jev 的表现，但不等于 Jev 的实现与 Kev 相同。

## 一、输入与输出

一次请求由 **state** 和一组 **question** 组成。state 是待判断的内容，可以是字符串或 JSON；每个 question 带一个类型和它允许的答案：

```json
{
  "state": "My payouts have failed three times this week. The bank says everything is fine on their side. Can someone please fix this today?",
  "questions": {
    "queue": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "payments": "Payout failures and payment processing",
        "account": "Login and account access",
        "other": "Something else"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does this message require urgent human attention?"
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

问题有三种类型，返回的都是数：

| 类型 | 答案空间 | Jev 对这条工单的返回 |
| --- | --- | --- |
| Choice | 给定的选项里选一个，最多 255 个 | `payments` 1.00，`account` 0.00，`other` 0.00 |
| Noul | 是或否 | 0.87 |
| Score | 一组有序的档位，最多 10 档 | 1.09（Calm 0.00，Frustrated 0.91，Very angry 0.09） |

答案只能落在调用方给出的选项上，所以不会出现格式错误，也不会出现选项之外的内容。TypeSafe 说 Jev “不会 hallucinate”，指的就是这一点；答案本身仍然可能选错。

## 二、省掉的是 decode

LLM 处理一次请求分两段。**prefill** 阶段输入的 token 全部已知，一次 forward 就能算完所有位置。**decode** 阶段每个输出 token 要等前一个选定之后才能开始算，只能串行，输出多少个 token 就要多少次 forward。

::jev-two-paths::

上面那份答案写成 JSON 是 70 个 token。LLM 要在 prefill 之后再做 70 次 forward，其中大部分花在引号、字段名和小数点上。Jev 在 prefill 结束时直接从 hidden state 读出三个分布，JSON 由普通代码拼出来。TypeSafe 的发布文章把这一点写作“并行输出所有概率，而不是逐 token 自回归生成”。

Jev 的响应里仍然有一个 `output_tokens` 字段，但它只是把返回的 JSON 按 tokenizer 数了一遍。一个 Choice 问题从 2 个选项加到 200 个，`output_tokens` 从 33 变成 2,103，我测到的往返耗时是 665 ms 和 677 ms（大部分是网络），没有随输出变长。

## 三、一条序列装下所有问题

多个问题要共用同一段 state，又不能互相影响：问“客户是否生气”不应该改变“分给哪个团队”的答案。Kev 的做法是把 state 和所有问题排成一条序列：

```text
<state> …state…
<q> 指令 <opt> 选项 1 </opt> <opt> 选项 2 </opt> … <decide>
<q> 指令 <opt> 选项 1 </opt> <opt> 选项 2 </opt> … <decide>
```

下文把一个问题自己的那段 token（从 `<q>` 到 `<decide>`）叫作它的 **branch**。隔离靠 attention mask 完成：一个 token 只能读 state，以及自己 branch 里排在它前面的 token。

::jev-mask::

Kev 生成这张 mask 的规则只有两行，`seg` 记录每个 token 属于 state（0）还是第几个问题：

```python
same = (seg[None, :] == seg[:, None]) | (seg[None, :] == 0)   # 同一个 branch，或者是 state
allow = torch.tril(torch.ones(L, L, dtype=torch.bool)) & same
```

position ID 也要配合：每个 branch 的编号都从 state 的末尾接着数，而不是接在上一个 branch 后面。这样每个问题看到的序列与“state 后面只跟着它自己”完全一样，把问题打包在一起问和分开问，Kev 给出的概率相差不超过 $4\times10^{-6}$。

Jev 的行为与这个结构一致，我在 API 上测了两件事：

- **state 只计一次。** 同一个 Noul 问题重复 1、2、10 遍，`input_tokens` 是 298、311、415，每多一个问题加 13。把 state 加长到原来的 20 倍，这三个数变成 792、805、909，每个问题还是加 13。
- **问题之间互不可见。** 在一个问题的指令里写上“本次请求的暗号是 ZEBRA-7741”，再用另一个问题问暗号是什么，`ZEBRA-7741` 的概率是 0.00；把同一句话挪进 state，概率变成 1.00。

Jev 的 context 上限也是按这个结构定的：整个请求 64k tokens，state 加上最长的一个问题 32k tokens。官方文档的说法是“state 只读入一次，所有问题并行地对它求值”。

## 四、从 hidden state 读出概率

每个 branch 的最后一个 token 是 `<decide>`。它排在所有选项之后，能读到 state、指令和整张选项表。Kev 在 backbone 上接了一个很小的 **pointer head**：取 `<decide>` 的 hidden state 和每个选项末尾 `</opt>` 的 hidden state，各过一个线性层，做点积得到每个选项的 logit。

::jev-pointer::

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)

    def forward(self, h_decide, h_opts):                  # [d], [K, d] -> [K]
        return (self.k(h_opts) @ self.q(h_decide)) / math.sqrt(dp) / self.temperature
```

三种问题共用这一个 head。Noul 是 `false`、`true` 两个选项，返回 `true` 的概率；Score 把每一档当作一个选项，返回档位的期望，图中的 $0\times0.00+1\times0.91+2\times0.09$ 约等于返回的 1.09。选项的含义来自它自己的文本，所以换一组从没见过的选项也能用，不需要为每个任务训练新的分类层。backbone 不带 vocab head，模型从结构上就无法生成文本。

响应里的 `confidence` 不是模型的另一个输出，而是由分布算出来的。Choice 的公式是 $(p_{\max}-1/K)/(1-1/K)$，衡量最高的那一项比均匀分布高出多少。

这个读法的代价是选项之间会互相影响。后面的选项能读到前面的选项，`<decide>` 读到的是整张表，所以选项的顺序会改变结果。我用一条归属不明确的工单测了三个选项的六种排列，Jev 给 `payments` 的概率在 0.72 到 0.86 之间变动。如果业务阈值正好卡在 0.8，同一条工单会因为选项顺序不同而走向不同的分支。

## 五、耗时随什么增长

::jev-latency::

图中是 [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) 测到的 Jev 服务端耗时。问题从 1 个加到 100 个，耗时一直在 80 ms 左右；加到 1,500 个是 610 ms。state 从 360 tokens 加到约 3 万 tokens，耗时从 57.5 ms 升到 218 ms。

这两条曲线正是“state 算一次，branch 成批算”的样子。state 的 KV cache 算好之后，每个 branch 只需要算自己那十几个 token，而且各 branch 互不依赖，可以放进同一个 batch。问题很少时，耗时主要是 state 的 prefill 和固定开销；问题多到占满一个 batch 之后，耗时才随问题数增长。

Kev 的 serving 就是这样实现的：先单独对 state 做一次 forward 并保存 KV cache，再让所有 branch 接在这份 cache 后面成批计算。同一段 state 再次出现时可以直接复用 cache，只付问题的开销，这就是 prefix caching。

## 六、概率可信吗

返回概率而不是标签，前提是概率可以拿来设阈值。这个性质叫 **calibration**：在模型给出 0.8 的那些判断里，应该有大约 80% 是对的。它描述的是一批预测，不保证单次判断正确。

TypeSafe 把自己的训练方法叫作 RLCD（Reinforcement Learning for Calibrated Decisions），公开的信息只有目标：让概率与结果的实际发生频率一致。具体的 loss 和算法没有公布。Hume 在 1,200 道 MMLU 题上测得 Jev 的 expected calibration error（ECE，各概率区间内预测概率与实际正确率之差的加权平均）是 0.031。

Kev 没有用强化学习。它用普通的 cross-entropy 训练，然后在留出的数据上拟合一个 temperature $T$，读出时把 logits 除以它。$T$ 不改变哪个选项排第一，只把过于自信的分布拉平：Kev-9B 在没见过的数据集上，ECE 加 $T$ 之前是 0.103，之后是 0.041。Kev 的作者还试过 label smoothing、Brier loss 等专门针对 calibration 的 loss，都没有比 cross-entropy 更好。

calibration 是统计意义上的，不保证逻辑自洽。我对同一条工单分别问“客户生气吗”和“客户没有生气吗”，Jev 返回 0.83 和 0.24，加起来是 1.07。

## 七、能力边界

**知识量由底座决定。** pointer head 和 LoRA 只教会模型按这种格式作答，不增加知识。在 MMLU-Pro 上，未训练的 Qwen3.8-27B 是 0.635，Kev-27B 是 0.675，Jev 是 0.840。在 Kev 没训练过的 11 类数据里，Kev-27B 有 9 类与 Jev 相差不到三个点或更高，差距主要在知识题。

**不擅长需要多步计算的问题。** 没有 decode 就没有地方写中间步骤，所有推理都要在一次 forward 里完成。TypeSafe 的文档列出的弱项是计数、算术、日期比较和多跳推理，建议把这些留给代码。Kev 提供了同样思路的开关：服务端先用代码算出 state 里每两个日期相隔的天数，写成句子追加进去，Kev-9B 在截止日期类问题上的正确率不开是 0.80，打开是 0.90。

**对 state 里的内容不设防。** state 中夹带的指令会影响判断，无关内容多了也会干扰结果。TypeSafe 建议在送进模型之前先过滤。

## 八、和 classifier 的关系

“给定文本和候选标签，输出一个分布”就是分类，zero-shot classification 也不是新东西。发布后最常见的批评正是这一点。区别在于拿什么当底座、怎么读出：

| 路线 | 代表 | 底座 | 读出方式 |
| --- | --- | --- | --- |
| encoder | [Laya](https://github.com/NandhaKishorM/laya) | ModernBERT-large（4.2 亿参数）等双向 encoder | head 给每个选项的槽位打分 |
| 冻结的 LLM | [AnyJev](https://github.com/nokia-applied-research/AnyJev) | 现成的 Qwen，不训练 backbone | 直接读答案 token 的 logits，或用少量标注拟合一个线性 head |
| 训练过的 decoder | Kev，以及按上文证据推测的 Jev | 几十亿到几百亿参数的 causal LM | state 共享，branch 隔离，pointer head |

encoder 路线最快，但知识量和 context 长度受限于小模型。直接读 LLM 的 logits 不需要训练，但受选项位置影响大：AnyJev 在 Qwen3-8B 上测到，把选项倒过来排，23% 的题答案会变。Jev 这一类的做法是保留大模型预训练得到的知识，再把读出方式、state 复用和 calibration 当作一个整体来训练和 serving。它没有发明新的模型结构，新的是这套组合，以及把“带概率的判断”做成了一个可以按 token 计价的接口。

## 小结

- Jev 只做 prefill：输入处理完就从 hidden state 读出概率，省掉了逐 token 的 decode。
- state 和所有问题排成一条序列，attention mask 让每个问题只读 state 和自己，于是 state 只算一次，问题之间互不干扰。
- 概率由 `<decide>` 与各选项的 hidden state 做点积得到；选项互相可见，顺序会影响结果。
- 概率能不能用来设阈值取决于 calibration，知识量取决于底座，多步计算要留给代码或会生成文本的模型。
