# 从 MLE 到 KL 散度：理解 NLL 与交叉熵

训练分类器时，我们希望模型给正确答案更高的概率。**NLL 把正确答案的概率变成 loss，CE 把各个答案的负对数概率按目标分布加权，KL 则衡量两个概率分布的差异。** 三者关系紧密，理解它们有助于读懂分类、语言模型和强化学习中的训练目标。

本文先用一个三分类神经网络走通 Maximum Likelihood Estimation（MLE）、NLL 与 CE，再通过编码长度理解 KL，最后回到 SFT、蒸馏和 GRPO。正文中的 $\log$ 默认是自然对数，数值单位为 nats；引用的编码图使用 $\log_2$，单位为 bits。两者只相差常数倍：$1\ \mathrm{nat}=1/\ln 2\ \mathrm{bits}$。

## 一、从分类器的 loss 理解 NLL、CE 与 MLE

考虑一个区分「猫」「狗」「鸟」的神经网络。输入一张图片 $x$，网络输出三个 logits，经 softmax 得到三个类别的概率。$q_\theta(c\mid x)$ 表示参数为 $\theta$ 的网络看到图片 $x$ 后，给类别 $c$ 的概率；$\theta$ 就是网络权重，真实类别记为 $y$。

### 初始化时，loss 应该是多少

先看一个输出均匀分布的基准：三个 logits 相等，模型给每个类别的概率都是 $1/3$。输入一张猫的图片，真实类别对应的概率就是 $1/3$，该样本的 loss 为：

$$
\mathcal L=-\log q_\theta(y=\text{猫}\mid x)
=-\log\frac{1}{3}=\ln 3\approx1.0986.
$$

这是这个样本的 **negative log-likelihood（NLL）**：取模型给真实类别的概率，再取负对数。

如果一个 batch 中每张图片的预测都均匀，无论标签是猫、狗还是鸟，每个样本的 loss 都是 $1.0986$，因此平均 loss 也是 $1.0986$。推广到 $C$ 个类别，均匀预测的平均 loss 为 $\ln C$。

**随机初始化不保证输出均匀。** 当初始化后的 logits 彼此接近时，loss 通常接近这个基准；实际值还取决于输入和初始化得到的 logits。这里讨论普通硬标签、没有类别权重、没有 label smoothing 的平均 CE loss。

### 训练如何降低单个样本的 loss

仍然看同一张猫的图片，比较模型给真实类别的概率：

| 猫的预测概率 | 该样本的 NLL |
| ---: | ---: |
| $1/3$ | $1.0986$ |
| $0.5$ | $0.6931$ |
| $0.9$ | $0.1054$ |
| $0.01$ | $4.6052$ |

**模型给真实类别的概率越高，loss 越小。** 即使两次预测都把猫排在第一位，给猫 $0.9$ 概率时的 loss 也比给 $0.5$ 时更小。如果模型确信图片是狗，只给猫 $0.01$ 的概率，loss 就会很大。

### 一个 batch 的 NLL 与 likelihood

现在取一个包含三张图片的 batch。网络对每张图片分别预测，下面列出三次输出：

| 图片的真实类别 | 猫的概率 | 狗的概率 | 鸟的概率 | 真实类别的概率 |
| --- | ---: | ---: | ---: | ---: |
| 猫 | $0.6$ | $0.3$ | $0.1$ | $0.6$ |
| 狗 | $0.2$ | $0.7$ | $0.1$ | $0.7$ |
| 鸟 | $0.2$ | $0.3$ | $0.5$ | $0.5$ |

平均 NLL 就是分别读取真实类别的概率，取负对数，再求平均：

$$
\mathcal L_{\mathrm{NLL}}=-\frac{\log0.6+\log0.7+\log0.5}{3}\approx0.5202.
$$

也可以先把这些概率乘起来。假设给定各自输入和模型参数后，样本标签独立，模型给这批实际标签分配的联合概率为：

$$
L(\theta)=\prod_{n=1}^{N}q_\theta(y_n\mid x_n)
=0.6\times0.7\times0.5=0.21.
$$

上面的 $0.21$ 是模型在给定这三张图片后，为这三个实际标签分配的联合概率。训练时，图片与标签保持不变，我们调整网络权重，比较这个数值如何变化。**把同一个概率表达式看作参数 $\theta$ 的函数，就叫 likelihood；MLE 的目标是找到让它最大的参数。** 它评价的是参数对这批标签的解释程度，不是“这组参数有多大概率正确”。

直接计算很多概率的乘积容易得到极小的数，因此通常先取对数，把乘积转成求和。对数保留大小顺序：likelihood 越大，log-likelihood 也越大。再取负号，把“最大化”改成“最小化”，最后除以样本数，就得到平均 NLL：

$$
\mathcal L_{\mathrm{NLL}}=-\frac{1}{N}\log L(\theta)
=-\frac{1}{N}\sum_{n=1}^{N}\log q_\theta(y_n\mid x_n).
$$

所以，**用梯度下降最小化平均 NLL，就是在朝提高训练数据 likelihood 的方向调整网络参数。**

### 为什么分类代码通常写 CE

对一张猫的图片，标签可以写成 one-hot 分布 $p=[1,0,0]$。若模型输出 $q=[0.6,0.3,0.1]$，**cross-entropy（CE）** 为：

$$
H(p,q)=-\sum_c p(c)\log q(c)
=-[1\log0.6+0\log0.3+0\log0.1]
=-\log0.6.
$$

只有真实类别的权重是一，其余为零，所以求和最后只留下真实类别的负对数概率。**对硬标签分类，CE 与 NLL 是同一个样本 loss；对 batch 采用相同的平均方式，结果也相等。**

在 PyTorch 中，`cross_entropy` 接收 logits，内部完成对应的 log-softmax 与目标类别 loss 计算；`nll_loss` 接收已经计算好的 log-prob。两种写法对应相同目标，具体代码见第四章。

### 软标签：让多个类别参与加权

CE 也可以接收教师模型给出的软标签。例如，对同一张图片，教师给出目标分布 $p=[0.5,0.3,0.2]$，学生预测 $q=[0.2,0.5,0.3]$。此时：

$$
H(p,q)=-[0.5\log0.2+0.3\log0.5+0.2\log0.3]\approx1.2535.
$$

这一次，猫、狗、鸟的负对数概率分别乘以 $0.5$、$0.3$、$0.2$，再相加。一般地：

$$
H(p,q)=-\sum_c p(c)\log q(c)
=\mathbb E_{c\sim p}[-\log q(c)].
$$

式中的 $\mathbb E_{c\sim p}$ 就是“按 $p(c)$ 加权求平均”。**CE 是目标分布下的平均 NLL。** One-hot 标签只有一个类别的权重为一，因此 CE 等于该类别的 NLL；软标签则让多个类别参与加权。后面的 entropy 与 KL 将继续使用这组概率。

### 为什么只读取真实类别，也会更新所有 logits

设模型输出 logits $z$，经 softmax 得到 $q$。对硬标签 $y$：

$$
-\log q(y)=-z_y+\log\sum_j\exp(z_j),\qquad
\frac{\partial\mathcal L}{\partial z_j}=q(j)-\mathbf 1[j=y].
$$

Softmax 中，猫的概率是猫的指数分数除以所有类别指数分数的总和，所以改变狗或鸟的 logit，也会改变猫的概率。式中的 $\mathbf 1[j=y]$ 在 $j$ 是真实类别时为一，否则为零。对均匀输出的猫样本，三个 logits 的梯度为 $[-2/3,1/3,1/3]$：把 logits 当作变量做梯度下降，会增大猫的 logit、减小另两个。实际训练通过这些梯度更新网络权重。使用归一化的软标签时，梯度是 $q(j)-p(j)$。

## 二、从编码长度理解 CE 与 KL

前面把 $-\log q$ 当作 loss，它也可以从信息编码的角度理解。假设要把一串猫、狗、鸟的结果编码成二进制：为了缩短总长度，应给常见结果安排短编码，给罕见结果安排长编码。按分布 $q$ 设计时，结果 $x$ 的理想编码长度是 $-\log_2 q(x)$。例如，概率为 $1/2$ 时对应 $1$ bit，概率为 $1/8$ 时对应 $3$ bits。

这里讨论理想的平均长度，长序列编码可以逼近它；单个结果的实际编码仍使用整数个位。训练神经网络时通常不生成这些编码，而是直接用负对数概率计算 loss。本节所说的“编码代价”，具体就是编码长度。

如果结果实际来自分布 $p$，并使用适配 $p$ 的理想编码，平均长度就是 entropy。沿用本文的 $\log$ 记法：

$$
H(p)=-\sum_xp(x)\log p(x).
$$

如果结果仍然来自 $p$，却按 $q$ 设计编码，就要把每个结果的长度 $-\log q(x)$ 乘以它实际出现的概率 $p(x)$，再相加。这正是 CE：$H(p,q)$。colah 的 [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) 用矩形面积表示这个加权过程：高度是事件概率，宽度是编码长度，所有矩形的面积相加就是平均长度。

::colah-cross-entropy::

沿图的第一行看，消息分布保持为 $p$。左图使用适配 $p$ 的编码，右图改用适配 $q$ 的编码，平均长度从 $1.75$ bits 增到 $2.375$ bits。平均多出的 $0.625$ bits，就是 $D_{\mathrm{KL}}(p\|q)$。第二行的消息来自 $q$，比较的是反方向：改用适配 $p$ 的编码后，平均多出 $0.5$ bits。

### KL 是 CE 超过 entropy 的部分

从 CE 中减去目标分布自身的 entropy，就得到 KL：

$$
\begin{aligned}
D_{\mathrm{KL}}(p\|q)
&=\sum_xp(x)\log\frac{p(x)}{q(x)}\\
&=-\sum_xp(x)\log q(x)+\sum_xp(x)\log p(x)\\
&=H(p,q)-H(p).
\end{aligned}
$$

因此：

$$
\boxed{H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)}.
$$

$H(p)$ 是使用适配真实分布的理想编码时的平均长度，KL 是改用 $q$ 的编码后平均增加的长度。因此 KL 非负，两个分布相同时为零。使用自然对数时单位是 nats，使用以二为底的对数时单位是 bits。

如果某个结果满足 $p(x)>0$，却有 $q(x)=0$，意味着真实分布认为它可能发生，模型却给它零概率。固定 $p(x)>0$，当 $q(x)$ 趋近零时，$\log[p(x)/q(x)]$ 会无限增大，因此这一项的贡献以及 KL 都趋向无穷大；在 $q(x)=0$ 时按此极限定义。$p(x)=0$ 的项则按零处理，因为这个结果不参与按 $p$ 加权的平均。

回到前面的软标签 $p=[0.5,0.3,0.2]$ 与预测 $q_\theta=[0.2,0.5,0.3]$，使用自然对数计算：

| 量 | 数值 / nats | 含义 |
| --- | ---: | --- |
| $H(p)$ | $1.0297$ | 目标分布自身的 entropy |
| $H(p,q_\theta)$ | $1.2535$ | 模型在该分布下的平均 NLL |
| $D_{\mathrm{KL}}(p\|q_\theta)$ | $0.2238$ | CE 比目标 entropy 高出的部分 |

把学生输出调成 $q_\theta=p$ 后，KL 降为零，CE 降为 $1.0297$。学生已经完全匹配教师，但教师仍给三个类别非零概率，各类别的负对数概率加权后仍大于零。因此，软标签的 CE 达到最小值时也可以不为零。

对一张确定标为猫的图片，one-hot 目标是 $[1,0,0]$，它的 entropy 为零，所以该样本的 CE、NLL 与 $D_{\mathrm{KL}}(p\|q)$ 数值相等。即使整个数据集同时有猫、狗、鸟，也不影响每张图片的 one-hot 目标具有零 entropy。

### 何时可以把 CE 与 KL 当作同一个优化目标

例如，教师对这张图片始终输出 $p=[0.5,0.3,0.2]$，训练只更新学生 $q_\theta$。此时 $H(p)=1.0297$ 不变，所以 CE 始终等于 KL 加上 $1.0297$。这个常数对学生参数求导为零，因此：

$$
\nabla_\theta H(p,q_\theta)
=\nabla_\theta D_{\mathrm{KL}}(p\|q_\theta).
$$

**梯度描述的是参数变化时，loss 如何变化，而不是 loss 本身有多大。** 例如，$\theta^2$ 与 $\theta^2+10$ 的数值始终相差 $10$，但对 $\theta$ 的导数都是 $2\theta$：常数只把曲线整体向上平移，不改变同一位置的斜率。这里的 $H(p)$ 就是这样的常数，因此 CE 与 KL 在相同参数处给出相同梯度；使用相同学习率做梯度下降，参数更新也相同。

硬标签监督和固定软标签监督都适用。关键在于：**本次求梯度时，目标分布 $p$ 要保持固定。** 如果 $p$ 也依赖正在优化的参数，且梯度会通过它传播，$H(p)$ 就可能变化，不能直接从 loss 中去掉。第五章的 GRPO 会遇到这种情况。

## 三、KL 的方向为什么重要

看 $D_{\mathrm{KL}}(p\|q)$ 时，可以先认清两个位置：**第一个分布 $p$ 提供平均权重，对数中比较的是 $p(x)/q(x)$。** 交换方向后，平均权重改成 $q$，概率比也反过来：

$$
D_{\mathrm{KL}}(p\|q)=\mathbb E_{x\sim p}\!\left[\log\frac{p(x)}{q(x)}\right],\qquad
D_{\mathrm{KL}}(q\|p)=\mathbb E_{x\sim q}\!\left[\log\frac{q(x)}{p(x)}\right].
$$

在猫、狗、鸟的例子里，$p=[0.5,0.3,0.2]$，$q=[0.2,0.5,0.3]$。计算 $D_{\mathrm{KL}}(p\|q)$ 时，三个对数比值分别乘以 $0.5$、$0.3$、$0.2$；计算 $D_{\mathrm{KL}}(q\|p)$ 时，反向的对数比值分别乘以 $0.2$、$0.5$、$0.3$。这是两种不同的加权平均。

两个方向的结果分别为 $0.2238$ 和 $0.1938$ nats。**交换两个分布，KL 通常会改变，这就是“不对称”。** 普通距离要求从 A 到 B 与从 B 到 A 一样远，KL 不满足这一要求，因此可以衡量分布差异，却不是严格数学意义上的距离。

### 用单峰分布拟合双峰分布

方向不同，还会影响模型最后拟合成什么样。Eric Jang 的 [A Beginner's Guide to Variational Methods](https://blog.evjang.com/2016/08/variational-bayes.html) 给出了一个例子：蓝色目标 $P$ 有两个峰，绿色模型 $Q$ 被限制为只有一个峰。训练可以调整绿色曲线的位置和宽度，但不能把它变成双峰。

::eric-forward-kl::

最小化 $D_{\mathrm{KL}}(P\|Q)$ 时，按蓝色 $P$ 的概率密度加权。两个峰附近都有较多概率，都会对加权结果产生影响。如果绿色 $Q$ 在其中一个峰附近很低，那里的 $\log(P/Q)$ 就很大，会增大 KL。由于绿色曲线只能有一个峰，它往往通过变宽，同时覆盖蓝色的两个峰。

::eric-reverse-kl::

最小化 $D_{\mathrm{KL}}(Q\|P)$ 时，改为按绿色 $Q$ 的概率密度加权。如果绿色曲线铺在蓝色两峰之间的低谷上，这里的 $P$ 很小，而 $Q$ 不小，$\log(Q/P)$ 就会产生较大的正贡献。绿色曲线集中到一个蓝色峰附近，可以减少这种贡献；另一个蓝色峰附近由于 $Q$ 很小，在平均中的权重也很小。

前一种倾向称为 mass-covering：尽量覆盖目标分布的主要概率区域。后一种称为 mode-seeking：集中到目标分布的某个峰附近。图中的差异与“绿色只能有一个峰”这个限制有关；如果模型能够完整表示蓝色分布，两个方向都可以在 $Q=P$ 时达到零。具体拟合结果仍取决于分布形状和优化过程。

## 四、在 PyTorch 中对齐数学量

### 硬标签：CE 与 LogSoftmax + NLL

把第一章的计算写成 PyTorch，有两种等价方式：直接把 logits 交给 `cross_entropy`，或者先用 `log_softmax` 得到 log-prob，再交给 `nll_loss`。下面使用普通硬标签、默认平均方式，不加类别权重或 label smoothing：

```python
logits = torch.tensor([[0.2, 1.1, -0.4]], requires_grad=True)
target = torch.tensor([0])

ce = F.cross_entropy(logits, target)
logq = F.log_softmax(logits, dim=-1)
nll = F.nll_loss(logq, target)
```

`ce` 与 `nll` 数值相等，梯度也相等。使用 `cross_entropy` 时直接传 logits，无需先调用 softmax。先 softmax 再把结果当 logits 传入，会改变预测分布和训练目标。[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html)

### 软标签：CE、entropy 与 KL

下面直接计算前面的软标签例子。为方便复现指定的模型概率，取它的对数作为 logits；再做 softmax 就恢复原分布。

```python
p = torch.tensor([[0.5, 0.3, 0.2]])
logits = torch.tensor([[0.2, 0.5, 0.3]]).log().requires_grad_()
logq = F.log_softmax(logits, dim=-1)

ce = -(p * logq).sum(-1).mean()
entropy = -(p * p.log()).sum(-1).mean()
kl = F.kl_div(logq, p, reduction="batchmean")
# ce = 1.2535, entropy = 1.0297, kl = 0.2238
```

这里 `F.kl_div(logq, p)` 计算的是 $D_{\mathrm{KL}}(p\|q)$。它用 `p` 作为权重，计算 `p * (log(p) - logq)` 并求和。**函数参数先写模型的 log-prob，KL 公式却先写目标分布 $p$，两者顺序不同。** 本例的 `p` 都大于零，可以直接计算 `p.log()`。

对于 `[B, V]` 的输入，`B` 是样本数，`V` 是类别数。`batchmean` 先把每个样本所有类别的贡献相加，再对 `B` 个样本取平均；`mean` 则除以 `B × V`，结果会再小 `V` 倍。语言模型的输入若是 `[B, T, V]`，还多了长度为 `T` 的 token 维度：`batchmean` 仍只除以 `B`。要得到每个有效 token 的平均 KL，需要先选出有效位置，再除以有效 token 数。[PyTorch KLDivLoss](https://docs.pytorch.org/docs/2.12/generated/torch.nn.KLDivLoss.html)

## 五、回到语言模型训练

### 预训练与 SFT：条件 NLL

可以把语言模型在每个位置的预测看成一次分类：输入是前面的文本，类别是词表中的 token，标签是文本中实际出现的下一个 token。把各位置给真实 token 的概率相乘，就得到整段文本的概率。其中 $x_{<t}$ 表示第 $t$ 个 token 之前的文本：

$$
q_\theta(x_{1:T})=\prod_{t=1}^{T}q_\theta(x_t\mid x_{<t}).
$$

取负对数后，乘积变成求和。每个位置的 loss 仍然是“读取真实类别的概率，再取负对数”，与第一章的分类器一样。实际训练通常只对参与训练的位置取平均，用 $\mathcal M$ 表示这些位置的集合：

$$
\mathcal L=-\frac{1}{|\mathcal M|}\sum_{t\in\mathcal M}
\log q_\theta(x_t\mid x_{<t}).
$$

预训练时，模型根据文本中实际出现的前缀预测下一个 token。常见的 SFT 做法是把 prompt 作为输入，只对 assistant response 中指定的 token 计算 loss。被 mask 的位置不参与这个平均。因此，看训练代码时还要确认哪些 token 被计入 loss，以及最后除以的是 token 数还是序列数。

在固定 tokenizer 和相同评价口径下，perplexity 是平均 NLL 的指数 $\exp(\mathcal L)$。它反映模型给目标文本分配概率的能力；任务正确率还取决于生成与评价方式。

### 蒸馏：让学生拟合教师分布

教师在同一上下文下给出分布 $p_{\mathrm{teacher}}$，学生给出 $q_\theta$。如果教师分布固定，最小化 CE 与最小化 $D_{\mathrm{KL}}(p_{\mathrm{teacher}}\|q_\theta)$ 有相同梯度。软标签把教师对其他 token 的相对偏好也传递给学生。

这与第一章的软标签分类相同，只是猫、狗、鸟变成了词表中的各个 token。蒸馏中还常用 temperature 调整分布的集中程度；比较 CE 与 KL 时，要使用同一组温度处理后的分布。

### GRPO：当前策略位于 KL 的第一项

在 GRPO 训练中，常见的 KL 约束方向是：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=H(\pi_\theta,\pi_{\mathrm{ref}})-H(\pi_\theta).
$$

这里比较的是同一上下文下，当前模型与 reference 对下一个 token 的概率分布。$\pi_\theta$ 是正在训练的模型，$\pi_{\mathrm{ref}}$ 是固定的 reference。**这次参与加权的是会随训练变化的 $\pi_\theta$。** 因此，$H(\pi_\theta)$ 也会变化，不能像固定教师分布的蒸馏那样，把 KL 中的 entropy 项当常数去掉。

对整个词表求和可以计算这个 KL，也可以用采样近似：从当前模型 $\pi_\theta$ 抽取 token，对每个抽到的 token 计算 $\log\pi_\theta-\log\pi_{\mathrm{ref}}$，再取平均。抽到哪些 token、各出现多少次，就近似实现了按 $\pi_\theta$ 加权。单个 token 的差值可能为负，少量样本的平均也可能为负；保证非负的是完整分布上的 KL。如果 token 来自更新前的 old policy，采样频率就不再对应当前模型，不能直接把这个平均当作当前 KL 的无偏估计。

阅读训练目标时，可以先确认三个问题：**按哪个分布加权，计算哪些概率的对数，训练会改变哪些分布。** 例如，SFT 使用固定标签、计算当前模型的负对数概率；固定教师的蒸馏按教师分布加权；这里的 GRPO KL 则按当前模型分布加权。这些区别决定了哪些 loss 等价、哪些项必须保留。

## 延伸阅读

- [colah · Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/)：适合建立 entropy、CE 与 KL 的编码直觉，矩形面积图尤其清晰。
- [Eric Jang · A Beginner's Guide to Variational Methods](https://blog.evjang.com/2016/08/variational-bayes.html)：适合继续理解 KL 的方向，以及受限分布族的拟合行为。
- [Lilian Weng · From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)：把 reconstruction 与 KL 放进完整的生成模型目标中，适合作为后续拓展。
