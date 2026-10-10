Gemma 的 RMSNorm 把每个通道的缩放系数写成 $1 + w$：参数 $w$ 初始化为 0，计算时再加上常数 1。本文称这个常数为 offset。weight 以 BF16 存储时，offset 在哪一步相加会改变结果：在 BF16 中相加，$1 + 0.00390625$ 得 1.0；先把 $w$ 转成 FP32 再相加，得 1.00390625。笔者在 Liger Kernel 的 CuTe DSL 后端中把这次加法移进 kernel、在 FP32 中完成，BF16 Gemma RMSNorm 的输出 Y 相对 PyTorch 参考实现的相对 L2 误差修复前为 2.9e-3，修复后为 2.3e-5。

## 一、1 + w 参数化

RMSNorm 对一行输入 $x \in \mathbb{R}^N$ 计算：

$$
y = \hat x \odot (1 + w), \qquad \hat x = r \cdot x, \qquad r = \Big(\tfrac{1}{N}\textstyle\sum_i x_i^2 + \epsilon\Big)^{-1/2}
$$

$r$ 是这一行的 rstd，$\odot$ 表示逐元素相乘。Llama 等模型直接乘一个初始化为 1 的缩放系数 $g$；Gemma 的官方实现把 $g$ 写成 $1 + w$，$w$ 初始化为 0。两种写法在数学上等价，区别在于存储和训练的参数是 $w$ 而不是 $g$。

这种写法有两个公开的理由。第一个与 weight decay 有关：weight decay 把参数拉向 0，作用在 $g$ 上会把缩放系数拉向 0，作用在 $w$ 上则把缩放系数拉向 1。Qwen 团队在 Qwen3-Next 的介绍中写到，Qwen3 部分层的 norm weight 出现异常高的值，Qwen3-Next 因此改用 Zero-Centered RMSNorm，并对 norm weight 施加 weight decay。第二个与浮点精度有关，Ceramic.ai 的一篇文章提出：浮点数在 0 附近刻度最细，训练后的 $w$ 多是接近 0 的小数，存 $w$ 比存 $g$ 保留更多有效位。

| 缩放系数 | 模型 |
| --- | --- |
| $1 + w$，$w$ 初始化为 0 | Gemma、Gemma 2、Gemma 3，Qwen3-Next、Qwen3.5 |
| $g$，初始化为 1 | Llama、Qwen3、Gemma 3n、Gemma 4 |

Gemma 3n 与 Gemma 4 的官方实现回到了直接乘 $g$ 的写法，不涉及本文的问题。

## 二、BF16 的 ulp 与舍入

BF16 有 1 位符号、8 位指数、7 位尾数。尾数加上隐含的最高位共 8 位有效位，每个 2 倍区间 $[2^e, 2^{e+1})$ 等分为 128 份。相邻两个可表示数的间隔称为 ulp，在这个区间内为 $2^{e-7}$：

| 数值所在区间 | ulp |
| --- | --- |
| $[2^{-7}, 2^{-6})$，含 0.01 | $2^{-14} \approx 6.1 \times 10^{-5}$ |
| $[0.5, 1)$ | $2^{-8} = 0.00390625$ |
| $[1, 2)$ | $2^{-7} = 0.0078125$ |
| $[8, 16)$ | $2^{-4} = 0.0625$ |

运算结果不能精确表示时，取最近的可表示数；恰在两个可表示数正中时，取尾数末位为偶数的那个，即 round-half-to-even。$1 + 0.00390625 = 1.00390625$ 恰在 1 与 1.0078125 正中，1 的尾数末位为 0，结果为 1.0。$w = 0.003$ 时 $1 + w$ 同样舍入为 1.0，$w$ 的信息全部丢失；$w = 0.005$ 时结果为 1.0078125。

::offset-rulers::

$1 + w$ 落在 1 附近，舍入误差最大为半个 ulp：$w > 0$ 时为 $2^{-8}$，$w < 0$ 时为 $2^{-9}$。$w$ 本身在 0.01 附近的 ulp 为 $6.1 \times 10^{-5}$，在 BF16 中先加 1，$w$ 中小于约 0.004 的部分被舍去，$1 + w$ 参数化在精度上的好处随之消失。FP32 有 24 位有效位，1 附近的 ulp 为 $2^{-23}$，在 FP32 中相加的误差可以忽略。以 $w$ 服从标准差 0.01 的正态分布、N = 4096 为例（CPU 数值计算），99.3% 的 $1 + w$ 在 BF16 中被舍入改变。

::offset-checkpoint-placeholder::

## 三、误差在 Y、dX 与 dW 中的去向

设 BF16 中相加得到的缩放系数为 $1 + w + \delta$，$\delta$ 为舍入误差。RMSNorm 的三个输出为：

$$
\begin{aligned}
Y &= \hat x \odot (1 + w + \delta) \\
dX &= r\Big(u - \hat x \cdot \tfrac{1}{N}\textstyle\sum_i u_i \hat x_i\Big), \qquad u = dY \odot (1 + w + \delta) \\
dW &= \textstyle\sum_{\text{rows}} dY \odot \hat x
\end{aligned}
$$

Y 与 dX 都含缩放系数，每个元素多出约 $\delta / (1 + w)$ 的相对误差，最大约 $2^{-8}$，与输出写回 BF16 时的舍入误差同一量级，相当于每个元素多做了一次舍入。dW 不含 $w$，不受影响。FusedAddRMSNorm 先把残差加到输入上再做 RMSNorm，误差的去向相同。

PR 中的测量（M = 128，N = 4096，BF16；H100 与 B200 结果相同）：

| 对比基准 | 输出 | 修复前 | 修复后 |
| --- | --- | --- | --- |
| Liger Triton 实现，最大绝对误差 | Y | 0.0625 | 0 |
| | dX | 0.0625 | 0.0078125 |
| | dW | 0 | 0 |
| PyTorch 参考实现，相对 L2 误差 | Y | 2.9e-3 | 2.3e-5 |
| | dX | 2.9e-3 | 1.2e-5 |

Liger 的 Triton 实现一直在 FP32 中加 offset，修复后 CuTe DSL 的 Y 与 Triton 实现逐位一致。两组测量的 $w$ 分布不同。第一组来自 Liger 的一致性测试，$w$ 服从标准正态分布，$1 + w$ 可达 4 以上；Y 的绝对值在 $[8, 16)$ 时 ulp 为 0.0625，舍入结果相差一个 ulp，误差即为 0.0625。第二组 $w$ 的标准差为 0.01。CPU 上只在 $1 + w$ 与输出两处做 BF16 舍入，第二种分布下 Y 的相对 L2 误差为 2.9e-3，第一种分布下 Y 的最大绝对误差为 0.0625，与 PR 的测量一致。

这一误差只在 weight 以 BF16 存储时出现。推理时 weight 通常是 BF16；混合精度训练若以 FP32 保存 weight，`weight + offset` 本身就是 FP32 加法。

## 四、在 kernel 内的 FP32 中相加

修复前，Liger CuTe DSL 后端在调用 kernel 之前执行 `weight + offset`，得到一个新的 BF16 向量 W_eff；kernel 读入 W_eff，转成 FP32 参与乘法，backward 复用同一个 W_eff。舍入发生在 kernel 之外的这次加法中。修复后，kernel 接收原始的 $w$ 与 offset，在寄存器中把 $w$ 转成 FP32 后再加 offset：

```python
# 修复前（kernel 外）：BF16 加法，结果舍入为 BF16
w_kernel = weight + offset

# 修复后（kernel 内）
w = tXrW.load().to(cute.Float32)
if const_expr(self.weight_offset != 0.0):
    w += Float32(self.weight_offset)
y *= w
```

backward 中计算 $u = dY \odot (1 + w)$ 的两处做同样的修改。写法沿用 quack 的 `quack/rmsnorm.py`：offset 是编译期常量，`const_expr` 使 offset 为 0 的 kernel 不含这次加法；offset 因此也进入编译缓存的 key，每个 offset 值编译一个 kernel。

::offset-dataflow::

修复同时去掉了一个 kernel 和一个临时向量：每次 forward 少一次 N 个元素的 elementwise kernel，少分配一个 N × 2 字节的 BF16 向量，N 为 4096、8192、18432 时分别为 8、16、36 KiB。PR 在 H100 上测得每次 forward 少 2–3 µs，这部分接近固定开销，与 M 无关：1 × 4096 时 forward 修复前为 8.9 µs，修复后为 6.1 µs；8192 × 4096 时修复前为 54.0 µs，修复后为 52.0 µs。

::offset-trace-placeholder::

## 五、各实现的做法

| 实现 | $1 + w$ 在哪里计算 | 精度 |
| --- | --- | --- |
| HF transformers | forward 中 `output * (1.0 + self.weight.float())` | FP32 |
| Liger Triton | kernel 内，offset 为运行时参数 | FP32 |
| quack、Liger CuTe DSL | kernel 内，offset 为编译期常量 | FP32 |
| FlashInfer `gemma_rmsnorm` | kernel 内，`weight_bias` 为运行时参数 | FP32 |
| TransformerEngine | kernel 内；cuDNN 路径在 2025 年 4 月前以 gamma 的 dtype 相加 | FP32 |
| vLLM | 每次 forward 计算 `self.weight.float() + 1.0`，传入 FP32 weight | FP32 |
| SGLang | 主路径调用 FlashInfer；融合路径用加载时算好的 `gemma_weight` | FP32 / weight 的 dtype |
| Unsloth | Triton kernel 内 | FP32 |

$1 + w$ 可以在三个位置计算：加载权重时算一次并存下来、每次 forward 在 kernel 外计算、在 kernel 内计算。

- **加载时计算**：没有每步开销，但结果以 weight 的 dtype 存储，BF16 下带有本文所述的舍入，还多占一份 weight 大小的显存。SGLang 为 TRT-LLM 的 allreduce 融合等不支持 offset 的 kernel 保留了这份 `gemma_weight`。
- **kernel 外计算**：每次得到一个 FP32 向量，精度没有损失，代价是一次额外的计算，以及 weight 与输入的 dtype 不再相同。vLLM 中要求两者 dtype 相同的 `vllm_c` 与 oink 实现因此不会被选中，Gemma 回退到 native 实现。
- **kernel 内计算**：既没有舍入，也没有额外的 kernel 与显存，前提是 kernel 支持 offset 参数。HF 的 Gemma 实现中有一行注释指向 transformers#29402，该 PR 在 2024 年 3 月指出 $1 + w$ 应在 FP32 中计算；TransformerEngine 在 #1690 中把 cuDNN 路径的默认行为改为 FP32 相加，并保留环境变量 `NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE=1` 恢复旧行为。

## 参考

- luca-888. [fix(cutedsl): fuse Gemma RMSNorm weight offset in FP32](https://github.com/linkedin/Liger-Kernel/pull/1462). linkedin/Liger-Kernel#1462, 2026.
- Liger Kernel. [源码 d5f2817](https://github.com/linkedin/Liger-Kernel/tree/d5f281789aeb5d8d13ffb6825d8e74486af7442c)：`src/liger_kernel/ops/backends/_cutedsl/rms_norm.py`、`fused_add_rms_norm.py`、`_cute_lib/rmsnorm_fwd.py`，`src/liger_kernel/ops/rms_norm.py`，`test/cutedsl/test_rms_norm.py`。
- Dao-AILab. [quack 源码 35266c3](https://github.com/Dao-AILab/quack/tree/35266c3298f0e9bf6d5f46c30aace2eaeae517e3)：`quack/rmsnorm.py`。
- Google DeepMind. [gemma 源码 0f7e742](https://github.com/google-deepmind/gemma/tree/0f7e7420e8)：`gemma/gm/nn/_layers.py`、`gemma/gm/nn/gemma4/_layers.py`。
- Hugging Face. [transformers 源码 536ecc0](https://github.com/huggingface/transformers/tree/536ecc0)：`models/gemma/modeling_gemma.py`、`models/gemma3/modeling_gemma3.py`、`models/gemma4/modeling_gemma4.py`、`models/qwen3_next/modeling_qwen3_next.py`。
- Daniel Han. [Gemma bug fixes - Approx GELU, Layernorms, Sqrt(hd)](https://github.com/huggingface/transformers/pull/29402). huggingface/transformers#29402, 2024.
- Qwen Team. [Qwen3-Next：迈向更极致的训练推理性价比](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list). 2025.
- Yi Liu, Tom Costello, Sean Costello. [Zero-centered Re-parameterization of LayerNorm](https://www.ceramic.ai/blog/zerocentered). Ceramic.ai, 2025.
- NVIDIA. [Support computing zero-centered gamma in compute dtype for CuDNN](https://github.com/NVIDIA/TransformerEngine/pull/1690). TransformerEngine#1690, 2025.
- vLLM. [源码 c41b263](https://github.com/vllm-project/vllm/tree/c41b2639e29c3bc01add1d34bef3032a6d9d8aca)：`vllm/model_executor/layers/layernorm.py`、`vllm/ir/ops/layernorm.py`、`vllm/kernels/vllm_c.py`、`vllm/kernels/oink_ops.py`。
- SGLang. [源码 6fc8d9d](https://github.com/sgl-project/sglang/tree/6fc8d9da3288e9710a6f9a1f59503cacf8a984f0)：`python/sglang/srt/layers/layernorm.py`。
- FlashInfer. [源码 c5bb61f](https://github.com/flashinfer-ai/flashinfer/tree/c5bb61fbc524e0fa0e795273c1e89c414fd0bc03)：`include/flashinfer/norm.cuh`。
- Unsloth. [源码 f33fc6a](https://github.com/unslothai/unsloth/tree/f33fc6ad1dfe0ab102ccb6189dda5fd3e10c09a4)：`unsloth/kernels/rms_layernorm.py`。
