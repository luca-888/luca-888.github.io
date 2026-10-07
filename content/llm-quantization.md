量化是用更少的 bit 存储模型参数，目的是减少显存占用、加快 decode。Qwen3.8-27B 用 BF16 存储是 51.8 GiB，INT4 量化后是 18.1 GiB。

## 一、量化与舍入误差

4 bit 有 $2^4 = 16$ 种取值，按有符号整数解释为 −8 到 7，本文称为编号。编号乘以 scale $s$ 得到它表示的数值，16 个编号因此对应数轴上间隔为 $s$ 的 16 个刻度。$s$ 通常取 $\max|w| / 7$，使绝对值最大的数对应编号 7。

::quant-codes::

量化时每个数取最近的刻度，即 round-to-nearest（RTN）：

$$
\hat w = s \cdot \mathrm{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil,\,-8,\,7\right)
$$

::quant-rulers::

在同一区间 −2 到 2 内，8 bit 有 $2^8 = 256$ 个刻度，间隔约 0.016，0.37 的误差为 0.005；16 bit 有 65536 个刻度，误差小于 0.0001。4 bit 只有 16 个刻度，误差为 0.12，约为原值的三分之一。

以灰度图类比，每个像素的亮度对应一个权重。按 8 bit 存储时与原图没有可见差别；按 4 bit 存储时只有 16 级亮度，球面出现明显的色阶断层。

::quant-picture-bits::

bit 数确定后刻度数量随之固定，各种量化方法的区别在于两点：

1. **刻度的分布**：刻度如何排列，以及多少个数共享一个 scale。
2. **舍入方式**：RTN 对每个数独立取最近的刻度，误差可能朝同一方向累积；GPTQ 用尚未量化的权重补偿已产生的误差，部分权重因此不取最近的刻度，但整层输出的误差更小。

## 二、权重的分布

下图是 Qwen3.8-27B 第 0 层 MLP 的 `down_proj` 矩阵，共 8913 万个权重。标准差为 0.0106，99.99% 的权重绝对值小于 0.044，最大值却是 0.98，为标准差的 92 倍。这类远离主体分布的数称为 outlier。

::quant-weight-hist::

按最大值确定 scale：

$$
s = \frac{0.98}{7} \approx 0.14
$$

刻度间隔为 0.14，而 99.98% 的权重离 0 比离 ±0.14 更近，全部量化为 0。以“舍入误差的均方根 ÷ 权重的标准差”衡量，整层共用一个 scale、刻度等距时，这一层的误差为 99%。

刻度等距，而权重集中在零附近；整层共用一个 scale，一个 outlier 就使所有权重都使用粗刻度。

## 三、刻度的分布：浮点格式

权重集中在零附近，刻度也应在零附近更密，浮点格式正是这样排列的。浮点数由指数和尾数组成：指数决定数量级，尾数在每个数量级内等分，因此小数值的刻度细，大数值的刻度粗。

FP8 常用的格式是 E4M3：4 位指数、3 位尾数。3 位尾数把每个 2 倍区间等分为 8 份：1 到 2 之间的刻度间隔为 0.125，2 到 4 之间为 0.25，0.5 到 1 之间为 0.0625。数值越小，刻度越密。

FP8 同样需要 scale，用于把整层最大的数 0.98 对齐到 FP8 能表示的最大值 448。以这一层中典型大小的权重 0.01 为例：INT8 的刻度间隔为 0.98 / 127 ≈ 0.0077，0.01 附近只有 0.0077 和 0.0154 两个刻度；FP8 在 0.01 附近的刻度间隔约为 0.0011，是 INT8 的 1/7。对整层统计，同为 8 bit、整层一个 scale，INT8 的误差为 21%，FP8 为 2.7%。

在灰度图类比中，于球旁加一盏亮度为 100 的灯作为 outlier，球身亮度在 0 到 1.5 之间。等距的 INT8 需要覆盖到 100，刻度间隔约 0.8，球身只用到 3 个刻度；FP8 在零附近刻度密集，球身用到 79 个。

::quant-picture-fp8::

FP8 的刻度排列本身就能容纳大跨度的数值，因此不需要额外的算法。vLLM 文档推荐的 `FP8_DYNAMIC` 方案对权重直接做 RTN。[vLLM FP8 文档](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/) [Kurtic et al., 2024](https://arxiv.org/abs/2411.02355) 的评测中，这样得到的 FP8 模型在各规模的 Llama-3.1 上基本无损；Qwen 官方也直接发布了 Qwen3.8-27B 的 FP8 版本。

同样的排列方式可以用于 4 bit。FP4（E2M1）有 1 位符号、2 位指数、1 位尾数，可表示的绝对值只有 0、0.5、1、1.5、2、3、4、6 八个，零附近间隔 0.5，远处间隔 2。[NVIDIA, 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) 但在 4 bit 下只调整刻度排列并不够：八个值太少，一个 outlier 仍会使整层的刻度变粗。

## 四、scale 的粒度：分组量化

outlier 只有少数几个，不必让所有权重都使用粗刻度。分组量化把权重沿矩阵的行连续划分为小组，每组使用独立的 scale：

- 每组的 scale 由组内绝对值最大的数决定。
- 一个 outlier 只影响它所在的组。

仍以上述那一层为例，每 128 个值一组时，outlier 所在组的 scale 为 0.98 ÷ 7 ≈ 0.14，而各组 scale 的中位数为 0.0042，刻度间隔缩小到约 1/33。使用等距 4 bit 刻度时，整层一个 scale 的误差为 99%，每 128 个值一个 scale 时为 12%。

scale 本身也需要存储。INT4 通常每 128 个权重配一个 FP16 的 scale，平均每个权重 4.125 bit。组越小，outlier 的影响范围越小，scale 的存储开销越大。

NVFP4 同时采用这两种方法：刻度使用零附近密集的 FP4，每 16 个值配一个 FP8（E4M3）的 scale，整个 tensor 另有一个 FP32 的 scale，平均每个值 4.5 bit。[NVIDIA, 2025](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) 在同一层权重上，NVFP4 直接舍入的误差为 9.4%。

把带灯的灰度图量化到 4 bit：INT4 每 128 个像素一组，灯所在的整块区域都受影响；NVFP4 每 16 个像素一组，只影响灯周围的小块区域。

::quant-picture-groups::

::quant-nvfp4-levels::

## 五、舍入方式：GPTQ

RTN 使每个权重尽量接近原值，但模型需要的是这一层的输出接近原输出。

取矩阵中一行的 4 个权重：0.37、0.61、0.35、0.56。这一行的输出是各权重与对应输入的乘积之和；四个输入都取 1 时，输出为四个权重之和 1.89。scale 取 0.25 并各自取最近的刻度，四个数都向下舍入为 0.25、0.50、0.25、0.50，输出变为 1.50，比原值少 0.39。

GPTQ 按顺序逐个量化，每量化一个权重，就把它的误差分摊给尚未量化的权重。

| 步骤 | 舍入 | 误差补偿 | 剩余权重 |
| --- | --- | --- | --- |
| 第 1 个 | 0.37 → 0.25，输出少 0.12 | 后三个各加 0.04 | 0.65、0.39、0.60 |
| 第 2 个 | 0.65 → 0.75，输出多 0.10 | 后两个各减 0.05 | 0.34、0.55 |
| 第 3 个 | 0.34 → 0.25，输出少 0.09 | 最后一个加 0.09 | 0.64 |
| 第 4 个 | 0.64 → 0.75 | — | — |

量化结果为 0.25、0.75、0.25、0.75，每个权重仍只使用 16 个刻度，输出为 2.00，与原值相差 0.11。以上为示意数值。

图像处理中的误差扩散（Floyd–Steinberg dithering）采用相同的思路：按顺序处理像素，把每个像素的舍入误差分给尚未处理的相邻像素。同样只有 16 级亮度，色阶断层消失。区别在于 GPTQ 保持的不是人眼看到的平均亮度，而是这一层的输出。

::quant-picture-diffusion::

实际输入并非全为 1，且各不相同。GPTQ 的目标是在一批真实输入上最小化这一层的输出误差 $\lVert WX - \hat W X \rVert^2$，其中 $X$ 是几百条校准样本在这一层的输入。[Frantar et al., 2023](https://arxiv.org/abs/2210.17323) 误差不再平均分摊，而是按输入之间的相关性分摊：只有输入与当前权重的输入高度相关的权重，才能补偿它的误差。相关性由 $H = 2XX^\top$ 给出，即这个二次目标的 Hessian。

::quant-gptq-columns::

GPTQ 从左到右逐列量化。所有行共用同一个 $H$ 和同一个量化顺序，并按 128 列分块更新，175B 参数的模型约 4 个 GPU 小时即可完成量化。论文中 OPT-175B 量化到 4 bit 后 perplexity 上升 0.03，RTN 上升 2.2。

### AWQ：按 activation 缩放重要 channel

AWQ 从另一个角度保护输出：输入幅度大的 channel 会放大对应权重的舍入误差，本文称这些 channel 为重要 channel。[Lin et al., 2024](https://arxiv.org/abs/2306.00978) AWQ 把重要 channel 的权重乘以 $s > 1$，对应的输入除以 $s$，乘积不变。组内的 scale 基本不变，舍入的绝对误差也不变，但输入缩小到 $1/s$，该 channel 对输出的误差也降到约 $1/s$。$s$ 取每个 channel 输入平均幅度的 $\alpha$ 次幂，$\alpha$ 在 0 到 1 之间搜索。

两种方法可以叠加使用。RedHatAI 发布的 Qwen3.8-27B INT4 版本先做 AWQ 缩放，再用 GPTQ 量化到 4 bit，校准数据 512 条；GPQA Diamond 上 BF16 为 89.2，INT4 为 87.9。[RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)

## 六、显存占用

公开的量化版本不会把每一层都量化到同一位宽，因此体积不是恰好一半或四分之一。以下按 Hugging Face 上各 checkpoint 的 safetensors 元数据统计：

| 版本 | 体积 | 构成 |
| --- | --- | --- |
| BF16（Qwen） | 51.8 GiB | 全部 16 bit |
| FP8（Qwen） | 28.8 GiB | 大部分层 8 bit，23.0 GiB；embedding 等少数层未量化，5.7 GiB |
| NVFP4（NVIDIA） | 20.4 GiB | MLP 与 `lm_head` 用 FP4，8.6 GiB；它们的 scale，1.1 GiB；attention 用 FP8，6.7 GiB；未量化的层，4.1 GiB |
| INT4（RedHatAI） | 18.1 GiB | 4 bit 权重与 scale，12.5 GiB；未量化的层，5.6 GiB |

各版本在一张 32 GiB 的 RTX 5090 上的显存占用：

::quant-memory-budget::

BF16 超出约 20 GiB；FP8 可以放下，但只剩 3 GiB；两个 4 bit 版本剩余 12 到 14 GiB。剩余显存用于 KV cache，KV cache 越大，可同时服务的请求越多、上下文越长。vLLM recipes 的配置与此一致：FP8 版本使用两张 5090，NVFP4 版本一张即可启动。[vLLM recipes](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

KV cache 也可以量化：`--kv-cache-dtype fp8` 在写入时把 K 与 V 除以 scale 存为 8 bit，读取时再乘回。[vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) 对全部为 attention 层的模型，同样的显存可以存放两倍的 token。Qwen3.8-27B 的 64 层中只有 16 层是 attention，其余 48 层是 Gated DeltaNet，每个请求保存一份固定大小的状态，不受该参数影响，因此收益较小：recipes 在单张 RTX 5090 上记录的 KV cache 容量，BF16 为 7.6 万 token，FP8 为 9.1 万 token。

## 七、decode 速度

decode 时每个权重对每个 token 做一次乘加，计 2 FLOPs；一个 BF16 权重占 2 字节，因此单请求时每读取 1 字节只需 1 FLOP。GPU 的算力远高于此：RTX 4090 峰值约 165 TFLOPS，显存带宽约 1 TB/s，读取 1 字节的时间内可完成 165 FLOPs。[Lin et al., 2024](https://arxiv.org/abs/2306.00978) 计算单元大部分时间在等待数据，每一步的耗时约等于“权重字节数 ÷ 带宽”。

因此权重字节数越少，decode 越快。社区在同一台 DGX Spark 上测试了 Qwen3.8-27B 三种格式的单请求速度：

| 权重格式 | 体积 | 单请求速度 | perplexity |
| --- | --- | --- | --- |
| BF16 | 51.8 GiB | 14.6 tok/s | 7.99 |
| FP8（Qwen） | 28.8 GiB | 20.6 tok/s | 8.03 |
| NVFP4（NVIDIA） | 20.4 GiB | 38.6 tok/s | 8.14 |

数据来自 [Rieker, 2026](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)，测试时开启了 MTP speculative decoding。NVFP4 的体积是 BF16 的 39%，速度是 BF16 的 2.6 倍。速度还取决于 kernel：同为 NVFP4，不同的混合精度配置与 kernel 路径下，速度在 22.7 到 38.6 tok/s 之间。

## 八、大 batch 下的量化

batch 增大时，读取一次权重可以服务 batch 中的所有请求，每字节对应的计算量随 batch 线性增长。当它达到 GPU 每字节可完成的计算量时，瓶颈从带宽转为算力。按 RTX 4090 的规格，BF16 在 batch 约 165 时转为受算力限制；INT4 每个权重约半个字节，batch 约 43 时即转为受算力限制。

受算力限制后，INT4 不再有优势。在没有原生 4 bit 计算的 GPU 上，INT4 权重需要先还原为 16 bit 再计算：读出编号，乘以所在组的 scale。还原本身需要计算，乘法仍在 16 bit 上进行，计算量与 BF16 相同。下图是 Marlin 论文对 Llama-2-7B 的测量：batch 1 时 INT4 是 BF16 的 2.93 倍，batch 128 时是 1.20 倍。

::quant-marlin-speedup::

单层 kernel 的上限是 3.87 倍，即 16 bit 与 4.125 bit 的字节数之比。端到端还包含 attention、采样等开销，加速比更低。

### Marlin：还原与访存重叠

batch 较小时，只有把还原和计算都隐藏在显存读取的时间内，加速才能接近上限。早期的 4 bit kernel 在 batch 1 时都能做到，但 batch 增大后加速迅速下降。Marlin 的目标是在转为受算力限制之前始终接近上限。[Frantar et al., 2024](https://arxiv.org/abs/2408.11743)

整数转浮点的指令较慢，Marlin 改用位运算：FP16 的数值在 1024 到 2048 之间时，尾数最低位恰好表示 1，把 4 bit 编号 $n$ 直接写入尾数的低 4 位，得到的就是 $1024 + n$，再减去一个常数即可。一个 32-bit 寄存器存放两个 FP16，因此一条 `lop3` 指令处理两个权重；最后乘以所在组的 scale。

::quant-marlin-dequant::

读取方面，Marlin 在加载模型时对权重重排一次，使每个线程一次读取 16 字节，读到的正好是它接下来要用的 32 个权重。读取通过 `cp.async` 从显存直接写入 shared memory，并维持 4 级 pipeline：当前块在计算时，后续三块已在传输。activation 被反复使用，保留在 L2 cache；权重只用一次，读取时带 `evict_first` 提示。在单层 kernel 上，Marlin 在 batch 16 到 32 以内保持接近 3.87 倍的加速。

### activation 量化

要降低计算成本，矩阵乘的另一个操作数，即层间传递的 activation，也需要使用低位宽。量化方案因此按量化对象命名，W 后为权重的位宽，A 后为 activation 的位宽：

| 方案 | 量化对象 | 收益 |
| --- | --- | --- |
| W4A16（INT4） | 只有权重 | 读取的字节减少；计算仍在 16 bit |
| W8A8（FP8 或 INT8） | 权重与 linear 层的输入 | 读取减半，矩阵乘也在 8 bit 上进行 |
| W4A4（NVFP4） | 权重与 linear 层的输入 | 读取约为 1/4，矩阵乘也在 4 bit 上进行 |

activation 也有与权重相同的问题：少数几个 channel 的数值约为其余 channel 的 100 倍，且固定出现在这几个 channel 上。一个 tensor 共用一个 scale 时，scale 由 outlier 决定，其余 channel 只用到两三个刻度。[Xiao et al., 2023](https://arxiv.org/abs/2211.10438) activation 随请求变化，无法离线逐个处理，常用的方法有两种。

第一种是 SmoothQuant，采用与 AWQ 相同的等价缩放，但方向相反：把 activation 的第 $j$ 个 channel 除以 $s_j$，权重对应的行乘以 $s_j$。$\alpha = 0.5$ 时，缩放后 activation 与权重在每个 channel 上的最大值相等，量化难度被均分。

$$
s_j = \frac{\max|X_j|^{\alpha}}{\max|W_j|^{1-\alpha}}
$$

::quant-smooth-channels::

第二种是改用浮点刻度，并在运行时计算 scale。`FP8_DYNAMIC` 对 activation 按 token 实时计算 scale，不需要校准数据；SmoothQuant 主要用于只支持 INT8 的旧硬件。NVFP4 在 Blackwell 上把 activation 也量化到 4 bit，Tensor Core 原生支持 FP4 矩阵乘，读取和计算的成本同时降低。也有工作把 Qwen3.8-27B 全部 496 个 linear 层量化为 W4A4，体积 17.5 GiB，五项评测的平均分与 BF16 相差 0.52。[Kozyrev & Maiboroda, 2026](https://arxiv.org/abs/2609.04098)

[Kurtic et al., 2024](https://arxiv.org/abs/2411.02355) 的结论与此一致：单请求、低延迟场景下 W4A16 最划算，高并发吞吐场景下 W8A8 领先。

## 九、vLLM 中的量化

量化后的模型保存为 compressed-tensors 格式，`config.json` 中记录每层权重与 activation 的位宽、类型和分组方式。该格式由 llm-compressor 生成：

```python
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])               # 需要校准数据
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])  # 不需要
```

vLLM 加载时分两步选择 kernel：先由 `CompressedTensorsConfig` 把每层的配置匹配到一个 scheme，再由 scheme 在按性能排序的 kernel 列表中选取第一个当前硬件可用的。

| checkpoint | scheme | NVIDIA GPU 上的 kernel |
| --- | --- | --- |
| W4A16 | `CompressedTensorsWNA16` | Hopper 用 Machete，其余用 Marlin |
| FP8 W8A8 | `CompressedTensorsW8A8Fp8` | Ada 及更新的 GPU 用 FlashInfer 或 CUTLASS 的 FP8 矩阵乘 |
| NVFP4 W4A4 | `CompressedTensorsW4A4Fp4` | Blackwell 用 FlashInfer 或 CUTLASS 的 FP4 矩阵乘；其余退回只量化权重的 kernel，如 Marlin |
| FP8，旧 GPU | `CompressedTensorsW8A16Fp8` | Marlin 等 kernel，只按 FP8 读取权重，计算仍在 16 bit |
| INT8 W8A8 | `CompressedTensorsW8A8Int8` | CUTLASS 的 INT8 矩阵乘 |

硬件支持按架构划分：FP8 计算从 Ada 开始支持，FP4 计算从 Blackwell 开始。更早的 GPU 加载同一个 checkpoint 仍能节省显存，但没有计算上的收益。vLLM 也支持在加载时量化：`--quantization fp8_per_tensor` 把 BF16 模型直接转为 FP8。[vLLM Online Quantization](https://docs.vllm.ai/en/latest/features/quantization/online/) 早期的量化入口大多已被替代：AutoAWQ 已弃用，算法并入 llm-compressor；GGUF 与 bitsandbytes 的支持迁移到了独立 plugin。[vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)

其他系统的做法类似。SGLang 同样加载离线量化的 checkpoint，GPTQ 模型也使用 Marlin；其文档指出，KV cache 的还原如果没有与 attention kernel 融合，额外开销会抵消节省显存的收益。TensorRT-LLM 由 NVIDIA Model Optimizer 负责量化，重点是 FP8 与 NVFP4。llama.cpp 的 GGUF 允许同一模型的不同 tensor 使用不同位宽，例如 Q4_K_M 把 Llama-3.1-8B 从 32.1 GB 压缩到 4.9 GB，主要面向 CPU 与消费级设备。

## 十、小结

4 bit 只有 16 个刻度，量化方法的区别在于刻度的分布和舍入方式。在刻度的分布上，浮点格式使刻度在零附近更密，分组量化让每个 scale 只服务一小组数。在舍入方式上，GPTQ 不让每个权重独立取最近的刻度，而是用尚未量化的权重补偿已产生的误差，以保持这一层的输出。

Qwen3.8-27B 的两个公开 4 bit 版本分别依靠其中一点保持精度。NVIDIA 的 NVFP4 版本依靠刻度的分布：浮点刻度加每 16 个值一个 scale，scale 用 Local-Hessian 方法选取，GPQA Diamond 上量化前为 88.9，量化后为 88.0。[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) RedHatAI 的 INT4 版本依靠舍入方式：等距刻度，结合 AWQ 与 GPTQ，量化前为 89.2，量化后为 87.9。两个版本都能放进一张 32 GiB 的显卡；NVFP4 版本的单请求生成速度是 BF16 的 2.6 倍。

精度损失因模型与任务而异：Marlin 论文中的 Llama-2-7B INT4 模型，MMLU 量化前为 47.88，量化后为 43.59。量化模型上线前，需要在自己的任务上评测。

## 参考

- Frantar, Ashkboos, Hoefler, Alistarh. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). ICLR 2023.
- Lin et al. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). MLSys 2024.
- Xiao, Lin, Seznec, Wu, Demouth, Han. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438). ICML 2023.
- Frantar, Castro, Chen, Hoefler, Alistarh. [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743). 2024.
- Kurtic, Marques, Pandit, Kurtz, Alistarh. [“Give Me BF16 or Give Me Death”? Accuracy-Performance Trade-Offs in LLM Quantization](https://arxiv.org/abs/2411.02355). 2024.
- Alvarez et al. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference). NVIDIA, 2025.
- Kozyrev, Maiboroda. [Why Gated DeltaNet Survives 4-Bit Quantization](https://arxiv.org/abs/2609.04098). 2026.
- Qwen3.8-27B 的各个 checkpoint：[BF16](https://huggingface.co/Qwen/Qwen3.8-27B) · [FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [INT4（RedHatAI）](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) · [NVFP4（NVIDIA）](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)。
- Rieker. [NVFP4 Shootout (Quality and Speed)](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192). 2026.
- vLLM Team. [Recipes: Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).
- vLLM Team. [Quantization](https://docs.vllm.ai/en/latest/features/quantization/) · [FP8 W8A8](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/) · [INT4 W4A16](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4/) · [Online Quantization](https://docs.vllm.ai/en/latest/features/quantization/online/) · [Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`csrc/libtorch_stable/quantization/marlin/`、`vllm/model_executor/kernels/linear/`、`vllm/model_executor/layers/quantization/`。
- SGLang Team. [Quantization](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantization.mdx) · [Quantized KV Cache](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx).
- NVIDIA. [TensorRT-LLM Quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md).
- ggml-org. [llama.cpp quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
- 本站计算：[真实权重的分布与舍入误差](/measurements/llm-quantization/weights.json)（`scripts/llm-quantization/weight_hist.py`）· [各组的 scale](/measurements/llm-quantization/group-scales.json)（`scripts/llm-quantization/group_scale.py`）。
