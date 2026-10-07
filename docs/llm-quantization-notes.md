# LLM 量化（FP8、INT4 与 Marlin Kernel）文章资料

状态：正文与 7 张图已写完（`content/llm-quantization.md`、`src/articles/llm-quantization/LlmQuantizationFigures.tsx`，`status: draft`）。不做本站实测，数据全部引用公开来源；图为直接精画，未经草图确认。视频的概念与关键帧在 `videos/llm-quantization/`。

## 2026-10-03 视频关键帧方向

- 用户要求“快速做 3b1b 那种，有更强可视化、例子的关键帧”。新建 `videos/llm-quantization/keyframes-3b1b/`，六张 1080 × 1920 草稿，使用深色背景、数轴、等输出线和矩阵图；原视频保留。
- 贯穿示例：输入 (1,1)，权重 (0.37,0.61)，原输出 0.98；步长 0.25 的直接舍入输出 0.75。第一个权重少掉的 0.12 补给第二个，第二个从 0.61 变成 0.73、再舍入到 0.75，最终输出 1.00，误差 0.02。该例解释 GPTQ 的补偿思路，不代表完整 Hessian 校准算法或实测结果。
- 数值格式示例：相同 scale=1，0.6 在 INT4 上舍入为 1，在 E2M1 上舍入为 0.5。只比较该示例，不声称 FP4 对每个数都优于 INT4。
- 显存数字、E2M1 定义沿用已有资料；本轮未新增核心来源、未重新调研。各图问题、后续动效与例子边界记录在该目录的 `README.md`。当前交付范围为关键帧，尚未制作此方向的配音或成片。

## 2026-10-02 改动（用户确认）

- 算例模型由 Qwen3-8B 换成 **Qwen3.8-27B**（用户：Qwen3-8B 偏老偏小）；示例显卡由 RTX 4090 换成 RTX 5090（32 GiB，Blackwell）。
- **不做实测，引用别人的数据**（用户：“不实测吧，找别人的数据就行”）。原“实测”占位一节删除，原“按模型形状估算”的速度上限表与显存图换成公开数据。
- 新增第四节 NVFP4；主标题改为“FP8、INT4、NVFP4 与 Marlin Kernel”。
- 视频：`videos/llm-quantization/film.html` 为 12 幕的动效源文件，`film.mjs` 负责渲染与导出每幕末帧；已出 5 fps 样片，待用户看过后做正式版与音效。

## 范围与授权

- 独立文章，不挂在 Vllm 名下。Slug `llm-quantization`；主标题“LLM 量化”，子标题“FP8、INT4、NVFP4 与 Marlin Kernel”。
- 用户授权（2026-10-01，发起会话转述）：直接写到可发布水平，不等参考方案与草图确认；暂不做硬件测试，正文留“实测”占位；`status: 'draft'`，不 commit、不 push、不发布。
- 源码基准：vLLM v0.30.0（本地浅克隆 `~/Documents/ChatGPT/reference-repos/vllm`，HEAD e006d76 即该 tag）；SGLang main@8854857；TensorRT-LLM main@111687f9。
- 算例模型：Qwen3.8-27B（官方 `config.json`，2026-10-02 读取：64 层，其中 16 层 full attention、48 层 linear attention（Gated DeltaNet）；hidden 5120、intermediate 17408、24 heads / 4 KV heads、head_dim 256、vocab 248320、embedding 不共享、带视觉编码器与 1 层 MTP）。

## 文章主线

量化分两半。离线一半是通用算法：怎么把权重（和 activation）压到低比特而不掉精度（RTN、GPTQ、AWQ、SmoothQuant）。推理一半由引擎实现：低比特权重怎么在 GPU 上算得快（Marlin、FP8 scaled_mm）。连接两半的是 decode 的内存带宽模型：batch 小时每步耗时约等于“读一遍权重的字节数 ÷ 带宽”，所以字节数减少直接变成速度；省下的显存留给 KV cache，变成并发与上下文长度。

## 章节提纲

1. 量化什么：W4A16、W8A8、W4A4、KV cache 四种对象各省什么。
2. 把权重压到 4 bit：均匀量化与 group、GPTQ 的逐列误差补偿、AWQ 的 per-channel 缩放；RedHatAI 的 INT4 版本作实例。（图：GPTQ 误差去向）
3. 把 activation 也压到 8 bit：outlier channel、SmoothQuant；FP8 为何不需要校准。（图：SmoothQuant 缩放前后）
4. NVFP4：E2M1 的取值、16 个值一个 scale、Blackwell 上的 W4A4；公开 checkpoint 的混合精度与精度。（图：INT4 与 NVFP4 的取值）
5. decode 为什么变快：FLOPs/Byte；Qwen3.8-27B 三种格式的体积与速度；转为计算受限的 batch。
6. Marlin：位运算 dequant、离线重排、`cp.async` 四级 pipeline、striped partitioning；加速随 batch 的变化。（图：一个 4-bit 权重的 dequant；图：端到端加速比）
7. vLLM 中的执行路径：compressed-tensors → scheme → kernel 优先级；online quantization；已弃用或迁出的方案。
8. KV cache 量化与显存账。（图：四种 checkpoint 与 32 GiB 显卡）
9. 其他系统：SGLang、TensorRT-LLM、llama.cpp GGUF；公开的不同结论。
10. 小结。

## 引用的数据（Qwen3.8-27B）

| 数据 | 数值 | 来源与条件 |
| --- | --- | --- |
| checkpoint 体积 | BF16 55.56 GB = 51.75 GiB；FP8（官方）30.87 GB = 28.75 GiB；NVFP4（NVIDIA）21.92 GB = 20.42 GiB；NVFP4（RedHatAI）24.69 GB；INT4（RedHatAI）19.45 GB = 18.12 GiB | Hugging Face API 的文件大小，2026-10-02 读取，safetensors 求和。正文与图统一用 GiB |
| 各格式的参数构成 | BF16 27.78B；FP8：24.70B 为 F8_E4M3、3.08B 为 BF16；NVIDIA NVFP4：9.19B 个 U8（即 18.4B 个 FP4 权重，等于 64 层 MLP 加 `lm_head`）、7.21B 为 FP8、2.18B 为 BF16 | 同上（safetensors 元数据）；“等于 MLP 加 lm_head”是自己按 config 验算 |
| 单请求速度与质量 | BF16 14.6 tok/s、PPL 7.993；FP8 20.6、8.029；NVFP4（NVIDIA）38.6、8.139；其余 NVFP4 版本 22.7–34.6 tok/s | HF 讨论帖 “NVFP4 Shootout”（Rieker，2026-09-09，09-12 更新）：单台 DGX Spark，vLLM nightly，single-stream、small-context decode，10 条 prompt，MTP 5 token，`--kv-cache-dtype fp8`，temperature 0；PPL 为 wikitext-2 |
| 并发下的速度 | 16 并发，decode 为主：FP8 99.47 tok/s，NVFP4（unsloth）132.07 tok/s | NVIDIA 开发者论坛（shahizat，2026-08-15），DGX Spark，vLLM 0.27.1。正文未用，备查 |
| INT4 的精度 | IFEval 92.24 → 91.93；MMLU-Pro 84.46 → 83.45；GSM8K Platinum 95.75 → 96.77；MATH-500 83.73 → 83.33；GPQA Diamond 89.23 → 87.88；AIME 2025 95.42 → 94.17 | RedHatAI INT4 模型卡；AWQ smoothing + GPTQ W4A16，512 条 open-perfectblend 样本，忽略视觉部分、`lm_head`、`embed_tokens` 等 |
| NVFP4 的精度 | GPQA Diamond 88.92 → 88.01；Terminal-Bench 75.56 → 74.02；IFBench 80.07 → 78.93 等六项 | NVIDIA NVFP4 模型卡；ModelOpt v0.48.0，MLP 与 `lm_head` 为 NVFP4，attention 与 linear attention 为 FP8 |
| KV cache 容量 | 单张 RTX 5090（可用 31.4 GiB）、NVFP4（Inferact 版）、32K 上下文、`--enforce-eager`：FP8 KV 91,022 token；BF16 KV 76,458；加 `--language-model-only` 135,926。两张 5090（TP2、262K 上下文）：FP8 权重 377,456 token；NVFP4（Inferact）445,875；NVFP4（unsloth）920,517 | vLLM recipes 的 Qwen3.8-27B 页面，启动日志数字 |
| 全 W4A4 的结果 | 496 个 linear 层全部 NVFP4 W4A4，17.5 GiB，五项平均 −0.52 | arXiv 2609.04098 摘要 |
| NVFP4 格式 | E2M1；取值 0、0.5、1、1.5、2、3、4、6；16 个值一个 FP8（E4M3）scale，另有 per-tensor FP32 scale；平均 4.5 bit；MXFP4 为 32 个值一个 2 的幂次 scale | NVIDIA blog（2025-06-24） |

## 核心图

| 图 | 回答的问题 | 数据来源 |
| --- | --- | --- |
| `quant-gptq-columns` | GPTQ 量化一列时，误差去了哪 | 示意，按论文 Algorithm 1 |
| `quant-smooth-channels` | per-channel 缩放把量化难度从哪移到哪 | 示意数值，按 SmoothQuant 的 $s_j$ 公式，α = 0.5 |
| `quant-marlin-dequant` | 一个 4-bit 权重怎么不经类型转换变成 FP16 | vLLM `dequant.h` 的常量，自己代入 n = 11 验算 |
| `quant-marlin-speedup` | INT4 的加速随 batch 怎么变 | Marlin 论文 Table 2，Llama-2-7B / A10 一行（论文数据，非本站实测） |
| `quant-rulers` | 16 bit、8 bit、4 bit 各有多少档，同一个权重落在哪 | 示意数值（−2 到 2，权重 0.37），与视频开场同一画面 |
| `quant-memory-budget` | Qwen3.8-27B 的四种 checkpoint 在一张 32 GiB 卡上各余多少显存 | Hugging Face 文件大小 |
| `quant-nvfp4-levels` | NVFP4 的 16 个编码代表哪些值，和 INT4 有什么不同 | NVIDIA blog 的 E2M1 取值 |

## 参考清单（按重要程度）

| 来源 | 作者 / 团队 | 阅读状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| [MARLIN](https://arxiv.org/abs/2408.11743)（arXiv 2408.11743） | Frantar、Castro、Chen、Hoefler、Alistarh | 源文件 `marlin.tex` 全文已读 | 核心 | 4、5、8 |
| vLLM v0.30.0 `csrc/libtorch_stable/quantization/marlin/`：`dequant.h`、`marlin.cuh`、`marlin_template.h` | vLLM 团队 | `dequant.h` 前 200 行（开头注释与 INT4→FP16/BF16 四个特化）与 `marlin.cuh` 全文已读；`marlin_template.h` 只检索了 stage、tile、`fetch_to_shared` 相关行；`marlin.cu` 与两个 repack 文件未读 | 核心 | 5 |
| vLLM v0.30.0 `vllm/model_executor/kernels/linear/__init__.py`、`mixed_precision/marlin.py`、`machete.py` | vLLM 团队 | 已读 `_POSSIBLE_KERNELS`、`_POSSIBLE_FP8_KERNELS`、`_POSSIBLE_WFP8A16_KERNELS`、backend 映射与两个 `can_implement` | 核心 | 6 |
| vLLM v0.30.0 `vllm/model_executor/layers/quantization/` | vLLM 团队 | 已读 `__init__.py` 全文、`compressed_tensors.py` 的 `_get_scheme_from_parts`、`compressed_tensors_wNa16.py` 前 200 行、`kv_cache.py` 全文、`input_quant_fp8.py` 前 120 行；`fp8.py`（853 行）未通读 | 核心 | 6、7 |
| vLLM 量化文档 `docs/features/quantization/` | vLLM 团队 | `README.md`、`llm_compressor/README.md`、`fp8.md`、`int4.md`、`online.md`、`quantized_kvcache.md` 全文已读；`gguf.md`、`auto_awq.md`、`gptqmodel.md`、`bnb.md` 读了开头 | 核心 | 3、6、7 |
| [GPTQ](https://arxiv.org/abs/2210.17323)（arXiv 2210.17323，ICLR 2023） | Frantar、Ashkboos、Hoefler、Alistarh | 源文件：摘要、引言、Background、算法三步与 Algorithm 1 已读；实验节只检索了关键数字 | 核心 | 2 |
| [AWQ](https://arxiv.org/abs/2306.00978)（arXiv 2306.00978，MLSys 2024） | Lin、Tang、Tang、Yang、Chen、Wang、Xiao、Dang、Gan、Han | 源文件：摘要、`3_approach.tex`、`4_system.tex` 全文已读；实验节未读 | 核心 | 2、4 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438)（arXiv 2211.10438，ICML 2023） | Xiao、Lin、Seznec、Wu、Demouth、Han | 源文件：摘要、`3_motivation_method.tex` 全文已读；实验节只检索了加速数字 | 核心 | 3 |
| [“Give Me BF16 or Give Me Death”?](https://arxiv.org/abs/2411.02355)（arXiv 2411.02355） | Kurtic、Marques、Pandit、Kurtz、Alistarh | 源文件：摘要、贡献列表、三种格式的设置、精度表、性能一节已读；附录未读 | 重点（相关系统与反对意见） | 3、4、8 |
| [Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)（2025-06-24） | NVIDIA（Alvarez 等） | 经 WebFetch 摘要读取格式定义、与 MXFP4 的对比、精度与硬件说明；未逐字通读 | 核心 | 4 |
| Qwen3.8-27B 的模型卡与元数据：[BF16](https://huggingface.co/Qwen/Qwen3.8-27B)、[FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)、[RedHatAI INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)、[RedHatAI NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)、[NVIDIA NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | Qwen、RedHatAI、NVIDIA | `config.json` 全文与 API 元数据已读；模型卡经 WebFetch 摘要读取量化做法与评测表 | 核心 | 引言、2、3、4、8 |
| [NVFP4 Shootout](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)（HF 讨论帖） | Rieker（社区） | 经 WebFetch 摘要读取结果表与测试方法 | 核心（速度数据） | 引言、5 |
| [vLLM recipes: Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | vLLM 团队 | 页面正文已读（至 client usage 之前） | 核心（显存数据） | 7、8 |
| [Why Gated DeltaNet Survives 4-Bit Quantization](https://arxiv.org/abs/2609.04098)（arXiv 2609.04098） | Kozyrev、Maiboroda | 只读了摘要 | 重点 | 4 |
| [NVIDIA 论坛：NVFP4 vs FP8 on DGX Spark](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258) | shahizat（社区） | 经 WebFetch 摘要读取 | 补充 | 未写入正文 |
| [Kaitchup：Fastest Qwen3.8 27B Quantization?](https://kaitchup.substack.com/p/fastest-qwen38-27b-quantization-benchmarking) | Kaitchup | 付费墙，只读到方法与结论；未引用 | 补充 | — |
| SGLang 文档 `advanced_features/quantization.mdx`、`quantized_kv_cache.mdx` | SGLang 团队 | 平台支持表与 KV cache 页开头已读 | 重点（相关系统） | 8 |
| TensorRT-LLM 文档 `docs/source/features/quantization.md` | NVIDIA | 前 70 行已读（recipe 列表、KV cache、ModelOpt 流程） | 重点（相关系统） | 8 |
| [llama.cpp `tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | ggml-org | 前半已读（流程、选项、Llama-3.1 体积表、bits/weight 表开头） | 重点（相关系统） | 8 |
| [Qwen3-8B `config.json`](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | Qwen 团队 | 已读 | 补充 | 4、7 的算例 |
| llm-compressor 官方文档站与仓库 | vLLM 团队 | 未读（只读了 vLLM 文档中 llm-compressor 的页面） | 重点 | 6 |
| Kim et al.，Who Says Elephants Can't Run（Marlin 与 AWQ 引用的 dequant 位运算出处） | NVIDIA | 未读 | 补充 | 5 |
| Machete、Humming kernel 的设计说明 | vLLM / Neural Magic 等 | 未读，只从 kernel 优先级列表与 `can_implement` 得知适用条件 | 补充 | 6 |
| NVIDIA 各代 GPU 的 FP8 / FP16 Tensor Core 规格 | NVIDIA | 未读 | 补充 | 4（未写入具体数字） |

## 阅读要点

- **GPTQ**：逐层最小化 $\lVert WX - \hat W X\rVert^2$，Hessian $H = 2XX^\top$ 只依赖层输入，所有行共享。相对 OBQ 的三处改动：所有行按同一固定顺序逐列量化；每 128 列为一块 lazy 更新；用 Cholesky 预先算出需要的 $H^{-1}$ 行并加 1% 阻尼。校准数据为 C4 中 128 段、每段 2048 token。OPT-175B 约 4 GPU 小时；4-bit 下 perplexity 只升 0.03，RTN 升 2.2。
- **AWQ**：按 activation 幅度选出 0.1%–1% channel 保留 FP16 就能显著恢复精度，但混合精度不利于硬件；改为对 input channel 乘 $s$、输入除以 $s$。误差比约为 $\Delta'/\Delta \cdot 1/s$。$s = s_X^{\alpha}$，$\alpha$ 在 $[0,1]$ 网格搜索，再加权重裁剪。不做回归或反向传播。OPT-6.7B INT3：$s=1$ 时 perplexity 23.54，$s=2$ 时 11.92。
- **AWQ 的系统部分（TinyChat）**：RTX 4090 峰值 165 TFLOPS、带宽 1 TB/s，算术强度低于 165 即带宽受限；FP16 decode 的算术强度约为 1。硬件没有 INT4 × FP16 的乘法指令，必须先 dequant，并且要融合进 matmul kernel。
- **SmoothQuant**：activation 的 outlier 比多数值大约 100 倍，且固定出现在少数 channel；per-channel 量化 activation 无法映射到 GEMM kernel（scale 只能加在外侧维度）。变换 $Y = (X\,\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)\,W)$，$s_j = \max|X_j|^{\alpha} / \max|W_j|^{1-\alpha}$，多数模型取 $\alpha = 0.5$。最高 1.56× 加速、2× 省内存。
- **Marlin**：A10 的 FLOPs/Byte 约 200；4-bit 权重下每读一个权重的时间可做约 100 FLOPs，$b_{\text{opt}} \approx 50$。单层 72k × 18k 矩阵上，batch 16–32 以内接近理想的 3.87×（group 128 的 scale 占 0.125 bit），batch 128 降到约 1.5×。既有 kernel 在 batch 1 也接近 3.87×，但 batch 增大后迅速下降。端到端（vLLM，Llama-2-7B，A10）：batch 1 / 16 / 128 为 2.93× / 2.74× / 1.20×。Llama-2-7B INT4 的 MMLU 47.88 → 43.59。
- **Marlin 的实现要点**：`lop3` 把 4-bit 值写进指数固定的 FP16 尾数再相减；一个 32-bit 寄存器同时处理两个 16-bit 操作数；权重离线重排成每线程一次 16 字节读取恰好是自己要用的数据；`cp.async` 从 global 直接到 shared memory，$P = 4$ 级 pipeline；activation 从 L2 反复读取，权重用 `evict_first`；每个 warp 子 tile 宽 64，多个 warp 累加同一输出再做归约；striped partitioning 让各 SM 的 tile 数均匀。
- **`dequant.h` 验算**：`EX = 0x6400` 是 FP16 的 $2^{10} = 1024$，此时尾数最低位的权重恰为 1；`(q & 0x000F) | 0x6400` 的值为 $1024 + n$；`SUB = 0x6408` 为 1032，相减得 $n - 8$。高 4 位一路用 `MUL = 0x2c00`（1/16）与 `ADD = 0xd480`（−72）：$(1024 + 16n)/16 - 72 = n - 8$。
- **vLLM 的 kernel 选择**：`choose_mp_linear_kernel` 按优先级取第一个 `can_implement` 的 kernel。CUDA 上混合精度的顺序是 CutlassW4A8、Machete（仅 compute capability 9.0）、Marlin、Conch、Exllama、TritonW4A16、Humming。FP8 W8A8 要求 capability ≥ 8.9，否则落到 W8A16（Humming、Marlin）。
- **KV cache**：`--kv-cache-dtype fp8` 时以 `uint8` 存储；scale 默认 1.0，或由 llm-compressor 校准后随 checkpoint 加载；FlashAttention 3 backend 下 query 也量化，attention 在 FP8 域内计算。v0.30.0 的 `CacheDType` 还包括 `int4_per_token_head`、`nvfp4`、`turboquant_*` 等，正文未展开。`gpu_memory_utilization` 默认 0.92。

## 现状变化（以 v0.30.0 源码与文档为准）

| 项目 | 现状 | 出处 |
| --- | --- | --- |
| 推荐入口 | llm-compressor，输出 compressed-tensors 格式 | 量化文档 `README.md` 的 tip |
| AutoAWQ | 库已弃用，AWQ 算法并入 llm-compressor；`awq`、`awq_marlin`、`auto_awq` 三个名字都映射到 `AutoAWQConfig` | `auto_awq.md`、`__init__.py` |
| GPTQ | `gptq`、`gptq_marlin`、`auto_gptq` 都映射到 `AutoGPTQConfig`；文档指向 GPTQModel | `__init__.py`、`gptqmodel.md` |
| GGUF、bitsandbytes | 迁到 out-of-tree plugin（`vllm-gguf-plugin`、`vllm-bnb-plugin`）；GGUF 标注为 experimental、under-optimized | `gguf.md`、`bnb.md` |
| `fbgemm_fp8`、`fp_quant` | 在 `DEPRECATED_QUANTIZATION_METHODS` 中 | `__init__.py` |
| Marlin | 不再只是 INT4 × FP16：支持 GPTQ / AWQ / FP8 / FP4 权重，输入可为 8-bit（模板参数 `is_a_8bit`） | 文档硬件表、`marlin_template.h` |
| Hopper 上的 W4A16 | 优先 Machete，Marlin 次之 | `_POSSIBLE_KERNELS` |
| Online quantization | `--quantization fp8_per_tensor / fp8_per_block / mxfp8 / mxfp4 …`，加载时量化，无需校准数据 | `online.md` |
| 更低比特的浮点格式 | NVFP4、MXFP4、MXFP8 已有 scheme 与 kernel | `schemes/`、`kernels/linear/` |

## 未核实起点的核对结果

| 起点 | 结论 |
| --- | --- |
| 量化对象三种：W4A16、W8A8、KV cache FP8，省的东西不同 | 成立。补充：老 GPU 上 FP8 checkpoint 会以 W8A16 运行；另有 W4A8 等组合。 |
| GPTQ 逐层用二阶信息把误差补偿到其余权重 | 成立。更准确的说法是逐列、所有行同一顺序，补偿给同一行中尚未量化的权重。 |
| AWQ 按 activation 幅度找重要 channel，先缩放再量化 | 成立。缩放系数由 activation 平均幅度的 $\alpha$ 次幂给出，$\alpha$ 网格搜索。 |
| SmoothQuant 把 activation 难量化的部分转移到权重 | 成立。$\alpha = 0.5$ 时是平分而不是全部转移。 |
| decode 受内存带宽限制，权重字节减少会近似同比提速 | 只在小 batch 的单层 kernel 上成立（3.87×）。端到端在 batch 1–16 约 2.7–2.9×，batch 128 只剩 1.2×（Marlin Table 2）。正文按此修改。 |
| GPU 没有原生 INT4 矩阵乘；Marlin 解决解包瓶颈 | 前半成立（AWQ 论文明确写出）。后半不准确：既有 kernel 在 batch 1 已接近理想，Marlin 解决的是 batch 增大后仍保持带宽受限时的最优，dequant 只是其中一环。正文按此修改。 |
| FP8 有硬件原生支持 | 限于 compute capability ≥ 8.9（Ada、Hopper、Blackwell）；Turing / Ampere 上退为 W8A16。 |
| 量化离线完成，vLLM 只负责加载和执行 | 不完全成立：vLLM 有 online quantization；FP8 的 activation 与 KV cache 在运行时量化；Marlin 的权重重排在加载时做。 |

## 相关系统与分歧

- **SGLang**：同样加载离线量化的 checkpoint（AWQ、GPTQ、compressed-tensors、ModelOpt、Quark），NVIDIA 上 `gptq` 已移除、改用 `gptq_marlin`；也有 online quantization，但文档建议优先离线。KV cache 支持 FP8 与实验性的 FP4，并警告 dequant 未与 attention kernel 融合时会很慢。
- **TensorRT-LLM**：量化交给 NVIDIA Model Optimizer，recipe 以 FP8、NVFP4 为主，另有 W4A16 / W4A8 的 AWQ 与 GPTQ；默认 PyTorch backend 面向 Hopper 与 Blackwell。
- **llama.cpp GGUF**：`llama-quantize` 把 GGUF 转成 K-quant 等格式，不同 tensor 可用不同类型，可选 importance matrix；Llama-3.1-8B 从 32.1 GB 到 4.9 GB（Q4_K_M）。
- **分歧 1：INT4 的精度**。Marlin 论文自己的 Llama-2-7B INT4 结果 MMLU 掉 4.3 分；2411.02355 在 Llama-3.1 上报告 W4A16 平均恢复约 99%，Leaderboard V2 上 8B 为 96.1%。两者模型、算法细节与校准数据都不同。
- **分歧 2：AWQ 还是 GPTQ**。AWQ 论文称不依赖回归、泛化更好；2411.02355 报告调过的 GPTQ 在真实任务上优于 AWQ。
- **分歧 3：哪种格式更快**。2411.02355：单请求同步部署 W4A16 最划算；高并发吞吐 W8A8 领先；交叉点随模型、硬件、任务变化。

## 待核实问题

- **KV cache FP8 的收益为什么只有约两成（2026-10-02 读源码，部分确认）。** `Qwen3_5ForConditionalGeneration` 是 `IsHybrid`；GDN 层的状态由 `MambaStateShapeCalculator.gated_delta_net_state_shape` 给出，每层一份 conv 状态（10240 × 3）加一份 temporal 状态（48 × 128 × 128 = 786,432 个数），以 `MambaSpec` 的形式放在同一个 KV cache 池里；状态的 dtype 来自 `mamba_cache_dtype` / `mamba_ssm_cache_dtype`，与 `--kv-cache-dtype` 无关。正文据此写“数据类型不受这个参数影响”。池内各类 block 的具体配比与 7.6 万 → 9.1 万这两个数字的换算没有核对。
- **NVFP4 的 scheme 与 kernel（2026-10-02 读源码确认）。** `_is_nvfp4_format`：tensor-group、float、4 bit、group size 16、symmetric；权重是 NVFP4 时匹配 `CompressedTensorsW4A4Fp4`，没有 input 量化时为 `use_a16=True`。`_POSSIBLE_NVFP4_KERNELS` 在 CUDA 上的顺序：FlashInfer CuteDSL、FlashInfer CUTLASS、FlashInfer b12x、CUTLASS，之后是 weight-only 的 fallback（FlashInfer CuteDSL W4A16、Marlin 等），最后是 emulation。各 kernel 的 `can_implement`（具体的 compute capability 条件）未读，“Blackwell”取自 NVIDIA blog 与 recipes。
- 速度表来自单个社区评测，开启了 MTP；INT4 在 Qwen3.8-27B 上的公开速度数据未找到（Kaitchup 的评测在付费墙后）。
- 第五节的转折 batch（BF16 约 165、INT4 约 43）是用 AWQ 论文给出的 4090 规格推出的理论值。
- recipes 的 NVFP4 是 Inferact 版（全 W4A4），显存图用的是 NVIDIA 版的体积，两者不是同一个 checkpoint；正文只引用 recipes 的 token 数，没有把它画进图里。
- Machete 相对 Marlin 的设计差异与实际收益（只确认了优先级与 Hopper 限定）。
- `kv_cache_dtype=fp8` 在各 attention backend 下是在 FP8 域内计算还是先 dequant，未逐个核对。

## 已搁置的待测项（单卡 RTX 4090，vLLM v0.30.0，Qwen3-8B）

2026-10-02 用户决定本文不做实测，以下为换模型之前的计划，保留备查。

| 待测 | 配置 | 记录 |
| --- | --- | --- |
| 权重显存与 KV 容量 | BF16、FP8（`FP8_DYNAMIC`）、W4A16（GPTQ，group 128）各启动一次，再各加 `--kv-cache-dtype fp8` | 启动日志中的模型权重占用、KV cache 可容纳 token 数；替换显存图 |
| decode 吞吐与 ITL | 三种权重格式 × 并发 1 / 8 / 32 / 128，输入 128、输出 256 | 输出吞吐、ITL 中位数；验证 INT4 加速随并发收窄、FP8 在高并发反超的位置 |
| prefill | 长输入（4096 token）单请求 TTFT | 验证 W4A16 在计算受限阶段无收益 |
| kernel 选择 | 记录日志中的 `Using … for CompressedTensorsWNA16` 与 `Selected <kernel>` | 确认 4090 上 W4A16 走 Marlin、FP8 走哪个 scaled_mm |
| online quantization 对照 | `--quantization fp8_per_tensor` 与离线 FP8 checkpoint | 吞吐与显存是否一致 |
| 精度抽查（可选） | gsm8k 250 条，三种格式 | 只作参考，不替代论文评测 |

量化 checkpoint 需先用 llm-compressor 生成（W4A16 需要校准数据，约 512 条），或直接取现成的公开 checkpoint。

## 封面（2026-10-02）

- 用户要“跟正文配套的 16:9 封面”，并说不确定比例。按 AGENTS.md 的约定默认出 9:16 长竖版，同时附一张 16:9 横版供选。
- 瑞士风格，与视频同一套配色与图形：三把尺子（16 bit 密成一片、8 bit 细齿、4 bit 的 16 根粗刻度），一条紫色虚线代表一个权重，在最后一把尺子上落到最近的一档。中文“量化”为最大标题，英文 LLM Quantization 为辅，小字 FP8 · INT4 · NVFP4；顶部约 12% 不放文字。
- 代码绘制（canvas），不是图像模型生成；源文件 `covers/llm-quantization/swiss-20261002/cover.html`，`node render.mjs` 导出。实际尺寸：`portrait-1080x1920.png` 为 1080 × 1920，`landscape-1920x1080.png` 为 1920 × 1080。待用户评估。

## 别人怎么讲量化（2026-10-03，为视频的讲法与比喻而读）

以下四篇都是经 WebFetch 摘要读取，没有逐字读原文；只用于对照讲解顺序与比喻，不作为事实来源。

| 来源 | 作者 | 讲解顺序 | 比喻 / 演示 |
| --- | --- | --- | --- |
| [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) | Maarten Grootendorst | LLM 太大 → 数怎么用 bit 表示 → 内存 → 量化与数据类型 → 对称 / 非对称 → 范围映射与 clipping（outlier）→ 校准 → PTQ → 4 bit（GPTQ、GGUF）→ QAT → BitNet | 一张图减到 8 种颜色，说明可表示的值变少会变“粗”；outlier 把范围撑大，其余的值挤到同一档；GPTQ 用 inverse Hessian 讲权重的重要性与误差再分配 |
| [Quantization from the ground up](https://ngrok.com/blog/quantization)（2026-03） | Sam Rose（ngrok） | 模型为什么大（参数）→ 计算机怎么存数（bit）→ 能不能用更小的浮点 → 量化是什么（对称 / 非对称）→ 实际怎么做（分块，隔离 outlier）→ 对精度的影响（perplexity、KL、GPQA、对话）→ 对速度的影响 | 没有生活化比喻，全是可交互演示：逐 bit 开关、浮点数浏览器、六个模型的参数直方图（挤在零附近、有 outlier 长尾）、一个 outlier 让平均误差到 116.8%、小网络上“直接取整”对“量化”的对比 |
| [LLM quantisation through interactive visualisations](https://smcleod.net/2024/07/understanding-ai/llm-quantisation-through-interactive-visualisations/) | Sam McLeod | 先给定义，再给比喻 | 把模型数据想成色谱：16 bit 减到 8 bit，仍能看到大范围的颜色，但丢了细节 |
| [Optimum：Quantization 概念指南](https://huggingface.co/docs/optimum/concept_guides/quantization) | Hugging Face | 定义 → 数据类型 → 仿射量化公式 → 对称 → per-tensor / per-channel → 校准 → 步骤；数的 bit 表示放在附录 | 无比喻，公式为主 |

对照视频第 3 版的结论：

- “模型是一堆数 → 一个数怎么用 bit 存 → 更少的 bit → 量化 → outlier 与分块”这条顺序，与 Sam Rose 和 Grootendorst 的前半段一致。
- 用图片讲“可选的值变少”是通行做法（8 种颜色、色谱）。视频里的球属于同一类。
- 用一盏灯当 outlier、用误差扩散对应 GPTQ，在这四篇里没有出现。搜索结果的摘要里有一句把 GPTQ 的步骤称作 error diffusion，但回到它列出的两篇论文（arXiv 2606.01412、2606.10890）核对，摘要与正文都没有这个说法，所以这一点没有出处，仍按“自己的类比”处理。
- 别人有而视频没有的：真实模型的权重直方图（证明“数大多挤在零附近”）；一个 outlier 造成多大误差的具体数字；精度怎么衡量（perplexity、KL）。
- 视频有而这四篇没有的：GPTQ 保的是输出而不是单个权重；公开 checkpoint 的体积拆解；带宽与 batch 对速度的影响。

## 真实权重的分布与舍入误差（2026-10-03，本站计算，供视频用）

脚本 `scripts/llm-quantization/weight_hist.py`，在 Modal 上用 CPU 运行（不占 GPU）：读 `Qwen/Qwen3.8-27B` 各 safetensors 分片的 header，用 HTTP Range 只取 16 个二维权重（第 0、31、63 层与 MTP 层的 MLP，第 31 层的 attention）。结果在 `public/measurements/llm-quantization/weights.json`。

- 口径：只做 RTN（取最近的一档），不做 GPTQ / AWQ；相对误差 = RMS(w − q) ÷ std(w)。INT8、FP8、“INT4 整层”用整个矩阵的最大绝对值定 scale；INT4 每行 128 个一组；NVFP4 每行 16 个一组、E2M1 档位、scale 取组内最大值 ÷ 6（没有把 scale 本身再量化成 FP8）。
- `layers.0.mlp.down_proj.weight`（5120 × 17408）：标准差 0.0106，绝对值最大 0.98，是标准差的 92 倍；99.99% 的数绝对值小于 0.044；99.98% 小于最大值的 1/16。
- 同一矩阵的相对误差：INT8 21%，FP8 2.7%，INT4 整层 99%，INT4 128 一组 12%，NVFP4 9.4%。
- 16 个矩阵的范围：最大值 ÷ 标准差从 12.9 到 92；INT8 2.9% 到 21%；FP8 2.6% 到 2.7%；INT4 整层 53% 到 99.9%；INT4 128 一组 11.8% 到 13.4%；NVFP4 9.3% 到 9.4%。
- 这些数只用于视频第 4 到 6 幕，正文未引用。

## 两个公开 4 bit 版本各用了什么算法（2026-10-03 核对，经 WebFetch 摘要读取模型卡）

- [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)：nvidia-modelopt v0.48.0；NVFP4 层用 Local-Hessian calibration，2,048 条 Nemotron-Post-Training-Dataset-v3 样本；MLP 与 `lm_head` 为 NVFP4，self-attention 与 linear-attention 为 FP8。Local-Hessian 是为每个 16 值的 block 挑 scale、使 Hessian 加权的输出误差最小的算法（[ModelOpt 公告](https://nvidia.github.io/Model-Optimizer/announcements/local-hessian.html)，只读了搜索摘要），不是 GPTQ 式的权重补偿。模型卡没有写明 activation 是否量化。
- [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)：LLM Compressor，`AWQModifier(duo_scaling="both")` 加 `GPTQModifier`，W4A16，512 条 open-perfectblend 样本，最大长度 4096，忽略视觉部分与 `embed_tokens`；GPQA Diamond 89.23 → 87.88。
- 结论：视频结尾不能把 GPTQ 的作用算到 NVFP4 版上；改成两个版本各对应一个选择。

## 2026-10-04 正文按视频的逻辑重写（用户要求）

- 主线改成视频第 3 版的顺序：一个数少用几个 bit（编号 × scale，0.37 → 0001）→ 真实权重的分布（新图 `quant-weight-hist`，本站数据）→ 刻度怎么摆（FP8、FP4）→ 一个 scale 管多少个数（分组、NVFP4）→ 每个数落到哪个刻度（GPTQ 的 4 权重示例，AWQ 作小节）→ 实际省多少（按 dtype 拆体积，KV cache 并入）→ 速度 → batch 变大（Marlin、W/A 命名、SmoothQuant 与 activation 量化并入）→ vLLM 与其他系统（压缩）→ 小结（两个公开版本各靠一个选择）。
- 用词统一为“刻度”与 scale，不再用“步长”“档”“量化级”；图里的旧说法同步改了。
- 正文开始引用本站计算的数：第 0 层 `down_proj` 的分布、INT8 21% / FP8 2.7% / INT4 整层 99% / INT4 128 一组 12% / NVFP4 9.4%，以及各组 scale（灯那组 0.14，中位数 0.0042，`scripts/llm-quantization/group_scale.py` → `group-scales.json`）。
- 没有新增外部来源，结论未变；删去了原第一节的独立命名表（移到第八节）。
- 同日补回视频的图片类比（用户问“颜色的类比没了吗”）：第一、三、四、五节各放一张视频三联图（16/8/4 bit 的球、灯对 INT8/FP8、灯对 INT4-128/NVFP4、误差扩散），从视频静帧裁出，来源记在 `src/assets/llm-quantization/SOURCES.md`。
