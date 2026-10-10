# BF16 下 Gemma RMSNorm 的 1 + w：调研笔记

## 范围与状态

- 2026-10-10 开始调研。案例为作者提交的 linkedin/Liger-Kernel#1462（2026-10-06 合并）。定位为科普：讲 1 + w 应该在哪一步相加，PR 作为真实案例，不写成 PR 报告。
- 2026-10-10 协调会话（代用户统筹五篇）确认方案：不做新实测、不下载 checkpoint，Gemma 3 norm weight 分布与 nsys trace 在正文留占位；删去延迟柱图，改为一句固定开销说明；图 1、图 2 跳过草图直接精画；理解检查不等用户回答，题目与参考答案写在本文末尾。
- 当前状态：正文、两张自绘图、两处占位、页面入口完成，`status: 'draft'`；已做成稿自查与 de-ai-edit（subagent 改 3 处，其中删去的“（CPU 数值计算）”标注按“演示数据须标明”恢复）。
- 本阶段不做 GPU 实测。PR 描述中的 H100/B200 数字可用，正文标明出自 PR。`scripts/bf16-weight-offset/cpu_rounding_demo.py` 是 CPU 数值演示（float64 计算，只在 1 + w 与输出处做 BF16 舍入），不算实测。
- 与站内文章的边界：`rmsnorm.md` 讲 RMSNorm 计算与 Triton 融合，`llm-quantization.md` 讲浮点刻度与量化。本篇只就地交代 BF16 刻度与 RMSNorm 公式，不重复融合与量化内容。
- 不越界：CUDA Graph 正确性、Qwen-VL 视频 token、Liger 接入 HF 模型、semantic cache 由其他会话负责。

## 源码版本

| 仓库 | commit | 日期 | 备注 |
| --- | --- | --- | --- |
| Liger-Kernel（上游） | `d5f2817` | 2026-10-07 | `reference-repos/Liger-Kernel-upstream`；已包含 #1462 |
| quack | `35266c3` | 2026-09-12 | 本次加入 sync.sh；`weight_offset` 由 `d6f421d9`（2026-07-09）引入 |
| transformers | `536ecc0` | 2026-10-09 | |
| vLLM | `c41b2639e2` | 2026-10-10 | |
| SGLang | `6fc8d9da` | 2026-10-10 | |
| FlashInfer | `c5bb61f` | 2026-09-29 | |
| Unsloth | `f33fc6a` | 2026-10-10 | |
| google-deepmind/gemma | `0f7e7420e8` | 2026-10-06 | 经 `gh api` 读单个文件，未克隆 |
| TransformerEngine | `76ae4b0981` | 2026-10-09 | 经 `gh api` 读单个文件，未克隆 |

## 参考清单（按重要程度）

| 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| [linkedin/Liger-Kernel#1462](https://github.com/linkedin/Liger-Kernel/pull/1462) 描述与 diff | luca-888（作者本人） | 已读全文与 diff | 核心 | 案例、修改前后代码、误差与延迟数字 |
| Liger 源码：`ops/backends/_cutedsl/rms_norm.py`、`fused_add_rms_norm.py`、`_cute_lib/rmsnorm_fwd.py`、`ops/rms_norm.py`（Triton）、`test/cutedsl/test_rms_norm.py` | Liger 团队 | 已读相关段落 | 核心 | 现状、Triton Gemma 路径、parity 测试的 weight 分布 |
| quack `quack/rmsnorm.py` | Tri Dao 等 | 已读 forward / backward 的 `weight_offset` 段落与 `rmsnorm_ref` | 核心 | FP32 加 offset 的参考做法、编译期常量 |
| transformers `GemmaRMSNorm` / `Gemma2` / `Gemma3` / `Gemma3n` / `Gemma4` / `Qwen3Next` | HF | 已读 | 核心 | 参考实现在 FP32 加 1；Gemma 3n/4 去掉 offset |
| [huggingface/transformers#29402](https://github.com/huggingface/transformers/pull/29402) | Daniel Han（Unsloth） | 已读描述与评论 | 重要 | 2024-03 指出 “(w+1) should be done in float32”；该 PR 关闭未合并，修复经其他 PR 进入主线 |
| [NVIDIA/TransformerEngine#1690](https://github.com/NVIDIA/TransformerEngine/pull/1690) | jberchtold-nvidia | 已读描述；读了 `common.cpp`、`rmsnorm_fwd_kernels.cuh` 的相关行 | 重要 | 同类问题的先例：cuDNN 路径原先在 gamma dtype 中加 1，2025-04 改为默认 FP32 |
| vLLM `layers/layernorm.py`（GemmaRMSNorm）、`ir/ops/layernorm.py`、`kernels/vllm_c.py`、`kernels/oink_ops.py` | vLLM | 已读 | 重要 | 每次 forward `weight.float() + 1.0`，FP32 weight |
| SGLang `layers/layernorm.py`（GemmaRMSNorm）、`model_loader/loader.py` | SGLang | 已读相关段落 | 重要 | 主路径交给 FlashInfer；另存 BF16 的 `gemma_weight = w + 1` 给融合路径 |
| FlashInfer `include/flashinfer/norm.cuh` | FlashInfer | 已读 | 重要 | `weight_bias` 作为运行时 float 参数，FP32 中相加 |
| google-deepmind/gemma `gm/nn/_layers.py`、`gm/nn/gemma4/_layers.py` | Google DeepMind | 已读 RMSNorm 类 | 重要 | 1 + scale、zeros 初始化；Gemma 4 改为 ones 初始化直接乘 |
| [Qwen3-Next 官方 blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) | Qwen 团队 | **未读原页**（JS 渲染，只经搜索摘要看到原句） | 重要 | 采用 Zero-Centered RMSNorm 并对 norm weight 做 weight decay 的动机 |
| [Ceramic.ai: zero-centered](https://www.ceramic.ai/blog/zerocentered) | Yi Liu, Tom Costello, Sean Costello | 已读（经 WebFetch 摘要） | 补充 | 参数化动机：浮点在 0 附近刻度细；weight decay 应拉向 1 |
| Unsloth `kernels/rms_layernorm.py` | Unsloth | 已读 `_gemma_rms_layernorm_forward` | 补充 | 全程 FP32 |
| TransformerEngine `pytorch/module/rmsnorm.py` 文档 | NVIDIA | 已读 | 补充 | `zero_centered_gamma` 的定义 |

## 阅读要点

### Liger #1462

- 修改前：CuTe DSL 的 Gemma 路径在 host 端执行 `w_eff = W + offset`。W 为 BF16 时，这是一次 BF16 加法，结果先舍入为 BF16，再在 kernel 中转 FP32 参与乘法。forward 和 backward 都用这个 `w_eff`（存在 ctx 中）。
- 修改后：Gemma 模式下传原始 W 和 `weight_offset`；forward kernel 中 `w = load().to(Float32); w += offset; y *= w`，backward 的两处 `wdy = dy * w` 同样处理。`weight_offset` 是 `const_expr` 编译期常量，进入编译缓存 key（`_FWD_COMPILE_CACHE`、`_BWD_COMPILE_CACHE`）。非 Gemma casting mode 仍在 host 端预加（Liger 内置的模型中，offset=1 只与 `casting_mode="gemma"` 搭配，Gemma 4 用 offset=0）。
- 顺带的收益：每次 forward 少一个 N 元素的 elementwise kernel 与一个 BF16 临时向量。N=4096/8192/18432 时临时向量为 8/16/36 KiB（N × 2 字节），与 PR 中 peak 分配的节省一致。
- PR 数字（H100 与 B200 相同）：BF16、M=128、N=4096，相对 Triton 的最大绝对误差 Y 0.0625 → 0，dX 0.0625 → 0.0078125，dW 0 → 0。相对 PyTorch 参考的相对 L2（weight 为 0.01 × randn）：RMSNorm 的 Y 0.00292 → 2.3e-05，dX 0.00288 → 1.22e-05（H100）。
- PR 延迟：forward 降低 3.8–31.4%（H100）、13.9–42.1%（B200）。按 PR 表格计算，H100 上七个 RMSNorm 形状的 forward 节省为 1.9–2.8 µs，B200 为 2.9–4.9 µs，大致是一个固定量，所以小 M 时比例大。
- 修复后 dX 相对 Triton 仍差 0.0078125：待核实原因（推测为 backward 中 `c = mean(wdy · x̂)` 的归约顺序不同）。
- CuTe DSL backend 在 Liger 中是 opt-in（`LIGER_KERNEL_IMPL=cutedsl`，`_cutedsl` 没有 `default_devices`），默认用户走 Triton。

### 现状变化

- Liger main（`d5f2817`）已包含 #1462 的写法，CuTe DSL 与 Triton 的 Gemma 路径一致。
- Liger Triton Gemma 路径一直在 FP32 中加 offset：forward `W_row.to(fp32)` 后 `offset + W_row`，backward `W_row + offset.to(tl.float32)`（BF16 + FP32 提升为 FP32）。所以 PR 中 “相对 Triton 的误差变为 0”。
- CuTe DSL 文件头注释 “gemma: pre-cast x to fp32 host-side” 已与代码不符：现在只有 x 与 W dtype 不同时才 host 端转 FP32（`_promote_gemma_fp32`）。正文不涉及。
- quack 在 2026-07-09（`d6f421d9`）加入 `weight_offset`，注释原文为 fp32 add so (1 + w) doesn't round through the weight dtype。Liger 的 CuTe DSL RMSNorm 后端（#1299 2026-07-15、#1417 2026-08-31）移植自更早的 quack，没有这个参数，于是在 host 端预加。
- 模型侧：Gemma 3n 与 Gemma 4（HF 与官方 JAX 实现）改为 ones 初始化、直接乘 scale，不再有 1 + w；Liger 的 Gemma 4 patch 用 offset=0。仍用 1 + w 的 HF 模型：gemma、gemma2、gemma3、recurrent_gemma、t5gemma、t5gemma2、vaultgemma、qwen3_next、qwen3_5、qwen3_5_moe、nemotron（LayerNorm1P）等。

### 相关系统

| 系统 | 1 + w 在哪里算 | 精度 |
| --- | --- | --- |
| Gemma 官方 JAX | `normed_inputs * (1 + scale)`，与输入同 dtype | 取决于调用方 dtype |
| HF transformers | forward 中 `1.0 + self.weight.float()` | FP32 |
| Liger Triton | kernel 内 | FP32 |
| Liger CuTe DSL（#1462 后） / quack | kernel 内，offset 为编译期常量 | FP32 |
| FlashInfer `GemmaRMSNorm` | kernel 内，`weight_bias` 为运行时 float 参数 | FP32 |
| TransformerEngine | kernel 内 `g_ij += 1`（compute_t）；cuDNN 路径 #1690 前在 gamma dtype，现在默认 FP32，`NVTE_ZERO_CENTERED_GAMMA_IN_WTYPE=1` 保留旧行为 | FP32 |
| vLLM | 每次 forward `self.weight.float() + 1.0`，传 FP32 weight 给 `ir.ops.rms_norm`（参考语义 `x.to(weight.dtype) * weight`）。`vllm_c` 与 oink 实现要求 weight 与 x 同 dtype，因此 Gemma 不走它们 | FP32 |
| SGLang | 主 CUDA 路径交给 FlashInfer（kernel 内 FP32）；加载时另存 `gemma_weight = w + 1`，dtype 为模型 dtype，供 TRT-LLM allreduce 融合、CuTe DSL layer boundary、ROCm vLLM kernel、NPU 大 hidden、MiniMax-M3 qk norm 使用 | 融合路径为 BF16（待核实运行时 dtype） |
| Unsloth | Triton kernel 内全程 FP32 | FP32 |

### 1 + w 参数化的动机

- Gemma 官方代码只有实现（zeros 初始化、乘 1 + scale），没有解释。
- Ceramic.ai：浮点在 0 附近刻度最细，scale 存在 0 附近更精确；weight decay 作用在 w 上等于把 scale 拉向 1。该文报告 Gemma 3 后几层的 gamma 均值约 300（未核实）。
- Qwen3-Next blog（经搜索摘要）：Qwen3 中观察到部分 norm weight 异常增大，Qwen3-Next 改用 Zero-Centered RMSNorm 并对 norm weight 加 weight decay。
- 本文的关联：1 + w 参数化本身就是把 scale 放到浮点刻度细的位置；在 BF16 中先加 1，又把它放回刻度粗的位置。

## CPU 数值演示（`scripts/bf16-weight-offset/cpu_rounding_demo.py`）

- BF16 有 8 位有效位（7 位存储尾数 + 隐含位）。刻度间隔：0.01 附近 6.1e-5（2^-14），[0.5, 1) 为 2^-8 = 0.00390625，[1, 2) 为 2^-7 = 0.0078125，[4, 8) 为 2^-5，[8, 16) 为 2^-4 = 0.0625。
- 1 + 0.00390625 = 1.00390625 恰在 1 与 1.0078125 正中，round-to-nearest-even 取尾数为偶的 1.0。w = 0.003 也得 1.0；w = 0.005 得 1.0078125；w = −0.003 得 0.99609375。
- M=128、N=4096、seed 42：
  - w ~ 0.01 × randn：99.3% 的 1 + w 被改变，最大舍入误差 3.9e-3；Y 相对 L2 2.85e-3、最大绝对差 0.03125；dX 相对 L2 2.88e-3。与 PR 的 0.00292 / 0.00288 量级一致。
  - w ~ randn（Liger parity 测试的分布）：43.9% 被改变，最大舍入误差 1.56e-2；Y 与 dX 最大绝对差 0.0625，与 PR 中相对 Triton 的 0.0625 一致（|1 + w| 可达 4 以上，刻度更粗）。
- dW = Σ dy · x̂ 不含 w，因此不受影响。

## 待核实（处理结果）

1. Qwen3-Next blog 原句：已用内置浏览器打开中文原页（2025-09-11）核对：“在Qwen3中我们采用了QK-Norm，我们发现部分层的 norm weight 值会出现异常高的现象……采用了 Zero-Centered RMSNorm [7]，并在此基础上，对 norm weight 施加 weight decay”。正文转述，不引原句。
2. SGLang `gemma_weight` 的 dtype：源码中 buffer 由 `torch.ones_like(self.weight)` 创建，与 weight 同 dtype（加载在 `set_default_torch_dtype(model_config.dtype)` 下）。正文只写“weight 的 dtype”，不写运行时实测。
3. vLLM：`IrOp.dispatch` 按优先级取第一个 `supports_args` 为真的实现，native 兜底；`vllm_c` 与 oink 都要求 weight 与 x 同 dtype，aiter 只在 ROCm。Gemma 传 FP32 weight，CUDA 上落到 native。是否被 torch.compile 融合未核实，正文不写。
4. 真实 checkpoint 的 norm weight 分布：未核实，正文留占位；Ceramic 的“约 300”不写进正文。
5. 修复后 dX 相对 Triton 仍差 0.0078125 的原因：未核实（推测为归约顺序），正文只列数字不解释。
6. HF 实际合并 #29402 修改的 PR：未查到，正文只写 HF 代码注释指向 #29402。

## 理解检查

供用户自测，先用自己的话回答，再对照参考答案。

1. **BF16 中 1 + 0.00390625 为什么得 1.0？1 − 0.00390625 为什么能精确表示？**
   参考答案：[1, 2) 的 ulp 为 2^-7 = 0.0078125，1.00390625 恰在 1 与 1.0078125 正中，round-half-to-even 取尾数末位为偶数的 1.0。1 − 0.00390625 = 0.99609375 落在 [0.5, 1)，那里 ulp 为 2^-8，它正好是一个刻度，不需要舍入。
2. **修复前后 dW 完全不变，dX 却变了，原因是什么？**
   参考答案：dW = Σ dY ⊙ x̂ 不含缩放系数；dX 中 u = dY ⊙ (1 + w)，既直接出现，也经过 Σ u·x̂ 进入每个元素，所以 1 + w 的舍入误差进入 dX。
3. **训练时 weight 以 FP32 保存（autocast），修复前的 Liger CuTe DSL 还会有这个误差吗？**
   参考答案：不会。`weight + offset` 是 FP32 加法；x 为 BF16、weight 为 FP32 的情形走 `_promote_gemma_fp32` 路径，weight 以 FP32 进入 kernel。误差只在 weight 以 BF16 存储时出现。
4. **forward 节省的时间为什么大致是 2–3 µs 的固定量，而不随 M × N 增长？**
   参考答案：省掉的是一个只处理 N 个元素的 elementwise kernel（与 M 无关），以及一次 N × 2 字节的分配。它的耗时主要是 kernel 启动与调度的固定开销，所以 M 小时占比大（1 × 4096 时 forward 修复前为 8.9 µs，修复后为 6.1 µs），M 大时占比小。
5. **SGLang 为什么在加载时预先算好一份 `w + 1`？它和修复前的 Liger 有什么相同之处？**
   参考答案：TRT-LLM allreduce 融合等 kernel 不支持 offset 参数，只能接收现成的缩放系数，预先算一次避免每步开销。相同之处是 `w + 1` 以 weight 的 dtype 存储，BF16 下同样舍去了 w 的低位；不同之处是 SGLang 只算一次，修复前的 Liger 每次 forward 都要算并分配临时向量。

## 交付（2026-10-10）

未 commit、未 push。

新增：

- `content/bf16-weight-offset.md`：正文（约 10.9K 字符，含参考）
- `docs/bf16-weight-offset-notes.md`：本笔记
- `posts/bf16-weight-offset/index.html`：页面入口
- `src/articles/bf16-weight-offset/bf16-weight-offset.tsx`：页面入口代码
- `src/articles/bf16-weight-offset/bf16-weight-offset.blocks.tsx`：正文与图表占位映射
- `src/articles/bf16-weight-offset/Bf16WeightOffsetFigures.tsx`：图 1 `OffsetRulers`、图 2 `OffsetDataflow`、两个实测占位
- `scripts/bf16-weight-offset/cpu_rounding_demo.py`：CPU 数值演示

修改的共享文件：

- `vite.config.ts`：`rollupOptions.input` 增加一条 `bf16WeightOffset: 'posts/bf16-weight-offset/index.html'`
- `src/posts.ts`：`posts` 数组开头增加 `slug: 'bf16-weight-offset'` 条目，`status: 'draft'`

仓库外（不随本仓库提交）：

- `~/Documents/ChatGPT/reference-repos/sync.sh`：kernel 库分组增加一行 `quack`；新浅克隆 `reference-repos/quack`
- `reference-repos/Liger-Kernel`（用户 fork 的完整克隆）：fetch 了上游 main，只更新了 `FETCH_HEAD`，分支与工作区未动

验证：`npx tsc --noEmit -p tsconfig.json` 通过（整个项目）；dev server（本会话启动，端口 5173）中打开 `/posts/bf16-weight-offset/`，无 console 错误，两张图正常渲染，KaTeX 无报错。
