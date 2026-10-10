Liger-Kernel 是 LinkedIn 开源的一组 Triton kernel，覆盖 RMSNorm、SwiGLU、RoPE 与 cross-entropy 等训练中的常见算子。它不修改 Hugging Face transformers 的源码，而是在运行时替换模型代码里的类、函数与 forward 方法，即 monkey patch。替换发生在模型构建之前还是之后，决定了哪些模块能换上 Liger 的实现；没有替换到的模块继续使用原实现，程序不报错，输出也在数值误差内一致。Qwen3 的 `q_norm` 与 `k_norm` 在 HF Trainer 的接入方式下以原实现运行了约 17 个月，本文作者在 Liger #1470 中补上后，H100 上 Qwen3-8B 一层的 Q/K norm（forward + backward）修复前为 2.44 ms，修复后为 0.56 ms。

**TL;DR**

- **三种替换**：改模块里的类名，只影响之后构建的实例；改类上的 forward，影响这个类的全部实例；改实例的 forward，只影响这一个对象。
- **两条入口**：构建模型前调用 `apply_liger_kernel_to_*` 时，由模型自己的构建代码换上 Liger 的类；HF Trainer 在模型构建之后才接入，只能按 Liger 写出的属性路径逐个替换。
- **静默回退**：漏写的属性路径、被过滤掉的参数、不支持的模型类型，都只让模块保留原实现，不报错。
- **接口契约**：Liger 替换后的 forward 要声明 `logits_to_keep` 并使用 `shift_labels`，否则生成时会算出整个序列的 logits，切分序列训练时会丢掉 label。
- **其他做法**：kernels hub 按类型遍历全部模块，transformers 的 patching API 在构建时按类名替换，Unsloth 改写整层的 forward。

## 一、Liger 替换的对象

以 Qwen3 为例，`apply_liger_kernel_to_qwen3()` 默认替换四类对象：

| 对象 | 原实现 | Liger 实现 | Qwen3-8B 中的数量 |
| --- | --- | --- | --- |
| RMSNorm | `Qwen3RMSNorm` | `LigerRMSNorm` | 每层 4 个，另有最后一个，共 145 个 |
| MLP | `Qwen3MLP` | `LigerSwiGLUMLP` | 每层 1 个 |
| RoPE | 函数 `apply_rotary_pos_emb` | `liger_rotary_pos_emb` | 每层调用 1 次 |
| lm_head 与 loss | `Qwen3ForCausalLM.forward` | `lce_forward` | 整个模型 1 个 |

每层的 4 个 RMSNorm 分别是 attention 前的 `input_layernorm`、MLP 前的 `post_attention_layernorm`，以及 attention 内的 `q_norm` 与 `k_norm`。后两个在 RoPE 之前对每个 head 的 query 与 key 各做一次 RMSNorm，长度为 head dim 128。

前三类替换的是模块或函数，签名与原实现相同。最后一类替换整个 CausalLM 的 forward：Fused Linear Cross Entropy（FLCE）把 lm_head 与 cross-entropy 合成一个按 token 分块的计算，不生成完整的 logits，所以要改写 forward 里从 hidden states 到 loss 的整段代码。本文把 Liger 替换后的 CausalLM forward 称为 patched forward。

## 二、三种替换方式

Liger 用到三种写法，改动的是三个不同的引用：

```python
modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm                         # ① 模块中的名字
modeling_qwen3.Qwen3ForCausalLM.forward = lce_forward              # ② 类属性
module.__dict__["forward"] = LigerRMSNorm.forward.__get__(module)  # ③ 实例属性
```

`Qwen3Attention.__init__` 中的 `self.q_norm = Qwen3RMSNorm(...)` 在执行时才按名字到 `modeling_qwen3` 的全局命名空间里查找类。① 改的是这个名字的指向，之后构建的模型拿到 `LigerRMSNorm`；已构建的实例在创建时就确定了类型，不受 ① 影响。RoPE 函数也按名字查找，但查找发生在每次 forward 时，所以 ① 对已构建的模型同样生效。

调用 `module(x)` 时，PyTorch 取 `self.forward`，Python 先查实例的 `__dict__`，再查类。② 改类上的 forward，这个类已构建和之后构建的实例都受影响，范围是整个进程。③ 写入实例 `__dict__` 的 forward 遮蔽类上的 forward，只作用于这一个对象。

::liger-binding-scope::

Liger 对已构建的模型使用 ③。参数对象保持不变，optimizer、FSDP 与 PEFT 持有的引用仍然有效；实例的类型仍是 `Qwen3RMSNorm`，所以 Liger 另外替换 `_get_name`，让 `print(model)` 显示 `LigerRMSNorm`，并给实例补上 Liger forward 读取的 `offset`、`casting_mode` 等属性。

## 三、两条入口

Liger 的接入入口分成两类：在模型构建之前替换类，本文称为类路径；在模型构建之后替换实例的 forward，本文称为实例路径。

| 入口 | 时机 | 实际起作用的替换 | 使用者 |
| --- | --- | --- | --- |
| `apply_liger_kernel_to_qwen3()`，再 `from_pretrained` | 构建前 | ①② | LlamaFactory |
| `AutoLigerKernelForCausalLM.from_pretrained` | 构建前 | 按 `model_type` 选函数，再 ①② | Liger 文档 |
| `apply_liger_kernel_to_qwen3(model=model)` | 构建后 | RoPE 靠 ①，模块靠 ③ | HF Trainer（v4.45 起） |

HF Trainer 开启 `use_liger_kernel=True` 时调用的是第三个入口。实例路径同样会执行 ①，但对已构建的模型，① 只对 RoPE 这种在 forward 中按名字调用的函数起作用，模块由随后的 ③ 替换。`apply_liger_kernel_to_qwen3` 遍历每个 decoder 层，对代码中写出的属性路径逐个绑定 forward：

```python
for decoder_layer in base_model.layers:
    if swiglu: _patch_swiglu_module(decoder_layer.mlp, LigerSwiGLUMLP)
    if rms_norm:
        _patch_rms_norm_module(decoder_layer.input_layernorm)
        _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
        _patch_rms_norm_module(decoder_layer.self_attn.q_norm)  # #1470 补上
        _patch_rms_norm_module(decoder_layer.self_attn.k_norm)  # #1470 补上
```

类路径替换哪些模块，由模型自己的构建代码决定：凡是按 `Qwen3RMSNorm` 这个名字构建的模块都会换上。实例路径替换哪些模块，由 Liger 列出的属性路径决定。同一个模型、同一个 Liger 版本，两条入口的覆盖范围可以不同。

## 四、静默回退

本文把 patch 没有覆盖某个模块、该模块继续使用原实现、程序也不报错的情况称为静默回退。Liger 的 RMSNorm 与原实现签名相同、计算相同，输出在 BF16 误差内一致，loss 曲线相同，两者只在速度与显存上不同。

Qwen3 在 2025-05 加入 Liger 时，实例路径只写了 `input_layernorm` 与 `post_attention_layernorm`。之后的 Qwen3-VL 写了 `q_norm` 与 `k_norm`，Qwen3 MoE、Qwen3 Next、Qwen3.5 与 Qwen3.5 MoE 都没有写。用 HF Trainer 训练 Qwen3-8B 时，145 个 RMSNorm 中有 72 个以原实现运行；走类路径则全部换上。

Qwen3.5 是混合架构，每 4 层中 3 层为线性 attention，只有 8 个 full attention 层有 `q_norm` 与 `k_norm`。线性 attention 层的输出 norm 是带 gate 的另一种 RMSNorm，Liger 没有对应实现，两条路径都保留原实现。

::liger-coverage::

#1470 在五个模型的实例路径中补上这两个 norm，复用各模型已有的 RMSNorm 参数（Qwen3.5 的 RMSNorm 用 $1+w$ 作缩放，`offset` 为 1），并受 `rms_norm` 开关控制。PR 中 H100 上的测量（batch 4、序列长度 2048、BF16，forward + backward）：

| 模型 | 覆盖的层 | Q+K norm 修复前 | 修复后 | 单层 attention 修复前 | 修复后 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 36 / 36 | 2.44 ms | 0.56 ms | 8.88 ms | 6.71 ms |
| Qwen3.5-9B | 8 / 32 | 2.56 ms | 0.74 ms | 10.76 ms | 8.80 ms |

原实现由几次逐元素运算组成，每次都把 query 或 key 从显存读写一遍；Liger 用一个 kernel 完成 forward，backward 同样只用一个 kernel。

::liger-measure-train::

漏写属性路径之外，静默回退还有两个来源：

- **参数过滤**：`_apply_liger_kernel_to_instance(model, **kwargs)` 只保留对应 apply 函数签名中有的参数。拼错的开关（例如把 `rms_norm` 写成 `rmsnorm`）被丢弃，模块按默认值替换。
- **模型类型**：`config.model_type` 不在 Liger 的映射表中时，只记一条 info 级日志。

数值比较无法区分两种实现，所以 #1470 的测试直接比较每个 norm 绑定的 forward：`rms_norm=False` 时 forward 不变；正常调用后 `inspect.getsource(norm.forward)` 与 `LigerRMSNorm.forward` 相同，权重对象与 eps 不变，线性 attention 的 norm 不被改动。

::liger-measure-audit::

## 五、patched forward 的接口

HF 的 CausalLM forward 从 base model 取得 hidden states，按 `logits_to_keep` 截取位置，经 lm_head 得到 logits，再调用 `self.loss_function` 计算 loss。patched forward 要保留同样的参数与行为，只在训练时把最后两步换成 FLCE：默认在 `self.training` 且有 label 时走 FLCE，不生成 logits。

生成时 prefill 只需要最后一个位置的 logits。transformers 的 `generate()` 检查 `"logits_to_keep" in inspect.signature(self.forward).parameters`，为真时才传入 `logits_to_keep=1`；参数只包含在 `**kwargs` 中时，检查结果为假。#1181 之前，Qwen3-VL 的 patched forward 签名中没有这个参数，生成时 prefill 会对整个 prompt 计算 logits，也不报错。#1181 按 HF 的写法补上：

```python
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
kept_hidden_states = hidden_states[:, slice_indices, :]
```

`logits_to_keep=0` 时 `slice(0, None)` 取全部位置；传入张量时按下标选取，用于 packed 格式。

::liger-logits::

Qwen3-VL-8B 的词表为 151936。prompt 为 10955 个 token 时，完整 logits 有 1.66 × 10⁹ 个元素，BF16 下为 3.3 GB；只保留最后一个位置时为 0.3 MB。PR 在 H20 上测量了这个长度下单次 forward 的耗时，完整 logits 为 1.67 s，只保留最后一个位置为 1.58 s，省去的是 lm_head 对其余 10954 个位置的矩阵乘。

causal LM 的第 $i$ 个位置预测第 $i+1$ 个 token，loss 函数默认先把 `labels` 左移一位。context parallelism 与 DeepSpeed Ulysses 把一条序列切到多个 rank 上，若在本地左移，每段最后一个位置的 label 变成 padding，下一段的第一个 token 不再被预测。所以 HF Trainer 与 Accelerate 在切分之前先生成 `shift_labels`，forward 拿到它时直接使用，不再左移。

::liger-shift-labels::

#1181 之前，Qwen3-VL 的 FLCE 分支已经使用 `shift_labels`，不走 FLCE 的分支（关闭 FLCE 或评估时）则忽略它。#1181 让这个分支也使用 `shift_labels`，并把 `**kwargs` 传给 `self.loss_function`，其中的 `num_items_in_batch` 是整个梯度累积步内有效 label 的总数，loss 按它归一化，梯度累积的结果才与一次大 batch 相同。

## 六、其他接入方式

| 方式 | 时机 | 发现模块的方式 | 替换对象 | 漏项时 |
| --- | --- | --- | --- | --- |
| Liger 类路径 | 构建前 | 构建代码按名字查找 | 类名、类的 forward | 同名类不会漏 |
| Liger 实例路径 | 构建后 | 手写的属性路径 | 实例的 forward | 保留原实现 |
| kernels hub | 构建后 | 遍历模块，看层名 | 实例的 forward | 警告，可设为报错 |
| transformers patching API | 构建时 | 类名或正则 | 类 | 对已构建实例无效 |
| Unsloth | 构建前 | 按模型手写 | 整层的 forward | 导入顺序不对时警告 |

**kernels hub。** transformers 用 `@use_kernel_forward_from_hub("RMSNorm")` 标记模型中的类，装饰器只给类加一个属性 `kernel_layer_name`。`from_pretrained(..., use_kernels=True)` 调用 `kernelize`，遍历全部模块，凡是类上有这个属性的，都用 Hub 上对应 kernel 的 forward 替换，方式与 ③ 相同。发现模块靠类型，`q_norm` 与 `k_norm` 不会漏掉。代价是对 kernel 的约束：Hub 上的层不能定义 `__init__` 或额外成员，forward 的参数个数与种类要与原类一致。层名也承载语义，Qwen3.5 的 norm 标为 `RMSNormZeroCentered`，不会拿到普通 RMSNorm 的 kernel。默认映射中的 `RMSNorm`、`SwiGLUMLP` 等指向 `kernels-community/liger-kernels`，Liger 的 kernel 也经这条途径分发；TRL 1.15 已把 `use_liger_kernel` 标为弃用，建议改用 `use_kernels=True`。

**transformers patching API。** 2026 年 3 月加入的 `register_patch_mapping` 登记类名到替换类的映射，`from_pretrained` 在构建期间把所有 `transformers.*` 模块中匹配的名字换掉，构建结束后恢复。它是官方提供的 ①，作用范围限定在构建过程内。

**Unsloth。** Unsloth 的 `from_pretrained` 在构建模型之前用 ② 替换 attention、decoder 层与 CausalLM 的 forward，自写的 attention forward 直接调用 `fast_rms_layernorm(self.q_norm, Q)`，不需要逐个替换叶子模块。代价是与 transformers 的版本强耦合；另有一部分补丁在 `import unsloth` 时完成，所以 Unsloth 要求先于 transformers 导入，否则发出警告。

多个工具改写同一个 forward 时会相互覆盖。TRL 的 `SFTTrainer` 给模型包上自己的 fused LM head，与 Liger 的 FLCE 都要替换 `model.forward`，所以 TRL 关闭 Liger 的 FLCE，只保留 RMSNorm、SwiGLU 等模块的 kernel。TRL 的包装用 `functools.wraps(type(model).forward)` 保留原签名，原因与 `logits_to_keep` 相同：`generate()` 按 forward 的签名决定传哪些参数。

## 参考

- Hsu et al. [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989). 2024.
- linkedin/Liger-Kernel [#1470: patch Q/K norms on existing Qwen models](https://github.com/linkedin/Liger-Kernel/pull/1470)，[#1181: logits_to_keep and shift_labels support for Qwen3-VL and Qwen3-VL-MoE](https://github.com/linkedin/Liger-Kernel/pull/1181).
- Liger-Kernel 源码：`src/liger_kernel/transformers/monkey_patch.py`、`auto_model.py`、`model/qwen3.py`、`model/loss_utils.py`（commit `d5f2817`）。
- transformers 源码：`integrations/liger.py`、`integrations/hub_kernels.py`、`monkey_patching.py`、`generation/utils.py`、`trainer.py`、`models/qwen3/modeling_qwen3.py`（commit `536ecc0`）；[#43917: Model patching API](https://github.com/huggingface/transformers/pull/43917).
- huggingface/kernels 源码：`kernels/src/kernels/layer/layer.py`、`kernelize.py`（commit `b899be0`）。
- Unsloth 源码：`unsloth/models/qwen3.py`、`llama.py`、`_gpu_init.py`（commit `f33fc6a`）。
- TRL 源码：`trl/trainer/sft_trainer.py`、`utils.py`（commit `72cf37d`）。
- LlamaFactory 源码：`src/llamafactory/model/model_utils/liger_kernel.py`（commit `1fded91`）。
- 模型配置：Hugging Face 上 Qwen3-8B、Qwen3-VL-8B-Instruct、Qwen3.5-9B 的 `config.json`（2026-10-10 读取）。
