# Liger-Kernel 接入 Hugging Face 模型：调研笔记

## 范围与状态

- 2026-10-10 开始调研并提交参考方案。同日协调会话（代用户统筹 5 篇）回复按方案执行、视为确认：实测占位 1、2 都不做，正文留占位；待核实项能从源码核实的核实，不能的不进正文；理解检查不等用户回答，题目与参考答案写进本文末尾；跳过草图确认，直接精画。正文、4 张图与页面代码已完成，状态为 draft。
- 本篇讲接入机制：类替换、实例 patch、静默回退、patched forward 与 HF 接口的契约。不讲 RMSNorm kernel 本身的优化（博客已有 `content/rmsnorm.md`），不展开 Qwen3.5 的 1+w 计算位置（另一篇的主题），只说明 `offset=1.0` 参数的存在。
- 案例为作者合并到 linkedin/Liger-Kernel 的两个 PR：#1181、#1470。正文以原理为主，PR 作为真实案例出现。
- 本阶段不做实测；实测占位见文末。

## 源码版本

| 仓库 | commit | 日期 | 备注 |
| --- | --- | --- | --- |
| Liger-Kernel（上游） | `d5f2817` | 2026-10-07 | `reference-repos/Liger-Kernel-upstream`，含 #1470 |
| Liger-Kernel（用户 fork，完整克隆） | `b297821` | 2026-10-02 | 只用来查历史（`git log -S`） |
| transformers | `536ecc0` | 2026-10-09 | v5.19.0 之后 |
| huggingface/kernels | `b899be0` | 2026-10-09 | 本次新增到 sync.sh |
| unsloth | `f33fc6a` | 2026-10-10 | 本次新增到 sync.sh；`unsloth_zoo` 未克隆 |
| LlamaFactory | `1fded91` | 2026-10-10 | 另一会话加入 sync.sh |

## 参考清单（按重要程度）

| 来源 | 作者 | 状态 | 重要程度 | 用于 |
| --- | --- | --- | --- | --- |
| Liger 源码 `transformers/monkey_patch.py`（`_bind_method_to_module`、`_patch_rms_norm_module`、`apply_liger_kernel_to_qwen3/_qwen3_5/_qwen3_vl`、`_apply_liger_kernel`、`_apply_liger_kernel_to_instance`）、`auto_model.py`、`model/qwen3.py`、`model/loss_utils.py` | LinkedIn | 已读相关函数 | 核心 | 三种替换、两条入口、kwargs 过滤、FLCE 分支 |
| [linkedin/Liger-Kernel#1470](https://github.com/linkedin/Liger-Kernel/pull/1470) fix(monkey_patch): patch Q/K norms on existing Qwen models | luca-888 | 已读描述、diff、review | 核心 | 漏 patch 案例；H100 前后对比表；测试写法 |
| [linkedin/Liger-Kernel#1181](https://github.com/linkedin/Liger-Kernel/pull/1181) logits_to_keep / shift_labels for Qwen3-VL(-MoE) | luca-888 | 已读描述、diff、评论（含 H20 测量） | 核心 | patched forward 与 HF 接口契约 |
| transformers 源码：`integrations/liger.py`、`trainer.py`（`use_liger_kernel`、CP 的 `shift_labels`、`num_items_in_batch`）、`generation/utils.py`（`_supports_logits_to_keep`）、`loss/loss_utils.py`（`ForCausalLMLoss`）、`models/qwen3/modeling_qwen3.py`、`models/qwen3_5/modeling_qwen3_5.py` | Hugging Face | 已读相关段落 | 核心 | Trainer 走实例路径；生成按签名决定是否传 `logits_to_keep`；`shift_labels` 来源 |
| transformers `monkey_patching.py` 与 [#43917 Model patching API](https://github.com/huggingface/transformers/pull/43917)（2026-03-02 合并） | Hugging Face | 已读源码与 PR 描述开头 | 重要 | 官方的构建期类替换，现状变化 |
| huggingface/kernels `layer/layer.py`、`layer/kernelize.py`；transformers `integrations/hub_kernels.py`、`modeling_utils.set_use_kernels` | Hugging Face | 已读 | 重要 | 按类型遍历发现模块、同样是实例 forward 绑定、签名校验、fallback |
| [Liger Kernel 论文](https://arxiv.org/abs/2410.10989) §3 API 与 Integrations | Hsu et al.（LinkedIn） | 已读 TeX 相关段落 | 重要 | 官方给出的三种用法；2024 年 TRL 走 `AutoLigerKernelForCausalLM` |
| Unsloth `models/qwen3.py`（`pre_patch`）、`models/llama.py`（`CausalLM_fast_forward`）、`_gpu_init.py`（导入顺序警告） | Unsloth AI | 已读相关段落 | 重要 | 整层 forward 改写 + 导入顺序警告 |
| LlamaFactory `model/model_utils/liger_kernel.py`、`model/loader.py` | hiyouga | 已读 | 补充 | 加载前调用类路径的代表 |
| Liger README “Getting Started” | LinkedIn | 已读 | 补充 | 用法说明 |

未读：HF 博客关于 kernels hub 的文章、TRL 当前 `SFTTrainer` 的 Liger 接入代码、`unsloth_zoo` 编译器（改写 modeling 源码再 exec 的路径）。

## 阅读要点

### Liger 的三种替换

- **模块全局名**：`modeling_qwen3.Qwen3RMSNorm = LigerRMSNorm`、`modeling_qwen3.Qwen3MLP = LigerSwiGLUMLP`、`modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb`。`Qwen3Attention.__init__` 在构建时按全局名查找 `Qwen3RMSNorm`（`modeling_qwen3.py:229-230` 构建 `q_norm`、`k_norm`），所以加载前调用时 q/k norm 自动换上。对已构建的模型无效（`_apply_liger_kernel` 的 docstring 也写了这一点）。RoPE 是函数，在 forward 时按全局名查找，所以构建后替换也生效。
- **类属性**：没有实例时 `Qwen3ForCausalLM.forward = qwen3_lce_forward`。类属性在调用时查找，对已有和未来的所有实例都生效，作用于整个进程。
- **实例属性**：`_bind_method_to_module` 写 `module.__dict__["forward"] = new_method.__get__(module, cls)`；有实例时 `model.forward = MethodType(lce_forward, model)`。只影响这一个对象，遮蔽类上的 forward。`type(module)` 不变（`isinstance` 仍是 `Qwen3RMSNorm`），所以另绑 `_get_name` 让 repr 显示 `LigerRMSNorm`；同时补上 `offset`、`casting_mode`、`in_place`、`row_mode` 等 Liger forward 需要的属性。参数对象不变（#1470 的测试断言 `norm.weight is weight`），optimizer、FSDP、PEFT 持有的引用仍有效。PEFT `ModulesToSaveWrapper` 要同时 patch `modules_to_save.default` 与 `original_module`。
- 历史：#692（2025-05-02 加入 Qwen3）时，实例路径的 FLCE 也是改类属性 `Qwen3ForCausalLM.forward`；现在有实例时改为实例绑定。

### 两条入口

- 加载前：`apply_liger_kernel_to_xxx()` 后再 `from_pretrained`；`AutoLigerKernelForCausalLM` 读 config 的 `model_type` 后调 `_apply_liger_kernel`，再走 `AutoModelForCausalLM.from_pretrained`。LlamaFactory 在 `loader.py:148` 加载前调用，属于这一类。
- 加载后：HF Trainer `use_liger_kernel=True` → `integrations/liger.py::apply_liger_kernel` → `_apply_liger_kernel_to_instance(model=base_model, **kernel_config)`。v4.45.0 起就是实例路径。Liger 论文（2024）写 TRL `use_liger=True` 通过 `AutoLigerKernelForCausalLM` 加载，当前 TRL 待核实。
- 实例路径对已经存在的模块只能逐个按属性路径 patch：`base_model.norm`、每层 `input_layernorm`、`post_attention_layernorm`、`mlp`。写漏的路径就保持原实现。

### #1470：漏掉的 q_norm / k_norm

- Qwen3（#692，2025-05）、Qwen3 MoE、Qwen3 Next、Qwen3.5（#1123，2026-03）、Qwen3.5 MoE 的实例路径都没有 patch `self_attn.q_norm/k_norm`；Qwen3-VL（#911，2025-11）的实例路径已经有。同一族模型之间不一致。
- 修复：每个函数在 `for decoder_layer in ...` 里加两行；Qwen3 Next / 3.5 用 `hasattr(decoder_layer, "self_attn")` 只覆盖 Full Attention 层，`linear_attn.norm`（RMSNormGated）不动；复用各模型已有的 partial（Qwen3.5 为 `offset=1.0, casting_mode="gemma", in_place=False`），并受 `rms_norm` 开关控制。
- 覆盖：Qwen3-8B 36/36 层，Qwen3-30B-A3B 48/48，Qwen3.5-9B 8/32（`full_attention_interval=4`），Qwen3.5-35B-A3B 10/40。
- 测试：先用 `rms_norm=False` 调一次，断言 forward 未变；再正常调用，断言 `inspect.getsource(norm.forward) == inspect.getsource(LigerRMSNorm.forward)`、权重对象与数值不变、eps / offset / casting_mode / in_place 正确、linear attention 的 norm forward 未变。13 个新断言修复前失败、修复后通过。新增 `base_model` 参数化，覆盖直接 patch 基座模型（不含 lm_head）的情况。
- PR 中的 H100 数据（BF16、SDPA、无 compile、forward+backward 中位数）：Qwen3-8B 4×2048，Q+K norm 2.439 → 0.562 ms（4.34×），单层 attention 8.883 → 6.705 ms（1.32×）；Qwen3.5-9B 4×2048，Q+K 2.560 → 0.739 ms，attention 10.763 → 8.802 ms。环境 PyTorch 2.9.1 / CUDA 12.8 / Triton 3.5.1 / Transformers 5.15.1。

### 为什么不报错

- Liger RMSNorm 与原实现签名相同、数学相同，输出在数值误差内一致；漏 patch 只影响速度与显存，loss 曲线看不出来。
- 发现模块靠手写属性路径，没有“遍历所有 RMSNorm 实例并核对”的步骤。
- 参数靠签名过滤：`_apply_liger_kernel(_to_instance)` 只保留 apply 函数签名中有的 kwargs，拼错或不支持的开关被丢弃；`LigerForCausalLMLoss` 也按 `liger_fused_linear_cross_entropy` 的签名过滤 kwargs。
- 不支持的 `model_type` 只打 `logger.info`。
- 生成时 transformers 用 `"logits_to_keep" in inspect.signature(self.forward).parameters` 决定是否传 `logits_to_keep=1`（`generation/utils.py:2326-2331, 2930-2934`）；`**kwargs` 不算。patched forward 签名里没有这个参数时，prefill 计算全部位置的 logits，不报错。

### #1181：patched forward 与 HF 接口

- 修复前 Qwen3-VL / Qwen3-VL-MoE 的 `lce_forward` 签名中没有 `logits_to_keep`；非 FLCE 分支只在 `labels is not None` 时算 loss，且不传 `shift_labels`。
- 修复后：按 HF 写法 `slice(-k, None)` 或张量索引截取 hidden states，FLCE 与普通分支都用截取后的 hidden states；普通分支在 `labels` 或 `shift_labels` 任一存在时调用 `self.loss_function(..., shift_labels=..., **kwargs)`（`num_items_in_batch` 由 kwargs 传入）。
- `skip_logits` 默认 `self.training and (labels or shift_labels)`：训练时走 FLCE，不生成 logits；推理时走 lm_head。
- `shift_labels` 的来源：torch CP 时 Trainer 在切分序列之前先 `pad + labels[:, 1:]` 生成 `shift_labels`（`trainer.py:2398-2403`），DeepSpeed Ulysses 由 Accelerate 的 `UlyssesSPDataLoaderAdapter` 生成；padding-free collator 也可能直接提供。序列切到各 rank 后再本地 shift，每段末尾会丢掉下一段第一个 token 作为 label。
- PR 评论的测量（单卡 H20，transformers 5.4.0，full logits 对比 `logits_to_keep=1`）：Qwen3-VL-8B 10955 token，延迟 1.6723 → 1.5753 s，峰值显存 22.660 → 20.655 GB；Qwen3-VL-30B-A3B 10955 token，1.4846 → 1.4352 s，64.161 → 63.585 GB。
- 按配置估算：Qwen3-VL-8B 词表 151936，10955 个位置的 BF16 logits 为 3.33 GB，与实测峰值差 2.0 GB 不一致，原因待核实。

### 相关系统

- **kernels hub**：`@use_kernel_forward_from_hub("RMSNorm")` 只在类上设 `kernel_layer_name`；`kernelize(model, mode=...)` 遍历 `named_modules()`，凡类上有该属性的实例都处理，替换方式同样是 `module.forward = MethodType(layer.forward, module)`。发现靠类型，不靠属性路径，所以 q_norm 不会漏。加载时校验 hub layer 无自有 `__init__`、无额外成员、forward 签名参数个数与种类一致。没有映射、设备不匹配、mode 不匹配时默认回退原 forward 并打 warning；`use_fallback=False` 时报错。层名承载语义：Qwen3.5 的 norm 标为 `RMSNormZeroCentered`，不与普通 `RMSNorm` 共用 kernel。transformers 默认映射里 `RMSNorm`、`SwiGLUMLP`、`GeGLUMLP` 等指向 `kernels-community/liger-kernels`，即 Liger 的 kernel 也经由 hub 分发。transformers 中由 `from_pretrained(use_kernels=True)` 触发。
- **transformers model patching API**（#43917，2026-03）：`register_patch_mapping({"Qwen2MoeExperts": Custom})`，键可为类名或正则；`from_pretrained` / `from_config` 构建时进入 `apply_patches()`，遍历 `sys.modules` 中所有 `transformers.*` 模块替换同名属性，退出后恢复；`patch_output_recorders` 处理 `_can_record_outputs` 中对类对象的引用。只作用于构建期，对已构建实例无效。动机是量化工具需要把 MoE 拼接权重拆回 `nn.Linear`（issue #43284）。
- **Unsloth**：`pre_patch()` 在类上替换 `Qwen3Attention.forward`、`Qwen3DecoderLayer.forward`、`Qwen3Model.forward`、`Qwen3ForCausalLM.forward`；自写的 attention forward 里直接调用 `fast_rms_layernorm(self.q_norm, Q)`，不需要逐个 patch 叶子模块。代价是与 transformers 版本强耦合，要求先 `import unsloth`，否则警告“unoptimized versions run”。`CausalLM_fast_forward` 自己处理 `logits_to_keep`（取 `max(num_logits_to_keep, logits_to_keep)`，张量形式退回 0）。

## 分歧与待核实（2026-10-10 处理结果）

1. **hub 映射中的 `ForCausalLMLoss`**：`hub_kernels.py` 中有 `ForCausalLMLoss → LigerForCausalLMLossLayer`（TRAINING | TORCH_COMPILE），但 transformers `536ecc0` 全仓库没有任何对象用 `use_kernel_forward_from_hub("ForCausalLMLoss")` 标记，`kernelize` 找不到对应模块，目前不生效。正文不写。
2. **#1181 的显存差**：评论中 10955 token 的峰值显存差 2.0 GB，小于按配置算出的 BF16 logits 3.33 GB，测量口径无法从源码确认。正文只引延迟（1.67 s 对 1.58 s），logits 大小用按配置的计算值，不把两者对比。
3. **FLCE 分支下 `logits_to_keep≠0` 且有 labels**：hidden states 被截成 k 个位置，labels 仍为 T 个位置，`LigerForCausalLMLoss` 中 `view(-1, hidden)` 与 `shift_labels.view(-1)` 的长度不一致，会在 kernel 中报错，不是静默错误；HF 原生 forward 同样只截 logits 不截 labels。训练默认 `logits_to_keep=0`。正文不写。
4. **TRL**：已核实（TRL main `72cf37d`，v1.15.0 起）。`SFTTrainer` 把 `use_liger_kernel=True` 标为弃用，建议 `model_init_kwargs={"use_kernels": True}`；显式设置 `liger_kernel_config={"fused_linear_cross_entropy": True}` 时报错，否则强制为 False，因为 TRL 用 `add_fused_lm_head` 自己包装 `model.forward`。包装用 `functools.wraps(type(model).forward)` 保留签名，注释写明 `generate` 按签名校验 model kwargs。已写入正文第六节。
5. **其他模型的实例路径漏项**：未做覆盖审计，正文留占位。

另：Unsloth 的 `pre_patch()` 在 `FastLlamaModel.from_pretrained` 中、模型构建前调用（`llama.py:2768`），不是 import 时；import 时完成的是另一部分补丁（`_gpu_init.py` 的导入顺序警告针对这部分）。正文按此写。

## 图的数据来源

- 图 2（Qwen3.5-9B 覆盖）是**由源码静态推导的结果，不是运行结果**：Qwen3.5-9B `config.json` 的 `layer_types`（`full_attention_interval=4`，第 4、8、…、32 层为 full attention）；transformers `modeling_qwen3_5.py` 中 `Qwen3_5RMSNorm` 用于 input / post-attention / q / k / 最后的 norm，线性 attention 层的输出 norm 为 `Qwen3_5RMSNormGated`；Liger 类路径替换 `modeling_qwen3_5.Qwen3_5RMSNorm`，实例路径在 #1470 前只 patch `norm`、`input_layernorm`、`post_attention_layernorm`。计数：Qwen3_5RMSNorm 共 1 + 64 + 16 = 81 个（图中不画最后的 norm，故为 80），#1470 前实例路径漏 16 个；gated norm 24 个，三条路径都不替换。
- 第四节 Qwen3-8B “145 个中 72 个”同样由源码与 config（36 层）推导：36 × 4 + 1 = 145，漏 36 × 2 = 72。
- 图 3 的尺寸按 Qwen3-VL-8B `config.json`（hidden 4096，词表 151936）与 #1181 评论中的序列长度 10955 计算；宽度按比例，高度不按比例。
- 第四节表格与开头的数字出自 #1470 描述（H100），第五节的延迟出自 #1181 评论（H20）。

## 实测占位（2026-10-10 决定都不做，正文留占位）

1. **覆盖审计（CPU 即可，无需 GPU）**：对 `MODEL_TYPE_TO_APPLY_LIGER_FN` 中每个模型用小 config 构建，分别走类路径与实例路径，列出每类模块的 forward 是否为 Liger 实现；在 #1470 前（`80fb156^`）与当前 main 各跑一次。用于覆盖图和“现状”段落。需用户确认后再做。
2. **端到端训练差异（1×H100）**：Qwen3-8B，HF Trainer `use_liger_kernel=True`，#1470 前后各跑若干 step，记录 step 时间与峰值显存，以及与类路径的对比。用于说明漏 patch 在整步中的占比。可选。
3. 正文的 kernel 级数字使用 #1470 描述中的 H100 表，标明出自 PR；`logits_to_keep` 的数字使用 #1181 评论中的 H20 测量，标明出自 PR。

## 理解检查

题目供用户自测，参考答案按 2026-10-10 读过的源码给出。

1. **如果在 `from_pretrained` 之后才调用 `apply_liger_kernel_to_qwen3()`，且不传 `model`，哪些部分会换成 Liger？为什么 `Qwen3ForCausalLM.forward` 会生效，而 `q_norm` 不会？**
   参考答案：生效的是 RoPE（forward 中每次按名字查 `apply_rotary_pos_emb`）与 FLCE（不传 model 时改的是类属性 `Qwen3ForCausalLM.forward`，已构建的实例调用时到类上查找，所以也生效）。全部 RMSNorm（含 `q_norm`、`k_norm`）与 MLP 都不会换：改的是 `modeling_qwen3` 中的名字，而这些模块在构建时已按旧名字创建，类型已经确定。

2. **#1470 之前，用 Trainer `use_liger_kernel=True` 训练 Qwen3-8B，loss 曲线和不开 Liger 时一致吗？为什么测试最后比较的是 `inspect.getsource(norm.forward)`，而不是数值？**
   参考答案：在 BF16 误差范围内一致。漏掉的 72 个 norm 用原实现，计算本来就相同；换上的部分与原实现数值等价。所以数值比较无法区分“换上了”与“没换上”，只能检查每个模块实际绑定的 forward 是哪一个实现。测试还先以 `rms_norm=False` 调用一次，确认开关能关闭替换。

3. **生成时 transformers 怎样决定传不传 `logits_to_keep=1`？如果 patched forward 只用 `**kwargs` 接收这个参数，会发生什么？**
   参考答案：`generate()` 检查 `"logits_to_keep" in inspect.signature(self.forward).parameters`，为真才设 `model_kwargs["logits_to_keep"] = 1`。`**kwargs` 不是名为 `logits_to_keep` 的参数，检查为假，generate 不传，prefill 对整个 prompt 计算 T × V 的 logits；即使用户手动传入，参数也只是进了 kwargs，没有被用于截取。结果正确但多花算力与显存，不报错。

4. **开启 context parallelism 后，为什么不能在每个 rank 上对本地的 labels 做 shift？如果 patched forward 在非 FLCE 分支忽略了 `shift_labels`，结果会怎样？**
   参考答案：序列被切成多段，每段最后一个位置的 label 是下一段的第一个 token，而它在别的 rank 上；本地 shift 只能补 -100（padding），这些位置的 loss 丢失，对应 token 不再被预测。忽略 `shift_labels` 时，loss 函数改用本地 `labels` 自己左移，正是上述情形：每个切分边界丢一个 label，不报错。若 `labels` 不在输入中，loss 为 None。

5. **kernels hub 的 `kernelize` 为什么不会漏掉 `q_norm`？它为此对 kernel 提出了哪些约束？**
   参考答案：`kernelize` 遍历 `named_modules()`，按模块的类上是否有 `kernel_layer_name` 判断，与模块挂在哪个属性路径下无关；`q_norm` 的类型是被标记的 `Qwen3RMSNorm`，自然会被找到。代价是替换只能换 forward、不能改模块状态：Hub 上的层不能自定义 `__init__`、不能有额外成员，forward 的参数个数与种类要与原类一致；层名还要准确表达语义（如 `RMSNormZeroCentered`），否则会拿到计算不同的 kernel。

## 交付（2026-10-10）

未 commit、未 push。

新增：
- `content/liger-hf-patching.md`：正文（已做成稿自查，并由不带写作上下文的 subagent 用 `de-ai-edit` 改过一遍）
- `posts/liger-hf-patching/index.html`：页面入口
- `src/articles/liger-hf-patching/liger-hf-patching.tsx`、`liger-hf-patching.blocks.tsx`、`LigerHfPatchingFigures.tsx`：页面代码与 4 张图（`LigerBindingScope`、`LigerCoverage`、`LigerLogits`、`LigerShiftLabels`）及 2 个实测占位
- `docs/liger-hf-patching-notes.md`：本笔记

修改的共享文件（只做了以下定点修改）：
- `vite.config.ts`：`rollupOptions.input` 中新增一行 `ligerHfPatching: 'posts/liger-hf-patching/index.html'`
- `src/posts.ts`：`posts` 数组开头新增 `liger-hf-patching` 条目（category `训练优化`，status `draft`）
- `docs/terminology.md`：在 `logits / lm_head` 行之后新增 `monkey patch / patch` 一行

仓库外（不在本仓库提交范围）：`~/Documents/ChatGPT/reference-repos/sync.sh` 的 REPOS 中新增 `kernels`、`unsloth` 两行，并浅克隆了这两个仓库。

验证：在 dev server（本会话启动于端口 54469，同一个 checkout）的桌面视口中打开 `/posts/liger-hf-patching/`，没有页面报错，4 张图与 2 个占位都正常显示，表格无横向溢出。控制台有一条 400，是浏览器面板对页面 URL 的重复请求，随后的同一请求返回 200，与页面资源无关。未运行完整 build。
