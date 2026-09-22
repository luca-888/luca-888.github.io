# Gradient Checkpointing：原理、实现与自动重计算策略

**Gradient Checkpointing（GC）减少 forward 时保存的 activation，在 backward 需要时重新计算。** 它用额外计算换显存；省下的显存若能容纳更大的 microbatch，也可能提高训练吞吐。

PyTorch 文档也称它为 Activation Checkpointing（AC）；这里的 checkpoint 指重算边界，与保存模型权重的 training checkpoint 不同。

## 一、GC 为什么能省显存

训练显存包含 params、grads、optimizer states、activation 和临时工作区。**GC 主要减少为 backward 保存的 activation。**

普通执行把所需的内部数值一直留到 backward；GC 保留区域入口，让内部结果在使用后释放，等 backward 需要时再恢复。

::gc-concept::

看一个具体例子：$a=\sin(x)$，$b=\sin(a)$，$y=2b$。两个 `sin` 的 backward 分别需要 $x$ 和 $a$，因为局部导数是 $\cos(x)$ 和 $\cos(a)$；末尾乘法的导数是常数 $2$，无需保存 $b$。

在 PyTorch 中，只需把这段计算交给 `checkpoint`：

```python
def fn(x):
    a = x.sin()
    b = a.sin()
    return b * 2

# x 为 requires_grad=True 的 tensor
y = checkpoint(fn, x, use_reentrant=False)
y.sum().backward()
```

这个例子保存入口 $x$，到 backward 时从 $x$ 恢复内部所需的 $a$。**求导关系仍然成立，改变的是数值的保存方式。**

## 二、重算区域与 PyTorch 实现

### 区域大小：边界存储与临时显存的取舍

下面每次 `checkpoint` 包住 **1 个 block，也就是 1 个区域**：

```python
for block in blocks:
    x = checkpoint(block, x, use_reentrant=False)
```

也可以把连续多个 block 放进同一次 `checkpoint` 调用。以 12 个串联 block 为例，每 1、2、4 个 block 一组，分别保留 12、6、3 个区域入口；组内其他 block 的输入由重算恢复。

区域越大，保留的入口越少，但一次重算需要恢复更多内部结果。**区域大小需要在边界存储、重算开销与临时显存之间取舍。**

### Reentrant 与 non-reentrant

两者的关键区别是 **backward 使用哪张计算图**：

| | Reentrant | Non-reentrant |
| --- | --- | --- |
| 首次 forward | 不记录区域内部图 | 记录 autograd 图 |
| Backward | 完整重跑函数，对新图求导 | 重算恢复数值，沿原图继续 |

Non-reentrant 默认还有 **early stop**：所需 saved tensors 恢复齐全后就停止重算。前面的 `sin` 例子中，它能跳过末尾乘法；reentrant 会重跑完整函数。

**新代码优先使用 `use_reentrant=False`。** 它支持 `torch.autograd.grad` 和 SAC，也不要求传入 checkpoint 的 tensor 本身必须带梯度；冻结 embedding 或使用 LoRA 时，这一点尤其有用。[PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)

推荐 non-reentrant 主要因为功能支持更完整；early stop 能减少部分重算，但不保证端到端更快。

## 三、减少重算：SAC 与编译器策略

### SAC：指定哪些算子结果值得保存

**Selective Activation Checkpointing（SAC）允许在 checkpoint 区域内部额外保存指定算子的结果。** 例如保留昂贵的矩阵乘结果，只重算便宜的逐元素操作。

::gc-official-sac::

**Autograd 决定 backward 需要哪些 tensor；GC / SAC 决定这些值如何保存或恢复。** 算子的求导公式不变，也无需逐个重写 backward。

下面的 policy 缓存矩阵乘输出。Backward 触发重算时，遇到匹配的矩阵乘就取缓存，其他操作按需重算；矩阵乘自身的梯度计算仍然执行。

```python
ops_to_save = {torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
               torch.ops.aten.addmm.default}

def policy(ctx, op, *args, **kwargs):
    if op in ops_to_save:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

context_fn = partial(create_selective_checkpoint_contexts, policy)
y = checkpoint(block, x, use_reentrant=False, context_fn=context_fn)
```

Policy 匹配的是 **ATen 算子**，如 `mm`，不是 `Linear` 这类 Python 模块。保存 Attention 时，也要匹配实际使用的后端。[SAC API](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)

### torch.compile：由编译器选择保存方案

手工 SAC 由用户指定“哪些算子的结果要保存”；memory budget 则由用户给出“允许保存多少”，编译器分析 forward 与 backward 的联合图，按依赖、tensor 大小和估计的重算成本选择方案。

```python
torch._functorch.config.activation_memory_budget = 0.5
compiled_block = torch.compile(block, fullgraph=True, dynamic=False)
```

这里的预算作用于 **`block` 编译区域内，为 backward 保存的 tensor**。理解 `0.5`，先看两个端点：

| Memory budget | 保存策略 |
| ---: | --- |
| `0.0` | 以只保留区域输入、内部结果重算为基准 |
| `1.0` | 默认 `torch.compile` 策略，已包含部分重算 |
| `0.5` | 在两个基准之间，分配一半的可调保存空间 |

**假设**只保留区域输入需要 100 MB，默认编译策略保存 500 MB，那么 `0.5` 对应的预算目标是：

\[
100 + 0.5 \times (500 - 100) = 300\ \text{MB}
\]

其中 100 MB 是输入基准，可调空间是剩余的 400 MB；`0.5` 允许使用其中 200 MB。因此它既不是默认保存量直接乘以 50%，也不是保留一半的算子。[对应版本的预算归一化实现](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_functorch/partitioners.py#L2937-L2950)

**编译器再决定这份预算花在哪里。** 例如，保存 matmul 输出可以避免昂贵的重算；释放 pointwise 结果通常只需较低成本就能恢复。它会结合大小与成本选择组合，目标是在预算内尽量减少重算耗时。预算越紧，通常需要重算越多。

这是编译器估计的保存预算，实际保存量不一定刚好用满；整卡峰值还包含 grads、optimizer states 与临时 tensor，所以 **`0.5` 不表示总显存减半**。该配置属于内部接口，使用时需核对[对应版本](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_functorch/config.py)。

SAC 的 selective 表示“选择性”；这里由编译器选择保存方案，才是自动策略。

## 四、官方结果：memory budget 与速度

[PyTorch 官方博客](https://pytorch.org/blog/activation-checkpointing-techniques/)给出了 Transformer 上的 memory budget 结果，对应上一节的 **`torch.compile` 自动策略**。

::gc-official-benchmark::

**沿曲线从右向左看：保存预算越紧，重算越多，速度逐步下降。** 在这组 Transformer 结果中，官方报告仅重算 pointwise 算子就能减少约 50% 的 activation 保存量；继续节省显存，需要逐步重算 matmul，Attention 的重算成本最高。

这组结果展示了**优先重算便宜算子、保留昂贵结果**的取舍。图中的策略由编译器选择，不是 eager SAC 与普通 GC 的直接测速对照。

## 五、实际配置与训练检查

### Megatron-LM / Megatron Core

```bash
# 默认关闭 GC；以下方案选一组，追加到 pretrain_gpt.py 的启动参数
# 关闭时移除显式开启重算的参数

# 每层独立 checkpoint；末尾改为 2 表示每两层一组
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1

# 只重算 core Attention
--recompute-granularity selective --recompute-modules core_attn

# 只重算 MLP；也可填 core_attn mlp 同时选择两者
--recompute-granularity selective --recompute-modules mlp

# 每个 pipeline stage 只 checkpoint 两层；virtual pipeline 按 virtual stage 计算
--recompute-granularity full --recompute-method block --recompute-num-layers 2
```

### LLaMA-Factory

```yaml
# 训练 YAML：开启 GC，使用 non-reentrant
# 关闭 GC 时将 disable_gradient_checkpointing 设为 true
# 目前不支持 SAC 或 memory budget
disable_gradient_checkpointing: false
use_reentrant_gc: false
```

### TorchTitan

```python
# 修改已有训练配置 config：每个 block 整段重算
config.activation_checkpoint = FullAC.Config()

# 其他方案：选择一组取消注释，替换上面的配置

# 内置算子级 SAC
# config.activation_checkpoint = SelectiveAC.Config()

# 自动预算：同时启用模型编译
# config.compile = CompileConfig(components=["model"])
# config.activation_checkpoint = MemoryBudgetAC.Config(memory_budget=0.5)

# 关闭显式 GC；编译器仍可能执行重算
# config.activation_checkpoint = None
```

### 保持重算一致

| 场景 | 关键条件 |
| --- | --- |
| Dropout | 保留 RNG 状态，让重算使用同一份 mask；PyTorch 默认开启 |
| LoRA / 冻结层 | 冻结 params 后，输入梯度仍可能需要穿过该层；检查 adapter 的 grads |
| 有状态代码 | 日志、计数器和缓存追加放在重算区域外，避免重复执行 |
| 标准整段训练 | 通常关闭 KV cache：`use_cache=False` |

## 参考资料

- [PyTorch checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [Megatron Core 配置定义](https://github.com/NVIDIA/Megatron-LM/blob/c711dc0a7f4f7460045cae18ec2f9ff5f4b92b5e/megatron/core/transformer/transformer_config.py)
- [TorchTitan 重算策略](https://github.com/pytorch/torchtitan/blob/6d1be0ab2ba25352e122b87f9a5eed23672a175e/torchtitan/distributed/activation_checkpoint.py) · [训练配置示例](https://github.com/pytorch/torchtitan/blob/6d1be0ab2ba25352e122b87f9a5eed23672a175e/torchtitan/models/llama3/config_registry.py)
