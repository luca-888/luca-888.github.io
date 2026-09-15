# RMSNorm 第四章：生成代码上的优化实验

## 来源与范围

- 基线为 `../rmsnorm-compile/4096x8192/backward.py` 中保存的真实 TorchInductor 输出，原文件未修改。来源及 SHA-256 见 `origin.json`。
- `kernels.py` 由 `scripts/prepare-rmsnorm-triton.py` 提取和修改；`changes.diff` 记录提取、dx 复用、dgamma 分组的逐项差异。新增的 `dgamma_finish` 完整保留在 `kernels.py`。
- 提取时把固定的 M、N、步长和 1/N 参数化；主样本常量不变。未改计算逻辑的提取版与原始 kernel 分别实测，验证提取过程没有制造一个较慢的基线。
- dx 复用版展开两个单次循环，将重复加载改为已有变量，并复用 dx_hat。它只支持 `XBLOCK=1, R0_BLOCK=N`，源码使用 `tl.static_assert` 限定，不能直接用于多 tile 行归约。
- 分组 dgamma 保留原有 64 列 × 64 行 tile、16 warps、计算顺序与缓存提示，仅改变循环范围和输出位置；新增第二阶段使用 128 列、4 warps。分组会改变跨行浮点累加顺序。
- 五组 workload 均为连续 BF16 x、gamma、g，FP32 r；y、dx、dgamma 为 BF16。没有加入 autograd 接口，也未测任意 stride、其他 dtype 或其他形状。
- Forward 原样使用 `torch.compile` 参考函数。最终修改版为 dx 复用 / 16 warps + dgamma 直接归约，全部形状使用同一配置，没有逐 shape 分派。

## 测量

`measurements.json` 保存全部原始计时样本、输出校验、峰值显存样本及软件信息，与 `content/data/rmsnorm-triton.json` 完全相同。完整运行环境另见 `environment.json`。

### GPU 耗时

使用 Triton 3.5.1 的 `triton.testing.do_bench_cudagraph(fn, rep=50, return_mode='all')`：在同一 CUDA Graph / CUDA event 路径测量所有候选，每轮返回 10 个样本，轮转候选顺序测 3 轮，报告各轮中位数的中位数。输入重复使用，没有显式清空缓存。原始 Inductor launcher 在回调内取得当前 CUDA stream，确保进入捕获流。

- 消融实验：预先分配同样的输入、输出和工作区；只比较 GPU 执行。dgamma 分组时间包括第一阶段和第二阶段，不能只计算 partial kernel。
- 五 shape 汇总：捕获完整 backward callable，包括生成包装层及修改版的 kernel 序列，输出分配由 CUDA Graph 的内存池处理；event 时间不包含 Python 的主机执行时间。
- 单 kernel 消融与完整序列的缓存状态和运行上下文不同，不应把两张表的耗时直接相加或混用。GPU event 数值也不能直接与 Python Timer 数值计算加速比。

### 完整调用耗时

与前两章使用同样的 `torch.utils.benchmark.Timer`：30 次预热，单 CPU 线程，`blocked_autorange(min_run_time=0.3)`，轮转实现顺序测 3 轮，报告轮中位数的中位数。包含 Python 包装层、输出与工作区分配，排除首次编译与校验。每个 shape 前重置 Dynamo 捕获缓存，避免形状 guard 累积和 recompile limit；每个 shape 独立构建默认 `torch.compile(fullgraph=True, dynamic=False)`。

完整调用与 GPU 耗时独立报告。直接 Triton launcher 的 Python 开销与 torch.compile 包装层不同，所以小输入的调用加速不能被当作 kernel 加速。

### 峰值显存增量

预热后，记录输入和 r 已分配时的 `torch.cuda.memory_allocated()`，重置峰值统计，执行一次完整调用并同步，再取最大已分配字节数的增量。测 3 次取最大值；不是 reserved memory，也不是 nvidia-smi 显示的进程显存。包含输出、存活的临时张量，排除预先存在的输入、编译缓存和其他已存在分配。

修改版先计算 dgamma，再释放 partial，最后分配并计算 dx；使用同一 stream 保证顺序。因此独立 partial 的大小不必等于完整 Backward 的峰值增量差。最终选用直接归约，工作区为零。

### 数值校验与指令检查

全部候选在 4096×8192 上对照 eager；最终五组输入分别比较共享 eager r 和 compile r 串联计算的结果。BF16 使用 `rtol=0.016, atol=0.001`，r 使用 `rtol=1e-5, atol=1e-6`。主样本另测 x 缩放为 0、1e-3、1e2；记录每项最大绝对误差及相对 L2 误差。

`inspection.json` 与压缩 trace 由 `scripts/inspect-rmsnorm-triton.py` 生成：另一个随机种子检查全部 shape 的结果，并记录三次调用的 kernel 序列。trace 时间用于诊断，正文数据以 `measurements.json` 为准。

PTX 文件及寄存器、spill、shared memory 信息来自实际编译结果。静态 PTX load 指令数量不是运行时显存事务数量；本实验未采集缓存命中率，不能据此断言具体缓存瓶颈。

## 复现

在解压后的仓库根目录、与环境记录一致的 Python / CUDA 环境执行：

```sh
python scripts/prepare-rmsnorm-triton.py
python scripts/benchmark-rmsnorm-triton.py
python scripts/inspect-rmsnorm-triton.py
```

若已完成消融，可用 `python scripts/benchmark-rmsnorm-triton.py --final-only` 沿用保存的消融结果与配置，重测五组输入。它会覆盖汇总测量记录。

脚本从文章的前两个 Python 代码块提取参考实现，并校验 SHA-256。完整包包含原始生成模块、参考文章、数据和上述脚本。
