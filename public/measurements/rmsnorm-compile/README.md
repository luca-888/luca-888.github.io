# RMSNorm torch.compile 原始产物

对应文章第三章。正文只展示结果与分析；环境、计时和数值检查细节保存在这些原始记录中。

## 文件

- 仓库 `content/data/rmsnorm-compile.json`：五组 BF16 shape 的稳态耗时、峰值显存增量、首次调用墙钟时间、数值误差及 profiler 样本。
- 仓库 `content/data/rmsnorm-eager.json`：第二章基线及完整 reference；两章 reference 的 SHA-256 相同。
- 各 shape 文件夹的 `forward.py`、`backward.py`：TorchInductor 原样导出的完整模块，包含 Triton 源码和分配、释放、启动 kernel 的 wrapper。
- `*-trace.json.gz`：压缩后的 Chrome trace。解压后可用 Perfetto 查看；`measured_0` 至 `measured_2` 是独立调用。
- `kernel-details.json`：执行生成模块后取得的最终 launch 配置、寄存器数量、spill 数量、shared memory 和 PTX 哈希。
- 各 shape 文件夹的 `*.ptx`：上述配置对应的 PTX。
- `latency-diagnostic.json`、`latency-cprofile.txt`：针对 `1024×4096` Forward 的 CPU 调用开销诊断。直接调用生成 wrapper 会跳过 Dynamo 和 AOT 运行时，因此该路径的耗时不用于正文的 compile 对比。

## 环境与测量

使用与第二章相同的 RTX 4090、Python 3.12.14、PyTorch 2.9.1+cu128、CUDA 12.8、驱动 570.153.02；Triton 3.5.1。CPU 为 AMD EPYC 7543，16 vCPU KVM guest，GPU 使用默认频率。

编译配置为 `backend="inductor"`、`fullgraph=True`、`dynamic=False`，默认 mode，无 CUDA Graph。Forward 与第一章手写 Backward 独立编译，均在 `no_grad` 下运行。

耗时沿用第二章：`torch.utils.benchmark.Timer.blocked_autorange(min_run_time=0.3)`，三轮中位数的中位数，30 次预热，1 个 CPU 线程，连续 CUDA 输入，复用输入，不清空缓存。Backward 不含 Forward。首次调用包含图捕获、编译或编译缓存加载及首次执行，单独记录，不是冷缓存编译耗时，也不进入稳态耗时与峰值显存数据。

峰值显存为每次调用的 `max_memory_allocated - 调用前 memory_allocated`，包含新输出和临时张量，排除已有输入、Backward 已保存的 r、未使用的 reserved memory 与 CUDA context。每次测量前重置峰值，测量后释放输出，独立于计时；三次预热、三次采样，取最大值。MiB = 2²⁰ 字节。

Profiler 内先预热三次，再记录三次带标记并同步的调用。kernel 数量仅统计每个标记范围内的 CUDA kernel 事件，排除 memcpy 与 memset。trace 的 kernel 时长与间隔不替代 Timer 的调用耗时。

## 数值检查

随机种子为 20260915，五组 shape 使用正态随机 BF16 输入。比较 y、r、dx、dgamma；Backward 既检查相同 eager r 的输入，也检查编译版 Forward 产生的 r。代表 shape 额外覆盖全零、缩放 1e-3 和 1e2 的 x。

BF16 采用 `rtol=0.016, atol=0.001`，FP32 的 r 采用 `rtol=1e-5, atol=1e-6`；记录最大绝对误差及相对 L2 误差。所有检查通过。

## 复现

在仓库根目录、上述 Python 环境中依次执行：

```bash
python scripts/benchmark-rmsnorm-compile.py
python scripts/inspect-rmsnorm-compile.py
```

第一个脚本检查 Python、PyTorch、GPU 型号和 reference 哈希，更新五组测量与生成代码；第二个脚本收集最终 launch 配置和 PTX。脚本使用当前 PyTorch 版本的内部代码导出接口。CPU 诊断是另行记录的补充，不由这两个脚本覆盖。

本次生成代码的 shape、线程配置和融合结构是实际编译结果，不保证其他版本或 GPU 采用同一策略。
