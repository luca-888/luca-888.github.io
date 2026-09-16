# CUDA Graph diamond 实验

比较同一个四节点计算的两种提交方式：普通 Python 调用与 CUDA Graph replay。输入及四个输出缓冲区均为 4096 元素 FP32 CUDA 张量，计算为：

```text
A: x = 2 * static_input
B: b = x + 1
C: c = x + 2
D: y = b + c
```

依赖为 `A → B`、`A → C`、`B → D`、`C → D`，结果为 `y = 4 * static_input + 3`。A、B、D 提交到当前 stream，C 提交到 side stream；两次 `wait_stream` 分别建立分叉与汇合依赖。普通路径与 Graph 路径复用同一组缓冲区和相同依赖，均执行四个 kernel。

## 运行

需要可用 NVIDIA GPU 和带 CUDA 的 PyTorch。将两个 Python 文件放在同一目录后运行：

```bash
python example.py
python benchmark.py --iterations 1000 --rounds 5 --output results.json
```

本项目已有解释器为 `/root/.cache/liger-rms-bench-cu128/bin/python`。脚本不安装依赖。本次结果来自 NVIDIA GeForce RTX 4090，完整环境与逐轮样本保存在 `results.json`。

## 正确性

`example.py` 验证以下结果：

- capture 前将 `y` 清零，capture 后仍为 0。
- 输入为 7 时，两次 replay 均得到 31；本例每轮覆盖输出，不累加上一轮结果。
- 将输入 9 复制到原缓冲区后，replay 得到 39；之前 `clone()` 保存的输出仍为 31。
- 向原输入缓冲区复制一组不同的数值后，结果逐元素等于 `4 * new_input + 3`。
- 五个缓冲区的地址始终不变。

`benchmark.py` 另行验证 capture 后 `y = 0`、普通调用与 replay 均得到 31，并在每轮计时结束后检查结果。

## 计时方法

- 两种路径均在 `torch.no_grad()` 下运行，CPU 线程数为 1。
- `prepare()` 创建固定缓冲区、capture stream 和 side stream。在 capture stream 上预热 20 次，通过 stream 依赖等待完成后将 `y` 清零，并同步设备。
- graph 创建、capture、默认实例化及末尾同步的耗时单独记录，不计入稳态数据；本次为 0.505552 ms，不含此前预热及首次 replay。
- 每种路径在每轮计时前预热 20 次并同步。每轮连续调用 1000 次，共 5 轮；普通调用与 replay 的先后顺序逐轮交换。
- 使用 `perf_counter_ns()` 计时，在整批调用开始前和结束后 CUDA 同步。样本是每轮总耗时除以调用次数，展示值取 5 个样本的中位数。
- 普通路径包含 Python、四次 kernel 提交、stream context 切换，以及 `wait_stream` 内部的 event 创建、记录、等待与销毁；Graph 路径调用 `graph.replay()`，其依赖已在捕获时记录。
- 计时包含 CPU 提交、GPU 执行与末尾同步，CPU 与 GPU 可并行工作。数据表示连续调用的平均耗时，不是逐次同步的请求延迟或 CUDA event 测得的纯 GPU 时间。
- 计时内无输入 `copy_`、输出 `clone()`、每次调用后的同步，也不清空数据缓存或锁定 GPU 频率。输入始终为 7，每轮覆盖输出。

本次普通提交为 60.798878 μs，Graph replay 为 3.626734 μs，约 16.76 倍。该结果包含普通路径的 Python 与跨 stream 调度开销，不能外推为 CUDA 单次 launch 的固定成本或所有模型的加速比。`samples_us` 保留逐轮原始样本，`round_orders` 保留执行顺序。

## 实际 profiler trace 与图结构

`profile.py` 使用同一 `workload` 采集 CUDA/NVTX trace，并从同一次 capture 导出真实 DOT。原始 Nsight Systems 报告、截图、SQLite 和复现命令见 [profiling/README-profiling.md](profiling/README-profiling.md)；四节点、四条边、kernel 符号和节点句柄见 [structure/README.md](structure/README.md)。

采集确认普通路径每轮 4 次 kernel launch，Graph 路径每轮 1 次 `cudaGraphLaunch`，两者均执行 4 个 GPU kernel。节点级追踪用于检查执行结构，不替代独立 benchmark 的耗时。

## 依据

- [PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)：预热、capture、固定地址、跨 stream 依赖与 replay。
- [CUDAGraph API](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.CUDAGraph.html)：默认在 capture 结束时实例化；`keep_graph=True` 可保留图并显式实例化。
