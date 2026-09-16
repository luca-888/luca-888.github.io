# Diamond 的真实 Nsight Systems 记录

本目录保存 NVIDIA Nsight Systems 在 RTX 4090 上采集的原始 trace，工作负载直接复用 `../example.py` 的 `workload`：四个逐元素 kernel、五个含 4096 个 FP32 元素的 CUDA buffer，计算 `A: x = 2 * static_input`、`B: b = x + 1`、`C: c = x + 2`、`D: y = b + c`。A 后分叉为 B、C，两者完成后汇合到 D，最终 `y = 4 * static_input + 3`。没有模拟事件，也没有重写或平移原始事件时间戳。

## 文件

- `diamond.nsys-rep`：Nsight Systems 原生报告，可直接在 Nsight Systems GUI 中打开。
- `diamond.sqlite`：Nsight Systems 导出的完整 SQLite 事件数据库。
- `summary.json`：按 NVTX 范围提取的时间坐标、CUDA API 计数、GPU kernel 计数、correlation ID、依赖时序验证及原始文件 SHA-256。
- `extract_summary.py`：从 SQLite 重新生成汇总并检查上述关系，仅需 Python 标准库。
- `nsys-stats.csv`：`nsys stats` 导出的 CUDA API、GPU kernel 和 NVTX 汇总，包含三张 CSV 表。
- `collection.log`：采集工具输出与计算结果检查。
- `../profile.py`：可复现采集脚本；同时调用 `../export_graph.py`，从本次 capture 导出 `../structure/` 的真实图结构。
- `diamond-ordinary.png`、`diamond-replay.png`：原生 Nsight Systems GUI 时间轴截图，分别放大 `ordinary/iteration_1` 和 `graph/iteration_1`。

## 环境与命令

- GPU：NVIDIA GeForce RTX 4090。
- Driver：570.153.02；CUDA：12.8；PyTorch：2.9.1+cu128。
- Nsight Systems：2024.6.2.225-246235244400v0。
- 采集日期：2026-09-16。

从仓库根目录执行，使用本地已安装的 CUDA PyTorch 解释器：

```bash
env -i HOME=/root LANG=C.UTF-8 \
  PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 \
  /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  --force-overwrite=true \
  --output=public/measurements/cuda-graph/profiling/diamond \
  /root/.cache/liger-rms-bench-cu128/bin/python \
  public/measurements/cuda-graph/profile.py \
  > public/measurements/cuda-graph/profiling/collection.log 2>&1

python3 public/measurements/cuda-graph/profiling/extract_summary.py

/usr/local/cuda/bin/nsys stats \
  --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum \
  --format csv \
  public/measurements/cuda-graph/profiling/diamond.sqlite \
  > public/measurements/cuda-graph/profiling/nsys-stats.csv
```

`cudaProfilerStart/Stop` 将采集限制在明确的区间内，排除 import、CUDA 初始化和普通调用预热。`--cuda-graph-trace=node` 逐节点采集 Graph 的 GPU kernel；默认的 graph 粒度不能用于这里的 kernel 数量对照。

采集使用最小环境，不继承会话环境；Nsight 可注入自身所需的库环境。已核查原始报告仅记录 `HOME`、`LANG`、`LD_LIBRARY_PATH`、`LD_PRELOAD`、`PATH`、`PWD` 和 `PYTHONDONTWRITEBYTECODE`。

## 采集范围

`prepare()` 先在专用 capture stream 上预热 20 轮，用 side stream 执行 C；每轮都建立 A → C 与 C → D 的依赖。等待完成并清零 `y` 后，采集依次记录 capture、检查 capture 未修改输出、显式 instantiate、第一次 replay、10 次 replay 预热、3 轮普通调用和 3 轮 Graph replay。比较轮次在提交完整 diamond 后同步一次，确保其 GPU 工作完整落在对应 NVTX 范围内；四个 kernel 之间只有必要的 stream 依赖，没有设备同步。

以下时间是原生报告坐标（ms），用于 GUI 中定位，不是 benchmark 数据。

| NVTX 范围 | 起点 | 终点 | 提交或准备 API | GPU kernel |
| --- | ---: | ---: | --- | ---: |
| `capture/diamond` | 18.416945 | 18.648270 | 4 × `cudaLaunchKernel`，处于 capture 状态 | 0 |
| `instantiate` | 18.821056 | 18.896287 | 1 × `cudaGraphInstantiateWithFlags` | 0 |
| `first_replay` | 18.900645 | 18.952843 | 1 × `cudaGraphLaunch` | 4 |
| `ordinary/iteration_0` | 19.081956 | 19.242178 | 4 × `cudaLaunchKernel` | 4 |
| `ordinary/iteration_1` | 19.248479 | 19.383974 | 4 × `cudaLaunchKernel` | 4 |
| `ordinary/iteration_2` | 19.389956 | 19.520431 | 4 × `cudaLaunchKernel` | 4 |
| `graph/iteration_0` | 19.530180 | 19.562291 | 1 × `cudaGraphLaunch` | 4 |
| `graph/iteration_1` | 19.568332 | 19.602536 | 1 × `cudaGraphLaunch` | 4 |
| `graph/iteration_2` | 19.606333 | 19.648763 | 1 × `cudaGraphLaunch` | 4 |

`capture` 外层范围包括 PyTorch 的初始化。PyTorch 在 `cudaStreamBeginCapture` **之前**执行两个 `FillFunctor<long>` kernel 来初始化内部元数据；它们不是 diamond 的四个 kernel。内层 `capture/diamond` 恰好覆盖四个工作节点的捕获过程，范围内没有 GPU kernel。

采集脚本使用 `keep_graph=True`，退出 capture context 后保留图模板，再在单独的 `instantiate` 范围内显式实例化。因此这份 trace 可区分 capture 与 instantiate；正文与 benchmark 中默认的 `CUDAGraph()` 则在 capture 结束时自动实例化。

脚本检查 capture 后 `y` 仍为 0，全部调用结束后 `y = 31`；该计算覆盖输出，不累加。随后停止 profiler，将同一张图导出至 `../structure/graph.dot` 及 `graph.json`。

## GUI 截图定位

在 Nsight Systems GUI 中打开 `diamond.nsys-rep`，展开线程的 NVTX、CUDA API，以及 GPU 的 CUDA HW 下各 stream 的 Kernels 轨道。分别选择以下 NVTX 范围并执行 **Fit to screen**：

- 普通执行：`ordinary/iteration_1`，19.248479–19.383974 ms。
- Graph replay：`graph/iteration_1`，19.568332–19.602536 ms。

定位后继续在原生时间轴上拖选工作区间，再执行 **Zoom Into Selection**。普通执行的视口约为 19.264–19.372 ms，包含 CPU 提交与 GPU 执行；replay 只放大 GPU 执行区间，视口约为 19.5857–19.5908 ms，四个 kernel 的实际执行范围为 19.586502–19.589958 ms。近似视口不代替上表准确的 NVTX 定位坐标。

两张截图放大倍率不同，图注已说明。整份报告保留 stream 7、17、142 三条轨道，但每阶段实际只用两条：普通路径 A、B、D 在 stream 7，C 在 stream 17；replay 中 A、B、D 在 stream 7，C 在 stream 142。第一图保留全部轨道，第二图隐藏此时空闲的 stream 17，仅保留 GPU 的 stream 7、142 与 `All Streams` 汇总行；`All Streams` 不是额外的 stream。Replay 保留依赖，实际调度不必复用 capture 时的 stream 分配。[PyTorch 多 stream 捕获](https://docs.pytorch.org/docs/2.9/notes/cuda.html#usage-with-multiple-streams)

普通图包含 CUDA API 与 GPU 轨道，可见 kernel launch 及 event 操作。Replay 图只包含 GPU 轨道，`cudaGraphLaunch` 位于当前窗口之前；一次 API 对应四个 kernel 的关联由原始报告和 `summary.json` 验证，不是第二图中的可见 API。各轮末尾用于划分轮次的 `cudaDeviceSynchronize` 也位于截图之外。

两图均在 Xvfb 中运行的 Nsight Systems GUI 上直接截取：普通图使用 `scrot -a 7,26,1106,351`，尺寸为 1106 × 351；replay 图使用 `scrot -a 7,26,1106,202`，尺寸为 1106 × 202。两图保留原始时间刻度，没有重画事件或改写截图文字。

## 验证关联关系与依赖

`summary.json` 同时采用范围内计数和 CUDA `correlationId` 关联验证，每一轮结果一致：

- 普通调用的四个 `cudaLaunchKernel` 各关联一个 GPU kernel。
- Graph 的一个 `cudaGraphLaunch` 关联四个 GPU kernel，这些 kernel 保留四个不同的 `graphNodeId`。
- capture 内的四个 launch API 均没有对应的 GPU kernel 执行记录。
- A 的结束时间不晚于 B、C 的开始时间；B、C 均结束后 D 才开始。

普通路径三轮中的 B、C 没有实际重叠；Graph 路径三轮的重叠分别为 832、832、864 ns。依赖图允许 B、C 并行，但不保证任意一次运行都重叠。该结果取自 kernel 的实际起止时间，不由截图像素宽度估算。

Nsight 的 `graphNodeId`、DOT 中的 `ID` 和节点句柄属于不同字段。虽然本次报告与 DOT 来自同一个 capture，也不能直接按数值将这些标识等同；kernel 符号、依赖与采集脚本共同用于解释节点含义。B、C 使用同一 kernel 函数，但输入标量、输出地址及图节点不同。

下面的 SQL 可以检查每个 API 对应的 GPU kernel 数量，并结合 `NVTX_EVENTS` 的起止时间选取轮次：

```sql
SELECT s.value AS api,
       r.start,
       r.end,
       r.correlationId,
       COUNT(k.start) AS gpu_kernels
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN StringIds AS s ON s.id = r.nameId
LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
  ON k.correlationId = r.correlationId
WHERE s.value LIKE 'cudaLaunchKernel_%'
   OR s.value LIKE 'cudaGraphLaunch_%'
GROUP BY r.start, r.end, r.correlationId, s.value
ORDER BY r.start;
```

## 解读限制

节点级追踪有额外开销，NVTX 标记与逐轮同步也改变了测量方式。此记录用于验证实际提交路径、依赖执行和 kernel 数量；不要把截图中的时间跨度当作文章 benchmark 的延迟，也不要根据截图重算加速比。性能数字沿用未开启 profiler 的 `../results.json`。

参考：[NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)，其中说明 `cudaProfilerApi` 采集范围和 CUDA Graph 的 graph/node 两种追踪粒度。
