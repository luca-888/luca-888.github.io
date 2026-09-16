# CUDA Graph：执行原理与优化实践

**CUDA Graph 将一组操作及其依赖组织成可重复执行的图，通过预先准备，减少每轮的提交与调度开销。** 节点可以表示 GPU kernel、内存复制或显式注册的 CPU 回调。本文用四个逐元素 kernel，结合代码、依赖图和真实 trace 说明其原理与使用边界。

## 一、执行原理

逐次启动 kernel，需要反复支付 CPU 侧的准备与提交开销，以及 GPU 侧的启动与调度开销。Kernel 越短，这些开销越容易成为瓶颈。即使 CPU 已提前提交，Graph 仍可能减少 GPU 侧的启动开销。[NVIDIA llama.cpp 案例](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)

当操作及其依赖反复出现时，可以将工作定义与执行分开，让准备结果被多次复用：

**Definition：定义操作与依赖。** 节点与依赖边组成图模板 `cudaGraph_t`。既可以显式添加节点，也可以通过 stream capture 构图：支持捕获的 CUDA 操作被记录为节点及依赖，暂不执行。

**Instantiation：准备可执行图。** CUDA 对图模板取快照，完成验证和大部分执行准备，生成 `cudaGraphExec_t`。这一步不执行图中的工作。

**Execution：启动并执行。** 通过 `cudaGraphLaunch` 启动可执行图，节点满足依赖后由 CUDA 调度。后续启动复用同一执行对象，无须重新构图和实例化。[NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#building-and-running-graphs)

::cuda-graph-lifecycle::

本文通过 PyTorch 的 stream capture 构图，再用 `graph.replay()` 重放。Capture 中普通的 Python 代码不会自动成为图节点；Host Node 则执行显式注册的 CPU 回调，回调内不能调用 CUDA API。[Host Node](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)

## 二、四节点 diamond

输入 `static_input` 是含 4096 个元素的 FP32 CUDA 张量。四个 kernel 分别计算 `A: x = 2 * static_input`、`B: b = x + 1`、`C: c = x + 2`、`D: y = b + c`，结果为 `y = 4 * static_input + 3`。

B、C 读取 A 的结果、写入不同缓冲区，彼此独立；D 等待两者完成，形成分叉后汇合的 diamond：

::cuda-graph-structure::

下面是**手动使用两条 stream 的 eager**：A、B、D 在 `origin`，C 在 `side`。Graph 从同一个 `workload` 捕获操作与依赖；所有缓冲区均提前分配。

```python
def workload(static_input, x, b, c, y, side):
    origin = torch.cuda.current_stream()
    torch.mul(static_input, 2, out=x)  # A：origin stream
    side.wait_stream(origin)           # A → C
    torch.add(x, 1, out=b)             # B：origin stream
    with torch.cuda.stream(side):
        torch.add(x, 2, out=c)         # C：side stream
    origin.wait_stream(side)           # C → D
    torch.add(b, c, out=y)             # D：origin stream
```

**两次 `wait_stream` 建立跨 stream 的依赖。** 第一次位于 A 之后、B 之前，让 C 只等待 A；第二次让 D 等待 C。A → B、B → D 由 `origin` 内的顺序保证。这些等待排入 stream，不阻塞 CPU 提交后续工作。

Eager 每轮逐次分派四个算子、调用四次 kernel launch，并设置 event 依赖。Capture 将这些操作及依赖记录下来；本例的内部 event 等待转化为图中的边，导出结果仍是四个 kernel 节点。

### 捕获与重放

输入初值为 7，`y` 初值为 0。`capture_stream` 和 `side` 已提前创建；完整准备代码见文末示例。

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=capture_stream):
    workload(static_input, x, b, c, y, side)

graph.replay()  # y = 31
graph.replay()  # y 仍为 31
```

默认 `CUDAGraph` 在 capture 结束时完成 instantiate，此时 `y` 仍为 0。四个 kernel 在 replay 时才执行，每轮读取当前输入并覆盖输出，因此重复重放仍得到 31。[PyTorch CUDAGraph](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.CUDAGraph.html)

## 三、执行轨迹与性能

下面是本例的 **NVIDIA Nsight Systems** 原生时间轴：

::cuda-graph-trace::

原始报告确认：eager 的四次 kernel launch 各对应一个 GPU kernel；replay 的一次 `cudaGraphLaunch` 对应全部四个 kernel。

本次 A、B、D 都在 stream 7 上执行；C 在 eager 中使用 stream 17，在 replay 中使用 stream 142。空白轨道在当前区间没有任务，`All Streams` 只是汇总行。**Replay 保留依赖关系，不要求沿用 capture 时的 stream 分配。** [PyTorch 多 stream 捕获](https://docs.pytorch.org/docs/2.9/notes/cuda.html#usage-with-multiple-streams)

两条路径都允许 B、C 并行。这次 trace 中，eager 的 B、C 没有重叠，replay 的 B、C 发生了重叠；D 均在两者结束后开始。分叉提供并行机会，是否重叠取决于提交时机、调度与资源。

在 **NVIDIA GeForce RTX 4090** 上，每轮耗时如下：

| 实现 | 每轮提交 | GPU kernel 数 | 每轮耗时（μs） |
| :--- | :--- | ---: | ---: |
| 多 stream eager | 4 次 kernel launch + event 操作 | 4 | 60.80 |
| Graph replay | 1 次 graph launch | 4 | 3.63 |

每轮耗时从 **60.80 μs 降至 3.63 μs，约 16.76×**。Replay 将逐次 Python 分派、kernel launch API 调用和 event 设置，替换为一次 graph launch；四个 kernel 的计算保持不变。这个倍数是整轮执行的收益，不能等同于单个 kernel 的加速。

## 四、应用场景与收益边界

CUDA Graph 适合**结构稳定、反复执行、提交与调度开销明显**的工作。小 batch 推理、LLM 逐 token decode 都是常见应用场景。

在 ML 中，Graph 作用于模型执行阶段：推理可捕获 forward；训练可覆盖 forward、backward 和支持 capture 的 optimizer 更新。[PyTorch 整轮训练捕获](https://docs.pytorch.org/docs/2.9/notes/cuda.html#whole-network-capture)

### 直接捕获

对于满足 capture 约束的固定形状计算，可以像本文一样使用 `torch.cuda.CUDAGraph`。调用方负责固定缓冲区、写入新输入，并在输出被覆盖前消费或复制它。

训练也可用 `torch.cuda.make_graphed_callables` 捕获子模块的 forward 与 backward，将数据读取、日志和动态控制留在外层。[PyTorch 局部捕获](https://docs.pytorch.org/docs/2.9/notes/cuda.html#partial-network-capture)

### 框架集成

**PyTorch：** `torch.compile(model, mode="reduce-overhead")` 会尝试用 CUDA Graph 降低兼容计算的运行开销。用户仍调用编译后的模型，捕获与重放由框架管理。[编译模式](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html)

**vLLM：** 框架根据 batch、attention backend 与配置选择整图、分段重放或 eager 执行。CUDA Graph 常用于 decode，也可覆盖受支持的 prefill 或混合 batch。[vLLM CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

### 成本与收益

本例创建和准备图约需 **0.51 ms**，首次 launch 也可能有额外开销，需要通过多次复用摊销。实际使用还需考虑输入复制、输出保留、固定缓冲区占用，以及图更新或重建的成本。

调优时，先确认 **CPU 提交或 GPU 侧启动开销是否影响性能**，排除数据未就绪或同步等待造成的空隙。如果长 kernel 已占据主要耗时，或图很少复用、频繁重建，收益就可能有限。

再权衡捕获范围和 shape 档位：整图减少提交次数，分段捕获容纳不兼容操作；更多档位能覆盖更多输入，也会增加准备成本，并可能占用更多显存。Kernel fusion 减少 kernel 数量与中间访存，可与 Graph 的提交和调度优化叠加。

[完整示例](../../measurements/cuda-graph/example.py) · [测量脚本](../../measurements/cuda-graph/benchmark.py) · [原始数据](../../measurements/cuda-graph/results.json) · [实验记录](../../measurements/cuda-graph/README.md)

[Nsight Systems 报告](../../measurements/cuda-graph/profiling/diamond.nsys-rep) · [Trace 采集记录](../../measurements/cuda-graph/profiling/README-profiling.md)

[全部节点文本](../../measurements/cuda-graph/structure/nodes.txt) · [图结构与导出记录](../../measurements/cuda-graph/structure/README.md)
