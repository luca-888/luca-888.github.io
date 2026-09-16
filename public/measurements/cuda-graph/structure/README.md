# Diamond 的真实 CUDA Graph 结构

工作负载复用 `../example.py` 的 `workload`，五个缓冲区均含 4096 个 FP32 元素。四个 kernel 为 A：`x = 2 * static_input`、B：`b = x + 1`、C：`c = x + 2`、D：`y = b + c`。真实依赖为 `A → B`、`A → C`、`B → D`、`C → D`。

本目录数据由 `../profile.py` 在生成 `../profiling/diamond.nsys-rep` 的同一次进程运行中导出，来自同一个被捕获和重放的图。结构导出发生在 `cudaProfilerStop()` 之后，不计入 trace。

## 文件与来源

- `graph.dot`：PyTorch `CUDAGraph.debug_dump()` 调用 CUDA `cudaGraphDebugDotPrint()` 生成的原始 DOT，未修改节点、边或属性。
- `graph.json`：从 DOT 提取四个节点、四条边、原始 ID、kernel 符号、launch 配置及句柄；另记录同次运行的五个缓冲区地址、环境及 DOT SHA-256。
- `nodes.txt`：将 `graph.json` 中的节点字段展开为可直接阅读的文本。
- `graph.svg`：Graphviz 对原始 DOT 的直接渲染，保留全部原始属性。
- `../export_graph.py`：`profile.py` 调用其中的 `export_structure()` 完成导出；也可作为独立脚本运行。
- `render_graph.cjs`：使用 Graphviz 的 WebAssembly 构建 `@viz-js/viz` 渲染完整图。

文章的 React + SVG 图读取 `content/data/cuda-graph-structure.json`，该文件是 `graph.json` 的副本。布局只重新安排节点位置；节点和边来自实采数据，点击节点可查看原始字段。A–D 是根据本例代码添加的语义标签，`NODE 0`–`NODE 3` 则保留 DOT 的原始 ID。

## 如何读原始字段

- `ID` 与 `topoId` 是 CUDA DOT 导出的标识，不能按其数值大小推断实际执行时序；依赖顺序以边为准。本次 A 的 `topoId` 为 3、D 为 0。
- `node handle` 标识图中的节点对象；`func handle` 标识该节点调用的 kernel 函数。B、C 的 node handle 不同，但 kernel 符号与 func handle 相同：它们使用同一实现、参数分别为 1 和 2。
- 句柄不是 tensor 数据地址。`buffer_addresses` 直接来自同次运行中五个 tensor 的 `data_ptr()`，不能从 node handle 或 func handle 推断。
- `<<<4,128,0>>>` 是本次各 kernel 的 grid、block 和动态共享内存配置，均来自原始 DOT。
- Nsight SQLite 中的 `graphNodeId` 是追踪记录的节点标识，不等同于 DOT 的小整数 ID 或 node handle。不要将这些不同字段直接按数值比较。

CUDA capture 将两次普通 event 等待表示成依赖边，所以原始图包含四个 kernel 节点和四条边，并没有额外的 event 节点。B、C 之间没有依赖，具备并行条件；是否实际重叠应查阅 trace，不能只凭图结构断言。

## 复现

要同时得到可关联的 trace 与结构，请按 [../profiling/README-profiling.md](../profiling/README-profiling.md) 运行 Nsight 采集命令。`profile.py` 使用 `keep_graph=True`、开启 debug mode，capture 后检查输出未变，再显式 instantiate 与 replay，最终验证结果为 31 并导出结构。

如果只需要独立导出结构，可以从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python public/measurements/cuda-graph/export_graph.py
```

独立运行会覆盖本目录结构文件，并将 `provenance` 标记为独立导出；此时不能再声称它与已有 Nsight 报告来自同次运行。需要保持关联时，应重新运行 `profile.py` 的完整采集。

完成相应采集后，更新文章读取的数据并渲染原始 DOT：

```bash
cp public/measurements/cuda-graph/structure/graph.json content/data/cuda-graph-structure.json
npm install --prefix /tmp/cuda-graph-structure-render --no-package-lock --ignore-scripts @viz-js/viz@3.30.0
node public/measurements/cuda-graph/structure/render_graph.cjs /tmp/cuda-graph-structure-render/node_modules/@viz-js/viz
```

Graphviz 工具只在单独目录安装，不加入站点依赖。本次使用 RTX 4090、PyTorch 2.9.1+cu128、CUDA 12.8；`@viz-js/viz` 3.30.0 内置 Graphviz 16.0.0。完整记录见 `graph.json`。

参考：[NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)、[PyTorch CUDAGraph](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.CUDAGraph.html)。
