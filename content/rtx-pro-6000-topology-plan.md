# RTX PRO 6000 多卡拓扑：从硬件连接到通信性能

写作提纲，等待实机资料与测量结果。本文聚焦用户提供的 RTX PRO 6000 多卡节点；H100 拓扑另文讨论。

## 硬件与文章范围

- 获取途径：用户已提供 [Modal 调用说明](../docs/modal.md) 与 [Runpod 调用说明](../docs/runpod.md)。Modal 首次八卡调用等待 10 分钟未收到远端事件；按用户要求延长后的[第二轮实验](../docs/rtx-pro-6000-experiment.md#第二轮延长-modal-等待)已正常等待 30 分钟，仍未观察到运行容器或取得实机资料，结束后确认应用与排队任务均已停止。用户已充值并明确改用 Runpod，[本轮计划](../docs/rtx-pro-6000-runpod-experiment.md)为一个同机八卡 Secure Pod，准备与测试合并一次执行，总期限 25 分钟、预算 $10。
- 实验需要同一物理节点上的多张完整 GPU，以及 CPU、NUMA、PCIe 设备树的可见性。能否对比同一 Switch、不同 Host Bridge 和跨 NUMA 路径，取决于该节点实际连接关系。
- 不需要加载大模型，显存主要用于通信缓冲区；具体规模在确认可用显存与节点占用后确定。
- 用户提供的图片对应 NVIDIA 八卡 RTX PRO 6000 Blackwell Server Edition 参考设计。它用于解释结构，不能替代实际节点的拓扑记录。
- 单节点可完成 GPU 通信与主机内存传输实验。GPU 到网卡的路径先按实际网卡位置解释；如需网络吞吐或 GPUDirect RDMA 实测，需另外确认对端节点与网络资源。

## 文章主线

同一台服务器中的 GPU，通过哪些设备交换数据？哪些传输会共用链路？这些关系如何影响多卡任务的分组与放置？先还原实机结构，再用通信结果解释路径差异。

### 一、读懂一台多卡服务器

以整机拓扑图介绍 GPU、PCIe Switch、CPU 的 PCIe Host Bridge、主机内存与 NUMA 归属。CPU 型号等一般测试环境信息留在测量记录；正文保留解释拓扑所需的插槽、NUMA 节点与设备连接。

区分 GPU 本地显存访问、主机内存与 GPU 之间的传输、GPU 之间的通信。显存带宽不能用作 GPU 间通信带宽。

图示：实际节点总览，固定 GPU 编号与 PCI bus ID 的对应关系，并注明编号仅对应本次采集。

### 二、从参考设计到实际连接

简要解读用户提供的八卡参考图，再以实机信息展开。参考设计名称 `2-8-5-200` 中，四项依次表示 CPU 数量、GPU 数量、NIC 数量、每 GPU 平均东西向网络带宽；末项不能直接解释成每张网卡的端口速率。

通过 `nvidia-smi topo -m`、PCIe 设备树和 NUMA 信息建立对应关系。解释 `PIX`、`PXB`、`PHB`、`NODE`、`SYS` 描述的路径类型；它们不是实测速度，也不能独自证明 CUDA P2P 可用。

RTX PRO 6000 Blackwell Server Edition 官方规格不支持 NVLink。参考图保留的 NVLink 图例没有对应连线，解读时以实际连接与产品规格为准。

图示：实际拓扑矩阵与简化设备树并排，通过相同编号对应设备。

### 三、数据经过哪些路径

沿实际存在的连接分别追踪：

1. 同一 PCIe Switch 下的 GPU 通信。
2. 经 Host Bridge 或跨 NUMA 的 GPU 通信。
3. 本地与远端主机内存到 GPU 的传输。

单独说明“经过 CPU 的 I/O 结构”和“数据复制到主机内存中转”的区别。P2P 可用性需要驱动能力与真实传输验证；不能仅凭物理连线推断。

只有数据确实经过 Switch 上行链路时，才用该上行链路分析共享瓶颈。同一 Switch 内可直接转发的流量，不应一概算到 CPU 上行带宽上。

交互图：选择实际存在的源端与目标端，高亮经过的设备和链路，在图旁直接显示路径说明。

### 四、NCCL 的 topology XML

以实际节点导出的 NCCL topology XML 为例，连接硬件拓扑与通信库的识别结果。正文保留一段覆盖两张 GPU 及其共同上游的 XML，完整文件放入原始记录，并将 XML 节点与拓扑图逐项对应。

重点说明 `cpu`、嵌套的 `pci`、`gpu`、`nic` / `net` 如何组织设备，以及 `numaid`、`busid`、`link_speed`、`link_width` 等字段的含义。`cpu` 节点与 NUMA 归属结合解读，不能直接把 XML 中的 `cpu` 数量当作物理 CPU socket 数量；每个 `pci` 元素也不能一概称为独立的物理 Switch。字段和采集行为以实机 NCCL 版本为准。

区分 `NCCL_TOPO_DUMP_FILE` 与 `NCCL_TOPO_FILE`：前者将检测后的拓扑导出为 XML，后者在检测前加载部分或全部拓扑描述。具体导出命令保留在采集记录；本篇先分析默认检测结果，不通过修改 XML 预设性能结论。

Topology XML 表示 NCCL 构建的硬件拓扑描述，不能直接当作最终 ring / tree、channel 排列或每次 collective 的执行轨迹。链路属性也不是实测吞吐。用 NCCL 日志与通信结果补足“看到了什么连接”和“实际使用什么传输方式”之间的证据。

图示：XML 节选与设备树并排，按 PCI bus ID 高亮相同设备，再选一对 GPU 解释共同上游和候选路径。

### 五、拓扑与通信结果

用实测回答三个问题：不同 GPU 对之间的带宽是否不同；同时传输时哪些设备争用链路；相同 GPU 数量、不同分组是否改变 collective 的耗时。

正文展示 GPU 两两带宽矩阵、与路径对应的延迟结果、本地与远端内存传输对比，以及代表性分组的 NCCL AllReduce / AllGather 结果。延迟图注明测量对象，区分远端内存访问延迟与整次传输耗时。没有实测前不填写数值或预设快慢排序。

图示：带宽热力图、共享路径下的并发传输对比、collective 消息大小与耗时曲线。使用真实数据，单位与方向明确；具体测量方法保留在原始记录。

### 六、多卡任务的放置

依据实测讨论频繁通信的 GPU 如何分组，进程与主机内存如何绑定 NUMA，以及 GPU 与网卡如何配对。区分通信密集的 tensor parallel 与相对独立的多副本任务；建议须对应本节点证据，不从某一次带宽测量直接推导所有模型的性能。

## 采集与实验准备（不进入文章正文）

- 在首次 GPU 申请前完成一次性测试计划：明确完整测试矩阵、镜像与代码准备情况、预计准备/排队/GPU 占用/回传时间；查询当前费率，列出预计总费用范围、预算上限、超时与失败停止条件，再向用户说明。首次设备与拓扑探测纳入同一次计划；本轮已完成[测试计划与费用估算](../docs/rtx-pro-6000-experiment.md)，CPU 镜像构建成功，远端调用因本地首次事件等待超时结束，未收到 GPU 启动事件或实测数据。
- 记录 GPU 型号、数量、PCI bus ID、驱动与 CUDA/NCCL 版本、主机与容器可见性、PCIe 链路最大能力和负载下实际速率、CPU/NUMA 设备归属。详细环境和原始输出保留在测量目录。
- 采集 `nvidia-smi topo -m`、`nvidia-smi topo -p2p r`、`nvidia-smi topo -p2p w`、`lspci -D -t`、`lspci -D -nn`、`numactl --hardware`；针对路径上的设备补充链路和桥信息。CUDA 侧验证 peer access 与数据正确性。
- 初始化实际 NCCL communicator 时通过 `NCCL_TOPO_DUMP_FILE` 导出 topology XML，记录 NCCL 版本、参与 GPU、设备编号与 rank 映射，以及是否加载了既有 topology 文件。结合日志核对 XML 的可见范围；不同运行的输出分别保存。
- 优先使用 NVIDIA nvbandwidth 与 nccl-tests。固定并记录工具版本；分别保留方向、消息大小、执行方式、绑定方式和原始结果。
- GPU 对带宽采用方向明确的矩阵；对角线不混入本地显存复制速度。并发实验与单对实验分开展示。
- 本地与远端主机内存对照需同时记录 CPU 亲和性和内存实际分配位置，避免只绑 CPU 后把结果归因于内存位置。
- 相同卡数的不同分组用于比较拓扑影响；不同卡数的 collective 另作扩展性观察。保留 NCCL 日志，确认实际使用的传输方式。
- 先观察现有配置。遇到 P2P 或性能异常，记录 ACS、IOMMU、虚拟化等相关状态，不为实验自动修改 BIOS、驱动或全局设备配置。

## 已核对资料

- [NVIDIA Enterprise Reference Architecture：Appendix D / Figure 10](https://docs.nvidia.com/enterprise-reference-architectures/white-paper/latest/appendix-d.html)：用户所给八卡参考图来源。
- [参考设计命名规则](https://docs.nvidia.com/enterprise-reference-architectures/white-paper/latest/introduction.html)：C-G-N-B 含义与 `2-8-5-200` 配置。
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：拓扑标记、P2P 能力与亲和性查询。
- [NVIDIA GPU 规格对照](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html)：RTX PRO 6000 Blackwell Server Edition 的 NVLink Support 为 No。
- [CUDA 多 GPU 与 Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-access)：按设备对查询 P2P 能力。
- [GPUDirect RDMA 的系统连接条件](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#supported-systems)：Switch、Host Bridge 与跨 CPU 路径的区别。
- [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)：主机与 GPU、GPU 对之间的传输测量；不同测试项的带宽与延迟含义以工具文档和固定版本实现为准。
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)：collective 正确性与性能测试。
- [NCCL topology XML 的加载与导出](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-topo-file)：`NCCL_TOPO_FILE` 与 `NCCL_TOPO_DUMP_FILE`。
- [NCCL XML 拓扑构建源码](https://github.com/NVIDIA/nccl/blob/master/src/graph/xml.cc)：字段与硬件探测关系；成文时改用与实测 NCCL 版本对应的源码引用。

## 成文条件

完成一次性测试计划与时间/费用估算 → 按当前获取途径申请同机八卡节点 → 还原真实拓扑并完成通信测量 → 释放资源 → 补齐正文与实测图表。参考设计与 XML 解读已接入博客草稿页。

当前状态：原测试套件已准备，Modal 尝试均已停止。用户充值后，Runpod 的 EUR-IS-2 八卡容量在 LOW/NONE 间变化；已成功分配同机八卡、$16.72/h，但首次因本地型号字段读取错误提前释放（已修正），后续成功分配的节点因宿主 Docker 拉取镜像错误未启动。就绪期限到达后已删除，所有本轮 Pod 列表核查为空，没有 SSH、XML 或通信实测结果。浏览器排除 EUR-IS-2 后，其余可见数据中心没有匹配八卡容量。完整原因、时间与费用估算见[Runpod 实验记录](../docs/rtx-pro-6000-runpod-experiment.md)，下一次申请前更新剩余预算与时间，不再依赖 Modal，不变更卡数。

停止后的只读查询确认实验函数 `backlog=0`、`num_total_tasks=0`、`num_running_inputs=0`，`TaskList` 返回容器数 0；当前没有继续等待或运行的本轮任务。统计接口没有排队位置或预计开始时间。日志仅记录本地 `TimeoutError()` 导致应用停止，尚不能解释首事件未返回的具体原因。
