# RTX PRO 6000 多卡拓扑：连接结构与通信路径

多卡通信性能取决于 **GPU 之间的连接路径，以及并发传输对共享资源的使用**。本文以 NVIDIA RTX PRO 6000 Blackwell Server Edition 的八卡参考设计为例，介绍 PCIe 连接、NUMA 内存组织及其对任务分组的影响。

## 一、硬件连接与内存组织

### 单卡硬件规格

RTX PRO 6000 Blackwell Server Edition 面向服务器中的 AI 计算与图形处理，主要参数如下。

| 参数 | 规格 | 参数 | 规格 |
| :--- | ---: | :--- | ---: |
| GPU 架构 | Blackwell | 显存 | 96 GB GDDR7 ECC |
| CUDA 核心 | 24,064 | 显存位宽 | 512 bit |
| Tensor Core | 752 个，第五代 | 显存带宽 | 1,597 GB/s |
| RT Core | 188 个，第四代 | 主机接口 | PCIe 5.0 x16 |
| FP32 峰值算力 | 120 TFLOPS | NVLink | 不支持 |
| 最大功耗 | 600 W，可配置 | MIG 分区 | 最多 4 个隔离实例 |

规格来源：[NVIDIA 官方产品页](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)、[Lenovo 服务器产品规格](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu)。

CUDA 核心承担通用并行计算，Tensor Core 加速矩阵运算，RT Core 加速光线追踪。对本文的多卡拓扑而言，重点是 **显存容量、显存带宽和 PCIe 接口**：分别对应单卡的数据容纳能力、本地数据读写能力和外部传输能力。

### PCIe 与八卡结构

每张 GPU 拥有独立显存。本地计算通过显存接口读写数据；GPU 与主机内存、其他 GPU 之间的数据传输则使用外部连接。RTX PRO 6000 Blackwell Server Edition 不支持 NVLink，卡间通信主要通过 PCIe。[NVIDIA GPU 规格对照](https://docs.nvidia.com/vgpu/sizing/virtual-workstation/latest/gpus-vws.html)

**PCIe 是连接 CPU、GPU、网卡等设备的高速互连标准。PCIe Switch 负责连接多个设备并转发数据，其通往 CPU 的连接称为上行链路。**

::rtx-reference::

参考设计的 GPU 连接结构如下：

- **两个 CPU**：记为 CPU 0 和 CPU 1，通过 CPU 间互连连接。
- **四个 PCIe Switch**：每个 CPU 连接两个 Switch。
- **八张 GPU**：每个 Switch 连接两张 GPU，并连接一张计算网卡。

每个 Switch 连接两张 GPU 是这套参考设计的分组方式；连接数量由 Switch 的端口、PCIe 通道数量及整机设计决定。[NVIDIA 八卡参考设计](https://docs.nvidia.com/enterprise-reference-architectures/white-paper/latest/appendix-d.html)

### NUMA 与内存位置

**NUMA（Non-Uniform Memory Access，非统一内存访问）将主机内存分布在不同节点。CPU 可以访问本地和远端内存，远端访问需要经过节点间互连，通常延迟更高。**

| 架构 | 内存组织 | 主要特点 |
| :--- | :--- | :--- |
| UMA：统一内存访问 | 多个 CPU 核心访问共享内存，访问代价大致一致 | 数据放置简单，扩展受共享内存带宽限制 |
| NUMA：非统一内存访问 | 内存分布在不同节点，CPU 可访问本地和远端内存 | 可扩展容量与总带宽，需要关注跨节点流量 |
| 分布式内存 | 各机器拥有独立内存，通过网络交换数据 | 适合多机扩展，需要组织跨机器通信 |

- **NUMA 节点**：通常由一组 CPU 核心及其本地内存组成，以 NUMA 0、NUMA 1 等编号标识。一个物理 CPU 可划分为多个 NUMA 节点。
- **GPU 的 NUMA 归属**：表示 GPU 与 CPU、主机内存的连接位置，用于选择邻近的计算与内存资源。

NUMA 描述主机内存的组织方式。GPU 自己的显存位于卡上，通过另一套接口访问。

## 二、GPU 间的通信路径

**GPU P2P 允许一张 GPU 直接访问另一张 GPU 的显存，省去主机内存中转。** 数据沿 PCIe 路径传输，经过的设备取决于两张 GPU 的连接位置。[CUDA P2P 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#peer-to-peer-memory-access)

### 三类连接路径

下表以 GPU A、GPU B 表示通信双方。路径中的 CPU 表示其内部的 I/O 硬件，由该硬件转发数据。

| GPU 位置 | 数据路径 | 使用的连接 |
| :--- | :--- | :--- |
| 同一个 Switch | GPU A → Switch → GPU B | 两张 GPU 与 Switch 的连接 |
| 不同 Switch、同一个 CPU | GPU A → Switch A → CPU 0 → Switch B → GPU B | 额外使用两个 Switch 的上行链路 |
| 不同 CPU | GPU A → Switch A → CPU 0 → CPU 间互连 → CPU 1 → Switch B → GPU B | 额外使用 CPU 间互连 |

同一 Switch 内的 GPU 通信可在 Switch 内完成转发。跨 Switch 或跨 CPU 通信则涉及更多共享连接。

### 拓扑标记

`nvidia-smi topo -m` 使用以下标记描述设备之间的路径。其中 Host Bridge 是 PCIe 与主机系统之间的桥接部分。

| 标记 | 路径含义 |
| :--- | :--- |
| `PIX` | 最多经过一个 PCIe bridge |
| `PXB` | 经过多个 PCIe bridge，不经过 Host Bridge |
| `PHB` | 经过 PCIe Host Bridge |
| `NODE` | 经过同一 NUMA 节点内不同 Host Bridge 之间的互连 |
| `SYS` | 经过 NUMA 节点之间的系统互连 |

这些标记描述连接路径；NUMA 0、NUMA 1 等编号则标识节点。[nvidia-smi 拓扑说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology)

### 实机拓扑示例

下面是一台 RTX PRO 6000 Blackwell Server Edition 八卡服务器的实机拓扑。GPU 0–3 属于 NUMA 0，GPU 4–7 属于 NUMA 1，每个节点内包含两组 PIX 配对。

::rtx-topology::

- **GPU 0 ↔ GPU 1：PIX**，属于同一配对。
- **GPU 0 ↔ GPU 2：NODE**，位于同一 NUMA 节点的不同分支。
- **GPU 0 ↔ GPU 4：SYS**，跨 NUMA 节点。

同一台机器的 GPU 间单向带宽如下，行表示源 GPU，列表示目标 GPU。

::rtx-bandwidth::

**同 NUMA 节点内的带宽约为 53 GB/s，跨 NUMA 节点约为 37–40 GB/s。** 本次 PIX 与 NODE 的带宽接近，主要差异出现在跨 NUMA 的 SYS 路径，矩阵中的分块与上图的 NUMA 分组对应。

## 三、共享带宽与通信开销

**单独传输时可用的带宽，在多路并发时需要由共享同一链路的流量分摊。** 例如，同一个 Switch 下的两张 GPU 同时从主机内存接收数据，会共同占用其上行链路；增加 GPU 数量，并不会同步增加这条链路的容量。

分析并发性能时，应先确定各路数据经过的资源，再找出重叠部分：

| 传输场景 | 主要关注的资源 |
| :--- | :--- |
| GPU 读取自己的显存 | 本地显存带宽 |
| 同一 Switch 内的 GPU P2P | GPU 的 PCIe 链路与 Switch 转发能力 |
| 跨 Switch、跨 CPU 的 GPU P2P | 沿途上行链路与 CPU 间互连 |
| 多张 GPU 同时访问主机内存 | 主机内存带宽与共享 PCIe 链路 |

通信开销还与消息大小有关：小消息更容易受到启动和路径延迟影响；大消息更关注可用带宽及并发争用。**单卡显存带宽、卡间传输带宽和多卡通信耗时，是不同层面的指标。**

## 四、任务分组与资源放置

### GPU 分组

Tensor parallel 将模型层的计算分配给多张 GPU，需要频繁交换中间结果。分组时可优先选择同一 Switch 下的 GPU，再根据卡数扩展到同一 CPU 连接的 GPU，以减少跨分支通信。

这些交换通常通过 NCCL 的集合通信完成：**AllReduce** 将各 GPU 的数据归约后分发给所有参与者，**AllGather** 将各 GPU 的数据汇集到所有参与者。NCCL 根据拓扑组织通信，GPU 分组会影响这些操作的整体耗时。

独立推理副本的卡间通信较少，主要关注模型加载、输入数据等流量对共享资源的占用。将并发任务分散到不同分支，有助于利用多条上行链路。

### CPU 与主机内存放置

- **CPU 亲和性**：限定线程可运行的 CPU 核心，使数据准备线程靠近对应 GPU。
- **内存放置**：选择主机缓冲区所在的 NUMA 节点，减少输入数据的跨节点传输。

两者分别控制执行位置与数据位置，应结合 GPU 的 NUMA 归属安排。

拓扑分析应围绕任务的数据流展开：识别通信双方、确定传输路径，再判断并发流量共用哪些链路。GPU 分组与 CPU、内存放置共同决定这些资源的使用方式。
