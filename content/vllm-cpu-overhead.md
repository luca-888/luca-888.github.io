Decode 的一步，GPU 上只要几毫秒；而每一步开始之前，CPU 都要决定这一步算哪些请求、输入张量是什么样、上一步的结果怎么发回给用户。这些工作的耗时基本不随模型大小变化，GPU 越快、模型越小，它占的比例就越大，直到 GPU 在两步之间空转等 CPU。

vLLM V1 用三个办法让 CPU 的活不挡在 GPU 前面：**挪走**（tokenize、detokenize、网络 IO 交给别的进程与线程）、**藏住**（async scheduling 让下一步的调度与这一步的 GPU 执行重叠）、**做薄**（MRV2 把输入准备搬到 GPU 上，CUDA Graph 把几百次 kernel launch 合成一次）。下文先用实测看 GPU 空等时 CPU 在做什么，再逐项说明这三个办法。源码基于 vLLM v0.30.0。

在 H100 上跑 Qwen3-8B：关掉 async scheduling，GPU 每步空转从 0.3 ms 升到 2.9 ms；关掉 CUDA Graph，ITL 从 7.03 ms 升到 13.95 ms，几乎翻倍。

## 一、一步 decode 里的 CPU 与 GPU

Decode 每步只为每个请求生成一个 token，计算量小，时间主要花在读权重与 KV cache 上。8B 模型在 H100 上一步约 6.8 ms，绝大部分是 36 层 Transformer。

同步执行时，GPU 算完一步后要等 CPU 做完三件事，下一步才能开始。下表取自关闭 async scheduling、并发 8 时的 profiler trace：

| CPU 工作 | 耗时 | 主要内容 |
| --- | --- | --- |
| 处理输出、调度 | 0.66 ms | 拷回采样结果、追加 token；选下一步的请求、分配 block |
| 准备输入 | 1.55 ms | 上百个小 tensor 操作与其间的 Python 开销 |
| 提交执行 | 0.44 ms | 一次 `cudaGraphLaunch`，含约 480 个 kernel |

三项合计约 2.65 ms，基本就是 GPU 空等的时间。最大的一块是准备输入，其中更新请求状态、拷贝变化约 0.94 ms，构造 attention metadata 约 0.30 ms。耗时的不是调度算法，而是这些杂务，UCSD WukLab 在 vLLM v0.5.4 上也得出同样结论。[WukLab, 2024](https://mlsys.wuklab.io/posts/scheduling_overhead/)

三个办法各管一部分：async scheduling 把前两项藏到上一步的 GPU 执行后面；MRV2 压缩准备输入；CUDA Graph 把提交从约 490 次 launch 压成一次，没有它这一项会涨到约 7 ms。

下图按比例画出连续三步：同步执行时每步空转约 2.9 ms，开启 async scheduling 后只剩约 0.3 ms。

::engine-step-timeline::

## 二、一个请求的路径

单卡部署时，vLLM 有两个进程：

- **API server**：收 HTTP 请求，做 tokenize 和多模态预处理；输出方向做 detokenize、处理 stop string、流式返回。
- **EngineCore**：调度器与模型执行，单卡时直接在进程内驱动 GPU。

拆成进程而不是线程，是因为 Python 的 GIL：同一进程里同一时刻只有一个线程在跑 Python 代码。tokenize 留在引擎进程里，就会和调度抢 CPU；放进另一个进程才能真正并行。

两个进程之间用 ZMQ 通信。EngineCore 内部再分三个线程：输入线程反序列化请求，主线程跑 busy loop，输出线程序列化结果。socket IO 会释放 GIL，所以收发消息能和主线程重叠。

多卡 TP 时每张卡再多一个 worker 进程，调度结果通过共享内存广播。[vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/)

::engine-process-path::

SGLang 拆法不同：tokenizer 在主进程，scheduler 与 detokenizer 各是一个子进程。目的相同，都是让驱动 GPU 的进程只做调度与执行。

拆分之后，EngineCore 主线程只剩第一节表中的三项工作，下面三节逐项处理。

## 三、藏住调度：async scheduling

同步的 busy loop 每步是 `schedule → execute_model → 等结果 → update_from_output`，下一步的调度必须等这一步的采样结果回来。Async scheduling 打破了这个依赖：这一步还在 GPU 上算时，CPU 就开始调度并准备下一步。它从 v0.14 起默认开启。

难点是调度下一步时还不知道这一步采样出了哪个 token。vLLM 分两处解决：

- **调度器一侧**：`AsyncScheduler` 给每个请求记一个占位数（`num_output_placeholders`），下一步直接按“多一个 token”排程，结果回来再扣掉。
- **GPU 一侧**：采样出的 token 留在 GPU 上，下一步的输入直接从那里读，不必绕回 CPU。

执行循环相应改成 `step_with_batch_queue`：最多一批在执行、一批已提交，队列满了才等结果。

关掉它，并发 8 的 ITL 从 7.03 ms 升到 9.02 ms，吞吐掉 22%。并发越高，每步 GPU 时间越长，差距越小。

Structured output 是例外：下一步的 grammar bitmask 要等上一步的 token，无法完全重叠。

SGLang v0.4 的 zero-overhead batch scheduler 思路相同：先调度下一批，用 future token 占位，结果就绪后回填。[SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## 四、做薄输入：从 persistent batch 到 Model Runner V2

准备输入是把每个请求的 token、位置、block table 整理成模型要的张量。MRV1 每步在 CPU 上用 numpy 算出这些数组，再拷到 GPU。它用 **persistent batch** 跨步复用张量、只改差异，但这份状态本身就是模型输入，请求增删要整表重排，在 async 下还要额外加锁保护。[Model Runner V2 设计文档](https://docs.vllm.ai/en/latest/design/model_runner_v2/)

Model Runner V2（MRV2）从 v0.29 起成为默认，做了三件事：

- **状态与输入分离**：请求状态常驻 GPU，每个请求占固定一行；每步按需把这些行 gather 成本步输入。
- **只传变化**：CPU 只把新请求、新分配的 block 写过去。
- **GPU 上生成输入**：4 个 Triton kernel 算出本步输入：`_prepare_pos_seq_lens` 生成 positions 与 seq_lens，`_combine_sampled_and_draft_tokens` 生成 input_ids，`_gather_block_tables` 取出 block table，`_compute_slot_mappings` 算出新 token 的 KV 写到哪。它们就是时间线空转里那几个小 kernel。

::engine-persistent-batch::

实测中 MRV2 每步的 CPU 时间从 2.52 ms 降到 2.15 ms，但 ITL 都是 7.03 ms：async 下这段 CPU 工作本来就藏在 6.8 ms 的 GPU 执行后面。模型越小、GPU 越快，这一段越关键，官方在 Qwen3-0.6B 上报告吞吐提升 56%。[MRV2 blog](https://vllm.ai/blog/2026-03-24-mrv2)

## 五、合并 launch：CUDA Graph 与 torch.compile

提交 forward 本身也是 CPU 工作：一次 forward 有约 490 个 kernel，逐个 launch。CUDA Graph 把这一串 launch 录成一张图，之后一次 replay。

图的 shape 是固定的，而每步 token 数在变。vLLM 启动时为 1、2、4、8、16、24 … 512 个 token 各录一张图，运行时把本步 token 数补到最近的一档，例如 13 补到 16。小 batch 档位密，因为补齐的浪费按比例算：3 补到 4 多了三分之一，500 补到 512 只多 2%。超过 512 就不用图。

Attention 的形状随每个请求的上下文长度变化，早期难以录进图。V1 最初用 **piecewise**：在 attention 处把计算图切开，其余部分录进图，attention 每层在图外单独 launch。v0.11 起默认改为 `FULL_AND_PIECEWISE`：

- 纯 decode 的 batch 走整图，连 attention 也在图里。
- 含 prefill 的 batch 走 piecewise。
- 超过最大档位时不用图。

::engine-cudagraph-dispatch::

代价是启动时要编译和录图，并多占一些显存。[vLLM CUDA Graphs 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

三种模式的 GPU 计算时间相同，差别在 CPU 每步发起多少次 launch（并发 8，trace 计数）：

| 模式 | 每步 launch | ITL |
| --- | --- | --- |
| 不用 CUDA Graph | 492 次，每个 kernel 一次 | 13.95 ms |
| `PIECEWISE` | 168 次：37 段图 + 每层 attention 约 3 个 | 7.57 ms |
| 默认（整图） | 25 次：1 张整图 + 图外的小 kernel | 7.03 ms |

整图也不是零成本：那一次 `cudaGraphLaunch` 本身仍要约 0.45 ms CPU，就是第一节表里的“提交执行”。

## 六、CPU 仍会成为瓶颈的地方

这三个办法管的是每步的固定成本，CPU 还会在别处卡住：

- **CPU 核数不足**：多卡推理时 CPU 核不够，GPU 照样空转，补足后 TTFT 可降低 1.47 至 7.11 倍。[Chung et al.](https://arxiv.org/abs/2603.22774)
- **前端预处理**：长 prompt 的 tokenize、多模态输入的预处理都在 API server 进程里，请求多时它会先成为瓶颈。

更激进的路线是干脆不要 host CPU：Blink 把调度与 KV 管理放进 GPU 上常驻的 kernel，由 SmartNIC 直接收发请求。[Blink, 2026](https://arxiv.org/abs/2604.07609)

## 七、实测

一张 H100，Qwen3-8B，vLLM v0.30.0，输入 128、输出 256 token，比较五种配置。

::engine-measured::

| 配置 | 并发 1 | 并发 8 | 并发 64 | 并发 256 |
| --- | --- | --- | --- | --- |
| 默认（async + MRV2 + `FULL_AND_PIECEWISE`） | 145 | 1118 | 6834 | 12149 |
| 关 async scheduling | 117 | 875 | 5170 | 11114 |
| MRV1 | 145 | 1115 | 6840 | 12989 |
| `PIECEWISE` | 132 | 1029 | 6634 | 12302 |
| 无 CUDA Graph | 71 | 569 | 4098 | 12421 |

表中为输出吞吐（tok/s）。各配置的 GPU 计算时间几乎一样，差别全在 CPU 那一侧；并发升到 256 后每步 GPU 时间变长，CPU overhead 被摊薄，各配置差距随之缩小。

## 八、小结

同步执行时，GPU 每步约有 2.9 ms 在等 CPU 处理输出、准备输入和提交。vLLM V1 先把 tokenize、detokenize 挪出主线程，再用 async scheduling 把剩下的 CPU 工作藏到 GPU 后面，最后用 MRV2 和 CUDA Graph 把它做薄。在 8B 上，async scheduling 与 CUDA Graph 分别省下约 2 ms 与 7 ms 的 ITL。

## 参考

- vLLM Team. [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) · [Model Runner V2](https://docs.vllm.ai/en/latest/design/model_runner_v2/) · [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) · [Optimization Levels](https://docs.vllm.ai/en/latest/design/optimization_levels/) · [vLLM V1 alpha release](https://vllm.ai/blog/2025-01-27-v1-alpha-release) · [Model Runner V2 blog](https://vllm.ai/blog/2026-03-24-mrv2).
- [vLLM v0.30.0 源码](https://github.com/vllm-project/vllm/tree/v0.30.0)：`vllm/v1/engine/core.py`、`vllm/v1/core/sched/async_scheduler.py`、`vllm/v1/worker/gpu/model_runner.py`、`vllm/config/vllm.py`、`vllm/config/compilation.py`。
- Srivatsa, Li, Zhang, Abhyankar. [Can Scheduling Overhead Dominate LLM Inference Performance?](https://mlsys.wuklab.io/posts/scheduling_overhead/) 2024.
- LMSYS. [SGLang v0.4: Zero-Overhead Batch Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). 2024.
- Chung, Jia, Jezghani, Kim. [Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference](https://arxiv.org/abs/2603.22774). IISWC 2026.
- Siavashi et al. [Blink: CPU-Free LLM Inference](https://arxiv.org/abs/2604.07609). 2026.
