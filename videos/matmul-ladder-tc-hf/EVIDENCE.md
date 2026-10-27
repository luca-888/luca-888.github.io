# 实测证据（v6 用）

视频里说的每个结论都要能在这里或 CODE.md、文章里找到依据；这里的数字是依据，**不是要念出来的内容**（念出来的数字只有 OUTLINE.md 列的几个）。
原始数据在 `public/measurements/matmul-ladder-tc/`，测量脚本在 `scripts/matmul-ladder/`。

## 实测数据（2026-10-27，Modal H100 SXM 700 W）

- 脚本：`scripts/matmul-ladder/modal_ncu_video.py`（`--part ladder / extra / nsys_soak`）、`video_ncu_summary.py`、`tc_peak.cu`；对照 kernel 在 `matmul_tc.cu`（`wmma_pad`、`mma_sync_ld`、`mma_noswz`）。
- 原始数据：`public/measurements/matmul-ladder-tc/` 下的 `20261027-134747/`（阶梯各版）、`20261027-140146-extra/`（对照与峰值）、`20261027-140640-nsys_soak/`（wmma 持续运行的 nsys）。前两个目录各有 `summary.json`。
- ncu 每个 kernel 一份 `--set full`，8192 阶；Modal 不能锁频，ncu 期间 SM 时钟 1.59–1.83 GHz。三次运行 GPU 合计约 207 s。
- 所有手写 kernel 的结果与 cuBLAS 逐位相同（`max_rel_err` 0）。

| 版本 | Tensor 管线忙 | 停顿第一 | 第二 | 显存读 | L2 命中 | L2 吞吐 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 wmma | 15.9% | 等访存返回 39%（约 2/3 等搬运的全局读取，约 1/3 等读片段） | 访存排队 21%（主要是读片段的 shared memory 读取） | 4.65 GB | 66% | 23% |
| 对照 wmma + padding | 23.0% | 等显存/L2 返回 46% | 固定延迟依赖 17% | 2.31 GB | 76% | 27% |
| 对照 128 B 行、不换位 | 11.2% | 等 shared memory 返回 45% | shared memory 排队 35% | 4.11 GB | 64% | 14% |
| 对照 换位 + mma.sync、搬运经寄存器 | 34.5% | 等显存/L2 返回 37% | 等 shared memory 返回 16% | 2.31 GB | 71% | 28% |
| 2 swizzle + cp.async + mma.sync | 45.2% | 等显存/L2 返回 21% | 固定延迟依赖 21% | 3.00 GB | 68% | 50% |
| 3 pipeline | 53.7% | 固定延迟依赖 30% | 计算管线忙 18% | 4.26 GB | 63% | 63% |
| 4 TMA + wgmma | 55.6% | 同步等待 57% | 固定延迟依赖 20% | 4.83 GB | 60% | 95% |
| 5 warp specialization | 50.9%（4096 阶 75.5%） | 等显存/L2 返回 45% | 同步等待 40% | 6.71 GB | 48% | 92% |
| 6 tile 分组 | 90.1% | 同步等待 64% | 等显存/L2 返回 15% | 1.37 GB | 79% | 80% |
| 7a 只加 PIPE | 90.7% | 同步等待 32% | 等显存/L2 返回 26% | 1.63 GB | 78% | 79% |
| 7b 再加常驻 | 86.4% | 等显存/L2 返回 61% | 同步等待 12% | 2.55 GB | 72% | 82% |
| 7 再加 cluster | 92.6% | 同步等待 56% | 固定延迟依赖 10% | 1.32 GB | 77% | 68% |
| 8 TMA store | 93.8% | 同步等待 41% | 等显存/L2 返回 30% | 1.22 GB | 74% | 65% |

停顿原因对应的 ncu 名字：等显存/L2 返回 = Long Scoreboard，全局访存排队 = LG Throttle，同步等待 = Barrier，固定延迟依赖 = Wait，等 shared memory 返回 = Short Scoreboard，shared memory 排队 = MIO Throttle，计算管线忙 = Math Pipe Throttle。L2 吞吐 = `lts__throughput`。

其他实测：

- bank conflict（每次 shared memory 读取的平均轮数）：wmma 4.00，padding 对照 1.01，128 B 行不换位 8.00，swizzle 1.00。
- 第 2 版的拆分（同一次运行轮流计时，TFLOPS）：wmma 100 → padding 148；不换位 71 → 换位 289；换位后搬运经寄存器 222 → 换成 cp.async 289；pipeline 337。
- 两种指令的上限（`tc_peak.cu`，只连续发指令、不读显存）：mma.sync.m16n8k16 643 TFLOPS，此时 Tensor 管线忙 99.6%；wgmma.m64n256k16 962 TFLOPS，忙 99.5%。
- wmma 版的 shared memory 读取：4.03 亿条普通读取指令（ldmatrix 0 条），每个 16×16 片段 4 条，每条 32 线程 × 4 字节 = 128 字节，平均 4.02 轮。
- 第 3 版：指令发射槽忙 43%（未占满）；shared memory 数据通路占峰值 57%（未占满）。
- wmma 持续运行 2 秒（nsys）：每次 10.63–10.94 ms，中位数 10.71 ms，102.6 TFLOPS；SM 时钟 1980 MHz，功耗 385 W。文章同口径的数字是 11.78 ms（另一次运行）。
- 最终版 ncu 期间时钟 1.62 GHz，此时钟下峰值 876 TFLOPS，实测 794 TFLOPS，为 90.6%。

![Tensor Core 忙的时间](concept-img/tensor-busy.png)

![停顿原因构成](concept-img/stalls.png)

![bank conflict](concept-img/bank.png)

![第 2 版的拆分对照](concept-img/ablation.png)

![两种指令的上限](concept-img/peak.png)

![显存与 L2](concept-img/memory.png)

![nsys 时间线](concept-img/nsys-wmma.png)

### 实测推翻或修正的旧说法

| 旧说法 | 实测 | 处理 |
| --- | --- | --- |
| （撤回）“第一种等待不是 bank conflict” | 读片段的 `load_matrix_sync` 编译成通用地址的 `LD.E`（ncu 标为 Shared Load），和全局读取走同一条访存队列；“访存排队 21%”里约 16 个百分点在读片段那两行 | 第一种等待仍是 shared memory 排队，s02 交代“和全局读取挤同一条队列” |
| wmma 用 ldmatrix 读 | ldmatrix 0 条；普通读取每条 128 字节，每片段 4 条 | s02 第 2 句按实测的读取方式 |
| mma.sync 卡在“每时钟要发一条” | 发射槽只忙 43% | 删去；改用 mma.sync 自身上限 643 TFLOPS |
| shared memory 每时钟需 192 B、只供 128 B | 第 3 版 shared memory 通路只占 57% | 删去 |
| 第 7 版只是 cluster | 代码里是 PIPE + 常驻 + cluster；单独加常驻反而慢 | s07 如实交代三项 |
| 等待一路从 SM 内部移到芯片之外 | 第 4 版 L2 吞吐已到 95% | 删去“一路外移”，只说每步消除了哪一项 |
| wmma 一次 11.78 ms | 这次持续运行 10.71 ms | 配音改为“约 11 毫秒、约 100 TFLOPS、峰值的一成” |

---

