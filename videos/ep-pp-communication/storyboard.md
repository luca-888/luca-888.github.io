# EP 通信与 PP：分镜表

16:9 横版（1920 × 1080），有配音、字幕与音效，全片约 5 分钟，14 幕。色彩与版式沿用系列的纸色风格（左上角色块标签 + 粗体标题、实心语义色块）。状态：用户确认方案（2026-10-20，一支片子、14 幕、不跑实测），直接做成片；配音暂不合成，只烧录字幕。文案与节拍以 `script.js` 为准（本表是分镜阶段的设计，个别数字已按 trace 更正：prefill 一层约 34 ms；decode 一个 micro-batch 一层的计算 0.56 ms，与通信 561 µs 相当）。

文件：`script.js`（文案与节拍，配音稿即字幕）、`data.py` → `data.js`（真实 trace）、`core.js` / `scenes.js`（画面）、`render.mjs`（12 fps 逐幕预览、分镜页、拼接预览）、`vo.py`（MiniMax 合成，已备好未运行；合成后生成 `vo.js`，字幕与音轨自动改用实测时长）。

## 全片的问题

DeepSeek-V3 的 FP8 权重 689 GB，一台 8 × H800 共 640 GB，必须跨机。跨机有两种切法：按 expert 切（EP），按层切（PP）。它们在线上各传什么、传多少？怎样传得起？推理里各自用在哪？

链条：放不下，要跨机（第 1 幕）→ 按 expert 切，token 要跟着 expert 走（第 2 幕）→ 传的量和计算一样耗时（第 3 幕）→ 少走 IB：转发（第 4 幕）→ 转发要先报数，decode 等不起（第 5 幕）→ decode 直发、形状固定（第 6 幕）→ 把通信藏到计算后面：prefill 分 SM，decode 不占 SM（第 7、8 幕）→ 藏住了，最慢的卡还在拖（第 9 幕）→ 2026 年的变化与链路的上限（第 10 幕）→ 另一种切法：按层，只传一次（第 11 幕）→ 代价：气泡与延迟（第 12 幕）→ 推理里 PP 用在哪（第 13 幕）→ 回到集群图（第 14 幕）。

## 贯穿全片的图形

**集群图**：每个节点是一个竖长的卡片，里面 8 个 GPU 格子；节点内的 NVLink 画成粗总线，节点之间是每卡一张网卡的 IB 线（同编号 GPU 连在同一条“轨”上）。线宽按真实带宽比例：NVLink 160 GB/s 对 IB 50 GB/s，3.2 : 1；第 10 幕加入 NVL72 的 900 GB/s，同一比例尺。

**token**：一个细长色条（7168 维），跟着它走完 dispatch 与 combine。全片只跟一个 token：节点 0、GPU 5 上的一个 prompt token。

**真实 trace 条**：第 3、7、8 幕下方出现 DeepSeek 在 H800 上的 profile（`data.js`），kernel 按真实时长画成色块，横轴是微秒或毫秒。示意内容在画面上标“示意”。

颜色：token 与 dispatch blue，combine purple，expert 计算 yellow，attention teal，等待与气泡 red，两个 micro-batch 用 yellow 与 green（与 DeepSeek 原图一致）。

## 分镜（14 幕）

每幕的“配音”按节拍排列，`→` 后是这一句落在的画面动作。字幕与配音同文。

### 1 开场：放不下（12 s）

- **画面问题**：一个模型为什么要跨机。
- **关键帧（幕尾）**：左上大标题“EP 通信与 PP”；右侧一台 8 卡 H800 的格子，权重色块 689 GB 灌进去，在 640 GB 处溢出，溢出的部分流向右边第二台机器，两台之间亮起 IB 线。随后标题缩到左上角成为页眉。
- **画面上的字**：DeepSeek-V3，FP8 权重 689 GB；8 × H800 = 640 GB。
- **配音**：
  - 一台机器装不下这个模型。→ 色块溢出
  - 只能分到几台机器上，机器之间靠网卡连接。→ IB 线亮起
  - 怎么分，决定了网线上每一步要传什么。→ 标题缩到页眉
- **音效**：灌入的流水声，溢出时一声低音。
- **出处**：Hugging Face `deepseek-ai/DeepSeek-V3` 权重大小；H800 80 GB。

### 2 一个 token 要去哪（22 s）

- **画面问题**：按 expert 切开以后，token 为什么要搬。
- **关键帧**：集群图 4 个节点 × 8 卡（EP32，prefill 的部署单位）。每张卡上 9 个 expert 小格（8 个 + 1 个冗余，冗余格先灰着）。节点 0、GPU 5 上的 token 经 router 选出 8 个 expert，分布在 3 个节点的 7 张卡上；8 条 blue 箭头出发（dispatch），在目标卡上变 yellow（expert 计算），8 条 purple 箭头返回并按权重相加（combine）。最后所有卡同时这样做，箭头铺满，画面右下角浮出一个 32 × 32 的流量矩阵，每格深浅不同。
- **画面上的字**：256 个 expert，每卡 8 个；router 选 8 个；dispatch；combine；每个 MoE 层两次，58 层。
- **配音**：
  - 每张卡只放一部分 expert。→ expert 小格亮起
  - router 选中的 expert，多半不在这张卡上。→ 8 个目标高亮
  - 所以 token 要先过去，算完再回来。→ dispatch / combine 箭头
  - 每张卡都在同时收发，数量由 router 当场决定。→ 流量矩阵浮出
  - 这种每对卡之间都有往来的通信，称为 all-to-all。→ 矩阵标名
- **音效**：每条箭头出发一声短音，铺满时连成一片。
- **出处**：DeepSeek-V3 报告（256 routed expert、top-8、每 token 最多 4 个节点）；DeepEP V1 README（EP32 的冗余 expert 配置见 V3 报告 §3.4）。

### 3 要传多少（20 s）

- **画面问题**：这些搬运有多重。
- **关键帧**：上半是 token 色条的拆解：7168 个数 × FP8 1 字节 ≈ 7 KB，复制 8 份 → 57 KB；回程 BF16 2 字节 → 115 KB。下半是两条管道按真实宽度并排：NVLink 160 GB/s、IB 50 GB/s。最下方一行真实 trace（prefill，EP32，每个 micro-batch 8192 个 token）：dispatch 4.6 ms、combine 8.6 ms 的通信条，与 attention 3.0 ms、expert GEMM 6.3 ms 的计算条在同一时间轴上并排，通信条不比计算短。
- **画面上的字**：每个 token：去 57 KB，回 115 KB；H800 实测，DeepSeek profile。
- **配音**：
  - 一个 token 去程按 FP8 发，回程是 BF16，回程大一倍。→ 色条拆解
  - 出了节点，带宽只剩三分之一。→ 两条管道
  - 在 DeepSeek 的线上配置里，一层的通信和计算一样长。→ trace 条对齐
  - 两条路：少走慢的链路，再把通信藏到计算后面。→ 两条路标在画面右侧
- **出处**：DeepSeek-V3 报告（NVLink 160 GB/s，IB 50 GB/s）；profile-data `prefill.json`（`data.py` 提取）。

### 4 少走 IB：先过网卡，再在节点内转发（22 s）

- **画面问题**：同一个节点的几个 expert，能不能只过一次网卡。
- **关键帧**：回到第 2 幕的 token。左半“直发”：8 份里 6 份各自穿过 IB。右半“转发”：每个目标节点只走一次 IB，落到同编号的 GPU 5，再沿 NVLink 分给本节点的目标卡；IB 线上的份数从 6 变成 2。一个计数器：IB 上 6 份 → 2 份。旁边标出 node-limited routing：每个 token 最多去 4 个节点。最下方：节点内 EP8 实测 153 GB/s，跨节点 43–58 GB/s；132 个 SM 里 24 个用于搬运。
- **画面上的字**：先走 IB 到同编号的卡，再走 NVLink；每个 token 最多 4 个节点；normal kernel（高吞吐）。
- **配音**：
  - 去同一个节点的几份，内容完全相同。→ 6 份 IB 线重合
  - 那就只过一次网卡，到了节点再分。→ 转发动画
  - NVLink 是 IB 的三倍多，节点内多分几份几乎不花时间。→ 两条管道的宽度对照
  - 为了让这个办法生效，DeepSeek 在模型里限制每个 token 最多去 4 个节点。→ 4 个节点的框
- **音效**：IB 一声，NVLink 分叉时几声轻响。
- **出处**：DeepSeek-V3 报告 §3.2.2（“3.2 experts per node”，20 SMs）；DeepEP V1 README 实测表；profile-data trace 中通信 kernel 的 grid 24 与 GEMM 的 grid 108。

### 5 转发的代价：先报数（18 s）

- **画面问题**：这个办法为什么不能用在 decode。
- **关键帧**：画面左侧放 DeepEP 原图 `normal.png`（CPU 与 GPU 两条泳道，红框圈出 CPU 的 Waiting 与 Tensor allocation）。右侧把它画成一条时间线：notify（交换各卡要收多少）→ CPU 等待 → 按数量分配 tensor → dispatch。底部两个刻度对照：prefill 一层约 34 ms（两个 micro-batch），一次 CPU 等待在里面只是一道细缝；decode 一层约 1.4 ms，一步要发出约 3600 个 kernel，平时靠 CUDA graph 一次提交，CPU 一停下来等数，这一步就只能由 CPU 逐个 launch。
- **画面上的字**：收多少要先知道，才能分配；形状每步不同，进不了 CUDA graph。
- **配音**：
  - 接收方得先知道会来多少个 token。→ notify 块
  - CPU 要停下来等这个数，再分配显存。→ 红框
  - prefill 一层要几十毫秒，这点等待不算什么。→ 上刻度
  - decode 一层只有一毫秒多，等不起；每步形状不同，CUDA graph 也用不上。→ 下刻度
- **出处**：DeepEP V1 README（“an implicit CPU wait for GPU received count signal”）与 `figures/normal.png`；profile-data（decode 一步的 kernel 数由 `decode.json` 统计）。

### 6 decode：直发，形状固定（22 s）

- **画面问题**：decode 怎么省掉报数与转发。
- **关键帧**：集群图缩小成 16 个节点 × 8 卡（EP128，decode 的配置），每卡 2 个 expert。同一个 token 的 8 份经 RDMA 直接发到目标卡，不经转发。目标卡上一个预先分好的接收框：`[本地 expert 数, 最多 token 数, 7168]`，框是固定大小，收到的 token 填进去，空位留着，计数留在 GPU 上。旁边一行真实代码 `buffer.low_latency_dispatch(x, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=True, return_recv_hook=True)`。底部：EP128、每卡 128 个 token 时，dispatch 192 µs、combine 369 µs，与这一层的计算约 0.7 ms 在同一刻度上。
- **画面上的字**：low-latency kernel；接收区预先分好，大小固定；不用等 CPU，可以进 CUDA graph。
- **配音**：
  - decode 每张卡只有一百多个 token。→ 集群缩成 16 个节点
  - 不再转发，每一份直接发到 expert 所在的卡。→ RDMA 直线
  - 接收区按最坏情况预先留好，不用先问数量。→ 固定大小的框
  - 代价是显存，换来的是每一步形状都不变。→ 空位闪一下
  - 即便如此，通信仍与这一层的计算一样长。→ 刻度对照
- **出处**：DeepEP V1 README 与 v1.2.1 `deep_ep/buffer.py`（`low_latency_dispatch` 的签名与输出形状）；DeepEP 首版 README 的 low-latency 表（H800 + CX7）；profile-data `decode.json`。

### 7 藏起来（一）：prefill 分出 24 个 SM（20 s）

- **画面问题**：通信这么长，计算能不能同时进行。
- **关键帧**：真实 trace 的两条泳道（prefill.json，一层）：上面计算 stream（108 个 SM），下面通信 stream（24 个 SM）。两个 micro-batch 分别着 yellow 与 green：micro-batch 0 在通信时，micro-batch 1 在算 attention 或 expert，反之亦然。时间轴 0–34 ms；两条泳道的忙碌比例标在右侧：计算 92%，通信 89%。右上角小图是 DeepSeek 原图 `profile-prefill.jpg`。
- **画面上的字**：把 batch 拆成两半；一半通信时，另一半计算；H800 实测 trace。
- **配音**：
  - 把一个 batch 拆成两半，交错执行。→ 一条泳道分成两色
  - 一半在网线上时，另一半正好在算。→ 两条泳道对齐
  - 搬运的 kernel 本身也要 SM，这里让出 24 个。→ 24 / 108 的标注
- **出处**：profile-data `prefill.json`、`assets/prefill.jpg`。

### 8 藏起来（二）：decode 不占 SM（22 s）

- **画面问题**：decode 的计算本来就短，还要分出 SM 吗。
- **关键帧**：真实 trace 的一条泳道（decode.json，一层两个 micro-batch，约 1.42 ms），132 个 SM 全部用于计算。`dispatch_ll` 与 `combine_ll` 是极窄的两种块：发出约 20 µs，接收约 20–70 µs；发出与接收之间，下方画一条虚线“在网线上”，它跨过另一个 micro-batch 的 attention（FlashMLA 243 µs）。右侧：stream 忙碌 97%。名字在这里出现：SGLang 叫 two-batch overlap，vLLM 叫 DBO。角落小字：batch 太小时拆半反而慢，SGLang 实测每卡 32 个 token 时 −27%。
- **画面上的字**：发出就返回，SM 全部让给计算；数据在另一半的 attention 期间到达。
- **配音**：
  - 发送的 kernel 只负责发出请求，随即退出。→ 窄块
  - 数据由网卡在后台传，这时 SM 已经去算另一半的 attention。→ 虚线跨过 FlashMLA
  - 等 attention 算完，再用一个很短的 kernel 把数据收进来。→ 接收块
  - 一整层下来，这条 stream 几乎没有空闲。→ 97%
- **出处**：profile-data `decode.json`（`dispatch_ll`、`combine_ll` 的时长与位置）与 README（“does not occupy GPU SMs”）；DeepEP `figures/low-latency.png`（备用）；SGLang 博客 2025-05-05（TBO 数字）；SGLang `batch_overlap/operations_strategy.py`、vLLM `--enable-dbo`。

### 9 最慢的那张卡（22 s）

- **画面问题**：通信藏住了，还有什么在拖慢一步。
- **关键帧**：左侧 SGLang 实测的 DeepSeek-V3 expert 分布热图（原图，58 层 × 256 个 expert，少数格子明显更深）。右侧 32 张卡的负载柱，按同一条基线排开，高低不齐，一条红线标出最高的那根：所有卡都等它。EPLB 动画：最热的几个 expert 复制到灰着的冗余格（256 + 32 = 288，每卡 9 个），柱子重新排平。右下：SGLang 实测，EPLB 后 prefill 吞吐 1.49 倍，decode 2.54 倍。
- **画面上的字**：每层所有卡同步，一步由最忙的卡决定；EPLB：复制热门 expert，重新装卡。
- **配音**：
  - 每一层所有卡都要同步，一步的时间取决于最忙的那张。→ 红线
  - router 偏爱少数 expert，它们所在的卡就成了瓶颈。→ 热图的深色格
  - 办法是统计负载，给最热的 expert 多放几份，再重新分到各卡。→ 复制与装卡
  - 冗余的位置在第二幕就留好了。→ 冗余格变色
- **占位（可选实测 M2）**：右侧的负载柱默认按热图示意并标“示意”；若要真实柱高，需要一次路由统计（见末尾）。
- **出处**：SGLang 博客 2025-05-05（热图、1.49 / 2.54 倍）；DeepSeek-V3 报告（32 个冗余 expert，约每 10 分钟重排）；EPLB `eplb.py`。

### 10 2026 年：合二为一，链路决定上限（18 s）

- **画面问题**：这套做法今天还是这样吗。
- **关键帧**：上半一条时间线：2025-02 DeepEP 开源（两种 kernel）→ 2026-04 V2：两种合并为同一个 dispatch / combine，改用 NCCL 的 GPU 直发网络接口，SM 用量至多减到四分之一，不再支持 0 SM 的 RDMA 模式 → 2026-09 V2.5 删除 V1。下半回到管道比例尺：IB 50、H800 NVLink 160、GB200 NVL72 的 NVLink 900 GB/s 三条管道按真实宽度；NVL72 的 72 张卡在一个 NVLink 域里，EP 不必出域。DeepSeek 的估算：只算 all-to-all，decode 每 token 至少 14.76 ms（IB）对 0.82 ms（NVL72）。
- **画面上的字**：DeepEP V2；GB200 NVL72：72 张卡一个 NVLink 域；理论下限 14.76 ms → 0.82 ms。
- **配音**：
  - 前面讲的，是 2025 年 DeepSeek 线上的做法。→ 时间线起点
  - 今年的新版把两种 kernel 合成了一套接口。→ V2 节点
  - 硬件也在变：整个 expert 组可以放进一个 NVLink 域。→ 900 GB/s 的管道
  - 网线有多粗，决定了 decode 最快能多快。→ 两个下限数字
- **出处**：DeepEP README（V2、V2.5 news）；SGLang GB200 Part 1（NVLink-only 时关掉 TBO）；DeepSeek ISCA '25（14.76 ms、0.82 ms）；NVIDIA GB200 NVL72 规格。

### 11 另一种切法：按层（20 s）

- **画面问题**：不按 expert 切，改按层切，线上传什么。
- **关键帧**：同样 4 个节点，但每个节点放一段连续的层：1–16、17–31、32–46、47–61（vLLM 默认的分法，多出的一层给前面的 stage）。token 从 stage 1 走到 stage 4，每过一个边界只传一次：hidden state 与 residual，2 × 7168 × 2 字节 ≈ 28 KB；3 个边界共约 86 KB。旁边同一比例尺下是 EP 的量：每个 MoE 层去 57 KB、回 115 KB，58 层约 10 MB。两根柱子并排，PP 的柱子几乎看不见。
- **画面上的字**：PP：按层切成 stage；每个边界传一次；每 token 每步：约 86 KB 对约 10 MB。
- **配音**：
  - 把层切成几段，每台机器放一段。→ 4 段层号
  - token 只在段与段之间传一次，而且只发给下一台。→ 3 个边界亮起
  - 同一个 token，按层切要传的量只有按 expert 切的百分之一。→ 两根柱子
- **出处**：vLLM `distributed/utils.py` `get_pp_indices`、`IntermediateTensors(hidden_states, residual)`；SGLang PP 博客的通信量公式；DeepSeek-V3 配置。

### 12 代价：气泡与延迟（22 s）

- **画面问题**：传得少，为什么推理不都用 PP。
- **关键帧**：4 行时间线（4 个 stage）。先只有一个 batch：它从 stage 1 走到 4，任一时刻只有一行在算，其余三行是 red 的空白（气泡），利用率 1/4。再放进 4 个 batch，像流水线一样错开，四行都填满。旁边一行真实代码注释：`# PP requires PP-size concurrent batches to fill the pipeline.`（vLLM）。最后跟住其中一个 token：它仍要依次走完 4 个 stage、3 次传输，从进入到产出的时间不变；而每个 batch 只有四分之一的请求。
- **画面上的字**：一个 batch：只有一个 stage 在算；4 个 batch：填满；单个 token 的延迟不变；每个 batch 变小。
- **配音**：
  - 一个 batch 进来，同一时刻只有一台机器在工作。→ 气泡
  - 推理没有反向传播，不同 batch 互不依赖，放进四个就能填满。→ 四行填满
  - 但每个 token 还是要依次走完所有的段。→ 跟住一个 token
  - PP 提高的是吞吐和容量，不是单个请求的速度。→ 两行结论
- **出处**：vLLM `config/vllm.py` `max_concurrent_batches`、`v1/engine/core.py` 的 batch queue；GPipe、Megatron-LM 的 bubble 公式只用于说明训练的不同。

### 13 推理里 PP 用在哪（22 s）

- **画面问题**：那推理里什么时候用 PP。
- **关键帧**：三栏，各一个场景：
  - 装不下、机器之间没有快速互联：节点内 TP，节点之间 PP（vLLM 文档的建议）。
  - 超长 prompt 的 prefill：一个 128K 的 prompt 切成 chunk，chunk 在 4 个 stage 间流水（动画，chunk 斜着依次进入）。SGLang 实测（H20，DeepSeek-V3.1）：TTFT 从 TP8 的 48.5 s 降到 PP4 TP8 的 15.5 s，吞吐比 TP32 高。
  - 大 MoE 的 decode：DP attention + EP，跨节点也是 EP，不用 PP（DeepSeek、SGLang、Kimi K2 的部署）。
- **画面上的字**：每栏一行标题与一个数字。
- **配音**：
  - 机器之间只有慢速网络时，按层切传得最少。→ 第一栏
  - 超长的 prompt 本身就能切成很多块，流水线自然被填满。→ chunk 流入
  - decode 每步只有一个 token，填不满流水线，大 MoE 模型在这里用的是 EP。→ 第三栏
- **出处**：vLLM `docs/serving/parallelism_scaling.md`；SGLang PP 博客 2026-01-15（48.5 s → 15.5 s，3.31 倍对 2.54 倍）；DeepSeek Day 6、SGLang Kimi K2 博客。

### 14 结尾：一张图上的三种流量（14 s）

- **画面问题**：回到开头，网线上到底在传什么。
- **关键帧**：回到第 1 幕的两台机器，扩成第 2 幕的集群图。三种流量同时在图上流动，按真实频率与大小：节点内 NVLink 上是 TP 的 all-reduce 或 EP 的转发；节点之间，EP 每个 MoE 层两次 all-to-all（密集的 blue / purple 点），PP 每步每个边界一次（稀疏的大块）。页眉回到大标题。
- **画面上的字**：节点内：TP / EP 转发；节点之间：EP 每层两次，PP 每段一次。
- **配音**：
  - 按 expert 切，每一层都要跨机两次，所以要转发、要重叠、要均衡。→ EP 流量
  - 按层切，每段只传一次，代价是流水线要填满。→ PP 流量
- **音效**：回到开场的低音。

## 实测占位

草稿与分镜不做实测。现有的实机材料（DeepSeek 的 H800 trace、DeepEP 官方实测表、SGLang 与 vLLM 的公开数字）已能支撑每一幕。下面两项可选，每项跑之前单独征求同意：

| 编号 | 用在 | 内容 | 硬件 | 预估 |
| --- | --- | --- | --- | --- |
| M1 | 第 3、4、10 幕 | 用 DeepEP V2 自带的测试，在同一套代码上测节点内（NVLink）与跨节点（RDMA）的 dispatch / combine 带宽，替换引用的 2025 年 V1 数字 | 2 节点 × 8 张 H100 / H200，节点间 InfiniBand（Modal 的多节点集群） | 约 0.5 h，按 16 卡计约 $30–40，需审批 |
| M2 | 第 9 幕 | 统计一个 MoE 模型在真实 prompt 上每层各 expert 收到的 token 数，按 EP32 的摆放算出每卡负载，画真实的负载柱与 EPLB 前后对比 | 单张 H100 / H200 跑一个能装下的 MoE（如 Qwen3-30B-A3B）；用 DeepSeek-V3 本身需要 8 × H200 | 单卡约 15 min，约 $1–2 |
