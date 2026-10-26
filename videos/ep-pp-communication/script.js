// EP 通信与 PP 影片的文案与节拍，画面（scenes.js）、配音（vo.py）与渲染（render.mjs）共用。纯数据，不依赖页面。
// 每幕：id 供 scenes.js 引用、tag 标签、head 标题（{t:…} teal、{p:…} purple、{r:…} red、{b:…} blue、{y:…} 深黄 为强调色）、src 出处、dur 时长（秒）、beats 节拍。
// 节拍：[幕内起点（秒）, 画面动作, 配音]。配音只讲“为什么”和“看哪里”，不念画面上的字和数字；配音稿即字幕。出处见 docs/ep-pp-communication-notes.md。
// hold：关键帧距幕尾的秒数（开场取在标题缩上去之前）。
const SCRIPT = [
  { id: 'open', tag: '开场', head: '', src: '', dur: 13.6, hold: 2.3, beats: [
    [0.8, '大标题；右侧一台 8 卡 H800，689 GB 的权重灌入 640 GB 的显存', 'DeepSeek-V3 的权重超过一台机器的显存。'],
    [4.4, '溢出的部分流向第二台机器，两台之间的网络亮起', '模型因此分布在多台机器上，机器之间经网卡连接。'],
    [9.0, '大标题缩到左上角', '切分方式决定了每一步在网络上传输的数据。'],
  ] },
  { id: 'route', tag: '按 expert 切分', head: '{b:发送}到 expert 所在的卡，计算后{p:取回}', src: 'DeepSeek-V3：256 个 expert，top-8；prefill 部署 EP32', dur: 25, beats: [
    [0.5, '集群图：4 台机器 × 8 张卡；每卡 8 个 expert 格与 1 个空着的冗余格', '以 prefill 的部署为例：4 台机器共 32 张卡，每张卡存放 8 个 expert。'],
    [7.5, '机器 0、GPU 5 上的 token；router 选出的 8 个 expert 在 3 台机器的 7 张卡上高亮', 'router 选中的 expert 大多位于其他卡。'],
    [12.5, '8 份 token 发往目标卡（dispatch），在那里计算，再发回并加权求和（combine）', '因此，token 先发送到这些卡，计算结果取回后按权重相加。'],
    [18.5, '所有卡同时收发；右侧浮出 32 × 32 的流量矩阵，深浅不一', '每张卡同时发送和接收，数量由 router 在运行时决定。这种通信称为 all-to-all。'],
  ] },
  { id: 'volume', tag: '通信量', head: '一层的通信与计算{r:耗时相当}', src: 'H800 实测：DeepSeek profile-data，prefill，EP32', dur: 24.5, beats: [
    [0.5, 'token 色条拆解：7168 个数，去程 FP8，8 份 57 KB；回程 BF16，115 KB', '去程按 FP8 发送，回程按 BF16 发送，回程数据量为去程的两倍。'],
    [7.0, '两条管道按真实宽度：NVLink 160 GB/s，网卡 50 GB/s', '跨机时，每张卡的带宽不到机器内的三分之一。'],
    [12.0, '真实 trace：一层里通信 stream 与计算 stream 的累计时长，并排比较', '在 DeepSeek 的线上配置中，一层的通信与计算耗时相当。'],
    [18.0, '右侧两条出路：少走慢的链路；把通信藏到计算后面', '优化方向一是减少经过慢速链路的数据，二是让通信与计算重叠。'],
  ] },
  { id: 'forward', tag: '减少跨机流量', head: '发往同一机器的数据只经过{b:一次}网卡，再由 NVLink 分发', src: 'DeepSeek-V3 报告 §3.2.2；DeepEP V1 实测（H800 + CX7）', dur: 28, beats: [
    [0.5, '直接发送：去往机器 1、机器 3 的 6 份各自经过网卡，计数 6', '直接发送时，发往同一台机器的多份数据各自经过网卡。'],
    [6.5, '转发：每台机器只过一次网卡，到同编号的 GPU 5，再沿 NVLink 分发；计数 6 → 2', '同一 token 的多份数据内容相同，只需经过一次网卡，到达后由 NVLink 分发。'],
    [14.0, '两条管道的宽度对照；下方实测带宽与 24 / 132 个 SM', 'NVLink 的带宽为网卡的三倍以上，机器内的额外分发几乎不增加耗时。'],
    [21.0, '名字出现：normal kernel，用于训练与 prefill；实测带宽', '这种先跨机、再在机器内分发的 kernel 称为 normal kernel，用于训练与 prefill。'],
  ] },
  { id: 'notify', tag: '转发的代价', head: '接收方需要{r:先获知数量}，CPU 因此等待', src: 'DeepEP figures/normal.png；H800 实测：DeepSeek profile-data', dur: 27, beats: [
    [0.5, 'DeepEP 原图：CPU 与 GPU 两条泳道', '转发之前，接收方须先获知将收到的 token 数量。'],
    [6.0, '红框圈出 CPU 的等待与分配', 'CPU 需要等待该数量到达，再据此分配显存。'],
    [11.0, 'prefill 一层约 34 ms，一次等待只是细缝', 'prefill 一层耗时几十毫秒，相比之下该等待可以忽略。'],
    [16.5, 'decode 一步约 3600 个 kernel 由 CUDA graph 一次提交；插入等待后被切碎', 'decode 依靠 CUDA graph 一次提交几千个 kernel。CPU 若需要等待，CUDA graph 无法使用。'],
  ] },
  { id: 'lowlat', tag: 'decode', head: '直接发送，接收区{t:大小固定}', src: 'DeepEP V1：low-latency kernel；H800 + CX7 实测，128 token/卡', dur: 28, beats: [
    [0.5, '集群扩成 16 台机器 × 8 张卡（EP128），每卡 2 个 expert', 'decode 时每张卡只有一百多个 token，部署扩大到 128 张卡。'],
    [6.5, '同一个 token 的 8 份经网卡直达目标卡，不经转发；8 份落在 4 台机器上（DeepSeek-V3 限制每个 token 最多 4 台）', '每一份数据不经转发，直接经网卡发往 expert 所在的卡。'],
    [12.0, '目标卡的接收区：每个 expert 预留 128 × 128 个位置；旁边是真实的调用', '接收区按最坏情况预留，无需先获知数量，每一步的形状都相同。'],
    [19.0, '收到的 token 填进去：约 3%', '代价是显存，预留的位置大部分为空。'],
    [23.0, '同一刻度：通信 192 + 369 µs，对一个 micro-batch 这一层的计算 0.56 ms', '即便如此，通信耗时仍与这一层的计算相当。'],
  ] },
  { id: 'overlap1', tag: '重叠 · prefill', head: 'batch 拆成两半：一半{b:通信}，另一半{y:计算}', src: 'H800 实测：DeepSeek profile-data，prefill，EP32，一层', dur: 22, beats: [
    [0.5, '真实 trace 两条 stream：计算（108 个 SM）、通信（24 个 SM）', '一个 batch 拆成两个 micro-batch，交错执行。'],
    [4.5, '两个 micro-batch 着色：一个在通信时，另一个在计算', '一个 micro-batch 通信时，另一个正在计算。'],
    [10.0, '24 / 108 的标注', '通信 kernel 同样占用 SM，此处分配了 24 个。'],
    [16.0, '两条 stream 的忙碌比例', '两条 stream 大部分时间都处于忙碌状态。'],
  ] },
  { id: 'overlap2', tag: '重叠 · decode', head: '发送后立即返回，SM {t:全部}用于计算', src: 'H800 实测：DeepSeek profile-data，decode，EP128，一层', dur: 27, beats: [
    [0.5, '真实 trace 一条 stream，132 个 SM 全部用于计算', 'decode 的计算时间短，再分出 SM，收益低于代价。'],
    [5.5, '极窄的发送块', '发送的 kernel 只负责发出请求，随即退出。'],
    [10.0, '虚线“在网络上”跨过另一半的 FlashMLA', '数据由网卡在后台传输，SM 此时计算另一半的 attention。'],
    [16.5, '接收块；stream 忙碌 97%', 'attention 计算完成后，一个很短的 kernel 接收数据。'],
    [22.5, '框架里的名字；batch 太小时 −27%', 'batch 过小时，拆成两半会降低吞吐。'],
  ] },
  { id: 'eplb', tag: '负载不均', head: '一步的耗时取决于{r:最忙}的那张卡', src: '左：SGLang 实测热图（DeepSeek-V3）；右：示意；SGLang，96 × H100', dur: 28, beats: [
    [0.5, 'SGLang 实测的 expert 分布热图', '少数 expert 被选中的频率更高；热图中颜色越深，收到的 token 越多。'],
    [7.0, '32 张卡的负载柱，红线标出最高的一根', '每一层所有卡同步执行，一步的耗时取决于负载最重的那张卡。'],
    [13.5, '最热的 expert 复制到冗余格，重新装卡，柱子变平', 'EPLB 统计负载，为最热的 expert 增加副本，再重新分配到各卡。'],
    [21.0, '冗余格亮起；SGLang 实测的提升', '每张卡上空置的第九个位置，即为副本预留。'],
  ] },
  { id: 'now', tag: '2026 年', head: '两种 kernel {b:合二为一}，链路决定{r:上限}', src: 'DeepEP README；SGLang GB200 博客；DeepSeek ISCA ’25（理论估算）', dur: 26, beats: [
    [0.5, '时间线：2025-02 DeepEP 开源，两种 kernel', '以上是 2025 年 DeepSeek 线上使用的做法。'],
    [6.0, '2026-04 V2：合并为同一套 dispatch / combine；2026-09 V2.5 删除 V1', '2026 年的新版本把两种 kernel 合并为同一套接口，占用的 SM 也更少。'],
    [13.5, '三条管道按真实宽度：网卡 50、H800 NVLink 160、GB200 NVL72 900 GB/s', '硬件同样在变化，GB200 NVL72 将 72 张卡置于同一个 NVLink 域。'],
    [20.0, 'DeepSeek 的估算：decode 每 token 至少 14.76 ms 对 0.82 ms', '链路的带宽决定了 decode 速度的上限。'],
  ] },
  { id: 'pp', tag: '按层切分', head: 'PP：每段之间只传输{t:一次}', src: 'vLLM get_pp_indices；hidden state 与 residual 按 BF16', dur: 22, beats: [
    [0.5, '同样 4 台机器，每台放连续的层：1–16、17–31、32–46、47–61', '另一种方式是按层切分，每台机器存放连续的若干层。'],
    [6.0, 'token 依次经过 4 段，每过一个边界传一次 28 KB', 'token 在每个段边界只传输一次，且只发送给下一台机器。'],
    [12.5, '同一刻度：每 token 每步，PP 约 86 KB，EP 约 10 MB', '同一个 token，按层切分的传输量约为按 expert 切分的百分之一。'],
  ] },
  { id: 'bubble', tag: '代价', head: '流水线须{r:填满}，单个请求{r:不会加速}', src: 'vLLM vllm/config/vllm.py；示意：4 段，每段耗时相同', dur: 26, beats: [
    [0.5, '一个 batch：任一时刻只有一行在算，其余是气泡', '只有一个 batch 时，同一时刻只有一台机器在计算。'],
    [6.5, '4 个 batch 错开进入，四行填满；vLLM 源码注释', '推理没有反向传播，batch 之间互不依赖，四个 batch 即可填满流水线。'],
    [13.5, '跟住一个 token：仍要依次走完 4 段、3 次传输', '每个 token 仍须依次经过所有段。'],
    [18.5, '两行结论', 'PP 提高的是吞吐与容量，而不是单个请求的速度。'],
  ] },
  { id: 'where', tag: '推理里的 PP', head: 'PP 用在{t:慢速网络}与{t:超长 prompt}', src: 'vLLM 文档；SGLang PP 博客（H20，DeepSeek-V3.1，128K）；DeepSeek、SGLang 部署', dur: 25, beats: [
    [0.5, '第一栏：机器内 TP，机器之间 PP', '机器之间只有慢速网络时，按层切分的传输量最少。'],
    [7.0, '第二栏：128K 的 prompt 切成 chunk 依次流入 4 段；TTFT 48.5 s → 15.5 s', '超长的 prompt 可以切成多个 chunk，在各段之间依次流水，从而填满流水线。'],
    [15.5, '第三栏：大 MoE 的 decode 用 DP attention + EP', 'decode 每步每个请求只有一个 token，大规模 MoE 的 decode 使用 EP。'],
  ] },
  { id: 'end', tag: '结尾', head: '', src: '', dur: 16, hold: 0.3, beats: [
    [0.5, '回到集群图：EP 每个 MoE 层两次跨机，密集的蓝紫点', '按 expert 切分时，每个 MoE 层跨机两次，因此需要转发、重叠与均衡。'],
    [8.0, 'PP 每段一次，稀疏的大块；大标题回到画面', '按层切分时，每段之间只传输一次，代价是流水线必须填满。'],
  ] },
];
