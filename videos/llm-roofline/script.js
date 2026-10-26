// LLM Roofline 影片的文案与节拍，画面（scenes.js）、渲染（render.mjs）与配音（vo.py）共用。纯数据，不依赖页面。
// 每幕：id 供 scenes.js 引用、tag 标签、head 标题（{t:…} teal、{p:…} purple、{r:…} red、{b:…} blue、{y:…} 深黄、{g:…} gray 为强调色）、src 出处、dur 时长（秒）、beats 节拍。
// 节拍：[幕内起点（秒）, 画面动作, 配音]。配音只讲“为什么”和“看哪里”，不念画面上的字和数字；配音为空的节拍只有画面。
// hold：关键帧距幕尾的秒数（开场取在标题缩上去之前）。数字与出处见 docs/llm-roofline-notes.md。
const SCRIPT = [
  { id: 'open', tag: '开场', head: '', src: '', dur: 12, hold: 2.6, beats: [
    [0.6, '对话窗口按实测速度逐 token 流出回答；右侧是 H100 的示意', '一块 H100，运行 Qwen3-8B，只服务一个请求。'],
    [5.0, '时间轴出现，第一个刻度落在 6.84 ms（实测）', '这个速度可以在运行之前估算出来。'],
    [9.6, '大标题缩到左上角，成为各幕的页眉', ''],
  ] },
  { id: 'work', tag: '一个 token 的工作量', head: '每个权重{b:读 2 字节}，只做 {t:2 次运算}', src: 'Qwen3-8B：36 层，hidden 4096，32 个 query head、8 个 KV head，FFN 12288，共 8.19 B 参数', dur: 18, beats: [
    [0.5, '36 层堆叠；一个 token 的向量从第 1 层走到第 36 层', '生成一个 token 须依次经过全部 36 层。'],
    [5.0, '放大其中一层：7 个权重矩阵按真实面积排开，向量逐个扫过', '每个权重从显存读出，只参与一次乘加。'],
    [11.5, '两个计数器走完 36 层：15.1 GB 与 15.1 GFLOP，二者之比为 1', '因此读出的字节数与运算次数几乎相等。'],
  ] },
  { id: 'speeds', tag: '两种速度', head: '读取权重需 {b:4.52 ms}，计算仅需 {t:0.015 ms}', src: 'NVIDIA H100 SXM 规格（dense BF16）；实测：vLLM 0.30，单请求', dur: 18, beats: [
    [0.5, 'H100 示意：中间的计算单元 989 TFLOPS，两侧 HBM 3.35 TB/s', 'GPU 有两个速度上限：算力与显存带宽。'],
    [6.0, '时间轴上长出两条：读取 4.52 ms；计算 0.015 ms，在放大框里才看得见', '在同一时间轴上，计算时间几乎不可见。'],
    [10.5, '下限刻度落在 4.52 ms', '每一步至少需要读完全部权重。'],
    [13.5, '实测刻度 6.84 ms，与下限之间的差标红', '实测多出的时间来自 kernel 效率与每一步的调度开销。'],
  ] },
  { id: 'roofline', tag: 'arithmetic intensity', head: 'H100 要每字节 {t:295 次运算}，decode 只有 {r:1 次}', src: 'Williams、Waterman、Patterson，Roofline，CACM 2009；NVIDIA H100 SXM 规格', dur: 20, beats: [
    [0.5, '公式：arithmetic intensity = 运算 ÷ 字节；decode 为 1', '运算次数与字节数之比称为 arithmetic intensity。'],
    [5.0, '989 ÷ 3.35 = 295', 'H100 每读取一个字节需做近三百次运算，才能用满算力。'],
    [10.0, '坐标系中先画斜线（带宽），再画平顶（算力），拐点标 295；右侧放 2009 年论文的原图', '这张图称为 roofline：拐点左侧受带宽限制，右侧受算力限制。'],
    [16.0, 'decode 的点落在最左端：3.35 TFLOPS，峰值的 0.3%', 'decode 落在最左端。'],
  ] },
  { id: 'batch', tag: 'batch', head: '8 个请求共用一次读取：每步用时{b:几乎不变}，吞吐为 {t:7.7 倍}', src: '实测：vLLM 0.30，并发 1 与 8，每个请求输入 128、输出 256 token', dur: 20, beats: [
    [0.5, '对话窗口从 1 个变成 8 个，同时流出文字', '八个请求同时计算，权重读取一次，产出八个 token。'],
    [6.5, '时间轴：读取不变，计算变为 8 倍仍很短；实测刻度从 6.84 移到 7.03 ms；吞吐 145 → 1118 tok/s', '每一步的用时几乎不变，吞吐接近原来的八倍。'],
    [13.0, 'roofline 上的点沿斜线从 1 移到 8；虚线标出到达拐点所需的 batch，约 295', 'batch 每增加一倍，点沿斜线上移一倍，直至拐点。'],
  ] },
  { id: 'prefill', tag: 'prefill', head: 'prefill 越过拐点，{t:受算力限制}', src: '估算：2 × 参数 × 2048 ÷ 989 TFLOPS；实测待补', dur: 16, beats: [
    [0.5, '一段 2048 token 的 prompt 一次送入模型', 'prefill 一次处理整段 prompt，每个权重被两千多个 token 共用。'],
    [5.5, '时间轴换刻度：计算 28.8 ms，读取 4.52 ms（估算）', 'prefill 的计算时间长于读取时间。'],
    [10.0, 'roofline 上 prefill 的点落在平顶上；两个点各标受什么限制', '两个阶段分处拐点两侧，由同一个比值决定。'],
  ] },
  { id: 'kv', tag: 'KV cache', head: 'batch 均摊了权重的读取，{p:KV cache 无法均摊}', src: '估算；Qwen3-8B 的 32 个 query head 共用 8 个 KV head（GQA）', dur: 20, beats: [
    [0.5, '每个请求旁长出自己的 KV cache，每 token 144 KiB', '权重由所有请求共用，KV cache 由各个请求独占。'],
    [6.5, 'batch 从 1 增到 64：矩阵乘法的点右移，attention 的点停在 4', '无论 batch 多大，attention 每读取一个字节只做几次运算。'],
    [13.0, '读取条拆成权重与 KV 两段：64 个请求 × 2048 token，KV 19.3 GB 超过权重 15.1 GB', '上下文变长后，每一步读取的主要部分由权重变为 KV cache。'],
  ] },
  { id: 'ledger', tag: '显存账', head: 'H100 的显存里，{p:55 GiB} 留给 KV cache', src: '实测：vLLM 0.30 启动日志，H100 80GB，gpu_memory_utilization 0.92', dur: 20, beats: [
    [0.5, '显存条：权重与非 torch 部分 16.17 GiB 填入，对应的日志行高亮', 'vLLM 启动时先加载权重。'],
    [4.0, 'activation 峰值 1.68 GiB、CUDA graph 0.45 GiB 依次填入', '随后 vLLM 按最大的一步试运行，测得 activation 的峰值。'],
    [9.0, 'KV cache 54.99 GiB 填满余下部分，末端 8% 预留', '其余显存全部分配给 KV cache。'],
    [13.5, 'KV 段换算为 400,448 token；按每个请求 2048 token，排成 195 格', '可同时服务的请求数由 KV cache 的容量决定。'],
  ] },
  { id: 'bound', tag: '一步的下限', head: '显存放满：每步至少 {b:22.1 ms}，仍受带宽限制', src: '估算：195 个请求，每个 2048 token；实测待补', dur: 18, beats: [
    [0.5, '公式：一步 ≥ max（字节 ÷ 带宽，运算 ÷ 算力）', '估算只需两个数：每一步读取的字节数与运算次数。'],
    [6.0, '代入 195 个请求 × 2048 token：读取 22.1 ms，其中四分之三是 KV；计算 3.0 ms', '显存放满后，计算时间仍然很短，下限由带宽决定。'],
    [12.0, '吞吐上限约 8800 tok/s，是 batch 1 的 40 倍；每个 token 的下限从 4.52 变为 22.1 ms', '吞吐提高，单个请求的速度则下降。'],
  ] },
  { id: 'levers', tag: '提高强度的手段', head: '增加运算或减少读取；{g:B200 的拐点几乎不变}', src: 'NVIDIA H100、B200 规格（dense）；DeepSeek，ISCA 2025；FlashMLA', dur: 20, beats: [
    [0.5, 'FP8 权重：字节减半，点右移一倍，平顶升到 1979 TFLOPS', '提高利用率，一是让每个字节多做运算，二是减少读取的字节。'],
    [6.0, 'MLA：每 token 的 KV 从 327 KB 降到 70 KB；attention 的点右移到拐点附近', 'MLA 让所有 query head 共用一份缓存。'],
    [11.0, 'speculative decoding：一次读取验证多个候选 token，点右移', 'speculative decoding 读取一次权重，验证多个候选 token。'],
    [15.0, '叠加 B200 的 roofline：平顶与斜线一起抬高，拐点 281', '新一代 GPU 的算力与带宽同步增长，拐点几乎不变。'],
  ] },
  { id: 'end', tag: '估算步骤', head: '{b:4.52 ms} 的下限，{r:6.84 ms} 的实测', src: '', dur: 10, hold: 0.05, beats: [
    [0.5, '回到开场：对话窗口与时间轴，刻度旁写出 15.1 GB ÷ 3.35 TB/s = 4.52 ms', '运行之前，先统计每步读取的字节数与运算次数。'],
    [5.0, '三行估算步骤', '速度受两者中较慢的一项限制。'],
  ] },
];
