// MoE Kernel 影片的文案与节拍，画面（scenes.js）、配音（vo.py）与渲染（render.mjs）共用。纯数据，不依赖页面。
// 每幕：tag 标签、head 标题（{t:…} {r:…} {p:…} {b:…} {y:…} 为强调色）、src 出处、dur 时长（秒）、beats 节拍。
// 节拍：[幕内起点（秒）, 画面动作, 配音]。配音只讲“为什么”和“看哪里”，不念画面上的字和数字；出处见 docs/moe-kernel-notes.md。
// 实测数字来自 data.js（H100，vLLM 0.31.0，第 24 层）。配音尚未合成：字幕按估计时长排，跑过 vo.py 生成 vo.js 后自动换成实测时长。
const SCRIPT = [
  { tag: '开场', head: '', src: 'Qwen3-30B-A3B；路由为 H100 实测', dur: 16, beats: [
    [0.6, '标题落位；128 个 expert 排成 8 × 16 的网格', 'Qwen3-30B-A3B 的每个 MoE 层有 128 个 expert。'],
    [5.4, '标题缩到左上角；一个 token 进来，网格里 8 格亮起（真实路由）', '每个 token 只激活 8 个 expert。'],
    [9.0, '16 个 token 依次进来，各自亮起的 8 格不同，网格上留下累计的颜色', '一批 token 激活的 expert 各不相同，GPU 需要一并完成计算。'],
  ] },
  { tag: '一个 expert', head: '一个 expert：{t:3 次矩阵乘法}', src: 'Qwen3-30B-A3B config：hidden 2048，moe_intermediate_size 768', dur: 18, beats: [
    [0.5, '一个 expert 展开：gate、up（2048 × 768）与 down（768 × 2048）三个权重按真实比例画出', '每个 expert 是一个小型前馈网络，包含三个权重矩阵。'],
    [6.2, '一个 token 的向量依次流过：乘 gate 与 up，逐元素相乘，再乘 down 回到 2048 维', '输入分别乘以 gate 和 up，两个结果逐元素相乘，再乘以 down 恢复原维度。'],
    [13.0, '大小：一个 expert 9.44 MB，一层 128 个共 1.21 GB；一个 token 只读 8 个', '一层权重为 1.2 GB，每个 token 只读取十六分之一。'],
  ] },
  { tag: 'router', head: 'router：{t:128 选 8}', src: 'H100 实测：第 24 层，请求 0 的 router 分数', dur: 16, beats: [
    [0.5, '128 根柱子：router 给每个 expert 的分数（真实数据）', 'router 先计算每个 expert 的分数。'],
    [4.6, '最高的 8 根留下并变色，其余变淡；8 个权重重新归一化，和为 1', '分数最高的 8 个被保留，权重重新归一化。'],
    [9.6, '8 个 expert 的输出按权重相加，得到这一层的输出', '8 个输出按权重相加，得到该层的结果。'],
  ] },
  { tag: '一批 token', head: '16 个 token，{t:65 个} expert', src: 'H100 实测：16 个请求的一步解码，第 24 层', dur: 18, beats: [
    [0.5, '16 行（每个请求一个 token），每行 8 个点落在 128 列上', '画面为一次实测的解码步，16 个请求各生成一个 token。'],
    [6.4, '按列统计：65 列有点，63 列是空的', '128 个 expert 中，该步激活了 65 个。'],
    [11.6, '每列的点数排成柱：51 个 expert 只分到 1 到 2 个 token', '大多数 expert 只分配到一至两个 token。'],
  ] },
  { tag: '逐个 expert', head: '逐个 expert 计算：{r:GPU 多数时间空闲}', src: 'transformers Qwen3MoeExperts.forward；H100 profiler 记录', dur: 19, beats: [
    [0.5, 'transformers 里逐个 expert 循环的源码原文', '最直接的实现是对每个 expert 循环一次。'],
    [6.2, 'profiler 时间线逐段展开：65 段，每段十几个小 kernel 和一次拷回 CPU', '每个 expert 需启动十几个小 kernel，并等待分配结果拷回 CPU。'],
    [13.4, '时间线上真正的矩阵乘法只是几条细线；一层 17.6 ms', 'GPU 大部分时间处于空闲。'],
  ] },
  { tag: '按 expert 排队', head: '先排序，{t:再统一计算}', src: 'vLLM moe_align_block_size；数据为 H100 实测的 16 个 token', dur: 20, beats: [
    [0.5, '128 个（token, expert）对展平成一列，按 token 的顺序', '先将 128 个选择展平为一列。'],
    [5.0, '按 expert 重新排序：同一个 expert 的行排在一起', '按 expert 排序后，同一 expert 的行彼此相邻。'],
    [10.0, '每段补齐到 16 行：65 段、1040 行，其中真实的 128 行', '每段补齐到 16 行，即一个 tile 的行数。'],
    [14.8, '得到两张表：sorted_token_ids（每行是谁）与 expert_ids（每块属于谁）', '两张表传给 kernel，记录每行的 token 与每块的 expert。'],
  ] },
  { tag: '一次 kernel', head: '一次启动，{t:所有 expert}', src: 'vLLM fused_moe.py fused_moe_kernel', dur: 20, beats: [
    [0.5, '输出被切成块，一块对应一个 program；其中一块高亮', '在一次 kernel 启动中，每个 program 计算输出的一块。'],
    [5.6, '这个 program 查 expert_ids 得到 expert 编号，取它的权重；按 sorted_token_ids 取出输入行', 'program 查表确定读取哪个 expert 的权重，再按下标读取对应的 token。'],
    [12.4, '补齐行被屏蔽：不读、不写', '补齐行被屏蔽，不读取也不写入。'],
    [15.6, '其余 64 块同时亮起，所有 expert 在同一次启动里完成', '所有 expert 在同一次启动中完成计算。'],
  ] },
  { tag: '整层', head: '整层 {t:7 个 kernel}', src: 'H100 profiler 记录；耗时为 50 次中位数', dur: 16, beats: [
    [0.5, '整层 7 个 kernel 按真实时长排开：两次矩阵乘法占 95%', '整层只有 7 个 kernel，耗时几乎全部来自两次矩阵乘法。'],
    [7.0, '与逐个循环放在同一时间轴：17.6 ms 对 0.40 ms', 'fused kernel 比逐个循环快四十多倍。'],
  ] },
  { tag: '瓶颈', head: '16 个 token：耗时集中于{r:读取权重}', src: 'H100 profiler 记录；显存带宽为 H100 规格', dur: 17, beats: [
    [0.5, '65 个 expert 的权重共 613 MB，从显存流进芯片', 'token 数为 16 时，耗时主要来自读取权重。'],
    [6.0, '两次矩阵乘法 212 µs 读完：约 2.9 TB/s，对比显存带宽 3.35 TB/s', '显存带宽接近用满，Tensor Core 利用率却很低。'],
    [11.6, '每个权重字节平均只乘了约 2 个 token', '每个权重只被使用一到两次。'],
  ] },
  { tag: 'batch 变大', head: 'token 越多，{t:权重复用率越高}', src: 'H100 实测：第 24 层；tile 为 vLLM 默认配置', dur: 20, beats: [
    [0.5, '横轴 token 数 1 到 4096：激活的 expert 数（实测）', 'token 越多，激活的 expert 越多，但增速随后迅速放缓。'],
    [7.0, '单层耗时（实测）：token 多 4096 倍，耗时约为 5 倍', '读入的权重由更多 token 共用，耗时增长缓慢。'],
    [13.6, '每个 expert 平均的行数，与 vLLM 选用的 tile 行数：16 → 128', 'tile 随之变大，kernel 由访存受限逐渐转为计算受限。'],
  ] },
  { tag: '各家做法', head: '共同思路：{t:按 expert 排序，一次完成计算}', src: 'vLLM、transformers、DeepGEMM、MegaBlocks 源码与论文', dur: 18, beats: [
    [0.5, '四种做法并排：补齐下标、精确偏移、分段对齐、block-sparse', '各实现细节不同，均将 token 按 expert 排序后一次计算。'],
    [7.4, 'B200 上 vLLM 默认换成 FlashInfer 的 TRT-LLM kernel，路由也融合进去（实测占位）', '在 B200 上，vLLM 默认改用另一套 kernel，路由计算也融合在内。'],
    [13.0, '回到开头的网格：16 个 token 的选择排成 65 块，一次算完', ''],
  ] },
];
