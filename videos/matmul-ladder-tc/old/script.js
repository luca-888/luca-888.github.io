// 矩阵乘法阶梯 · 下集（Tensor Core）的文案与节拍，画面（scenes.js，两集共用）、配音（vo.py）与渲染（render.mjs）共用。不依赖页面。
// 结构：回顾 → 第二幕「单元自己到顶」（wmma 讲透，mma.sync / 三级缓冲 / wgmma 快剪）→ 第三幕「让搬运与计算同时跑」（warp 特化、L2 tile 顺序讲透，提前归还 / 常驻 / 集群 / TMA 写回快剪）→ 各规模与功耗墙 → 未完成的工作。
const EPISODE = 'tc';
// 节拍：[步（保留，非三栏幕为 0）, 画面动作, 配音, x]。三栏幕的 x = { v: 版本名, f: [起帧, 止帧], end: 点亮到第几级（本拍画整幕阶梯） }；
//   帧来自 frames.js（extract.mjs 从 1.3 讲解页交互版导出），代码行、数据格子与硬件状态按同一帧对齐。
// 节拍起点按配音时长自动排：录音后（vo.js 先于本文件载入）按实测时长 + 0.5 s，未录时按估计；每拍至少 2.6 s，配音为空的拍留白 2.2 s。
// 数字全部是 H100 SXM（700 W）实测，原样输出见 inference-study 仓库 labs/1.3-matmul-flops/out/。
const SPK = (x) => (x.match(/[一-鿿]/g) || []).length * 0.21 + (x.match(/[A-Za-z0-9][A-Za-z0-9._\-]*/g) || []).length * 0.32 + (x.match(/[，；：、]/g) || []).length * 0.15;
const MEAS = typeof VODUR === 'undefined' || !VODUR ? {} : VODUR;
function scene(tag, head, src, beats, { stage = false, min = stage ? 3.4 : 2.6, tail = 1.0 } = {}) {
  let t = 0.5; const out = [], stages = [], xs = [];
  for (const [st, act, vo, x = null] of beats) { out.push([+t.toFixed(1), act, vo]); stages.push(st); xs.push(x); t += vo ? Math.max(min, (MEAS[vo] !== undefined ? MEAS[vo] + 0.5 : SPK(vo) + 0.4)) : 2.2; }
  return { tag, head, src, stage, stages, xs, dur: +(t - 0.6 + tail).toFixed(1), beats: out };
}
const A2 = '单元自己到顶', A3 = '让搬运与计算同时跑';
const SCRIPT = [
  scene('回顾', '上集回顾：{b:FP32 的天花板}', 'H100 SXM 实测 · 8192 阶 · TFLOPS（对数刻度）', [
    [0, '上集的阶梯：FP32 段 7 级点亮，FP32 峰值 67 的虚线', '上一集的 FP32 版本达到 cuBLAS 的 86%，止于 FP32 单元的峰值。'],
    [0, '标出本集：Tensor Core，BF16 输入、FP32 累加；每 SM 每时钟 2048 对 128', '本集改用 Tensor Core，目标仍是追平同布局的 cuBLAS。'],
    [0, 'Tensor Core 段的灰色台阶依次闪过', ''],
  ], { tail: 1.4 }),
  scene(A2, '换用 {t:Tensor Core}：wmma', 'matmul_tc.cu：mm_wmma', [
    [0, '三栏：左 wmma 代码（改动行标 +），中栏先画每 SM 每时钟 128 对 2048 的方格，右 SM 剖面', 'Tensor Core 每个时钟完成的乘加是 FP32 单元的 16 倍。', { v: 'wmma' }],
    [0, '中栏换成 block 128 × 128、8 个 warp 的分工与 C = A·Wᵀ', 'wmma 即 warp matrix multiply-accumulate，由一个 warp 共同计算一块。'],
    [0, '代码中 fragment 与 load_matrix_sync 高亮；中栏标 BF16 输入、FP32 累加', '输入改为 BF16，累加仍使用 FP32；W 按行存放，与 Linear 层相同。'],
    [0, '底部阶梯点亮 wmma 93', '结果只有 Tensor Core 峰值的十分之一。'],
    [0, 'SM：共享内存标橙，Tensor Core 多数时间等待', '瓶颈又回到数据：读取片段时 bank 冲突，搬运还须经过寄存器。'],
  ], { stage: true }),
  scene(A2, '快剪：{t:mma.sync}、三级缓冲、wgmma', 'matmul_tc.cu：mm_mma / mm_wgmma', [
    [0, 'mma：数据路径与换位前后的共享内存', '按行号对块序号做异或，同一列分散到不同的 bank。', { v: 'mma' }],
    [0, 'cp.async 一段高亮；SM：L2→共享内存直接亮起', 'cp.async 即异步拷贝，数据不经寄存器，直接写入共享内存。'],
    [0, 'ldmatrix 与 mma.sync 高亮', 'ldmatrix 读取矩阵片段，再由 mma 即 matrix multiply-accumulate 计算。'],
    [0, 'mma_async：三个缓冲槽轮转，搬运与计算重叠', '三级缓冲让搬运与计算重叠，速度停在约 300。', { v: 'mma_async' }],
    [0, 'wgmma：一个线程发出 TMA，整块落入共享内存；mbarrier 计数', 'Hopper 的 TMA 即 Tensor Memory Accelerator，一个线程即可搬运整块数据。', { v: 'wgmma' }],
    [0, 'warpgroup 发出 wgmma，直接读取共享内存', 'wgmma 由四个 warp 共同发出，直接读取共享内存中的操作数。'],
    [0, '整幕阶梯：Tensor Core 段点亮到 wgmma 484', '第二幕到此结束：速度达到 484，每一块结束后搬运与计算仍相互等待。', { v: 'wgmma', end: 11 }],
  ], { stage: true, tail: 1.8 }),
  scene(A3, 'warp 特化：{g:生产者与消费者}', 'matmul_tc.cu：mm_wgmma → mm_wgmma_ws（GROUP = 0）', [
    [0, '先回到 wgmma：代码中 wgmma_wait<0> 与 __syncthreads 高亮；SM：调度器 0 标橙', '上一版每一块结束都要全体同步，TMA 与 Tensor Core 相互等待。', { v: 'wgmma' }],
    [0, '代码换成 wgmma_ws；中栏出现 4 个槽位的环形缓冲；SM：生产者蓝点、消费者绿圈', '一个 warpgroup 只负责搬运，另外两个只负责计算。', { v: 'wgmma_ws', f: [0, 2] }],
    [0, 'setmaxnreg 两行高亮：生产者寄存器降到 40，消费者升到 232；块放大到 128 × 256', '生产者只用一个线程，把寄存器让给消费者，消费者因此可以计算更大的块。', { f: [3, 4] }],
    [0, '帧推进：TMA 往一个槽写入（橙），消费者读取另一个槽（绿）', '共享内存分成 4 个槽位，生产者写入，消费者读取。', { f: [5, 7] }],
    [0, '代码中等待 full、empty 的两行高亮', 'full 与 empty 两组 barrier 分别通知数据到齐与槽位空出。', { f: [8, 9] }],
    [0, '底部阶梯：wgmma_ws 480（8192 阶），旁注 4096 阶 635', '4096 阶提高约三分之一，8192 阶却没有变化。', { f: [7, 7] }],
    [0, '标注：W 在 4096 阶为 32 MB，在 8192 阶为 128 MB；L2 为 50 MB', 'warp 特化只在大矩阵上失效，原因在于 L2 的容量。', { f: [7, 7] }],
  ], { stage: true }),
  scene(A3, 'L2 的意外：{g:只改 tile 的发放顺序}', 'matmul_tc.cu：wgmma_ws → wgmma_l2', [
    [0, '代码换成 wgmma_l2：tile_mn 中 GROUP 分支标 +；中栏第一帧：一波 CTA', '每个 SM 同时运行一个 CTA，132 个 tile 同时进行。', { v: 'wgmma_l2', f: [0, 0] }],
    [0, '前三帧按行发放：同一波覆盖 A 的几行与 W 的全部列', '按行发放时，同时运行的 tile 需要读取整个 W。', { f: [1, 1] }],
    [0, 'L2 占用条溢出（橙）；标注 W 128 MB、L2 50 MB', '8192 阶的 W 为 128 MB，L2 只有 50 MB。', { f: [2, 2] }],
    [0, 'SM：显存→L2 通路标橙并加粗', 'W 被反复从显存读回，显存到 L2 的通路饱和。', { f: [2, 2] }],
    [0, '后三帧：16 行一组、组内按列发放，L2 占用条全部为绿', 'tile 改为 16 行一组、组内按列发放。', { f: [3, 4] }],
    [0, '同一波只用到 A 的少数行与 W 的少数列', '同时读取的 A 行与 W 行都减少，L2 放得下。', { f: [5, 5] }],
    [0, '底部阶梯点亮 wgmma_l2 740', '代码只改了编号的换算，速度提高约一半。', { f: [5, 5] }],
  ], { stage: true }),
  scene(A3, '快剪：{g:提前归还、常驻、集群、TMA 写回}', 'matmul_tc.cu：wgmma_pipe / persist / cluster / tstore；开发阶段对照取自 ladder-dev6', [
    [0, 'pipe：wgmma_wait<1> 一行；两个消费者交替', '提前归还槽位，速度几乎不变：两个消费者已互相填补空档。', { v: 'wgmma_pipe' }],
    [0, 'persist：132 个 CTA 循环领取 tile；阶梯上这一级下降（657）', '常驻 kernel 单独加上，速度下降约一成，原因尚未查明。', { v: 'wgmma_persist' }],
    [0, '标注：同一轮开发测量，集群版非常驻 686、常驻 722', '常驻 kernel 予以保留，集群与 TMA 写回以其为前提。'],
    [0, 'cluster：两个 CTA 各读一半 W 并多播；L2 流量约 8.7 → 5.9 TB/s（推算）', '集群多播让 W 从 L2 读出的量减半。', { v: 'wgmma_cluster' }],
    [0, 'tstore：结果经共享内存由 TMA 整块写回，与下一块计算重叠', 'TMA 写回与下一块的计算重叠。', { v: 'wgmma_tstore' }],
    [0, '整幕阶梯：全部点亮，777.6 与 cuBLAS 虚线齐平', '第三幕到此结束，手写版本达到 cuBLAS 的 99.5%。', { v: 'wgmma_tstore', end: 17 }],
  ], { stage: true, tail: 1.8 }),
  scene('各规模', '各规模：{t:最终版} 与 cuBLAS BF16', 'matmul_tc 原样输出；连续 2 秒时由 NVML 采样频率与功耗', [
    [0, '256 至 8192 阶，两条柱对齐（对数刻度），上方标出最终版 ÷ cuBLAS', '最终版与 cuBLAS 在六个规模上逐一对比。'],
    [0, '512、1024 阶落后：1024 阶只有 32 个 tile，132 个 SM 中 100 个空闲；256 阶两者都约 0.013 ms', '小矩阵的 tile 过少，大部分 SM 空闲。'],
    [0, '2048 阶 +7.7%', '2048 阶时，手写版本快 7.7%。'],
    [0, '8192 阶短时 99.5%；终端原样两行：连续 2 秒 682.34 对 681.14，功耗 699 W 与 697 W', '连续运行时，两者都受 700 瓦功耗上限约束，速度相同。'],
  ]),
  scene('未完成', '未完成的工作与{g:已去除的瓶颈}', 'H100 SXM 实测 · 8192 阶 · TFLOPS（对数刻度）', [
    [0, '清单：Hilbert 曲线调度 tile；FP32 线剩下的 14%；小矩阵的分块与 split-K', '调度顺序、FP32 线剩余的 14% 与小矩阵分块仍可改进。'],
    [0, '回到开场的阶梯，逐级标出去除的瓶颈', '每一级的提升都来自去除上一级的瓶颈。'],
    [0, '色带：显存、片上存储、计算单元与同步、L2 与写回', '瓶颈依次出现在显存、片上存储、计算单元与 L2。'],
    [0, '两条 cuBLAS 虚线，两端的 5.4 与 777.6', '矩阵乘法的速度取决于数据能否及时送达计算单元。'],
  ], { tail: 2 }),
];
