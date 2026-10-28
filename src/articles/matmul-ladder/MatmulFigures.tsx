// 矩阵乘法阶梯的自绘图。全文配色：FP32 路线 blue，Tensor Core 路线 teal，从显存读取 amber，
// 片上数据与正在使用的通路 green，瓶颈 red，选中的线程 blue、同一 warp 的线程浅蓝（blue 淡底）。
import type { ReactNode } from 'react'
import { Badge, Band, Cell, Figure, Legend, Link, useCompact, type Fill, type Hue } from '../../figure-kit'

// ---------------------------------------------------------------- 线程层级
export function HierarchyFigure() {
  const s = 4, ox = 30, oy = 40, n = 64
  const block = { bx: 2, by: 1 }, thread = { tx: 5, ty: 3 }
  const row = block.by * 16 + thread.ty, col = block.bx * 16 + thread.tx
  return <Figure title="grid、block、warp 与线程" desc="64 × 64 的 C 由 4 × 4 个 16 × 16 的 block 覆盖；选中 block (2, 1) 中的线程 (5, 3)，它属于 warp 1、lane 21，计算 C[19][37]。" height={340}
    caption="C 为 64 × 64 的缩小示意，每个 block 16 × 16 个线程、每个线程计算一个元素。浅蓝为与选中线程同一 warp 的 32 个线程，即 block 的第 2、3 行。">
    {() => <>
      <text x={ox + 128} y={20} className="fk-strong">C（64 × 64），grid = 4 × 4 个 block</text>
      <rect x={ox} y={oy} width={n * s} height={n * s} className="fk-fill-free" />
      <rect x={ox + block.bx * 64} y={oy + block.by * 64} width={64} height={64} className="fk-band-blue" />
      <rect x={ox + block.bx * 64} y={oy + block.by * 64 + 2 * s} width={64} height={2 * s} className="fk-fill-blue" opacity={0.3} />
      <rect x={ox + col * s} y={oy + row * s} width={s} height={s} className="fk-fill-blue" />
      {[1, 2, 3].map(i => <g key={i}>
        <line x1={ox + i * 64} y1={oy} x2={ox + i * 64} y2={oy + 256} className="fk-rule" />
        <line x1={ox} y1={oy + i * 64} x2={ox + 256} y2={oy + i * 64} className="fk-rule" />
      </g>)}
      {[0, 1, 2, 3].map(i => <text key={i} x={ox + i * 64 + 32} y={oy + 268} className="fk-muted">{i}</text>)}
      <text x={ox + 128} y={oy + 286} className="fk-muted">blockIdx.x</text>
      {[0, 1, 2, 3].map(i => <text key={`y${i}`} x={ox - 10} y={oy + i * 64 + 32} className="fk-muted">{i}</text>)}
      <text x={ox + block.bx * 64 + 32} y={oy + block.by * 64 + 40} className="fk-small fk-text-blue">block (2, 1)</text>
      <Link d={`M ${ox + col * s + 6} ${oy + row * s + 2} L 330 112`} hue="blue" />
      <g transform="translate(340, 60)">
        <text x={0} y={0} textAnchor="start" className="fk-strong">选中的线程</text>
        <text x={0} y={30} textAnchor="start" className="fk-mono">blockIdx = (2, 1)，threadIdx = (5, 3)</text>
        <text x={0} y={56} textAnchor="start" className="fk-mono">块内序号 = 3 × 16 + 5 = 53</text>
        <text x={0} y={82} textAnchor="start" className="fk-mono">warp = 53 / 32 = 1，lane = 53 mod 32 = 21</text>
        <text x={0} y={108} textAnchor="start" className="fk-mono">row = 1 × 16 + 3 = 19，col = 2 × 16 + 5 = 37</text>
        <line x1={0} y1={132} x2={400} y2={132} className="fk-rule" />
        <text x={0} y={156} textAnchor="start" className="fk-muted">block → 整体分配到一个 SM，共用共享内存与寄存器</text>
        <text x={0} y={180} textAnchor="start" className="fk-muted">warp → 32 个线程同时执行同一条指令</text>
        <text x={0} y={204} textAnchor="start" className="fk-muted">H100：132 个 SM，每个 SM 最多 2048 个线程（64 个 warp）</text>
      </g>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 合并访存
export function CoalesceFigure() {
  const sw = 34, ox = 150
  const strip = (y: number, used: (sector: number, slot: number) => boolean, count: number, gap: boolean) =>
    Array.from({ length: count }, (_, k) => {
      const x = ox + k * (sw + (gap ? 3 : 0))
      return <g key={k}>
        <rect x={x} y={y} width={sw - 2} height={22} rx={2} className="fk-fill-free" />
        {Array.from({ length: 8 }, (_, q) => used(k, q) && <rect key={q} x={x + 1 + q * 3.75} y={y + 1} width={3.4} height={20} className={k < 2 || !gap ? 'fk-fill-green' : 'fk-fill-amber'} />)}
      </g>
    })
  return <Figure title="一个 warp 的读取占用的扇区" desc="naive 读 B 的 16 个连续 float，占 2 个 32 字节扇区；naive_T 读 A 的 16 个地址相隔 N 个 float，占 16 个扇区，每个只用 4 字节。" height={150}
    caption="每格为一个 32 字节扇区（8 个 float），着色的竖条为 warp 实际使用的 float。naive 的 32 个线程分为两行，读同一组 16 个地址。">
    {() => <>
      <text x={10} y={32} textAnchor="start" className="fk-strong">naive</text>
      <text x={10} y={50} textAnchor="start" className="fk-muted">读 B[k][col]</text>
      {strip(30, (k) => k < 2, 4, false)}
      <text x={ox + 4 * sw + 14} y={41} textAnchor="start" className="fk-small">2 个扇区，64 B 全部使用</text>
      <text x={10} y={102} textAnchor="start" className="fk-strong">naive_T</text>
      <text x={10} y={120} textAnchor="start" className="fk-muted">读 A[row][k]</text>
      {strip(100, (_, q) => q === 0, 16, true)}
      <text x={ox} y={136} textAnchor="start" className="fk-small">16 个扇区，相邻两个相隔 N 个 float，每个只用 4 B</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 寄存器分块的外积
export function RegTileFigure() {
  const c = 24, ox = 300, oy = 50
  return <Figure title="寄存器分块的外积" desc="一个线程每一步读 A 的 8 个数与 B 的 8 个数，做 8 × 8 = 64 次 FMA。" height={290}
    caption="一个线程负责 C 的 8 × 8。沿 K 每走一步，rA 的一列与 rB 的一行做外积，累加到 64 个寄存器中：读 16 个数，做 64 次 FMA。">
    {() => <>
      {Array.from({ length: 8 }, (_, i) => <Cell key={`a${i}`} x={ox - 40} y={oy + i * c} w={c - 2} h={c - 2} fill="purple" label={i === 0 ? 'rA' : undefined} small />)}
      {Array.from({ length: 8 }, (_, j) => <Cell key={`b${j}`} x={ox + j * c} y={oy - 40} w={c - 2} h={c - 2} fill="purple" label={j === 0 ? 'rB' : undefined} small />)}
      {Array.from({ length: 64 }, (_, k) => <Cell key={k} x={ox + (k % 8) * c} y={oy + Math.floor(k / 8) * c} w={c - 2} h={c - 2} fill="blue" focus={k === 19} />)}
      <Link d={`M ${ox - 18} ${oy + 2 * c + 11} H ${ox + 3 * c - 2}`} hue="purple" />
      <Link d={`M ${ox + 3 * c + 11} ${oy - 18} V ${oy + 2 * c - 2}`} hue="purple" />
      <text x={ox + 8 * c + 20} y={oy + 60} textAnchor="start" className="fk-small">acc[2][3] += rA[2] × rB[3]</text>
      <text x={ox + 8 * c + 20} y={oy + 84} textAnchor="start" className="fk-muted">每步：读 8 + 8 = 16 个数</text>
      <text x={ox + 8 * c + 20} y={oy + 104} textAnchor="start" className="fk-muted">做 8 × 8 = 64 次 FMA</text>
      <text x={ox - 60} y={oy + 4 * c} textAnchor="end" className="fk-muted">来自 As</text>
      <text x={ox + 4 * c} y={oy - 52} className="fk-muted">来自 Bs</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- warp 分块与 bank
export function WarptileFigure() {
  const bw = 11, sx = 330
  const lanes = (y: number, items: [number, number, string][]) => items.map(([bank0, wave, label], i) =>
    <g key={i}><rect x={sx + bank0 * bw} y={y + wave * 26} width={4 * bw - 1} height={22} rx={2} className={wave > 0 ? 'fk-fill-red' : 'fk-fill-green'} />
      <text x={sx + bank0 * bw + 2 * bw} y={y + wave * 26 + 11} className="fk-cell-text fk-cell-small">{label}</text></g>)
  const ruler = (y: number) => <>
    {Array.from({ length: 33 }, (_, b) => <line key={b} x1={sx + b * bw} y1={y - 4} x2={sx + b * bw} y2={y} className="fk-rule" />)}
    {[0, 8, 16, 24].map(b => <text key={b} x={sx + b * bw + 2} y={y - 12} textAnchor="start" className="fk-muted">bank {b}</text>)}
  </>
  const p = 3, ox = 30, oy = 50
  return <Figure title="warp 分块与 bank 冲突" desc="左：warp 负责 64 × 32，32 个线程排成 8 × 4，lane 6 计算 4 个相隔的 4 × 4 小块。右：一组 8 个线程读 Bs 时 vec 需两轮，warptile 一轮完成。" height={260}
    caption="左为 warptile 中一个 warp 的输出块，浅蓝为 32 个线程各自的第一个 4 × 4 小块，深蓝为 lane 6 的 4 个小块。右为 LDS.128 一次服务的 8 个线程读 Bs 时占用的 bank：红色表示同一 bank 上的第二个地址，须第二轮读取。">
    {() => <>
      <text x={ox + 48} y={22} className="fk-strong">warptile：一个 warp 的 64 × 32</text>
      <rect x={ox} y={oy} width={32 * p} height={64 * p} className="fk-fill-free" />
      <rect x={ox} y={oy} width={16 * p} height={32 * p} className="fk-fill-blue" opacity={0.18} />
      {[0, 1].map(mi => [0, 1].map(ni => <rect key={`${mi}${ni}`} x={ox + (ni * 16 + 2 * 4) * p} y={oy + (mi * 32 + 1 * 4) * p} width={4 * p} height={4 * p} className="fk-fill-blue" />))}
      <text x={ox + 32 * p + 10} y={oy + 30} textAnchor="start" className="fk-small">lane 6</text>
      <text x={ox + 32 * p + 10} y={oy + 48} textAnchor="start" className="fk-muted">lRow 1，lCol 2</text>
      <text x={ox + 32 * p + 10} y={oy + 66} textAnchor="start" className="fk-muted">行相隔 32</text>
      <text x={ox + 32 * p + 10} y={oy + 84} textAnchor="start" className="fk-muted">列相隔 16</text>
      <text x={sx} y={24} textAnchor="start" className="fk-strong">vec：8 个线程读 rB，2 轮</text>
      {ruler(52)}
      {lanes(54, [[0, 0, 'L0'], [8, 0, 'L1'], [16, 0, 'L2'], [24, 0, 'L3'], [0, 1, 'L4'], [8, 1, 'L5'], [16, 1, 'L6'], [24, 1, 'L7']])}
      <text x={sx} y={150} textAnchor="start" className="fk-strong">warptile：8 个线程读 rB，1 轮</text>
      {ruler(178)}
      {lanes(180, [[0, 0, 'L0 L4'], [4, 0, 'L1 L5'], [8, 0, 'L2 L6'], [12, 0, 'L3 L7']])}
      <text x={sx + 16 * bw + 10} y={191} textAnchor="start" className="fk-muted">同一地址广播</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 时间线
type Seg = [number, number, Fill, string?]
type TRow = { label: string; segs: Seg[] }
type TGroup = { title: string; rows: TRow[] }
function TimelineFigure({ groups, title, desc, caption, legend }: { groups: TGroup[]; title: string; desc: string; caption: ReactNode; legend: [Fill, string][] }) {
  // 紧凑模式（520 宽）：时间轴缩短，行高与段内字号略增
  const compact = useCompact()
  const lx = compact ? 96 : 120, unit = compact ? 17.5 : 26, rh = compact ? 26 : 22, gapRow = 6
  let y = 10
  const placed = groups.map(g => { const top = y; y += 26 + g.rows.length * (rh + gapRow) + 10; return { g, top } })
  return <Figure title={title} desc={desc} width={compact ? 520 : 760} height={y} caption={caption} below={<Legend items={legend} />}>
    {() => placed.map(({ g, top }) => <g key={g.title}>
      <text x={10} y={top + 10} textAnchor="start" className="fk-strong">{g.title}</text>
      {g.rows.map((r, i) => {
        const ry = top + 26 + i * (rh + gapRow)
        return <g key={r.label}>
          <text x={lx - 10} y={ry + rh / 2} textAnchor="end" className="fk-muted">{r.label}</text>
          <line x1={lx} y1={ry + rh / 2} x2={lx + 24 * unit} y2={ry + rh / 2} className="fk-rule" />
          {r.segs.map(([a, b, fill, label], k) => <Cell key={k} x={lx + a * unit} y={ry} w={(b - a) * unit - 2} h={rh} fill={fill} label={label} small fs={compact ? 12 : undefined} />)}
        </g>
      })}
    </g>)}
  </Figure>
}

const rep = (n: number, f: (i: number) => Seg[]) => Array.from({ length: n }, (_, i) => f(i)).flat()

export function TimelineFp32() {
  return <TimelineFigure title="warptile 与双缓冲的时间线" desc="warptile 每块先读显存、等同步再计算；dbuf 计算第 t 块时已在读第 t+1 块。"
    groups={[
      { title: 'warptile', rows: [
        { label: '读显存', segs: rep(3, i => [[i * 8, i * 8 + 3, 'amber', `读 ${i}`]]) },
        { label: 'FP32 单元', segs: rep(3, i => [[i * 8 + 3, i * 8 + 3.6, 'gray'], [i * 8 + 3.6, i * 8 + 7.4, 'blue', `算 ${i}`], [i * 8 + 7.4, i * 8 + 8, 'gray']]) },
      ] },
      { title: 'dbuf', rows: [
        { label: '读显存', segs: [[0, 3, 'amber', '读 0'], ...rep(4, i => [[3 + i * 4.3, 6 + i * 4.3, 'amber', `读 ${i + 1}`]] as Seg[])] },
        { label: 'FP32 单元', segs: rep(5, i => [[3 + i * 4.3, 7 + i * 4.3, 'blue', `算 ${i}`], [7 + i * 4.3, 7.3 + i * 4.3, 'gray']]).filter(s => s[1] <= 24) },
      ] },
    ]}
    caption="示意，长度不按比例。灰色为在 __syncthreads() 处的等待。dbuf 中第 t+1 块的读取在计算第 t 块之前发出，结果在计算结束后写入另一份共享内存。"
    legend={[['amber', '从显存读取'], ['blue', 'FP32 计算'], ['gray', '同步等待']]} />
}

export function TimelineMma() {
  return <TimelineFigure title="单个 stage 与 pipeline" desc="STAGES = 1 时每块搬完才算；STAGES = 3 时拷贝提前两块发出，计算连续。"
    groups={[
      { title: 'mma（STAGES = 1）', rows: [
        { label: 'cp.async', segs: rep(3, i => [[i * 8, i * 8 + 3.5, 'amber', `搬 ${i}`]]) },
        { label: 'Tensor Core', segs: rep(3, i => [[i * 8 + 3.5, i * 8 + 8, 'teal', `算 ${i}`]]) },
      ] },
      { title: 'mma pipeline（STAGES = 3）', rows: [
        { label: 'cp.async', segs: [[0, 3.5, 'amber', '搬 0'], [0.4, 3.9, 'amber', '搬 1'], ...rep(4, i => [[4 + i * 4.5, 7.5 + i * 4.5, 'amber', `搬 ${i + 2}`]] as Seg[])] },
        { label: 'Tensor Core', segs: rep(5, i => [[3.9 + i * 4.5, 8.2 + i * 4.5, 'teal', `算 ${i}`]]).filter(s => s[1] <= 24.2) },
      ] },
    ]}
    caption=""
    legend={[['amber', '显存 → shared memory'], ['teal', 'Tensor Core 计算']]} />
}

export function TimelineWs() {
  return <TimelineFigure title="单个 warpgroup 与 warp specialization" desc="wgmma 版每块之后全体同步才发下一次 TMA；warp specialization 后 producer 连续搬运，两个 consumer 交替使用 Tensor Core。"
    groups={[
      { title: 'wgmma：一个 warpgroup', rows: [
        { label: 'TMA', segs: [[0, 2, 'amber', 'TMA'], ...rep(3, i => [[6 + i * 6, 8 + i * 6, 'amber', 'TMA']] as Seg[])] },
        { label: 'Tensor Core', segs: rep(4, i => [[2 + i * 6, 5.4 + i * 6, 'teal', `块 ${i}`], [5.4 + i * 6, 6 + i * 6, 'gray']]).filter(s => s[1] <= 24) },
      ] },
      { title: 'wgmma_ws：一个 producer、两个 consumer', rows: [
        { label: 'producer', segs: rep(10, i => [[i * 2.2, i * 2.2 + 2, 'amber', 'TMA']]).filter(s => s[1] <= 24) },
        { label: 'consumer 1', segs: rep(5, i => [[2 + i * 4.4, 4.2 + i * 4.4, 'teal', 'wgmma']]).filter(s => s[1] <= 24) },
        { label: 'consumer 2', segs: rep(5, i => [[4.2 + i * 4.4, 6.4 + i * 4.4, 'teal', 'wgmma']]).filter(s => s[1] <= 24) },
      ] },
    ]}
    caption=""
    legend={[['amber', 'TMA 搬运'], ['teal', 'wgmma 计算'], ['gray', '等待与同步']]} />
}

export function TimelineTstore() {
  return <TimelineFigure title="零散写回与 TMA store" desc="cluster 版每个 tile 结束时由 consumer 逐个线程写回显存，期间不发 wgmma；TMA store 由一个线程发出后与下一块计算重叠。"
    groups={[
      { title: 'wgmma_cluster：每个线程直接写回', rows: [
        { label: 'consumer', segs: rep(3, i => [[i * 8, i * 8 + 6, 'teal', `tile ${i}`], [i * 8 + 6, i * 8 + 8, 'red', '写回']]) },
        { label: '写回显存', segs: rep(3, i => [[i * 8 + 6, i * 8 + 8, 'amber']]) },
      ] },
      { title: 'wgmma_tstore：TMA store', rows: [
        { label: 'consumer', segs: rep(4, i => [[i * 6.4, i * 6.4 + 6, 'teal', `tile ${i}`], [i * 6.4 + 6, i * 6.4 + 6.4, 'green']]).filter(s => s[1] <= 24.4) },
        { label: 'TMA store', segs: rep(3, i => [[6.4 + i * 6.4, 9.4 + i * 6.4, 'amber', `C ${i}`]]) },
      ] },
    ]}
    caption="绿色为写入 shared memory 的停顿。"
    legend={[['teal', 'wgmma 计算'], ['red', 'consumer 在写回'], ['green', '写入 shared memory'], ['amber', '写回显存']]} />
}

// ---------------------------------------------------------------- 异或换位
export function SwizzleFigure() {
  const c = 26, top = 56
  const grid = (ox: number, swz: boolean) => Array.from({ length: 64 }, (_, k) => {
    const r = Math.floor(k / 8), phys = k % 8, logical = swz ? phys ^ r : phys
    const hit = logical === 0
    return <Cell key={k} x={ox + phys * c} y={top + r * c} w={c - 2} h={c - 2} fill={hit ? (swz ? 'teal' : 'red') : 'gray'} label={logical} small />
  })
  const head = (ox: number, t: string, sub: string) => <>
    <text x={ox + 4 * c} y={14} className="fk-strong">{t}</text>
    <text x={ox + 4 * c} y={32} className="fk-muted">{sub}</text>
    {Array.from({ length: 8 }, (_, i) => <text key={i} x={ox + i * c + c / 2 - 1} y={top + 8 * c + 12} className="fk-muted">{i}</text>)}
    <text x={ox + 4 * c} y={top + 8 * c + 32} className="fk-muted">物理位置（bank 组）</text>
  </>
  return <Figure title="128 字节行的异或换位" desc="一行 8 个 16 字节块。换位前 ldmatrix 读取的 8 个第 0 块都在 bank 组 0；第 r 行第 c 块存到 c ⊕ r 后分散到 8 组。" height={330}
    caption="每格为 16 字节（8 个 BF16，占 4 个 bank），格内数字为逻辑块号，行为矩阵的 8 行。ldmatrix 读取 8 行的同一逻辑块：换位前全部落在 bank 组 0，8 路冲突；换位后落在 8 个不同的组。">
    {() => <>
      {head(80, '换位前', '第 c 块存在位置 c')}
      {head(450, '换位后', '第 c 块存在位置 c ⊕ (r mod 8)')}
      {grid(80, false)}
      {grid(450, true)}
      {Array.from({ length: 8 }, (_, r) => <text key={r} x={66} y={top + r * c + c / 2 - 1} textAnchor="end" className="fk-muted">r = {r}</text>)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- tile 发放顺序
export function TileOrderFigure() {
  const cw = 6, ch = 3, numM = 64, numN = 32, wave = 132
  const panel = (ox: number, grouped: boolean, title: string, sub: string) => {
    const on = new Set<number>()
    for (let id = 0; id < wave; id++) {
      let m: number, n: number
      if (!grouped) { m = Math.floor(id / numN); n = id % numN }
      else { const r = id % (16 * numN); m = Math.floor(id / (16 * numN)) * 16 + r % 16; n = Math.floor(r / 16) }
      on.add(m * numN + n)
    }
    const cols = new Set([...on].map(k => k % numN)), rows = new Set([...on].map(k => Math.floor(k / numN)))
    const oy = 50
    return <g>
      <text x={ox + numN * cw / 2} y={14} className="fk-strong">{title}</text>
      <text x={ox + numN * cw / 2} y={32} className="fk-muted">{sub}</text>
      <rect x={ox} y={oy} width={numN * cw} height={numM * ch} className="fk-fill-free" />
      {[...on].map(k => <rect key={k} x={ox + (k % numN) * cw} y={oy + Math.floor(k / numN) * ch} width={cw - 0.6} height={ch - 0.6} className="fk-fill-teal" />)}
      {[...cols].map(n => <rect key={n} x={ox + n * cw} y={oy + numM * ch + 8} width={cw - 0.6} height={8} className="fk-fill-amber" />)}
      <text x={ox + numN * cw / 2} y={oy + numM * ch + 30} className="fk-small">用到的 W：{cols.size} / 32 列，{grouped ? '33' : '128'} MB</text>
      {[...rows].map(m => <rect key={m} x={ox - 16} y={oy + m * ch} width={8} height={ch - 0.6} className="fk-fill-amber" />)}
      <text x={ox + numN * cw / 2} y={oy + numM * ch + 48} className="fk-small">用到的 A：{rows.size} / 64 行，{grouped ? '32' : '8'} MB</text>
    </g>
  }
  return <Figure title="tile 的发放顺序" desc="8192 阶共 64 × 32 个 tile。按行发放时一轮 132 个 tile 覆盖约 4 行、全部 32 列；16 行一组时覆盖 16 行、约 8 列。" height={310}
    caption="8192 阶，tile 为 128 × 256，共 64 行 × 32 列。青色为同时运行的一轮 132 个 tile，橙色为它们需要的 A 行与 W 行。">
    {() => <>
      {panel(150, false, '按行发放', 'm = id / 32，n = id mod 32')}
      {panel(470, true, '16 行一组，组内按列', 'GROUP = 16')}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 集群多播
export function ClusterFigure() {
  return <Figure title="2 个 block 的集群与 TMA 多播" desc="上下相邻的两个 tile 使用同一块 W；rank 0 搬 W 的前 128 行、rank 1 搬后 128 行，各自多播给两个 block。" height={250}
    caption="每一步的 W 块为 256 × 64。两个 block 各从 L2 读一半并多播，每个 block 的共享内存中仍是完整的 W 块；A 块各读各的。">
    {arrow => <>
      <text x={110} y={18} className="fk-strong">L2</text>
      <Cell x={40} y={40} w={140} h={36} fill="amber" label="W[0:128]" />
      <Cell x={40} y={84} w={140} h={36} fill="amber" label="W[128:256]" />
      <Cell x={40} y={150} w={66} h={30} fill="gray" label="A₀" />
      <Cell x={114} y={150} w={66} h={30} fill="gray" label="A₁" />
      <rect x={400} y={26} width={330} height={92} rx={10} className="fk-band-teal" />
      <text x={414} y={44} textAnchor="start" className="fk-strong fk-text-teal">block rank 0（tile 上半）</text>
      <Cell x={414} y={58} w={72} h={44} fill="green" label="A₀" />
      <Cell x={500} y={58} w={100} h={20} fill="green" label="W 前半" small />
      <Cell x={500} y={82} w={100} h={20} fill="green" label="W 后半" small />
      <rect x={400} y={136} width={330} height={92} rx={10} className="fk-band-teal" />
      <text x={414} y={154} textAnchor="start" className="fk-strong fk-text-teal">block rank 1（tile 下半）</text>
      <Cell x={414} y={168} w={72} h={44} fill="green" label="A₁" />
      <Cell x={500} y={168} w={100} h={20} fill="green" label="W 前半" small />
      <Cell x={500} y={192} w={100} h={20} fill="green" label="W 后半" small />
      <Link d="M 180 58 L 340 58 L 340 68 L 500 68" hue="blue" arrow={arrow} />
      <Link d="M 340 68 L 340 178 L 500 178" hue="blue" arrow={arrow} />
      <Link d="M 180 102 L 320 102 L 320 92 L 500 92" hue="purple" arrow={arrow} />
      <Link d="M 320 102 L 320 202 L 500 202" hue="purple" arrow={arrow} />
      <text x={258} y={48} className="fk-small fk-text-blue">rank 0 发出，掩码 0b11</text>
      <text x={250} y={122} className="fk-small fk-text-purple">rank 1 发出，掩码 0b11</text>
      <text x={660} y={80} className="fk-muted">empty：</text>
      <text x={660} y={98} className="fk-muted">等 16 个</text>
      <text x={660} y={116} className="fk-muted">消费者 warp</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- SM 剖面
type Edge = 'hbm-l2' | 'l2-lsu' | 'lsu-reg' | 'lsu-smem' | 'smem-reg' | 'reg-fp32' | 'reg-tc' | 'smem-tc'
type Node = 'hbm' | 'l2' | 'lsu' | 'smem' | 'reg' | 'fp32' | 'tc'
type Warp = 'run' | 'mem' | 'sync' | 'prod' | 'cons'
type HwState = { title: string; on: Edge[]; hot?: (Edge | Node)[]; warps: [Warp, number][]; note: string; lsu?: string; s2r?: string }

const PATHS: Record<Edge, string> = {
  'hbm-l2': 'M 100 112 H 132',
  'l2-lsu': 'M 212 112 H 254',
  'lsu-reg': 'M 294 87 V 64 H 545 V 87',
  'lsu-smem': 'M 334 112 H 372',
  'smem-reg': 'M 462 112 H 500',
  'reg-fp32': 'M 590 104 L 640 74',
  'reg-tc': 'M 590 120 L 640 150',
  'smem-tc': 'M 417 137 V 186 H 690 V 170',
}
const NODES: Record<Node, [number, number, number, number, string]> = {
  hbm: [20, 87, 80, 50, 'HBM'],
  l2: [132, 87, 80, 50, 'L2 50 MB'],
  lsu: [254, 87, 80, 50, 'LSU · TMA'],
  smem: [372, 87, 90, 50, '共享内存'],
  reg: [500, 87, 90, 50, '寄存器'],
  fp32: [640, 54, 106, 40, 'FP32 单元'],
  tc: [640, 130, 106, 40, 'Tensor Core'],
}
const WARP_FILL: Record<Warp, Fill> = { run: 'green', mem: 'amber', sync: 'gray', prod: 'blue', cons: 'teal' }

export const HW: Record<string, HwState> = {
  naive: { title: 'naive_T', on: ['hbm-l2', 'l2-lsu', 'lsu-reg', 'reg-fp32'], hot: ['lsu', 'lsu-reg'], warps: [['mem', 14], ['run', 2]], lsu: '1 次读取 → 16 次访存',
    note: 'LSU 把一个 warp 的读取拆成 16 次访存，warp 大多在等显存；naive 读取连续，瓶颈是每次 FMA 都经 L1、L2 读两个数。' },
  smem: { title: 'smem T = 32', on: ['hbm-l2', 'l2-lsu', 'lsu-reg', 'lsu-smem', 'smem-reg', 'reg-fp32'], hot: ['smem-reg'], warps: [['run', 8], ['sync', 4], ['mem', 4]],
    note: '经 L2 的读取降为 1/32；内层循环每次 FMA 仍读两次共享内存，共享内存到寄存器的通路成为瓶颈。' },
  dbuf: { title: 'dbuf', on: ['hbm-l2', 'l2-lsu', 'lsu-reg', 'lsu-smem', 'smem-reg', 'reg-fp32'], hot: ['fp32'], warps: [['run', 14], ['sync', 2]],
    note: '显存读取与 FMA 同时进行；FP32 单元的峰值成为这一路线的上限。' },
  wgmma: { title: 'wgmma', on: ['hbm-l2', 'l2-lsu', 'lsu-smem', 'smem-tc'], hot: ['smem-tc'], warps: [['run', 2], ['sync', 2]], lsu: 'TMA',
    note: 'TMA 整块搬运，wgmma 直接读共享内存；每块结束时全体同步，搬运与计算相互等待。' },
  ws: { title: 'wgmma_ws（8192 阶）', on: ['hbm-l2', 'l2-lsu', 'lsu-smem', 'smem-tc'], hot: ['hbm-l2'], warps: [['prod', 4], ['cons', 8]], lsu: 'TMA',
    note: '生产者与两个消费者同时工作；8192 阶时 W 放不下 L2，显存到 L2 的通路成为瓶颈。' },
  l2: { title: 'wgmma_l2', on: ['hbm-l2', 'l2-lsu', 'lsu-smem', 'smem-tc'], hot: [], warps: [['prod', 4], ['cons', 8]], lsu: 'TMA',
    note: '同一轮 tile 需要的数据减半，L2 能留住共用的 W；剩下的空档在槽位归还与 tile 之间的写回。' },
}

export function HwProfile({ v }: { v: string }) {
  const st = HW[v]
  const hot = new Set<string>(st.hot ?? [])
  const dots = st.warps.flatMap(([w, n]) => Array.from({ length: n }, () => w))
  return <Figure title={`SM 剖面：${st.title}`} desc={st.note} height={232}
    caption={<><b>{st.title}</b>：{st.note}<br />示意。绿色为这一版主要使用的通路，红色为瓶颈；底部圆点为 warp 的状态。</>}
    below={<Legend items={[['green', '执行'], ['amber', '等显存'], ['gray', '等同步'], ['blue', '生产者'], ['teal', '消费者'], ['red', '瓶颈']]} />}>
    {() => <>
      <rect x={236} y={14} width={524} height={204} rx={12} className="fk-band-gray" />
      <text x={250} y={32} textAnchor="start" className="fk-strong">SM</text>
      {(Object.keys(PATHS) as Edge[]).map(e => {
        const active = st.on.includes(e), isHot = hot.has(e)
        return <path key={e} d={PATHS[e]} fill="none" strokeWidth={isHot ? 5 : active ? 3 : 1.2}
          stroke={isHot ? 'var(--fk-red)' : active ? 'var(--fk-green)' : 'var(--color-border)'} strokeLinejoin="round" />
      })}
      {(Object.keys(NODES) as Node[]).map(k => {
        const [x, y, w, h, label] = NODES[k]
        const dim = (k === 'fp32' && !st.on.includes('reg-fp32')) || (k === 'tc' && !st.on.includes('reg-tc') && !st.on.includes('smem-tc'))
        const fill: Fill = hot.has(k) ? 'red' : dim ? 'void' : k === 'fp32' ? 'blue' : k === 'tc' ? 'teal' : k === 'hbm' || k === 'l2' ? 'amber' : 'green'
        return <Cell key={k} x={x} y={y} w={w} h={h} fill={fill} label={label} sub={k === 'lsu' ? st.lsu : undefined} />
      })}
      {st.s2r && <text x={481} y={128} className="fk-small">{st.s2r}</text>}
      <text x={250} y={204} textAnchor="start" className="fk-muted">warp</text>
      {dots.map((w, i) => <circle key={i} cx={296 + i * 16} cy={204} r={5.5} className={`fk-fill-${WARP_FILL[w]}`} />)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 共享内存的 bank（重绘，替代视频静帧 bank.png）
// 左：32 列 bank，第 r 行第 b 列 = 字节地址 128r + 4b；一个 warp 的 32 个线程读第 3 行，各落一列。
// 右：bank 0 一列放大，两个线程读同一 bank 的不同地址，分两轮。
export function BankFigure() {
  // 紧凑模式（520 宽）：左块略窄，右块（冲突）移到下方
  const compact = useCompact()
  const gx = 78, cw = compact ? 13.5 : 14, top = 52, rh = 26, rows = 4, RD = 3, dotY = 200
  const cx = 640, cwide = 48, row2 = top + 2 * rh + rh / 2, row3 = top + 3 * rh + rh / 2
  return <Figure title="共享内存的 32 个 bank" desc="地址每 4 字节换下一个 bank，128 字节后回到 bank 0。左：一个 warp 的 32 个线程读同一行，32 个地址落在 32 个 bank。右：两个线程读 bank 0 的地址 256 与 384，分两轮完成。" width={compact ? 520 : 760} height={compact ? 456 : 226}
    caption={`每格 4 字节，左侧为每行的字节地址。${compact ? '上' : '左'}：warp 的 32 个线程读第 3 行，各占一个 bank。${compact ? '下' : '右'}：bank 0 上的两个不同地址。`}>
    {arrow => <>
      <text x={gx} y={14} textAnchor="start" className="fk-strong">无冲突</text>
      <text x={70} y={38} textAnchor="end" className="fk-muted">bank</text>
      {[0, 8, 16, 24, 31].map(b => <text key={b} x={gx + b * cw + cw / 2} y={38} className="fk-muted">{b}</text>)}
      {Array.from({ length: rows }, (_, r) => <text key={r} x={70} y={top + r * rh + rh / 2} textAnchor="end" className="fk-mono">{128 * r}–{128 * r + 124}</text>)}
      {Array.from({ length: rows * 32 }, (_, i) => {
        const r = Math.floor(i / 32), b = i % 32
        return <rect key={i} x={gx + b * cw + 1} y={top + r * rh + 1} width={cw - 2} height={rh - 2} rx={2} className={r === RD ? 'fk-fill-blue' : 'fk-fill-gray'} />
      })}
      {Array.from({ length: 32 }, (_, i) => {
        const x = gx + i * cw + cw / 2
        return <g key={i}>
          <line x1={x} y1={dotY - 5} x2={x} y2={top + rows * rh} className="fk-line-blue" strokeWidth={1.2} />
          <circle cx={x} cy={dotY} r={3.5} className="fk-fill-blue" />
        </g>
      })}
      <text x={70} y={dotY} textAnchor="end" className="fk-muted">warp</text>
      <text x={gx + cw / 2} y={dotY + 17} className="fk-muted fk-small">0</text>
      <text x={gx + 31 * cw + cw / 2} y={dotY + 17} className="fk-muted fk-small">31</text>

      {compact ? <line x1={10} y1={234} x2={510} y2={234} className="fk-rule" /> : <line x1={566} y1={4} x2={566} y2={220} className="fk-rule" />}
      <g transform={compact ? 'translate(-390, 236)' : undefined}>
      <text x={590} y={14} textAnchor="start" className="fk-strong">冲突</text>
      <text x={cx + cwide / 2} y={38} className="fk-muted">bank 0</text>
      {Array.from({ length: rows }, (_, r) => <Cell key={r} x={cx} y={top + r * rh + 1} w={cwide} h={rh - 2} rx={2} fill={r === 2 ? 'blue' : r === 3 ? 'red' : 'gray'} label={128 * r} small />)}
      <Link d={`M 618 ${dotY - 5} V ${row2} H ${cx}`} hue="blue" arrow={arrow} />
      <Link d={`M 710 ${dotY - 5} V ${row3} H ${cx + cwide}`} hue="blue" arrow={arrow} />
      <circle cx={618} cy={dotY} r={3.5} className="fk-fill-blue" />
      <circle cx={710} cy={dotY} r={3.5} className="fk-fill-blue" />
      <text x={618} y={dotY + 17} className="fk-muted fk-small">lane 0</text>
      <text x={710} y={dotY + 17} className="fk-muted fk-small">lane 16</text>
      <text x={610} y={row2} textAnchor="end" className="fk-small fk-text-blue">第 1 轮</text>
      <text x={718} y={row3} textAnchor="start" className="fk-small" style={{ fill: 'var(--fk-red)' }}>第 2 轮</text>
      </g>
    </>}
  </Figure>
}


// ================================================================ 重绘的静帧（2026-10-22）：图内只放标签，推导与数字留在正文

// ---------------------------------------------------------------- tile 发放顺序（替代 ws-l2.png、tile-order.png）
// 8192 阶的 C 共 64 行 × 32 列个 tile，只画上 24 行（下方截断）。tile 的深浅为开始时间（第几轮，每轮 132 个），
// 描边框出共用第 2 列 W 的 tile：按行发放时它们分属不同轮，16 行一组时同组的 16 个同时开始。
export function TileOrderFigure2() {
  const numN = 32, rows = 24, wave = 132, cw = 10, ch = 5, FOCUS = 2, maxW = 7
  const tileId = (m: number, n: number, grouped: boolean) => grouped ? Math.floor(m / 16) * 16 * numN + n * 16 + m % 16 : m * numN + n
  const op = (t: number) => 1 - t * 0.1
  const panel = (ox: number, grouped: boolean, title: string) => {
    const W = numN * cw, H = rows * ch, top = 56, ax = ox - 22, wy = top - 22, zig = top + H + 7
    const zz = (x0: number, x1: number) => `M ${x0} ${zig} ` + Array.from({ length: Math.round((x1 - x0) / 8) }, (_, i) => `L ${x0 + (i + 0.5) * 8} ${zig + (i % 2 ? -3 : 3)}`).join(' ') + ` L ${x1} ${zig}`
    return <g>
      <text x={ox + W / 2} y={14} className="fk-strong">{title}</text>
      <rect x={ax} y={top} width={14} height={H} className="fk-fill-gray" />
      <rect x={ox} y={wy} width={W} height={14} className="fk-fill-gray" />
      <rect x={ox + FOCUS * cw} y={wy} width={cw} height={14} className="fk-fill-amber" />
      <rect x={ox} y={top} width={W} height={H} className="fk-fill-free" />
      {Array.from({ length: rows * numN }, (_, k) => {
        const m = Math.floor(k / numN), n = k % numN, w = tileId(m, n, grouped) / wave
        return <rect key={k} x={ox + n * cw + 0.5} y={top + m * ch + 0.5} width={cw - 1} height={ch - 1} className="fk-fill-teal" opacity={op(w)} />
      })}
      {grouped && <line x1={ox - 26} y1={top + 16 * ch} x2={ox + W + 4} y2={top + 16 * ch} className="fk-rule" strokeDasharray="3 3" style={{ stroke: 'var(--color-muted)' }} />}
      <rect x={ox + FOCUS * cw - 1.5} y={wy - 1.5} width={cw + 3} height={top + H - wy + 3} rx={2} className="fk-focus" strokeWidth={1.2} />
      <path d={zz(ax - 2, ox + W + 2)} className="fk-link fk-line-muted" strokeWidth={1} />
      <text x={ax + 7} y={zig + 16} className="fk-muted">A</text>
      <text x={ox + W + 8} y={wy + 7} textAnchor="start" className="fk-muted">W</text>
      <text x={ox + W / 2} y={zig + 16} className="fk-muted">C</text>
      {grouped && <text x={ox + W + 8} y={top + 8 * ch} textAnchor="start" className="fk-muted fk-small">16 行</text>}
    </g>
  }
  const ly = 228
  return <Figure title="tile 的发放顺序与开始时间" desc="8192 阶 C 的上 24 行 tile，深浅表示开始时间。按行发放时同一列的 tile 逐行错开；16 行一组、组内按列时同一列的 16 个 tile 同时开始。" height={244}
    caption="只画 C 的上 24 行，A、W 沿 K 压缩显示；颜色越深开始越早，描边为共用同一块 W 的 tile。">
    {() => <>
      {panel(80, false, '按行发放')}
      {panel(440, true, '16 行一组，组内按列')}
      <text x={300} y={ly} textAnchor="end" className="fk-muted">开始时间：早</text>
      {Array.from({ length: maxW + 1 }, (_, w) => <rect key={w} x={308 + w * 18} y={ly - 6} width={16} height={12} rx={2} className="fk-fill-teal" opacity={op(w)} />)}
      <text x={308 + (maxW + 1) * 18 + 4} y={ly} textAnchor="start" className="fk-muted">晚</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 外积之和（替代 matrix.png）
export function OuterFigure() {
  // 紧凑模式（约 520 宽）：C = k0 + k1 一行，+ k2 + k3 换到第二行，与 k0、k1 对齐。
  const compact = useCompact()
  const s = compact ? 18 : 16, top = 62, ox = compact ? 76 : 90, step = compact ? 150 : 136, row2 = compact ? 132 : 0
  const pos = (k: number) => compact ? { x: ox + 120 + (k % 2) * step, y: top + Math.floor(k / 2) * row2 } : { x: ox + 120 + k * step, y: top }
  const grid = (x: number, y: number, cls: string, opacity = 1) => Array.from({ length: 16 }, (_, i) =>
    <rect key={i} x={x + (i % 4) * s + 1} y={y + Math.floor(i / 4) * s + 1} width={s - 2} height={s - 2} rx={2} className={cls} opacity={opacity} />)
  return <Figure title="C = AB 写成外积之和" desc="4 × 4 的 C 等于 4 个外积矩阵之和：A 的第 k 列乘 B 的第 k 行得到一个 4 × 4 矩阵。" width={compact ? 520 : 760} height={compact ? 166 + row2 : 166}
    caption="以 4 × 4 为例：第 k 层外积由 A 的第 k 列与 B 的第 k 行相乘，4 层相加得到 C。">
    {() => <>
      {grid(ox, top, 'fk-fill-purple')}
      <text x={ox + 2 * s} y={top + 4 * s + 20} className="fk-muted">C</text>
      <text x={ox + 4 * s + 28} y={top + 2 * s} className="fk-strong">=</text>
      {[0, 1, 2, 3].map(k => {
        const { x, y } = pos(k), gx = x + s + 6
        return <g key={k}>
          {k > 0 && <text x={x - 24} y={y + 2 * s} className="fk-strong">+</text>}
          {[0, 1, 2, 3].map(i => <rect key={`a${i}`} x={x + 1} y={y + i * s + 1} width={s - 2} height={s - 2} rx={2} className="fk-fill-blue" />)}
          {[0, 1, 2, 3].map(j => <rect key={`b${j}`} x={gx + j * s + 1} y={y - s - 6 + 1} width={s - 2} height={s - 2} rx={2} className="fk-fill-amber" />)}
          {grid(gx, y, 'fk-fill-purple', 0.4)}
          <text x={x + s / 2} y={y + 4 * s + 20} className="fk-muted fk-small">A</text>
          <text x={x + s / 2} y={y - s / 2 - 6} className="fk-muted fk-small">B</text>
          <text x={gx + 2 * s} y={y + 4 * s + 20} className="fk-muted">k = {k}</text>
        </g>
      })}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- grid、block 与 warp（替代 grid.png、block-warp.png 与 HierarchyFigure）
export function HierarchyFigure2() {
  const gx = 70, gy = 70, b = 48, zx = 420, zy = 36, t = 14
  const bx = 2, by = 1, tx = 5, ty = 3
  return <Figure title="grid、block、warp 与线程" desc="64 × 64 的 C 由 4 × 4 个 block 覆盖；放大 block (2, 1)：16 × 16 个线程，每两行为一个 warp，选中线程 (5, 3) 为 warp 1 的 lane 21。" height={300}
    caption="左为 grid：4 × 4 个 block，各负责 C 的一块。右为放大的 block (2, 1)：16 × 16 个线程按行编号，连续 32 个即两行为一个 warp。">
    {() => <>
      <text x={gx + 2 * b} y={gy - 30} className="fk-strong">grid</text>
      {Array.from({ length: 16 }, (_, i) => {
        const x = i % 4, y = Math.floor(i / 4), on = x === bx && y === by
        return <rect key={i} x={gx + x * b + 1} y={gy + y * b + 1} width={b - 2} height={b - 2} rx={3} className={on ? 'fk-fill-blue' : 'fk-fill-gray'} opacity={on ? 0.35 : 1} />
      })}
      {[0, 1, 2, 3].map(i => <text key={i} x={gx + i * b + b / 2} y={gy - 10} className="fk-muted">{i}</text>)}
      {[0, 1, 2, 3].map(i => <text key={`y${i}`} x={gx - 12} y={gy + i * b + b / 2} className="fk-muted">{i}</text>)}
      <text x={gx + 2 * b} y={gy + 4 * b + 18} className="fk-muted">blockIdx.x</text>
      <text x={gx - 30} y={gy + 2 * b} className="fk-muted" transform={`rotate(-90 ${gx - 30} ${gy + 2 * b})`}>blockIdx.y</text>
      <line x1={gx + (bx + 1) * b} y1={gy + by * b} x2={zx} y2={zy} className="fk-rule" />
      <line x1={gx + (bx + 1) * b} y1={gy + (by + 1) * b} x2={zx} y2={zy + 16 * t} className="fk-rule" />
      {Array.from({ length: 256 }, (_, i) => {
        const x = i % 16, y = Math.floor(i / 16), warp1 = y === 2 || y === 3, sel = x === tx && y === ty
        return <rect key={i} x={zx + x * t + 0.5} y={zy + y * t + 0.5} width={t - 1} height={t - 1} rx={1.5}
          className={sel ? 'fk-fill-blue' : warp1 ? 'fk-fill-blue' : 'fk-fill-gray'} opacity={sel ? 1 : warp1 ? 0.35 : 1} />
      })}
      <rect x={zx + tx * t - 1} y={zy + ty * t - 1} width={t + 2} height={t + 2} rx={2} className="fk-focus" strokeWidth={1.5} />
      {Array.from({ length: 8 }, (_, w) => <g key={w}>
        <line x1={zx + 16 * t + 6} y1={zy + w * 2 * t + 2} x2={zx + 16 * t + 6} y2={zy + (w + 1) * 2 * t - 2} className="fk-rule" style={{ stroke: w === 1 ? 'var(--fk-blue)' : undefined }} strokeWidth={2} />
        <text x={zx + 16 * t + 14} y={zy + w * 2 * t + t} textAnchor="start" className={`fk-small ${w === 1 ? 'fk-text-blue' : 'fk-muted'}`}>warp {w}</text>
      </g>)}
      <text x={zx + tx * t + t / 2} y={zy + 16 * t + 14} className="fk-small fk-text-blue">lane 21</text>
      <line x1={zx + tx * t + t / 2} y1={zy + (ty + 1) * t + 1} x2={zx + tx * t + t / 2} y2={zy + 16 * t + 6} className="fk-line-blue" strokeWidth={1} />
      <text x={zx + 8 * t} y={zy - 14} className="fk-strong">block (2, 1)</text>
      <text x={zx + 12 * t} y={zy + 16 * t + 30} className="fk-muted">threadIdx.x</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 合并访存（替代 naive.png 与 CoalesceFigure）
export function CoalesceFigure2() {
  const sw = 38, ox = 120, fw = (sw - 2) / 8
  const row = (y: number, n: number, gap: number, used: (k: number, q: number) => boolean, hue: string) => Array.from({ length: n }, (_, k) => {
    const x = ox + k * (sw + gap)
    return <g key={k}>
      <rect x={x} y={y} width={sw - 2} height={24} rx={2} className="fk-fill-free" />
      {Array.from({ length: 8 }, (_, q) => used(k, q) && <rect key={q} x={x + q * fw + 0.6} y={y + 1} width={fw - 1.2} height={22} className={hue} />)}
    </g>
  })
  return <Figure title="一个 warp 的读取占用的扇区" desc="naive 读 B 的连续 16 个 float，占 2 个 32 字节扇区；naive_T 读 A 的 16 个地址相隔 N 个 float，各占一个扇区。" height={146}
    caption="每格为一个 32 字节扇区（8 个 float），着色为 warp 实际用到的 float。">
    {() => <>
      <text x={ox - 14} y={34} textAnchor="end" className="fk-strong">naive</text>
      <text x={ox - 14} y={52} textAnchor="end" className="fk-muted fk-mono">B[k][col]</text>
      {row(30, 4, 0, k => k < 2, 'fk-fill-green')}
      <text x={ox - 14} y={104} textAnchor="end" className="fk-strong">naive_T</text>
      <text x={ox - 14} y={122} textAnchor="end" className="fk-muted fk-mono">A[row][k]</text>
      {row(100, 16, 2.5, (_, q) => q === 0, 'fk-fill-amber')}
      <path d={`M ${ox + 2} 132 V 136 H ${ox + sw + 2.5 + 2} V 132`} className="fk-link fk-line-muted" strokeWidth={1} />
      <text x={ox + 20 + sw / 2} y={142} className="fk-muted fk-small" textAnchor="start" dx={14}>相隔 N 个 float</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 共享内存分块（替代 smem.png）
export function SmemTileFigure() {
  const T = 56, ax = 160, ay = 266, cx = ax + 4 * T + 20, by = ay - 4 * T - 6, K0 = 2
  return <Figure title="共享内存分块" desc="block 负责 C 的一个 T × T 块；沿 K 每一步把 A 带与 B 带的第 k0 块读入共享内存的 As、Bs。" height={350}
    caption="block 负责 C 的一个 T × T 块，需要 A 的 T 行与 B 的 T 列。沿 K 每一步读入一对 T × T 小块到 As、Bs（图为第 2 步）。">
    {arrow => <>
      <Band x={ax} y={by + 20} w={220} h={150} hue="gray" title="共享内存" />
      <Cell x={ax + 30} y={by + 70} w={T} h={T} fill="green" label="As" />
      <Cell x={ax + 140} y={by + 70} w={T} h={T} fill="green" label="Bs" />
      {[0, 1, 2, 3].map(k => <Cell key={`a${k}`} x={ax + k * T + 1} y={ay + 1} w={T - 2} h={T - 2} rx={2} fill={k === K0 ? 'amber' : 'gray'} />)}
      {[0, 1, 2, 3].map(k => <Cell key={`b${k}`} x={cx + 1} y={by + k * T + 1} w={T - 2} h={T - 2} rx={2} fill={k === K0 ? 'amber' : 'gray'} />)}
      <Cell x={cx + 1} y={ay + 1} w={T - 2} h={T - 2} rx={2} fill="blue" label="C 块" />
      {[0, 1, 2, 3].map(k => <text key={`ka${k}`} x={ax + k * T + T / 2} y={ay + T + 14} className="fk-muted fk-small">{k}</text>)}
      {[0, 1, 2, 3].map(k => <text key={`kb${k}`} x={cx + T + 12} y={by + k * T + T / 2} className="fk-muted fk-small">{k}</text>)}
      <text x={ax - 12} y={ay + T / 2} textAnchor="end" className="fk-muted">A</text>
      <text x={cx + T / 2} y={by - 10} className="fk-muted">B</text>
      <text x={ax + 2 * T} y={ay + T + 32} className="fk-muted">k0</text>
      <text x={cx + T + 30} y={by + 2 * T} className="fk-muted">k0</text>
      <Link d={`M ${ax + K0 * T + T / 2} ${ay} V ${ay - 20} H ${ax + 30 + T / 2} V ${by + 70 + T}`} hue="blue" arrow={arrow} />
      <Link d={`M ${cx} ${by + K0 * T + T / 2} H ${cx - 10} V ${by + 70 + T / 2} H ${ax + 140 + T}`} hue="blue" arrow={arrow} />
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 1/8：共享内存供数与 FP32 单元（替代 eighth.png）
export function EighthFigure() {
  // 紧凑模式（520 宽）：格宽 14，FP32 单元的点略小
  const compact = useCompact()
  const gx = compact ? 72 : 100, cw = compact ? 14 : 20, top = 30, uy = 108, ur = compact ? 20 : 22
  return <Figure title="每周期的共享内存读取与 FP32 单元" desc="32 个 bank 每周期送出 32 个 float，两两一组供一次 FMA，128 个 FP32 单元中只有 16 个在工作。" width={compact ? 520 : 760} height={compact ? 270 : 292}
    caption="每周期 32 个 bank 各送出一个 float，As、Bs 各一个凑成一次 FMA；一个 SM 的 128 个 FP32 单元中只有一行在工作。">
    {() => <>
      <text x={gx - 12} y={top + 10} textAnchor="end" className="fk-muted">bank</text>
      {Array.from({ length: 32 }, (_, i) => <rect key={i} x={gx + i * cw + 1} y={top} width={cw - 2} height={20} rx={2} className="fk-fill-green" opacity={i % 2 ? 0.6 : 1} />)}
      <text x={gx + cw / 2} y={top + 10} className="fk-cell-text" style={{ fontSize: 9 }}>As</text>
      <text x={gx + cw * 1.5} y={top + 10} className="fk-cell-text" style={{ fontSize: 9 }}>Bs</text>
      {Array.from({ length: 16 }, (_, p) => {
        const x = gx + p * 2 * cw + cw
        return <path key={p} d={`M ${x - cw / 2} ${top + 22} V ${top + 30} H ${x + cw / 2} V ${top + 22} M ${x} ${top + 30} V ${uy - 9}`} className="fk-link fk-line-blue" strokeWidth={1.2} />
      })}
      <text x={gx - 12} y={uy + 3.5 * ur} textAnchor="end" className="fk-muted">FP32 单元</text>
      {Array.from({ length: 128 }, (_, i) => <circle key={i} cx={gx + (i % 16) * 2 * cw + cw} cy={uy + Math.floor(i / 16) * ur} r={compact ? 5.5 : 7} className={i < 16 ? 'fk-fill-blue' : 'fk-fill-gray'} />)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 寄存器分块的外积（替代 RegTileFigure 的带句子版本；reg-issue.png 并入正文）
export function RegTileFigure2() {
  const c = 24, ox = 330, oy = 70
  return <Figure title="寄存器分块的外积" desc="一个线程每一步读 rA 的 8 个数与 rB 的 8 个数，做外积累加到 8 × 8 个寄存器。" height={266}
    caption="一个线程负责 C 的 8 × 8。沿 K 每走一步，rA 与 rB 做外积，累加到 64 个寄存器；描边为 acc[2][3] += rA[2] × rB[3]。">
    {() => <>
      {Array.from({ length: 8 }, (_, i) => <Cell key={`a${i}`} x={ox - 40} y={oy + i * c} w={c - 2} h={c - 2} rx={2} fill="purple" focus={i === 2} />)}
      {Array.from({ length: 8 }, (_, j) => <Cell key={`b${j}`} x={ox + j * c} y={oy - 40} w={c - 2} h={c - 2} rx={2} fill="purple" focus={j === 3} />)}
      {Array.from({ length: 64 }, (_, k) => <Cell key={k} x={ox + (k % 8) * c} y={oy + Math.floor(k / 8) * c} w={c - 2} h={c - 2} rx={2} fill="blue" focus={k === 19} />)}
      <text x={ox - 56} y={oy + 4 * c} textAnchor="end" className="fk-muted">rA ← As</text>
      <text x={ox + 4 * c} y={oy - 52} className="fk-muted">rB ← Bs</text>
      <text x={ox + 8 * c + 14} y={oy + 4 * c} textAnchor="start" className="fk-muted">acc（8 × 8）</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 每 SM 每时钟的乘加与一条指令的工作量（构图同视频第 1 幕）
export function TcRateFigure() {
  // 左：一个完整 SM 的缩略（按 H100 白皮书的 SM 图，部件画成色块）；右：放大处理块 0，单元画成格子、乘加画成点，两种点距相同
  const SX = 14, SY = 30, BW = 104, BH = 120, BG = 8, BX0 = SX + 10, BY0 = SY + 24
  const chip = (x: number, y: number, w: number, h: number, cls: string, op = 1) => <rect x={x} y={y} width={w} height={h} rx={1.5} className={cls} opacity={op} />
  const block = (k: number) => {
    const x = BX0 + (k % 2) * (BW + BG), y = BY0 + Math.floor(k / 2) * (BH + BG), iw = BW - 12
    return <g key={k}>
      <rect x={x} y={y} width={BW} height={BH} rx={4} style={{ fill: PAGE }} stroke={k === 0 ? 'var(--fk-teal)' : 'var(--color-border)'} strokeWidth={k === 0 ? 2 : 1} />
      {chip(x + 6, y + 6, iw, 7, 'fk-fill-gray', 0.6)}
      {chip(x + 6, y + 16, iw, 9, 'fk-fill-gray')}
      {chip(x + 6, y + 28, iw, 9, 'fk-fill-purple', 0.3)}
      {chip(x + 6, y + 41, iw, 7, 'fk-fill-int')}
      {chip(x + 6, y + 51, iw, 14, 'fk-fill-blue')}
      {chip(x + 6, y + 68, iw, 7, 'fk-fill-blue', 0.38)}
      {chip(x + 6, y + 79, iw, 22, 'fk-fill-teal')}
      {chip(x + 6, y + 105, iw * 0.55, 8, 'fk-fill-amber')}
      {chip(x + 8 + iw * 0.55, y + 105, iw * 0.45 - 2, 8, 'fk-fill-purple', 0.6)}
    </g>
  }
  const sw = 2 * BW + BG + 20, by = BY0 + 2 * BH + BG + 6, sh = by + 34 - SY
  // 右侧放大
  const ZX = 300, ZY = 30, ZW = 446, ZH = 270, d = 6
  const ux = ZX + 150
  const dot = (cx: number, cy: number, cls: string) => <circle cx={cx} cy={cy} r={2} className={cls} />
  const label = (y: number, t: string, sub: string, cls: string) => <g>
    <text x={ZX + 14} y={y} textAnchor="start" className={`fk-strong fk-small ${cls}`}>{t}</text>
    <text x={ZX + 14} y={y + 17} textAnchor="start" className="fk-muted fk-small">{sub}</text>
  </g>
  return <Figure title="一个 SM 与它的一个处理块" desc="左：H100 一个 SM 的结构，4 个处理块。右：放大一个处理块：32 个 FP32 单元每个每时钟 1 次乘加；1 个 Tensor Core 每时钟 512 次乘加。整个 SM 每时钟 FP32 共 128 次，Tensor Core 共 2048 次。" height={SY + sh + 40}
    caption="左按 H100 白皮书的 SM 图缩略；右为一个处理块，格子为单元，点为每时钟的一次乘加。">
    {arrow => <>
      <text x={SX} y={12} textAnchor="start" className="fk-strong">一个 SM（H100 共 132 个）</text>
      <rect x={SX} y={SY} width={sw} height={sh} rx={8} className="fk-fill-teal" opacity={0.06} />
      <rect x={SX} y={SY} width={sw} height={sh} rx={8} fill="none" stroke="var(--fk-teal)" strokeWidth={1.4} />
      {chip(BX0, SY + 8, 2 * BW + BG, 9, 'fk-fill-gray', 0.6)}
      {[0, 1, 2, 3].map(block)}
      {chip(BX0, by, 2 * BW + BG, 9, 'fk-fill-gray')}
      {chip(BX0, by + 12, 2 * BW + BG, 12, 'fk-fill-green', 0.3)}
      <text x={BX0 + BW + BG / 2} y={by + 18} style={{ fill: '#2f6b4a', fontSize: 10, fontWeight: 600 }}>L1 / shared memory 256 KB</text>
      {/* 放大引线 */}
      <line x1={BX0 + BW} y1={BY0} x2={ZX} y2={ZY} stroke="var(--fk-teal)" strokeDasharray="4 3" />
      <line x1={BX0 + BW} y1={BY0 + BH} x2={ZX} y2={ZY + ZH} stroke="var(--fk-teal)" strokeDasharray="4 3" />
      <rect x={ZX} y={ZY} width={ZW} height={ZH} rx={8} style={{ fill: PAGE }} stroke="var(--fk-teal)" strokeWidth={2} />
      <text x={ZX + 14} y={ZY + 18} textAnchor="start" className="fk-strong">处理块 0（共 4 个）</text>
      {[['L0 指令缓存', 0.6], ['warp 调度器', 1], ['分发单元', 0.6]].map(([t, o], i) => <g key={t as string}>
        <rect x={ZX + 14 + i * 104} y={ZY + 32} width={98} height={16} rx={2} className="fk-fill-gray" opacity={o as number} />
        <text x={ZX + 14 + i * 104 + 49} y={ZY + 40} style={{ fill: '#4b5563', fontSize: 10.5 }}>{t}</text>
      </g>)}
      <rect x={ZX + 326} y={ZY + 32} width={106} height={16} rx={2} className="fk-fill-purple" opacity={0.25} />
      <text x={ZX + 379} y={ZY + 40} className="fk-text-purple" style={{ fontSize: 10.5 }}>寄存器 64 KB</text>
      {/* FP32：32 个单元，每格一个点 */}
      {label(ZY + 72, 'FP32 单元 × 32', '每个每时钟 1 次', 'fk-text-blue')}
      {Array.from({ length: 32 }, (_, i) => <g key={i}>
        <rect x={ux + i * d} y={ZY + 66} width={d - 0.8} height={d - 0.8} className="fk-fill-blue" opacity={0.25} />
        {dot(ux + i * d + d / 2 - 0.4, ZY + 66 + d / 2 - 0.4, 'fk-fill-blue')}
      </g>)}
      <text x={ux + 32 * d + 10} y={ZY + 69} textAnchor="start" className="fk-strong fk-text-blue">32 次</text>
      {/* Tensor Core：1 个单元，框内 512 个点 */}
      {label(ZY + 118, 'Tensor Core × 1', '每时钟 512 次', 'fk-text-teal')}
      <rect x={ux - 4} y={ZY + 104} width={32 * d + 7} height={16 * d + 7} rx={4} className="fk-fill-teal" opacity={0.16} />
      <rect x={ux - 4} y={ZY + 104} width={32 * d + 7} height={16 * d + 7} rx={4} fill="none" stroke="var(--fk-teal)" strokeWidth={1.4} />
      {Array.from({ length: 512 }, (_, i) => <g key={i}>{dot(ux + (i % 32) * d + d / 2 - 0.4, ZY + 107 + Math.floor(i / 32) * d + d / 2, 'fk-fill-teal')}</g>)}
      <text x={ux + 32 * d + 10} y={ZY + 156} textAnchor="start" className="fk-strong fk-text-teal">512 次</text>
      <text x={ux + 32 * d + 10} y={ZY + 174} textAnchor="start" className="fk-muted fk-small">16 倍</text>
      {/* 其余单元 */}
      <text x={ZX + 14} y={ZY + 228} textAnchor="start" className="fk-muted fk-small">另有 INT32 × 16、FP64 × 16、LD/ST × 8、SFU × 4</text>
      <text x={ZX + 14} y={ZY + ZH - 18} textAnchor="start" className="fk-strong">整个 SM 每时钟：FP32 <tspan className="fk-text-blue">128</tspan> 次，Tensor Core <tspan className="fk-text-teal">2048</tspan> 次</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 一条指令的工作量：FP32 一次乘加与 wmma 的 16 × 16 × 16
export function WmmaInstrFigure() {
  const FS = 48, u = FS / 16, R = 250
  const frag = (x: number, y: number, cls: string, hl: 'row' | 'col' | 'cell') => <g>
    <rect x={x} y={y} width={FS} height={FS} rx={2} className={cls} opacity={0.16} />
    {hl === 'row' && <rect x={x} y={y + 15 * u} width={FS} height={u} className={cls} />}
    {hl === 'col' && <rect x={x + 15 * u} y={y} width={u} height={FS} className={cls} />}
    {hl === 'cell' && <><rect x={x + 15 * u} y={y + 15 * u} width={u} height={u} className={cls} /><rect x={x + 15 * u - 3} y={y + 15 * u - 3} width={u + 6} height={u + 6} rx={1.5} fill="none" stroke="var(--fk-purple)" strokeWidth={1.4} /></>}
    <rect x={x} y={y} width={FS} height={FS} rx={2} fill="none" stroke="var(--color-border)" />
  </g>
  const cp = iso(R + 300, 94, 10.5 / 4)
  return <Figure title="一条指令的工作量" desc="FP32 单元一条 FFMA 做 1 次乘加；一条 wmma 做 16 × 16 × 16 的矩阵块乘加：C 的每个输出沿 K 做 16 次乘加，256 个输出共 4096 次。" height={196}
    caption="高亮为算出 C 的一个输出所用的 A 的一行、B 的一列，以及立方体中沿 K 的 16 次乘加。">
    {() => <>
      <rect x={20} y={10} width={210} height={176} rx={6} className="fk-fill-gray" opacity={0.35} />
      <text x={34} y={28} textAnchor="start" className="fk-strong fk-text-blue">FP32 单元：一条 FFMA</text>
      <text x={34} y={86} textAnchor="start" className="fk-mono" style={{ fontSize: 16, fontWeight: 600 }}>d = a × b + c</text>
      <circle cx={52} cy={130} r={7} className="fk-fill-blue" />
      <text x={66} y={130} textAnchor="start" className="fk-muted">1 次乘加</text>
      <rect x={R - 10} y={10} width={500} height={176} rx={6} className="fk-fill-gray" opacity={0.35} />
      <text x={R + 4} y={28} textAnchor="start" className="fk-strong fk-text-teal">Tensor Core：一条 wmma（16 × 16 × 16）</text>
      {frag(R + 4, 64, 'fk-fill-blue', 'row')}
      <text x={R + 62} y={88} className="fk-strong">×</text>
      {frag(R + 74, 64, 'fk-fill-amber', 'col')}
      <text x={R + 132} y={88} className="fk-strong">+</text>
      {frag(R + 144, 64, 'fk-fill-purple', 'cell')}
      <text x={R + 206} y={88} className="fk-strong">=</text>
      {[['A 16×16', 'blue', 0], ['B 16×16', 'amber', 70], ['C 16×16', 'purple', 140]].map(([t, c, dx]) =>
        <text key={t as string} x={R + 28 + (dx as number)} y={128} className={`fk-small fk-strong fk-text-${c}`} style={c === 'amber' ? { fill: '#a87a14' } : undefined}>{t}</text>)}
      <IsoCuboid p={cp} x={0} y={0} sx={16} sy={16} h={16} cls="fk-fill-teal" op={0.35} />
      <IsoCuboid p={cp} x={15} y={15} sx={1} sy={1} h={16} cls="fk-fill-purple" />
      <text x={R + 300} y={150} className="fk-small fk-muted">M × N × K</text>
      <text x={R + 4} y={170} textAnchor="start" className="fk-strong fk-text-teal">每个输出沿 K 做 16 次乘加，256 个输出共 4096 次</text>
      <text x={R + 350} y={86} textAnchor="start" className="fk-muted fk-small">一个 Tensor Core</text>
      <text x={R + 350} y={104} textAnchor="start" className="fk-muted fk-small">约 8 个时钟完成</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- wmma 的分工：完整矩阵 → block → warp → 片段（构图同视频第 2 幕左侧）
export function WmmaTileFigure() {
  const S = 112, g = 8, ax = 44, wx = ax + S + g, ty = 22, cy = ty + S + g
  const bi = 0.62, bj = 0.38, bs = 8                        // block 在 C 中的位置（相对），放大显示的边长
  const bx = wx + bj * S, by = cy + bi * S
  const zx = 352, zy = 56, Z = 176, u = Z / 128             // 放大的 128 × 128，u = 每个元素的像素
  const WR = 0, WC = 1                                       // 选中的 warp（2 × 4 中的第 0 行第 1 列）
  return <Figure title="wmma 的分工：block、warp 与 fragment" desc="C 按 128 × 128 切给 block；一个 block 的 8 个 warp 排成 2 × 4，各负责 64 × 32，即 4 × 2 个 16 × 16 片段。block 沿 K 读 A 的一条行带与 Wᵀ 的一条列带。" height={286}
    caption="紫色小块为一个 block 的 128 × 128（放大显示），蓝、黄细条为它沿 K 读取的 A、Wᵀ。">
    {() => <>
      <rect x={wx} y={ty} width={S} height={S} className="fk-fill-amber" opacity={0.28} />
      <rect x={ax} y={cy} width={S} height={S} className="fk-fill-blue" opacity={0.2} />
      <rect x={wx} y={cy} width={S} height={S} className="fk-fill-purple" opacity={0.18} />
      <rect x={ax} y={by} width={S} height={bs} className="fk-fill-blue" opacity={0.75} />
      <rect x={bx} y={ty} width={bs} height={S} className="fk-fill-amber" opacity={0.9} />
      <rect x={bx} y={by} width={bs} height={bs} className="fk-fill-purple" stroke="#fff" strokeWidth={1.2} />
      <text x={ax - 12} y={cy + S / 2} className="fk-strong fk-text-blue">A</text>
      <text x={wx + S + 14} y={ty + 10} className="fk-strong" style={{ fill: '#a87a14' }}>Wᵀ</text>
      <text x={wx + S + 12} y={cy + 10} className="fk-strong fk-text-purple">C</text>
      <text x={wx + S / 2} y={cy + S + 14} className="fk-muted">8192 × 8192</text>
      <text x={ax + S / 2} y={cy + S + 14} className="fk-muted">M × K</text>
      <text x={wx - g - S / 2} y={ty + S / 2} className="fk-muted">C = A · Wᵀ</text>
      {/* 放大引线 */}
      <line x1={bx + bs} y1={by} x2={zx} y2={zy} stroke="var(--fk-purple)" strokeDasharray="4 3" strokeWidth={1} />
      <line x1={bx + bs} y1={by + bs} x2={zx} y2={zy + Z} stroke="var(--fk-purple)" strokeDasharray="4 3" strokeWidth={1} />
      <rect x={zx} y={zy} width={Z} height={Z} className="fk-fill-purple" opacity={0.12} />
      {[0, 1, 2, 3, 4, 5, 6, 7].filter(k => k !== WR * 4 + WC).map(k => <rect key={k} x={zx + (k % 4) * 32 * u} y={zy + Math.floor(k / 4) * 64 * u} width={32 * u} height={64 * u} fill="none" stroke="var(--fk-purple)" strokeWidth={1.2} />)}
      {Array.from({ length: 8 }, (_, f) => <rect key={f} x={zx + (WC * 32 + (f % 2) * 16) * u + 1} y={zy + (WR * 64 + Math.floor(f / 2) * 16) * u + 1} width={16 * u - 2} height={16 * u - 2} className="fk-fill-purple" />)}
      <rect x={zx + WC * 32 * u} y={zy + WR * 64 * u} width={32 * u} height={64 * u} fill="none" stroke="var(--fk-purple)" strokeWidth={2.2} />
      <text x={zx} y={zy - 30} textAnchor="start" className="fk-strong fk-text-purple">一个 block：128 × 128，8 个 warp（2 × 4）</text>
      <text x={zx} y={zy - 12} textAnchor="start" className="fk-muted">每个 warp 64 × 32</text>
      {/* 右侧：一个 warp 的 8 个片段 */}
      <text x={zx + Z + 26} y={zy + 14} textAnchor="start" className="fk-strong">一个 warp</text>
      <text x={zx + Z + 26} y={zy + 36} textAnchor="start" className="fk-muted">= 4 × 2 个 16 × 16 片段</text>
      <text x={zx + Z + 26} y={zy + 58} textAnchor="start" className="fk-muted">每片一条 mma_sync</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- wmma 的数据通路与 4 路 bank 冲突（构图同视频第 2 幕结尾）
export function WmmaPathFigure() {
  const st: TowerV = { ...TOWER.wmma, hot: { lv: [2], hop: [1, 2] }, tags: [{ hop: 2, text: '寄存器中转' }, { lv: 2, text: 'bank conflict' }] }
  const x0 = 40, w = 300, top = 70
  // bank 面板：每行 64 字节，一排 32 个 bank 放两行矩阵；读 8 行的第 1 块（16 字节 = 4 个 bank）
  const PX = 540, PY = 134, cw = 6.5, ch = 20, rg = 8
  const hit = (b: number) => (b >= 4 && b < 8) || (b >= 20 && b < 24)
  const H = top + 4 * TOWER_GAP + TOWER_H + 18
  return <Figure title="wmma 的数据通路与 bank conflict" desc="显存经 L2 读入寄存器，写入 shared memory，再读回寄存器交给 Tensor Core；shared memory 每行 64 字节，读取 8 行的同一块时落在两组 bank，每组 4 个请求，分 4 轮完成。" height={H}
    caption="绿色为数据走过的路径，红色为瓶颈。">
    {() => <>
      <TowerBody st={st} x0={x0} w={w} top={top} stack />
      <text x={PX - 46} y={PY - 52} textAnchor="start" className="fk-strong">shared memory：每行 64 字节</text>
      <text x={PX - 46} y={PY - 33} textAnchor="start" className="fk-muted">一排 32 个 bank 放两行</text>
      {[0, 8, 16, 24].map(b => <text key={b} x={PX + b * cw + cw / 2} y={PY - 12} className="fk-muted fk-small fk-mono">{b}</text>)}
      <text x={PX + 8 * cw} y={PY + 4 * (ch + rg) + 2} className="fk-muted fk-small">偶数行</text>
      <text x={PX + 24 * cw} y={PY + 4 * (ch + rg) + 2} className="fk-muted fk-small">奇数行</text>
      {[0, 1, 2, 3].map(r => <g key={r}>
        <text x={PX - 8} y={PY + r * (ch + rg) + ch / 2} textAnchor="end" className="fk-muted fk-small fk-mono">{`行 ${2 * r}|${2 * r + 1}`}</text>
        {Array.from({ length: 32 }, (_, b) => <rect key={b} x={PX + b * cw} y={PY + r * (ch + rg)} width={cw - 0.8} height={ch} rx={1}
          className={hit(b) ? 'fk-fill-red' : b < 16 ? 'fk-fill-blue' : 'fk-fill-purple'} opacity={hit(b) ? 1 : 0.14} />)}
      </g>)}
      {[4, 20].map(b => <g key={b}>
        <line x1={PX + b * cw} y1={PY + 4 * (ch + rg) + 16} x2={PX + (b + 4) * cw - 0.8} y2={PY + 4 * (ch + rg) + 16} stroke="var(--fk-red)" strokeWidth={2} />
        <text x={PX + (b + 2) * cw} y={PY + 4 * (ch + rg) + 30} className="fk-small" style={{ fill: 'var(--fk-red)', fontWeight: 600 }}>4 个请求</text>
      </g>)}
      <text x={PX - 46} y={PY + 4 * (ch + rg) + 62} textAnchor="start" style={{ fill: 'var(--fk-red)', fontWeight: 600 }}>读 8 行的同一块：4 轮（4 路冲突）</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 数据通路（替代 wmma.png、wgmma-path.png、tstore-path.png）
type PNode = 'hbm' | 'l2' | 'tma' | 'smem' | 'reg' | 'tc'
const PNODES: Record<PNode, [number, number, number, number, string, Fill]> = {
  hbm: [20, 96, 80, 44, '显存', 'amber'],
  l2: [130, 96, 80, 44, 'L2', 'amber'],
  tma: [270, 150, 80, 44, 'TMA', 'gray'],
  smem: [410, 150, 100, 44, '共享内存', 'green'],
  reg: [410, 46, 100, 44, '寄存器', 'purple'],
  tc: [610, 96, 110, 44, 'Tensor Core', 'teal'],
}
const PEDGES: Record<string, [string, number, number]> = {
  'hbm-l2': ['M 100 118 H 130', 115, 102],
  'l2-reg': ['M 210 110 H 380 V 68 H 410', 300, 96],
  'reg-smem': ['M 446 90 V 150', 430, 120],
  'smem-reg': ['M 474 150 V 90', 490, 120],
  'reg-tc': ['M 510 68 H 665 V 96', 590, 54],
  'l2-tma': ['M 210 126 H 310 V 150', 256, 140],
  'tma-smem': ['M 350 172 H 410', 380, 158],
  'smem-tc': ['M 510 172 H 665 V 140', 590, 186],
  'smem-tma': ['M 410 180 H 350', 380, 194],
  'tma-l2': ['M 270 172 H 170 V 140', 220, 186],
  'l2-hbm': ['M 130 118 H 100', 115, 102],
}
const PATH_V: Record<string, { title: string; desc: string; caption: string; edges: string[]; nodes: PNode[]; acc?: boolean }> = {
  wmma: { title: 'wmma 的数据通路', desc: '显存经 L2 读入寄存器，写入共享内存，再读回寄存器交给 Tensor Core。', edges: ['hbm-l2', 'l2-reg', 'reg-smem', 'smem-reg', 'reg-tc'], nodes: ['hbm', 'l2', 'smem', 'reg', 'tc'],
    caption: 'wmma：数据先读入寄存器、写入共享内存，计算前再读回寄存器，两次经过寄存器。' },
  wgmma: { title: 'TMA 与 wgmma 的数据通路', desc: 'TMA 把整块从 L2 搬入共享内存，wgmma 直接读共享内存，寄存器只存累加结果。', edges: ['hbm-l2', 'l2-tma', 'tma-smem', 'smem-tc'], nodes: ['hbm', 'l2', 'tma', 'smem', 'reg', 'tc'], acc: true,
    caption: 'TMA 把整块从 L2 搬入共享内存，wgmma 直接从共享内存读操作数；寄存器只存累加结果。' },
  tstore: { title: 'TMA 写回的数据通路', desc: '累加结果从寄存器写入共享内存，再由 TMA 经 L2 写回显存。', edges: ['reg-smem', 'smem-tma', 'tma-l2', 'l2-hbm'], nodes: ['hbm', 'l2', 'tma', 'smem', 'reg'],
    caption: '写回：寄存器中的结果按 128 字节换位写入共享内存，再由 TMA 写回显存。' },
}

export function PathFigure({ v }: { v: 'wmma' | 'wgmma' | 'tstore' }) {
  const st = PATH_V[v]
  return <Figure title={st.title} desc={st.desc} height={226} caption={st.caption}>
    {arrow => <>
      <Band x={240} y={16} w={500} h={196} hue="gray" title="SM" />
      {st.acc && <path d={PEDGES['reg-tc'][0]} className="fk-link fk-line-muted" strokeDasharray="4 3" />}
      {st.edges.map(e => <Link key={e} d={PEDGES[e][0]} hue="blue" arrow={arrow} />)}
      {(Object.keys(PNODES) as PNode[]).map(k => {
        const [x, y, w, h, label, fill] = PNODES[k]
        return <Cell key={k} x={x} y={y} w={w} h={h} fill={st.nodes.includes(k) ? fill : 'void'} label={label} sub={k === 'reg' && st.acc ? '累加结果' : undefined} />
      })}
      {st.edges.map((e, i) => <Badge key={e} x={PEDGES[e][1]} y={PEDGES[e][2]} n={i + 1} />)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 128 字节行的异或换位（替代 swizzle.png 与 SwizzleFigure 的带公式版本）
export function SwizzleFigure2() {
  const c = 26, top = 40
  const grid = (ox: number, swz: boolean) => Array.from({ length: 64 }, (_, k) => {
    const r = Math.floor(k / 8), phys = k % 8, logical = swz ? phys ^ r : phys
    return <Cell key={k} x={ox + phys * c} y={top + r * c} w={c - 2} h={c - 2} rx={2} fill={logical === 2 ? (swz ? 'teal' : 'red') : 'gray'} label={logical} small />
  })
  const head = (ox: number, t: string) => <>
    <text x={ox + 4 * c} y={14} className="fk-strong">{t}</text>
    {Array.from({ length: 8 }, (_, i) => <text key={i} x={ox + i * c + c / 2 - 1} y={top + 8 * c + 12} className="fk-muted">{i}</text>)}
    <text x={ox + 4 * c} y={top + 8 * c + 32} className="fk-muted">bank 组</text>
  </>
  return <Figure title="128 字节行的 swizzle" desc="一行 8 个 16 字节块。swizzle 前 8 行的逻辑块 2 都在 bank 组 2；swizzle 后第 r 行的块 c 存到 c ⊕ r，分散到 8 组。" height={292}
    caption="每格 16 字节（4 个 bank），数字为逻辑块号；着色为 ldmatrix 一次读取的 8 行同一块。">
    {() => <>
      {head(110, 'swizzle 前')}
      {head(440, 'swizzle 后')}
      {grid(110, false)}
      {grid(440, true)}
      {Array.from({ length: 8 }, (_, r) => <text key={r} x={96} y={top + r * c + c / 2 - 1} textAnchor="end" className="fk-muted">{r}</text>)}
      <text x={60} y={top + 4 * c} className="fk-muted">行</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 一条指令的计算量（替代 wgmma-parts.png）
export function WgmmaSizeFigure() {
  const u = 2.4, top = 40, mx = 150, wx = 300
  return <Figure title="一条指令的输出块" desc="mma.sync 一条指令算 16 × 8 的输出块，wgmma.m64n128k16 一条算 64 × 128，K 方向都是 16，按比例绘制。" height={250}
    caption="按比例绘制，K 方向都是 16；细格为一条 mma.sync 的 16 × 8。">
    {() => <>
      <rect x={mx} y={top} width={8 * u} height={16 * u} rx={1} className="fk-fill-purple" />
      <text x={mx + 4 * u} y={top + 16 * u + 18} className="fk-strong">mma.sync</text>
      <text x={mx + 4 * u} y={top + 16 * u + 36} className="fk-muted fk-mono">m16n8k16</text>
      <rect x={wx} y={top} width={128 * u} height={64 * u} rx={2} className="fk-fill-teal" />
      {Array.from({ length: 15 }, (_, i) => <line key={`v${i}`} x1={wx + (i + 1) * 8 * u} y1={top} x2={wx + (i + 1) * 8 * u} y2={top + 64 * u} stroke="#fff" strokeOpacity={0.45} />)}
      {Array.from({ length: 3 }, (_, i) => <line key={`h${i}`} x1={wx} y1={top + (i + 1) * 16 * u} x2={wx + 128 * u} y2={top + (i + 1) * 16 * u} stroke="#fff" strokeOpacity={0.45} />)}
      <rect x={wx} y={top} width={8 * u} height={16 * u} className="fk-fill-purple" />
      <text x={wx + 64 * u} y={top + 64 * u + 18} className="fk-strong">wgmma</text>
      <text x={wx + 64 * u} y={top + 64 * u + 36} className="fk-muted fk-mono">m64n128k16</text>
      <text x={wx + 64 * u} y={top - 12} className="fk-muted">N = 128</text>
      <text x={wx + 128 * u + 10} y={top + 32 * u} textAnchor="start" className="fk-muted">M = 64</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- warp 特化的结构（替代 ws.png）
export function WsFigure() {
  const sx = 200, sw = 104, gap = 16
  return <Figure title="warp 特化：生产者、槽位与消费者" desc="生产者 warpgroup 用一个线程发 TMA，把 A、W 块写入 4 个槽位；每个槽位有 full、empty 两个 mbarrier；两个消费者 warpgroup 各自发 wgmma，Tensor Core 直接读槽位。" height={348}
    caption="一个 block 有 3 个 warpgroup。槽位的 full 表示 TMA 已到齐，由消费者等待；empty 表示消费者已用完，由生产者等待。">
    {arrow => <>
      <Band x={sx} y={10} w={4 * sw + 3 * gap} h={60} hue="blue" title="warpgroup 0：生产者" />
      <Cell x={sx + 4 * sw + 3 * gap - 120} y={26} w={104} h={30} fill="blue" label="TMA" />
      <Cell x={60} y={124} w={80} h={44} fill="amber" label="L2" />
      <Link d={`M 140 146 H ${sx}`} hue="blue" arrow={arrow} />
      <Link d={`M ${sx + 4 * sw + 3 * gap - 68} 56 V 96`} hue="blue" arrow={arrow} />
      {[0, 1, 2, 3].map(i => {
        const x = sx + i * (sw + gap)
        return <g key={i}>
          <Cell x={x} y={124} w={sw} h={44} fill="green" label={`槽 ${i}`} />
          <Cell x={x} y={100} w={sw / 2 - 2} h={20} rx={2} fill="free" label="full" small />
          <Cell x={x + sw / 2 + 2} y={100} w={sw / 2 - 2} h={20} rx={2} fill="free" label="empty" small />
        </g>
      })}
      {[0, 1].map(j => {
        const x = sx + j * (2 * sw + 2 * gap), w = 2 * sw + gap
        return <g key={j}>
          <Band x={x} y={204} w={w} h={56} hue="teal" title={`warpgroup ${j + 1}：消费者`} />
          <Link d={`M ${x + w / 2} 168 V 204`} hue="teal" arrow={arrow} />
          <Link d={`M ${x + w / 2} 260 V 296`} hue="teal" arrow={arrow} />
        </g>
      })}
      <Cell x={sx} y={296} w={4 * sw + 3 * gap} h={40} fill="teal" label="Tensor Core" />
    </>}
  </Figure>
}

// ---------------------------------------------------------------- L2 读取与集群多播（替代 l2-read.png、cluster.png 与 ClusterFigure）
export function ClusterFigure2() {
  const panel = (ox: number, cluster: boolean, title: string, arrow: string) => {
    const b0 = ox, b1 = ox + 180, by = 140
    const block = (bx: number, rank: number) => <g>
      <rect x={bx} y={by} width={160} height={84} rx={12} className="fk-band-teal" />
      <text x={bx + 80} y={by + 68} className="fk-strong fk-text-teal">block {rank}</text>
      {cluster
        ? <><Cell x={bx + 12} y={by + 14} w={64} h={34} fill="green" label="W 前半" small /><Cell x={bx + 84} y={by + 14} w={64} h={34} fill="green" label="W 后半" small /></>
        : <Cell x={bx + 12} y={by + 14} w={136} h={34} fill="green" label="W" small />}
    </g>
    return <g>
      <text x={ox + 170} y={14} className="fk-strong">{title}</text>
      <text x={ox + 60} y={56} textAnchor="end" className="fk-muted">L2</text>
      {cluster
        ? <><Cell x={ox + 80} y={38} w={88} h={36} fill="amber" label="W 前半" small /><Cell x={ox + 172} y={38} w={88} h={36} fill="amber" label="W 后半" small /></>
        : <Cell x={ox + 80} y={38} w={180} h={36} fill="amber" label="W" />}
      {block(b0, 0)}{block(b1, 1)}
      {cluster ? <>
        <Link d={`M ${ox + 124} 74 V 100 H ${b0 + 44} V ${by + 14}`} hue="blue" arrow={arrow} />
        <Link d={`M ${ox + 124} 100 H ${b1 + 44} V ${by + 14}`} hue="blue" arrow={arrow} />
        <Link d={`M ${ox + 216} 74 V 118 H ${b0 + 116} V ${by + 14}`} hue="purple" arrow={arrow} />
        <Link d={`M ${ox + 216} 118 H ${b1 + 116} V ${by + 14}`} hue="purple" arrow={arrow} />
      </> : <>
        <Link d={`M ${ox + 130} 74 V 108 H ${b0 + 80} V ${by + 14}`} hue="blue" arrow={arrow} />
        <Link d={`M ${ox + 210} 74 V 108 H ${b1 + 80} V ${by + 14}`} hue="blue" arrow={arrow} />
      </>}
    </g>
  }
  return <Figure title="L2 读取与集群多播" desc="上下相邻的两个 tile 用同一块 W。无集群时两个 block 各从 L2 读整块 W；2 个 block 的集群中各读一半，用 TMA 多播写入两个 block。" height={240}
    caption="上下相邻的两个 tile 用同一块 W。左：两个 block 各从 L2 读整块 W。右：block 0 读前半、block 1 读后半，各自多播给两个 block。">
    {arrow => <>
      {panel(20, false, '无集群', arrow)}
      <line x1={380} y1={4} x2={380} y2={232} className="fk-rule" />
      {panel(400, true, '2 个 block 的集群', arrow)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- grid → block → warp 的等轴测 3D 层级（构图同视频 grid、block 段落）
// 等轴测投影：x 轴向右下，y 轴向左下，z 向上。左上 grid，右上放大的 block (2, 1)，下方放大的 warp 1。
// 引线先画，被方块盖住，只在缝隙与空白处露出。
type Iso = (x: number, y: number, z?: number) => [number, number]
const iso = (X0: number, Y0: number, k: number): Iso => (x, y, z = 0) => [X0 + (x - y) * k * 0.866, Y0 + (x + y) * k * 0.5 - z * k]
const poly = (ps: [number, number][]) => ps.map(p => p.join(',')).join(' ')
type Tone = 'gray' | 'tint' | 'blue'
function IsoBox({ p, x, y, s, h, tone }: { p: Iso; x: number; y: number; s: number; h: number; tone: Tone }) {
  const top = poly([p(x, y, h), p(x + s, y, h), p(x + s, y + s, h), p(x, y + s, h)])
  const right = poly([p(x + s, y, h), p(x + s, y + s, h), p(x + s, y + s, 0), p(x + s, y, 0)])
  const left = poly([p(x, y + s, h), p(x + s, y + s, h), p(x + s, y + s, 0), p(x, y + s, 0)])
  const cls = tone === 'gray' ? 'fk-fill-gray' : 'fk-fill-blue', op = tone === 'tint' ? 0.4 : 1
  const face = (pts: string, shade: number) => <>
    <polygon points={pts} style={{ fill: 'var(--color-page)' }} />
    <polygon points={pts} className={cls} opacity={op} />
    {shade > 0 && <polygon points={pts} fill="#3a3320" opacity={shade} />}
    <polygon points={pts} fill="none" stroke="var(--color-page)" strokeWidth={0.6} strokeLinejoin="round" />
  </>
  return <g>{face(left, 0.1)}{face(right, 0.2)}{face(top, 0)}</g>
}

export function HierarchyFigure3D() {
  const BX = 2, BY = 1, TX = 5, TY = 3, WARP = 1
  // 紧凑模式（520 宽）：grid 与 block 缩小并排在上方，warp 在下方
  const compact = useCompact()
  const G: [number, number, number] = compact ? [134, 92, 20] : [160, 58, 34], B: [number, number, number] = compact ? [392, 58, 7] : [572, 58, 10.6], WP: [number, number] = compact ? [196, 236] : [232, 300]
  // grid：4 × 4 个 block，单位 = 一个 block 的边长（含缝）
  const gp = iso(...G), gs = 1.16
  // block：16 × 16 个线程
  const bp = iso(...B)
  // warp：16 × 2 个线程
  const wp = iso(WP[0], WP[1], 20)
  const leader = (a: [number, number], b: [number, number], key: string) =>
    <line key={key} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="fk-line-blue" strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
  const order = (n: number, m: number) => Array.from({ length: n * m }, (_, i) => [i % n, Math.floor(i / n)] as const).sort((a, b) => a[0] + a[1] - b[0] - b[1])
  const g0 = BX * gs, g1 = BY * gs
  const lane = (x: number, y: number) => y * 16 + x
  const sel = lane(TX, TY - 2 * WARP)
  const reg = wp(TX + 0.5, 1.92, 0)
  return <Figure title="grid、block、warp 与线程" desc="grid 为 4 × 4 个 block；放大 block (2, 1)：16 × 16 个线程，每两行为一个 warp，共 8 个；再放大 warp 1：32 个线程排成两行，lane 0–31，选中的 lane 21 即线程 (5, 3)，每个线程有自己的寄存器。" width={compact ? 540 : 760} height={compact ? 410 : 500}
    caption="左上为 grid：4 × 4 个 block。右上放大 block (2, 1)：16 × 16 个线程，连续 32 个即两行为一个 warp。下方放大 warp 1：lane 0–31，深蓝为线程 (5, 3)，即 lane 21。">
    {() => <>
      {/* 引线：grid 的 block (2, 1) → block；block 的 warp 1 → warp */}
      {leader(gp(g0, g1, 0.15), bp(0, 0), 'l1')}
      {leader(gp(g0 + 1, g1 + 1, 0), bp(16, 16), 'l2')}
      {leader(bp(0, 2 * WARP + 2, 0.75), wp(0, 2), 'l3')}
      {leader(bp(16, 2 * WARP, 0.75), wp(16, 0), 'l4')}

      {/* grid */}
      <text x={G[0]} y={compact ? 30 : 16} className="fk-strong">grid</text>
      {order(4, 4).map(([x, y]) => <IsoBox key={`g${x}${y}`} p={gp} x={x * gs} y={y * gs} s={1} h={0.15} tone={x === BX && y === BY ? 'tint' : 'gray'} />)}
      {[0, 1, 2, 3].map(i => { const [x, y] = gp(i * gs + 0.5, -0.4); return <text key={`gx${i}`} x={x + 4} y={y - 2} className="fk-muted">{i}</text> })}
      {[0, 1, 2, 3].map(i => { const [x, y] = gp(-0.4, i * gs + 0.5); return <text key={`gy${i}`} x={x - 4} y={y - 2} className="fk-muted">{i}</text> })}
      {(() => { const [x, y] = gp(2.3, -1.15); return <text x={x + 14} y={y} textAnchor="start" className="fk-muted" style={compact ? HALO : undefined}>blockIdx.x</text> })()}
      {(() => { const [x, y] = gp(-1.15, 2.3); return <text x={x - 14} y={y} textAnchor="end" className="fk-muted">blockIdx.y</text> })()}
      {(() => { const [x, y] = gp(g0 + 0.5, g1 + 0.5, 0.15); return <text x={x} y={y} className="fk-small fk-text-blue">(2, 1)</text> })()}

      {/* block (2, 1) */}
      <text x={B[0]} y={16} className="fk-strong">block (2, 1)</text>
      {order(16, 16).map(([x, y]) => {
        const inW = Math.floor(y / 2) === WARP, me = x === TX && y === TY
        return <IsoBox key={`b${x}_${y}`} p={bp} x={x + 0.08} y={y + 0.08} s={0.84} h={0.75} tone={me ? 'blue' : inW ? 'tint' : 'gray'} />
      })}
      {(() => { const [x, y] = bp(8, -1.3); return <text x={x + 14} y={y} textAnchor="start" className="fk-muted">threadIdx.x</text> })()}
      {(() => { const [x, y] = bp(-1.3, 8); return <text x={x - 14} y={y} textAnchor="end" className="fk-muted" style={compact ? HALO : undefined}>threadIdx.y</text> })()}
      {Array.from({ length: 8 }, (_, w) => {
        const a = bp(16.5, 2 * w + 0.25), b = bp(16.5, 2 * w + 1.75), [tx, ty] = bp(17.6, 2 * w + 1)
        const on = w === WARP
        return <g key={`w${w}`}>
          <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={on ? 'fk-line-blue' : 'fk-rule'} style={on ? undefined : { stroke: 'var(--color-muted)' }} strokeWidth={2} />
          <text x={tx} y={ty} textAnchor="start" className={`fk-small ${on ? 'fk-text-blue' : 'fk-muted'}`} style={{ paintOrder: 'stroke', stroke: 'var(--color-page)', strokeWidth: 3 }}>{w === 0 ? 'warp 0' : w}</text>
        </g>
      })}

      {/* warp 1 */}
      <text x={WP[0] - 46} y={WP[1] - 8} textAnchor="end" className="fk-strong">warp 1</text>
      {order(16, 2).map(([x, y]) => {
        const l = lane(x, y), me = l === sel, [cx, cy] = wp(x + 0.5, y + 0.5, 0.8)
        return <g key={`t${l}`}>
          <IsoBox p={wp} x={x + 0.08} y={y + 0.08} s={0.84} h={0.8} tone={me ? 'blue' : 'tint'} />
          <text x={cx} y={cy + 0.5} className={me ? 'fk-cell-text fk-cell-small' : 'fk-cell-text fk-on-gray fk-cell-small'} style={{ fontSize: compact ? 10 : 9 }}>{l}</text>
        </g>
      })}
      {/* 选中线程的寄存器 */}
      {(() => {
        const rx = compact ? reg[0] - 118 : reg[0] - 30, ry = reg[1] + (compact ? 40 : 44)
        return <g>
          <line x1={reg[0]} y1={reg[1]} x2={reg[0]} y2={ry} className="fk-line-blue" strokeWidth={1} />
          {/* 一条寄存器条，末尾渐隐加省略号：数量不固定，最多 255 个 */}
          {Array.from({ length: 14 }, (_, i) => <rect key={i} x={rx + i * 9} y={ry + 1} width={8} height={11} rx={1.5} className="fk-fill-green" opacity={0.8 - Math.max(0, i - 8) * 0.11} />)}
          <text x={rx + 14 * 9 + 4} y={ry + 6} textAnchor="start" className="fk-small" style={{ fill: 'var(--fk-green)' }}>…</text>
          <text x={rx - 6} y={ry + 7} textAnchor="end" className="fk-small fk-text-blue">lane 21</text>
          <text x={rx} y={ry + 26} textAnchor="start" className="fk-small" style={{ fill: 'var(--fk-green)' }}>寄存器（每线程最多 255 个）</text>
        </g>
      })()}
    </>}
  </Figure>
}

// ================================================================ 立体重绘（2026-10-22 第二批）：硬件塔、完整矩阵与等轴测，图内只放标签
const PAGE = 'var(--color-page)'
type Pt = [number, number]
// 带箭头的直线（任意颜色；figure-kit 的 marker 只有四种色）
function Arrow({ a, b, color, w = 1.6, dash, head = 7 }: { a: Pt; b: Pt; color: string; w?: number; dash?: string; head?: number }) {
  const t = Math.atan2(b[1] - a[1], b[0] - a[0]), c = Math.cos(t), s = Math.sin(t)
  const bx = b[0] - head * c, by = b[1] - head * s, hw = head * 0.55
  return <g>
    <line x1={a[0]} y1={a[1]} x2={bx} y2={by} stroke={color} strokeWidth={w} strokeDasharray={dash} />
    <polygon points={poly([b, [bx - hw * s, by + hw * c], [bx + hw * s, by - hw * c]])} fill={color} />
  </g>
}
// 等轴测长方体（IsoBox 的一般形式：任意长宽高、任意填色）
function IsoCuboid({ p, x, y, z = 0, sx, sy, h, cls, op = 1, stroke }: { p: Iso; x: number; y: number; z?: number; sx: number; sy: number; h: number; cls: string; op?: number; stroke?: string }) {
  const t = z + h
  const top = poly([p(x, y, t), p(x + sx, y, t), p(x + sx, y + sy, t), p(x, y + sy, t)])
  const right = poly([p(x + sx, y, t), p(x + sx, y + sy, t), p(x + sx, y + sy, z), p(x + sx, y, z)])
  const left = poly([p(x, y + sy, t), p(x + sx, y + sy, t), p(x + sx, y + sy, z), p(x, y + sy, z)])
  const face = (pts: string, shade: number) => <>
    <polygon points={pts} style={{ fill: PAGE }} />
    <polygon points={pts} className={cls} opacity={op} />
    {shade > 0 && <polygon points={pts} fill="#3a3320" opacity={shade} />}
    <polygon points={pts} fill="none" stroke={stroke ?? PAGE} strokeWidth={stroke ? 1.4 : 0.6} strokeLinejoin="round" />
  </>
  return <g>{face(left, 0.1)}{face(right, 0.2)}{face(top, 0)}</g>
}
// 等轴测顶面上的一块（贴在 z 高度的平面上）
const isoFace = (p: Iso, x: number, y: number, sx: number, sy: number, z: number) => poly([p(x, y, z), p(x + sx, y, z), p(x + sx, y + sy, z), p(x, y + sy, z)])
// 2.5D 板：正面矩形 + 右上方的顶面与侧面（视频硬件塔的画法）
const SD: Pt = [14, -9]
const HALO = { paintOrder: 'stroke', stroke: PAGE, strokeWidth: 4, strokeLinejoin: 'round' } as const
type SlabTone = 'idle' | 'on' | 'hot' | 'dim'
function Slab({ x, y, w, h, tone = 'idle' }: { x: number; y: number; w: number; h: number; tone?: SlabTone }) {
  const [dx, dy] = SD
  const tint = tone === 'hot' ? 'fk-fill-red' : tone === 'on' ? 'fk-fill-green' : null
  const stroke = tone === 'hot' ? 'var(--fk-red)' : tone === 'on' ? 'var(--fk-green)' : 'var(--color-border)'
  const top = poly([[x, y], [x + dx, y + dy], [x + w + dx, y + dy], [x + w, y]])
  const side = poly([[x + w, y], [x + w + dx, y + dy], [x + w + dx, y + h + dy], [x + w, y + h]])
  return <g opacity={tone === 'dim' ? 0.45 : 1}>
    <polygon points={top} className="fk-fill-gray" />
    {tint && <polygon points={top} className={tint} opacity={0.3} />}
    <polygon points={side} className="fk-fill-gray" />
    <polygon points={side} fill="#3a3320" opacity={0.13} />
    <rect x={x} y={y} width={w} height={h} style={{ fill: PAGE }} />
    {tint && <rect x={x} y={y} width={w} height={h} className={tint} opacity={0.1} />}
    <rect x={x} y={y} width={w} height={h} fill="none" stroke={stroke} strokeWidth={tint ? 1.6 : 1.2} />
  </g>
}
// 瓶颈标签：红底“瓶颈” + 红字
function BnTag({ x, y, text }: { x: number; y: number; text: string }) {
  return <g>
    <rect x={x} y={y - 9} width={34} height={18} rx={3} className="fk-fill-red" />
    <text x={x + 17} y={y} className="fk-cell-text" style={{ fontSize: 11 }}>瓶颈</text>
    <text x={x + 40} y={y} textAnchor="start" className="fk-small" style={{ fill: 'var(--fk-red)', fontWeight: 600 }}>{text}</text>
  </g>
}

// ---------------------------------------------------------------- 硬件塔（替代 ml-hw:* 的 SM 剖面与 ml-path:* 的数据通路）
// 五层：0 计算单元、1 寄存器、2 共享内存、3 L2、4 显存；路径为经过的层，每一跳一条竖线，相邻两跳在层内横向相接。
type TowerTag = { lv?: number; hop?: number; text: string }
type TowerV = {
  title: string; desc: string; caption: string; tc: boolean
  path: number[]; hops: string[]; lanes?: [number, number]
  hot?: { lv?: number[]; hop?: number[] }; tag?: TowerTag; tags?: TowerTag[]
  xpath?: number[]; xhop?: string; xl?: number
  sub?: Record<number, string>; warps?: string; purple?: boolean
}
const TOWER: Record<string, TowerV> = {
  naive: { title: 'SM 剖面：naive_T', desc: '显存经 L2 直接读入寄存器，再交给 FP32 单元；一个 warp 的读取拆成 16 个扇区，warp 大多在等显存。', tc: false,
    path: [4, 3, 1, 0], hops: ['', 'LDG', 'FFMA'], hot: { hop: [1] }, tag: { hop: 1, text: '每次读取 16 个扇区' }, warps: 'wwwwwwwwwwwwwwrr',
    caption: 'naive_T：每次 FMA 从显存经 L2 读两个数进寄存器，一次读取拆成 16 个扇区。示意。' },
  smem: { title: 'SM 剖面：smem T = 32', desc: '块先经 L2 搬进共享内存；每次 FMA 再从共享内存读两个数，共享内存到寄存器成为瓶颈。', tc: false,
    path: [4, 3, 2, 1, 0], hops: ['', '搬入', 'LDS', 'FFMA'], hot: { hop: [2] }, tag: { hop: 2, text: '共享内存带宽' }, sub: { 2: 'As、Bs' }, warps: 'rrrrrrrrsssswwww',
    caption: 'smem（T = 32）：经 L2 的读取降为 1/32，每次 FMA 仍从共享内存读两个数。示意。' },
  dbuf: { title: 'SM 剖面：dbuf', desc: '显存读取与计算重叠，共享内存开两份；FP32 单元的峰值成为上限。', tc: false,
    path: [4, 3, 2, 1, 0], hops: ['', 'LDG → STS', 'LDS.128', 'FFMA'], hot: { lv: [0] }, tag: { lv: 0, text: 'FP32 峰值' }, sub: { 2: 'As、Bs × 2' }, warps: 'rrrrrrrrrrrrrrss',
    caption: 'dbuf：读显存与计算重叠，共享内存开两份；上限变为 FP32 单元的峰值。示意。' },
  wmma: { title: 'wmma 的数据通路', desc: '显存经 L2 读入寄存器，写入 shared memory，再读回寄存器交给 Tensor Core。', tc: true,
    path: [4, 3, 1, 2, 1, 0], hops: ['', '读入', '写入', '读回', 'mma'], lanes: [0.5, 0.96],
    caption: 'wmma：数据先读入寄存器、写入 shared memory，计算前再读回寄存器，两次经过寄存器。' },
  mma: { title: 'mma.sync 与 pipeline 的数据通路', desc: 'cp.async 由 L2 直接写入 shared memory，ldmatrix 读入寄存器，mma.sync 交给 Tensor Core；每条 mma.sync 只有 2048 次乘加。', tc: true,
    path: [4, 3, 2, 1, 0], hops: ['', 'cp.async', 'ldmatrix', 'mma.sync'], hot: { hop: [3] }, tag: { hop: 3, text: '指令小，停在约 300' },
    caption: '' },
  wgmma: { title: 'TMA 与 wgmma 的数据通路', desc: 'TMA 把整块从 L2 搬入 shared memory，wgmma 直接读 shared memory，寄存器只存累加结果；同一个 warpgroup 既搬运又计算，两块之间 Tensor Core 空闲。', tc: true,
    path: [4, 3, 2, 0], hops: ['', 'TMA', 'wgmma'], sub: { 1: '累加结果' }, hot: { lv: [0] }, tag: { lv: 0, text: '两块之间空闲' },
    caption: '' },
  tstore: { title: 'TMA store 的数据通路', desc: '累加结果从寄存器 swizzle 写入 shared memory，再由 TMA 经 L2 写回显存；上一版由每个线程零散写回。', tc: true,
    path: [1, 2, 3, 4], hops: ['swizzle 写入', 'TMA store', ''], lanes: [0.64, 0.95], xpath: [1, 4], xhop: '零散写回', xl: 0.45, sub: { 1: '累加结果' }, purple: true,
    caption: '划掉的虚线为上一版的零散写回。' },
}
const WARP_HUE: Record<string, Fill> = { r: 'green', w: 'amber', s: 'gray' }

export function TowerFigure({ v }: { v: string }) {
  const st = TOWER[v]
  // 紧凑模式（520 宽）：塔靠左、宽 300，warp 状态点紧贴塔的右侧
  const compact = useCompact()
  const x0 = compact ? 20 : st.warps ? 150 : 200, w = compact ? 300 : 320
  return <Figure title={st.title} desc={st.desc} width={compact ? 520 : 760} height={TOWER_TOP + 4 * TOWER_GAP + TOWER_H + 18} caption={st.caption}
    below={st.warps ? <Legend items={[['green', '执行'], ['amber', '等显存'], ['gray', '等同步'], ['red', '瓶颈']]} /> : undefined}>
    {() => <TowerBody st={st} x0={x0} w={w} top={TOWER_TOP} compact={compact} wx={compact ? x0 + w + 48 : 650} wdy={compact ? 18 : 0} />}
  </Figure>
}

const TOWER_TOP = 46, TOWER_H = 34, TOWER_GAP = 72
// 塔本体：可单独嵌进组合图（x0、top 为塔的左上角）
function TowerBody({ st, x0, w, top, compact = false, wx = 650, wdy = 0, stack = false }: { st: TowerV; x0: number; w: number; top: number; compact?: boolean; wx?: number; wdy?: number; stack?: boolean }) {
  const h = TOWER_H, gap = TOWER_GAP
  const Y = (i: number) => top + i * gap
  const LV: [string, Hue, string][] = [
    st.tc ? ['Tensor Core', 'teal', '×4'] : ['FP32 单元', 'blue', '128 FMA / 时钟'],
    ['寄存器', 'purple', '256 KB'], [st.tc ? 'shared memory' : '共享内存 / L1', 'green', '228 KB'], ['L2', 'amber', '50 MB'], ['显存', 'amber', '80 GB'],
  ]
  const onLv = new Set(st.path), hotLv = new Set(st.hot?.lv ?? []), hotHop = new Set(st.hot?.hop ?? [])
  const [l0, l1] = st.lanes ?? [0.56, 0.95], n = st.path.length - 1
  const lane = (i: number) => x0 + w * (n > 1 ? l0 + (l1 - l0) * i / (n - 1) : (l0 + l1) / 2)
  const mid = (L: number) => Y(L) + h / 2
  const col = st.purple ? 'var(--fk-purple)' : 'var(--fk-green)'
  // 每一跳的标注与数据方块放在目的层一侧的缝隙里（向上时在源层上方的缝，向下时在目的层上方的缝）
  const gc = (L: number) => Y(L) - (gap - h - SD[1]) / 2   // 层 L 上方缝隙（扣去顶面）的中线
  const gapY = (a: number, b: number) => a > b ? gc(a) : gc(b)
  const hops = Array.from({ length: n }, (_, i) => {
    const a = st.path[i], b = st.path[i + 1], lx = lane(i), up = a > b
    return { a, b, lx, up, edge: up ? Y(b) + h : Y(b), gy: gapY(a, b), hot: hotHop.has(i), label: st.hops[i] }
  })
  const crossed = (a: number, b: number) => Array.from({ length: Math.abs(a - b) - 1 }, (_, k) => Math.min(a, b) + 1 + k)
  const xr = x0 + w + SD[0] + 18
  const tags = [...(st.tag ? [st.tag] : []), ...(st.tags ?? [])]
  const tagY = (t: TowerTag) => t.lv !== undefined ? mid(t.lv) : hops[t.hop!].gy
  return <>
      {/* SM 边界：计算单元、寄存器、共享内存在片上 */}
      {stack ? (() => {
        // 132 个 SM 叠成一摞（后面两张为示意），共用下方的 L2 与显存
        const fx = x0 - 16, fy = Y(0) - 34, fw = w + SD[0] + 32, fh = Y(2) + h + 10 - fy
        return <g>
          {[2, 1].map(k => <rect key={k} x={fx + 9 * k} y={fy - 7 * k} width={fw} height={fh} rx={8} style={{ fill: PAGE }} stroke="var(--fk-teal)" strokeOpacity={0.45} strokeWidth={1.2} />)}
          <rect x={fx} y={fy} width={fw} height={fh} rx={8} style={{ fill: PAGE }} />
          <rect x={fx} y={fy} width={fw} height={fh} rx={8} className="fk-fill-teal" opacity={0.07} />
          <rect x={fx} y={fy} width={fw} height={fh} rx={8} fill="none" stroke="var(--fk-teal)" strokeWidth={1.6} />
          <text x={fx + 10} y={fy + 11} textAnchor="start" className="fk-small fk-strong fk-text-teal">一个 SM（共 132 个）</text>
          <text x={fx + fw + 26} y={fy - 8} textAnchor="start" className="fk-small fk-muted">× 132</text>
          <text x={x0 + w + SD[0] + 12} y={mid(3)} textAnchor="start" className="fk-small fk-muted">132 个 SM 共用</text>
        </g>
      })() : <>
        <rect x={x0 - 14} y={Y(0) - 22} width={w + SD[0] + 28} height={Y(2) + h + 5 - (Y(0) - 22)} rx={6} fill="none" stroke="var(--color-muted)" strokeDasharray="5 4" strokeWidth={1} />
        <text x={x0 - 14} y={Y(0) - 32} textAnchor="start" className="fk-muted">一个 SM（共 132 个）</text>
      </>}
      {LV.map(([name, hue, sub0], i) => {
        const tone: SlabTone = hotLv.has(i) ? 'hot' : onLv.has(i) ? 'on' : 'idle'
        const sub = st.sub?.[i] ?? sub0, subPurple = st.sub?.[i] === '累加结果'
        return <g key={i}>
          <Slab x={x0} y={Y(i)} w={w} h={h} tone={tone} />
          <rect x={x0 + 8} y={Y(i) + 9} width={4} height={h - 18} rx={1} className={`fk-fill-${hue}`} />
          <text x={x0 + 20} y={mid(i)} textAnchor="start">
            <tspan className="fk-strong" style={tone === 'hot' ? { fill: 'var(--fk-red)' } : undefined}>{name}</tspan>
            <tspan dx={8} className="fk-muted fk-small" style={subPurple ? { fill: 'var(--fk-purple)' } : undefined}>{sub}</tspan>
          </text>
        </g>
      })}
      {/* 越过的层：线从它前面经过，留一道缝 */}
      {hops.flatMap((hp, i) => crossed(hp.a, hp.b).map(L => <rect key={`c${i}${L}`} x={hp.lx - 5} y={Y(L) - 0.5} width={10} height={h + 1} style={{ fill: PAGE }} />))}
      {st.xpath && crossed(st.xpath[0], st.xpath[1]).map(L => <rect key={`xc${L}`} x={x0 + w * st.xl! - 5} y={Y(L) - 0.5} width={10} height={h + 1} style={{ fill: PAGE }} />)}
      {/* 划掉的旧路径 */}
      {st.xpath && (() => {
        const xx = x0 + w * st.xl!, ya = mid(st.xpath[0]), yb = mid(st.xpath[1]), my = gc(4)
        return <g>
          <line x1={xx} y1={ya} x2={xx} y2={yb} stroke="var(--color-muted)" strokeWidth={2} strokeDasharray="5 4" />
          <path d={`M ${xx - 7} ${my - 7} L ${xx + 7} ${my + 7} M ${xx - 7} ${my + 7} L ${xx + 7} ${my - 7}`} stroke="var(--fk-red)" strokeWidth={2.2} />
          <text x={xx - 12} y={my} textAnchor="end" className="fk-small fk-muted" style={HALO}>{st.xhop}</text>
        </g>
      })()}
      {/* 路径 */}
      {hops.map((hp, i) => {
        const c = hp.hot ? 'var(--fk-red)' : col, next = hops[i + 1]
        return <g key={`h${i}`}>
          <line x1={hp.lx} y1={mid(hp.a)} x2={hp.lx} y2={mid(hp.b)} stroke={c} strokeWidth={hp.hot ? 3.4 : 2.4} />
          {next && <line x1={hp.lx - 1.2} y1={mid(hp.b)} x2={next.lx + 1.2} y2={mid(hp.b)} stroke={col} strokeWidth={2.4} />}
          <polygon points={poly(hp.up ? [[hp.lx, hp.edge], [hp.lx - 4.5, hp.edge + 8], [hp.lx + 4.5, hp.edge + 8]] : [[hp.lx, hp.edge], [hp.lx - 4.5, hp.edge - 8], [hp.lx + 4.5, hp.edge - 8]])} fill={c} />
        </g>
      })}
      {hops.map((hp, i) => <g key={`d${i}`}>
        {hops[i + 1]?.gy !== hp.gy && <rect x={hp.lx - 4} y={hp.gy - 4} width={8} height={8} rx={1} className={st.purple ? 'fk-fill-purple' : i % 2 ? 'fk-fill-amber' : 'fk-fill-blue'} stroke={PAGE} strokeWidth={1} />}
        {hp.label && <text x={hp.lx - 9} y={hp.gy} textAnchor="end" className="fk-small fk-mono" style={{ ...HALO, fill: hp.hot ? 'var(--fk-red)' : 'var(--color-text)', fontSize: compact ? 12 : 11 }}>{hp.label}</text>}
      </g>)}
      {tags.map(t => <BnTag key={t.text} x={xr} y={tagY(t)} text={t.text} />)}
      {/* warp 状态：4 个调度器 × 4 个 warp */}
      {st.warps && <g>
        <text x={wx} y={Y(3) - 8 + wdy} textAnchor="start" className="fk-muted">warp</text>
        {st.warps.split('').map((c, i) => <circle key={i} cx={wx + 6 + (i % 4) * 20} cy={Y(3) + 14 + wdy + Math.floor(i / 4) * 18} r={6} className={`fk-fill-${WARP_HUE[c]}`} />)}
      </g>}
  </>
}

// ---------------------------------------------------------------- 合并访存：在完整矩阵上看 warp 读哪些元素（替代 CoalesceFigure2）
export function CoalesceFigure3() {
  // 紧凑模式（520 宽）：两块缩窄并排，naive 的 k 标在左侧，给 naive_T 的线程点让位
  const compact = useCompact()
  const cs = compact ? 7.4 : 9, cols = 32, rows = 16, K = 5, my = compact ? 60 : 66, W = cols * cs, H = rows * cs
  const panel = (ox: number, T: boolean) => {
    const used: Pt[] = T ? Array.from({ length: rows }, (_, r) => [r, K]) : Array.from({ length: 16 }, (_, j) => [K, 8 + j])
    const sectors: Pt[] = T ? Array.from({ length: rows }, (_, r) => [r, 0]) : [[K, 8], [K, 16]]
    return <g>
      <text x={ox} y={16} textAnchor="start" className="fk-strong">{T ? 'naive_T' : 'naive'}</text>
      <text x={ox + W} y={16} textAnchor="end" className="fk-mono fk-muted">{T ? 'A[row][k]' : 'B[k][col]'}</text>
      <rect x={ox} y={my} width={W} height={H} className="fk-fill-free" />
      {Array.from({ length: cols - 1 }, (_, c) => <line key={`v${c}`} x1={ox + (c + 1) * cs} y1={my} x2={ox + (c + 1) * cs} y2={my + H} className="fk-rule" opacity={(c + 1) % 8 ? 0.45 : 1} strokeWidth={(c + 1) % 8 ? 0.6 : 1.4} />)}
      {Array.from({ length: rows - 1 }, (_, r) => <line key={`h${r}`} x1={ox} y1={my + (r + 1) * cs} x2={ox + W} y2={my + (r + 1) * cs} className="fk-rule" opacity={0.45} strokeWidth={0.6} />)}
      {sectors.map(([r, c0]) => <rect key={`s${r}${c0}`} x={ox + c0 * cs} y={my + r * cs} width={8 * cs} height={cs} className="fk-fill-amber" opacity={0.25} />)}
      {used.map(([r, c]) => <rect key={`u${r}${c}`} x={ox + c * cs + 0.5} y={my + r * cs + 0.5} width={cs - 1} height={cs - 1} className="fk-fill-amber" />)}
      {sectors.map(([r, c0]) => <rect key={`o${r}${c0}`} x={ox + c0 * cs} y={my + r * cs} width={8 * cs} height={cs} fill="none" stroke="var(--fk-amber)" strokeWidth={1.3} />)}
      {/* warp 的线程：naive 沿 col 排开，naive_T 沿 row 排开 */}
      {T
        ? Array.from({ length: rows }, (_, r) => <circle key={`t${r}`} cx={ox - 12} cy={my + r * cs + cs / 2} r={compact ? 2.6 : 3} className="fk-fill-blue" />)
        : Array.from({ length: 16 }, (_, j) => <circle key={`t${j}`} cx={ox + (8 + j) * cs + cs / 2} cy={my - 12} r={compact ? 2.6 : 3} className="fk-fill-blue" />)}
      {T
        ? <text x={ox - 12} y={my - 14} className="fk-small fk-text-blue">warp</text>
        : <text x={ox + 8 * cs - 8} y={my - 12} textAnchor="end" className="fk-small fk-text-blue">warp</text>}
      {T
        ? <text x={ox + K * cs + cs / 2} y={my - 12} className="fk-muted fk-small">k</text>
        : <text x={compact ? ox - 6 : ox + W + 8} y={my + K * cs + cs / 2} textAnchor={compact ? 'end' : 'start'} className="fk-muted fk-small">k</text>}
      <text x={ox} y={my + H + 16} textAnchor="start" className="fk-muted">{T ? 'A' : 'B'}（行主序）</text>
      <text x={ox + W} y={my + H + 16} textAnchor="end" className="fk-muted fk-small">地址 →</text>
      <text x={ox + W / 2} y={my + H + 40} className="fk-strong">{T ? '16 个扇区' : '2 个扇区'}</text>
      <text x={ox + W / 2} y={my + H + 58} className="fk-muted">用到 64 / {T ? 512 : 64} B</text>
    </g>
  }
  return <Figure title="一个 warp 的读取在矩阵上的位置" desc="naive 的 warp 读 B 第 k 行的 16 个连续 float，落在 2 个 32 字节扇区；naive_T 的 warp 读 A 第 k 列的 16 个数，每个在不同的行，各占一个扇区、只用 4 字节。" width={compact ? 520 : 760} height={my + H + 68}
    caption="每格一个 float，8 格一个 32 字节扇区（粗竖线分隔）；深橙为 warp 用到的 float，浅橙为同扇区一并读出的部分。">
    {() => <>{panel(compact ? 16 : 40, false)}{panel(compact ? 280 : 440, true)}</>}
  </Figure>
}

// ---------------------------------------------------------------- 共享内存分块：显存中的 A 带、B 带与 C 块，片上的 As、Bs（替代 SmemTileFigure）
export function SmemFigure3D() {
  const S = 4, g = 0.7, o = S + g, BI = 1, BJ = 2, K0 = 2, Z = 3.3, th = 0.14
  // 紧凑模式（520 宽）：同一构图按 0.8 倍缩小
  const compact = useCompact()
  const p = compact ? iso(258, 114, 24) : iso(380, 142, 30)
  type T = { x: number; y: number; cls: string; op: number; k0?: boolean; c?: boolean }
  const tiles: T[] = []
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) {
    tiles.push({ x: c, y: o + r, cls: r === BI ? 'fk-fill-amber' : 'fk-fill-gray', op: r === BI ? (c === K0 ? 1 : 0.32) : 1, k0: r === BI && c === K0 })   // A：行 r、K 块 c
    tiles.push({ x: o + c, y: r, cls: c === BJ ? 'fk-fill-amber' : 'fk-fill-gray', op: c === BJ ? (r === K0 ? 1 : 0.32) : 1, k0: c === BJ && r === K0 })  // B：K 块 r、列 c
    const on = r === BI && c === BJ
    tiles.push({ x: o + c, y: o + r, cls: on ? 'fk-fill-blue' : 'fk-fill-gray', op: 1, c: on })                                                          // C
  }
  tiles.sort((a, b) => a.x + a.y - b.x - b.y)
  const As = { x: 0.35, y: 2.05 }, Bs = { x: 2.05, y: 0.35 }, sz = 1.6, bh = 0.5, pz = Z + 0.16
  const lbl = (pt: Pt, s: string, cls: string, anchor: 'start' | 'end' | 'middle' = 'middle') => <text x={pt[0]} y={pt[1]} textAnchor={anchor} className={cls}>{s}</text>
  return <Figure title="共享内存分块" desc="显存中的 A、B、C；block 负责 C 的一块，需要 A 的一条带与 B 的一条带。沿 K 每一步把两条带的第 k0 块搬进 SM 片上的共享内存 As、Bs（图为第 2 步）。" width={compact ? 520 : 760} height={compact ? 352 : 440}
    caption="下层为显存中的 A、B、C（各 4 × 4 块），浅橙为 block 要读的 A 带、B 带；沿 K 每步把一对小块搬进片上的 As、Bs（图为 k0 = 2）。">
    {() => <>
      {tiles.map((t, i) => <IsoCuboid key={i} p={p} x={t.x + 0.04} y={t.y + 0.04} sx={0.92} sy={0.92} h={th} cls={t.cls} op={t.op} />)}
      {tiles.filter(t => t.k0 || t.c).map((t, i) => { const [x, y] = p(t.x + 0.5, t.y + 0.5, th); return <text key={`t${i}`} x={x} y={y} className={`fk-cell-text fk-cell-small ${t.k0 ? 'fk-on-amber' : ''}`}>{t.c ? 'C 块' : 'k0'}</text> })}
      {/* 片上：一个 SM 的共享内存 */}
      <IsoCuboid p={p} x={0} y={0} z={Z} sx={S} sy={S} h={0.16} cls="fk-fill-green" op={0.18} stroke="var(--fk-green)" />
      <Arrow a={p(K0 + 0.5, o + BI + 0.5, th)} b={p(As.x + sz / 2, As.y + sz / 2, pz)} color="var(--fk-amber)" w={1.8} dash="5 3" head={8} />
      <Arrow a={p(o + BJ + 0.5, K0 + 0.5, th)} b={p(Bs.x + sz / 2, Bs.y + sz / 2, pz)} color="var(--fk-amber)" w={1.8} dash="5 3" head={8} />
      <IsoCuboid p={p} x={As.x} y={As.y} z={pz} sx={sz} sy={sz} h={bh} cls="fk-fill-green" />
      <IsoCuboid p={p} x={Bs.x} y={Bs.y} z={pz} sx={sz} sy={sz} h={bh} cls="fk-fill-green" />
      {lbl(p(As.x + sz / 2, As.y + sz / 2, pz + bh), 'As', 'fk-cell-text')}
      {lbl(p(Bs.x + sz / 2, Bs.y + sz / 2, pz + bh), 'Bs', 'fk-cell-text')}
      {(() => { const [x, y] = p(0, S, Z); return <text x={x - 12} y={y} textAnchor="end" className="fk-strong" style={{ fill: 'var(--fk-green)' }}>共享内存（片上）</text> })()}
      {(() => { const [x, y] = p(0, o + S); return <><text x={x - 12} y={y - 10} textAnchor="end" className="fk-strong">显存</text><text x={x - 12} y={y + 8} textAnchor="end" className="fk-muted">A</text></> })()}
      {(() => { const [x, y] = p(o + S, 0); return <text x={x + 12} y={y} textAnchor="start" className="fk-muted">B</text> })()}
      {(() => { const [x, y] = p(o + S, o + S); return <text x={x} y={y + 18} className="fk-muted">C</text> })()}
      {(() => { const [x, y] = p(-0.3, o + 2.2); return <text x={x - 6} y={y} textAnchor="end" className="fk-muted fk-small">A 带</text> })()}
      {(() => { const [x, y] = p(o + 2.6, -0.3); return <text x={x + 6} y={y} textAnchor="start" className="fk-muted fk-small">B 带</text> })()}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 寄存器分块的外积（立体，替代 RegTileFigure2）
export function RegTileFigure3D() {
  // 紧凑模式（520 宽）：同一构图按 0.64 倍缩小
  const compact = useCompact()
  const KK = 3, I = 2, J = 3, p = compact ? iso(260, 100, 13.5) : iso(380, 140, 21)
  const order = (n: number, m: number) => Array.from({ length: n * m }, (_, i) => [i % n, Math.floor(i / n)] as const).sort((a, b) => a[0] + a[1] - b[0] - b[1])
  const ax = -10.5, rr = -1.5
  return <Figure title="寄存器分块的外积" desc="一个线程负责 C 的 8 × 8。沿 K 每走一步，从 As 读一列 8 个数到 rA、从 Bs 读一行 8 个数到 rB，外积累加到 8 × 8 个寄存器 acc。" width={compact ? 520 : 760} height={compact ? 240 : 340}
    caption="一个线程沿 K 走第 k 步：As 的第 k 列进 rA，Bs 的第 k 行进 rB，外积累加到 64 个 acc；深蓝为 acc[2][3] += rA[2] × rB[3]。">
    {() => <>
      {/* As：本线程的 8 行 × BK；Bs：BK × 本线程的 8 列 */}
      {order(8, 8).map(([c, r]) => <IsoCuboid key={`a${c}${r}`} p={p} x={ax + c + 0.05} y={r + 0.05} sx={0.9} sy={0.9} h={0.15} cls={c === KK ? 'fk-fill-green' : 'fk-fill-gray'} />)}
      {order(8, 8).map(([c, r]) => <IsoCuboid key={`b${c}${r}`} p={p} x={c + 0.05} y={ax + r + 0.05} sx={0.9} sy={0.9} h={0.15} cls={r === KK ? 'fk-fill-green' : 'fk-fill-gray'} />)}
      <Arrow a={p(ax + KK + 1, 8.7, 0.15)} b={p(rr, 8.7, 0.15)} color="var(--color-muted)" w={1.3} />
      <Arrow a={p(8.7, ax + KK + 1, 0.15)} b={p(8.7, rr, 0.15)} color="var(--color-muted)" w={1.3} />
      {/* rA、rB 与 acc */}
      {Array.from({ length: 8 }, (_, i) => <IsoCuboid key={`ra${i}`} p={p} x={rr + 0.08} y={i + 0.08} sx={0.84} sy={0.84} h={0.6} cls="fk-fill-purple" op={i === I ? 1 : 0.55} />)}
      {Array.from({ length: 8 }, (_, j) => <IsoCuboid key={`rb${j}`} p={p} x={j + 0.08} y={rr + 0.08} sx={0.84} sy={0.84} h={0.6} cls="fk-fill-purple" op={j === J ? 1 : 0.55} />)}
      {order(8, 8).map(([j, i]) => <IsoCuboid key={`c${j}${i}`} p={p} x={j + 0.08} y={i + 0.08} sx={0.84} sy={0.84} h={0.6} cls="fk-fill-blue" op={i === I && j === J ? 1 : 0.35} />)}
      <line x1={p(rr + 0.5, I + 0.5, 0.6)[0]} y1={p(rr + 0.5, I + 0.5, 0.6)[1]} x2={p(J + 0.5, I + 0.5, 0.6)[0]} y2={p(J + 0.5, I + 0.5, 0.6)[1]} stroke="var(--fk-purple)" strokeWidth={1.4} strokeDasharray="3 3" />
      <line x1={p(J + 0.5, rr + 0.5, 0.6)[0]} y1={p(J + 0.5, rr + 0.5, 0.6)[1]} x2={p(J + 0.5, I + 0.5, 0.6)[0]} y2={p(J + 0.5, I + 0.5, 0.6)[1]} stroke="var(--fk-purple)" strokeWidth={1.4} strokeDasharray="3 3" />
      {(() => { const [x, y] = p(ax, 4); return <text x={x - 10} y={y} textAnchor="end" className="fk-muted">As</text> })()}
      {(() => { const [x, y] = p(4, ax); return <text x={x + 10} y={y} textAnchor="start" className="fk-muted">Bs</text> })()}
      {(() => { const [x, y] = p(ax + KK + 0.5, -0.2, 0.15); return <text x={x + 4} y={y - 8} className="fk-muted fk-small">k</text> })()}
      {(() => { const [x, y] = p(-0.2, ax + KK + 0.5, 0.15); return <text x={x - 4} y={y - 8} className="fk-muted fk-small">k</text> })()}
      {(() => { const [x, y] = p(rr + 0.5, -0.2, 0.6); return <text x={x - 6} y={y - 8} textAnchor="end" className="fk-strong" style={{ fill: 'var(--fk-purple)' }}>rA</text> })()}
      {(() => { const [x, y] = p(-0.2, rr + 0.5, 0.6); return <text x={x + 6} y={y - 8} textAnchor="start" className="fk-strong" style={{ fill: 'var(--fk-purple)' }}>rB</text> })()}
      {(() => { const [x, y] = p(8, 8); return <text x={x} y={y + 18} className="fk-strong fk-text-blue">acc 8 × 8</text> })()}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- warp 分块（立体，替代 WarptileFigure）
export function WarptileFigure3D() {
  // 紧凑模式（520 宽）：上方立体图按 0.73 倍缩小，下方两组 bank 上下排列
  const compact = useCompact()
  const p = compact ? iso(260, 50, 22) : iso(380, 70, 30), LIFT = 1.1, WR = 0, WC = 1
  const wx = 2 * WC, wy = 4 * WR, top = LIFT + 0.3
  const warps = Array.from({ length: 8 }, (_, i) => [i % 4, Math.floor(i / 4)] as const).filter(([c, r]) => !(c === WC && r === WR)).sort((a, b) => a[0] * 2 + a[1] * 4 - b[0] * 2 - b[1] * 4)
  // lane 6：lRow 1、lCol 2 → 行 mi*32 + 4、列 ni*16 + 8（单位 16 个元素）
  const subs = [0, 1].flatMap(mi => [0, 1].map(ni => [wx + (ni * 16 + 8) / 16, wy + (mi * 32 + 4) / 16] as Pt))
  const bw = compact ? 14 : 10.5
  const B1: Pt = compact ? [30, 272] : [30, 362], B2: Pt = compact ? [30, 372] : [410, 362]
  const banks = (sx: number, y: number, title: string, items: [number, number, string][]) => <g>
    <text x={sx} y={y} textAnchor="start" className="fk-strong">{title}</text>
    {Array.from({ length: 33 }, (_, b) => <line key={b} x1={sx + b * bw} y1={y + 30} x2={sx + b * bw} y2={y + 34} className="fk-rule" />)}
    {[0, 8, 16, 24].map(b => <text key={b} x={sx + b * bw + 2} y={y + 22} textAnchor="start" className="fk-muted fk-small">bank {b}</text>)}
    {items.map(([b0, wave, label], i) => <Cell key={i} x={sx + b0 * bw} y={y + 36 + wave * 26} w={4 * bw - 1} h={22} rx={2} fill={wave ? 'red' : 'green'} label={label} small />)}
  </g>
  return <Figure title="warp 分块与 bank 冲突" desc="上：C 块 128 × 128 分给 8 个 warp，各 64 × 32；抬起的 warp 中 lane 6 计算 4 个相隔的 4 × 4 小块，对应 As 的两段行与 Bs 的两段列。下：一组 8 个线程读 Bs，vec 需两轮，warptile 一轮。" width={compact ? 520 : 760} height={compact ? 440 : 472}
    caption="上为 C 块的 8 个 warp 与 As、Bs，深蓝为 lane 6 的 4 个小块；下为 8 个线程读 Bs 占用的 bank，红色须第二轮。">
    {() => <>
      {/* As（左）与 Bs（右）：lane 6 用到的行、列 */}
      <IsoCuboid p={p} x={-1.6} y={0} sx={1.1} sy={8} h={0.2} cls="fk-fill-gray" />
      <IsoCuboid p={p} x={0} y={-1.6} sx={8} sy={1.1} h={0.2} cls="fk-fill-gray" />
      <polygon points={isoFace(p, -1.6, wy, 1.1, 4, 0.2)} className="fk-fill-green" opacity={0.3} />
      <polygon points={isoFace(p, wx, -1.6, 2, 1.1, 0.2)} className="fk-fill-green" opacity={0.3} />
      {[0, 1].map(mi => <polygon key={`ar${mi}`} points={isoFace(p, -1.6, wy + (mi * 32 + 4) / 16, 1.1, 0.25, 0.2)} className="fk-fill-blue" />)}
      {[0, 1].map(ni => <polygon key={`bc${ni}`} points={isoFace(p, wx + (ni * 16 + 8) / 16, -1.6, 0.25, 1.1, 0.2)} className="fk-fill-blue" />)}
      {/* 其余 7 个 warp */}
      {warps.map(([c, r]) => <IsoCuboid key={`w${c}${r}`} p={p} x={2 * c + 0.05} y={4 * r + 0.05} sx={1.9} sy={3.9} h={0.3} cls="fk-fill-gray" />)}
      {/* 抬起的 warp：投影虚线 + 本体 */}
      {[[wx, wy], [wx + 2, wy], [wx + 2, wy + 4], [wx, wy + 4]].map(([x, y], i) => <line key={`pj${i}`} x1={p(x, y, 0)[0]} y1={p(x, y, 0)[1]} x2={p(x, y, LIFT)[0]} y2={p(x, y, LIFT)[1]} className="fk-line-blue" strokeDasharray="3 3" strokeWidth={1} opacity={0.6} />)}
      <polygon points={isoFace(p, wx, wy, 2, 4, 0)} className="fk-fill-blue" opacity={0.12} />
      <IsoCuboid p={p} x={wx + 0.05} y={wy + 0.05} z={LIFT} sx={1.9} sy={3.9} h={0.3} cls="fk-fill-blue" op={0.25} />
      <polygon points={isoFace(p, wx, wy, 1, 2, top)} className="fk-fill-blue" opacity={0.25} />
      {subs.map(([x, y], i) => <polygon key={`s${i}`} points={isoFace(p, x, y, 0.25, 0.25, top)} className="fk-fill-blue" />)}
      {(() => { const [x, y] = p(wx, wy, top + 0.3); return <text x={x} y={y - 14}><tspan className="fk-strong fk-text-blue">warp (0, 1)</tspan><tspan dx={8} className="fk-muted">64 × 32</tspan></text> })()}
      {(() => { const [x, y] = p(subs[3][0] + 0.25, subs[3][1] + 0.25, top); return <><line x1={x} y1={y} x2={x + 60} y2={y + 24} className="fk-line-blue" strokeWidth={1} /><text x={x + 64} y={y + 24} textAnchor="start" className="fk-small fk-text-blue">lane 6</text></> })()}
      {(() => { const [x, y] = p(-1.6, 6); return <text x={x - 8} y={y} textAnchor="end" className="fk-muted">As</text> })()}
      {(() => { const [x, y] = p(6, -1.6); return <text x={x + 8} y={y} textAnchor="start" className="fk-muted">Bs</text> })()}
      {(() => { const [x, y] = p(-1.6, wy + 1.4); return <text x={x - 8} y={compact ? y - 8 : y} textAnchor="end" className="fk-muted fk-small" style={compact ? HALO : undefined}>行隔 32</text> })()}
      {(() => { const [x, y] = p(4.6, -1.6, 0.2); return <text x={x + 4} y={y - 8} textAnchor="start" className="fk-muted fk-small">列隔 16</text> })()}
      {(() => { const [x, y] = p(8, 8); return <text x={x} y={y + 18} className="fk-muted">C 块 128 × 128</text> })()}
      {banks(B1[0], B1[1], 'vec：8 个线程读 rB，2 轮', [[0, 0, 'L0'], [8, 0, 'L1'], [16, 0, 'L2'], [24, 0, 'L3'], [0, 1, 'L4'], [8, 1, 'L5'], [16, 1, 'L6'], [24, 1, 'L7']])}
      {banks(B2[0], B2[1], 'warptile：8 个线程读 rB，1 轮', [[0, 0, 'L0 L4'], [4, 0, 'L1 L5'], [8, 0, 'L2 L6'], [12, 0, 'L3 L7']])}
      <text x={B2[0] + 16 * bw + 10} y={B2[1] + 47} textAnchor="start" className="fk-muted fk-small">同一地址广播</text>
    </>}
  </Figure>
}

// ---------------------------------------------------------------- warp 特化（硬件塔画法，替代 WsFigure）
export function WsFigure2() {
  const x0 = 236, w = 290, tcY = 44, smY = 124, smH = 98, l2Y = 274, h = 34
  const sw = 58, sg = 10, sx0 = x0 + 14, slotX = (i: number) => sx0 + i * (sw + sg)
  const SLOT: [string, Fill, boolean, boolean][] = [['空', 'free', false, true], ['计算中', 'green', true, false], ['已到', 'green', true, false], ['写入中', 'amber', false, false]]
  const tx = slotX(3) + sw / 2, cx = slotX(1) + sw / 2, cy = tcY + h + 14, py = smY + smH + 8, mid = (a: number, b: number) => (a + b) / 2
  return <Figure title="warp specialization：producer、stage 与 consumer" desc="producer warpgroup 的一个线程发 TMA，把 A、W 块从 L2 写入 shared memory 的 4 个 stage；每个 stage 有 full、empty 两个 mbarrier；两个 consumer warpgroup 发 wgmma，Tensor Core 直接读 stage。" height={322}
    caption="producer 为 warpgroup 0，两个 consumer 为 warpgroup 1、2。full：TMA 已到齐；empty：consumer 已用完。">
    {() => <>
      <rect x={18} y={20} width={724} height={smY + smH + 14 - 20} rx={6} fill="none" stroke="var(--color-muted)" strokeDasharray="5 4" />
      <text x={18} y={10} textAnchor="start" className="fk-muted">一个 SM</text>
      <Slab x={x0} y={tcY} w={w} h={h} tone="on" />
      <rect x={x0 + 8} y={tcY + 9} width={4} height={h - 18} rx={1} className="fk-fill-teal" />
      <text x={x0 + 20} y={tcY + h / 2} textAnchor="start" className="fk-strong">Tensor Core</text>
      <Slab x={x0} y={smY} w={w} h={smH} tone="on" />
      <rect x={x0 + 8} y={smY + 9} width={4} height={14} rx={1} className="fk-fill-green" />
      <text x={x0 + 20} y={smY + 16} textAnchor="start" className="fk-strong">shared memory</text>
      {SLOT.map(([s, fill, full, empty], i) => <g key={i}>
        <Cell x={slotX(i)} y={smY + 28} w={sw} h={40} rx={3} fill={fill} label={`stage ${i}`} sub={s} />
        <Cell x={slotX(i)} y={smY + 74} w={sw / 2 - 1} h={15} rx={2} fill={full ? 'green' : 'free'} label="full" fs={9} />
        <Cell x={slotX(i) + sw / 2 + 1} y={smY + 74} w={sw / 2 - 1} h={15} rx={2} fill={empty ? 'blue' : 'free'} label="empty" fs={9} />
      </g>)}
      <Slab x={x0} y={l2Y} w={w} h={h} />
      <rect x={x0 + 8} y={l2Y + 9} width={4} height={h - 18} rx={1} className="fk-fill-amber" />
      <text x={x0 + 20} y={l2Y + h / 2} textAnchor="start" className="fk-strong">L2</text>
      {/* 数据：L2 → 槽 2（TMA），槽 0 → Tensor Core（wgmma） */}
      <Arrow a={[tx, l2Y + h / 2]} b={[tx, smY + smH]} color="var(--fk-green)" w={2.4} />
      <Arrow a={[cx, smY + 28]} b={[cx, tcY + h]} color="var(--fk-green)" w={2.4} />
      <text x={tx + 8} y={mid(smY + smH + 14, l2Y - 9)} textAnchor="start" className="fk-small fk-mono">TMA</text>
      <text x={cx + 8} y={cy + 14} textAnchor="start" className="fk-small fk-mono">wgmma</text>
      {/* producer warpgroup：线程 0 发 TMA */}
      <rect x={34} y={smY} width={176} height={smH} rx={8} className="fk-band-blue" />
      <text x={46} y={smY + 16} textAnchor="start" className="fk-strong fk-text-blue">producer</text>
      <text x={198} y={smY + 16} textAnchor="end" className="fk-muted fk-small">warpgroup 0</text>
      {Array.from({ length: 64 }, (_, i) => <circle key={i} cx={52 + (i % 16) * 9.4} cy={smY + 40 + Math.floor(i / 16) * 11} r={2.6} className="fk-fill-blue" opacity={i === 0 ? 1 : 0.28} />)}
      <text x={52} y={smY + 88} textAnchor="start" className="fk-muted fk-small">线程 0</text>
      <path d={`M 48 ${smY + 40} H 40 V ${py} H ${tx - 3}`} fill="none" className="fk-line-blue" strokeWidth={1.4} strokeDasharray="4 3" />
      {/* 两个 consumer warpgroup */}
      {[0, 1].map(j => {
        const y = 32 + j * 96
        return <g key={j}>
          <rect x={574} y={y} width={160} height={80} rx={8} className="fk-band-teal" />
          <text x={586} y={y + 16} textAnchor="start" className="fk-strong fk-text-teal">consumer {j + 1}</text>
          {Array.from({ length: 64 }, (_, i) => <circle key={i} cx={590 + (i % 16) * 8.6} cy={y + 36 + Math.floor(i / 16) * 10} r={2.4} className="fk-fill-teal" opacity={0.75} />)}
          <path d={`M 574 ${y + 40} H 556 V ${cy}`} fill="none" className="fk-line-teal" strokeWidth={1.4} strokeDasharray="4 3" />
        </g>
      })}
      <path d={`M 556 ${cy} H ${cx + 3}`} fill="none" className="fk-line-teal" strokeWidth={1.4} strokeDasharray="4 3" />
    </>}
  </Figure>
}

// ---------------------------------------------------------------- L2 读取与 cluster 多播（硬件塔画法，替代 ClusterFigure2）
export function ClusterFigure3() {
  const panel = (ox: number, cl: boolean) => {
    const by = 62, bh = 92, bw = 160, b0 = ox + 6, b1 = ox + 180, ly = 222, lh = 44
    const block = (bx: number, r: number) => <g>
      <Slab x={bx} y={by} w={bw} h={bh} tone="on" />
      <text x={bx + 10} y={by + 14} textAnchor="start" className="fk-small fk-strong">block {r} · SM {r}</text>
      <Cell x={bx + 10} y={by + 28} w={28} h={54} rx={2} fill="gray" label="A" small />
      {cl ? <>
        <Cell x={bx + 44} y={by + 28} w={52} h={54} rx={2} fill="blue" label="W 前半" small />
        <Cell x={bx + 100} y={by + 28} w={52} h={54} rx={2} fill="purple" label="W 后半" small />
      </> : <Cell x={bx + 44} y={by + 28} w={108} h={54} rx={2} fill="blue" label="W" small />}
    </g>
    const wc = (bx: number, half: 0 | 1 | -1) => [half < 0 ? bx + 98 : bx + 70 + half * 56, by + 82] as Pt
    return <g>
      <text x={ox + 170} y={14} className="fk-strong">{cl ? '2 个 block 的 cluster' : '无 cluster'}</text>
      {cl && <><rect x={ox} y={by - 34} width={360} height={bh + 44} rx={6} fill="none" stroke="var(--color-muted)" strokeDasharray="5 4" /><text x={ox + 8} y={by - 22} textAnchor="start" className="fk-muted fk-small">cluster：相邻的两个 SM</text></>}
      {block(b0, 0)}{block(b1, 1)}
      <Slab x={ox + 6} y={ly} w={334} h={lh} />
      <rect x={ox + 14} y={ly + 9} width={4} height={lh - 18} rx={1} className="fk-fill-amber" />
      <text x={ox + 26} y={ly + lh / 2} textAnchor="start" className="fk-strong">L2</text>
      {cl ? <>
        <Cell x={ox + 120} y={ly + 8} w={56} h={28} rx={2} fill="blue" label="W 前半" small />
        <Cell x={ox + 182} y={ly + 8} w={56} h={28} rx={2} fill="purple" label="W 后半" small />
        {[b0, b1].map((bx, i) => <Arrow key={`f${i}`} a={[ox + 148, ly + 8]} b={wc(bx, 0)} color="var(--fk-blue)" w={1.8} />)}
        {[b0, b1].map((bx, i) => <Arrow key={`b${i}`} a={[ox + 210, ly + 8]} b={wc(bx, 1)} color="var(--fk-purple)" w={1.8} />)}
      </> : <>
        <Cell x={ox + 130} y={ly + 8} w={96} h={28} rx={2} fill="blue" label="W" small />
        {[b0, b1].map((bx, i) => <Arrow key={i} a={[ox + 178, ly + 8]} b={wc(bx, -1)} color="var(--fk-blue)" w={1.8} />)}
      </>}
      <text x={ox + 336} y={ly + lh / 2} textAnchor="end" className="fk-muted fk-small">{cl ? '读出 1 份 W' : '读出 2 份 W'}</text>
    </g>
  }
  return <Figure title="L2 读取与 cluster 多播" desc="上下相邻的两个 tile 用同一块 W。无 cluster 时两个 block 各从 L2 读整块 W；2 个 block 的 cluster 中各读一半，用 TMA multicast 写入两个 block 的 shared memory。" height={286}
    caption="左：两个 block 各读整块 W；右：各读一半并多播。">
    {() => <>
      {panel(14, false)}
      <line x1={390} y1={4} x2={390} y2={280} className="fk-rule" />
      {panel(398, true)}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- warp specialization 在 8192 阶的瓶颈：W 放不下 L2（构图同视频第 6 幕结尾）
export function WsL2Figure() {
  const st: TowerV = { title: '', desc: '', caption: '', tc: true, path: [4, 3, 2, 0], hops: ['', 'TMA', 'wgmma'], sub: { 1: '累加结果' },
    hot: { hop: [0], lv: [3] }, tag: { hop: 0, text: 'W 反复从显存读入' } }
  const top = 70, H = top + 4 * TOWER_GAP + TOWER_H + 18
  const PX = 486, k = 250 / 136, L2 = 50 * k
  const row = (y: number, mb: number, title: string, note: string) => <g>
    <text x={PX} y={y - 14} textAnchor="start" className="fk-small fk-strong">{title}</text>
    <rect x={PX} y={y} width={Math.min(mb, 50) * k} height={18} rx={2} className="fk-fill-green" opacity={0.75} />
    {mb > 50 && <rect x={PX + L2} y={y} width={(mb - 50) * k} height={18} rx={2} className="fk-fill-red" opacity={0.75} />}
    <text x={PX + mb * k + 6} y={y + 9} textAnchor="start" className="fk-small" style={{ fill: mb > 50 ? 'var(--fk-red)' : 'var(--fk-green)', fontWeight: 600 }}>{note}</text>
  </g>
  return <Figure title="8192 阶时 W 放不下 L2" desc="4096 阶的 W 为 32 MB，放得进 50 MB 的 L2；8192 阶为 128 MB，按行发放时一轮 132 个 tile 需要 136 MB，W 反复从显存读入。" height={H}
    caption="右：条长按数据量，虚线为 L2 的 50 MB。">
    {() => <>
      <TowerBody st={st} x0={40} w={300} top={top} stack />
      <line x1={PX + L2} y1={96} x2={PX + L2} y2={330} stroke="var(--color-muted)" strokeDasharray="4 3" />
      <text x={PX + L2} y={86} className="fk-small fk-muted">L2 50 MB</text>
      {row(140, 32, '4096 阶的 W', '32 MB')}
      {row(206, 128, '8192 阶的 W', '128 MB')}
      {row(272, 136, '8192 阶按行发放，一轮 132 个 tile', '136 MB')}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 持续满载：700 W 功耗墙（构图同视频第 10 幕）
export function PowerFigure() {
  const X = 230, k = 440 / 989, P = 743
  const rows: [string, string, number, string, string][] = [
    ['规格峰值', '1830 MHz', 989, 'fk-fill-gray', ''],
    ['按持续时钟的峰值', '约 1375 MHz', P, 'fk-fill-gray', ''],
    ['最终版，持续 2 秒', '1375 MHz · 700 W', 691.48, 'fk-fill-teal', '93%'],
    ['cuBLAS BF16，持续 2 秒', '1377 MHz · 697 W', 688.85, 'fk-fill-int', '93%'],
  ]
  return <Figure title="持续满载时的功耗墙" desc="8192 阶连续运行 2 秒，手写版与 cuBLAS 都达到 700 W，时钟降到约 1375 MHz；按这一时钟的峰值为 743 TFLOPS，两者都用到 93%。" height={196}
    caption="8192 阶，TFLOPS。">
    {() => <>
      {rows.map(([t, sub, v, c, pct], i) => {
        const y = 14 + i * 44
        return <g key={t}>
          <text x={X - 12} y={y + 8} textAnchor="end" className="fk-small fk-strong">{t}</text>
          <text x={X - 12} y={y + 24} textAnchor="end" className="fk-small fk-muted">{sub}</text>
          <rect x={X} y={y} width={v * k} height={30} rx={3} className={c} />
          <text x={X + v * k + 8} y={y + 15} textAnchor="start" className="fk-strong">{v}</text>
          {pct && <text x={X + v * k - 8} y={y + 15} textAnchor="end" className="fk-cell-text">{pct}</text>}
        </g>
      })}
      <line x1={X + P * k} y1={56} x2={X + P * k} y2={190} stroke="var(--color-muted)" strokeDasharray="4 3" />
    </>}
  </Figure>
}

// ---------------------------------------------------------------- 总结：每一层去掉的瓶颈（构图同视频结尾的硬件塔）
export function SummaryFigure() {
  const st: TowerV = { title: '', desc: '', caption: '', tc: true, path: [4, 0], hops: [''], lanes: [0.82, 0.82] }
  const top = 46, x0 = 40, w = 260, tx = 372
  const notes: [number, string, string][] = [
    [0, '3 wgmma', '指令更大，操作数直接从 shared memory 读'],
    [1, '2 cp.async；7 TMA store', '搬入与写回都不经寄存器中转'],
    [2, '1 swizzle；2 pipeline；4 warp specialization', '读取不冲突；搬运与计算重叠'],
    [3, '5 tile 分组；6 cluster 多播', '共用的 W 留在 L2；相邻 SM 只读一次'],
    [4, 'shared memory 分块（上篇）', '读进 shared memory 的数被复用多次'],
  ]
  return <Figure title="各层去掉的等待" desc="从显存到 Tensor Core 的五层，每层旁注明消除该层等待的步骤，数字为本文七步的编号。" height={top + 4 * TOWER_GAP + TOWER_H + 18}
    caption="">
    {() => <>
      <TowerBody st={st} x0={x0} w={w} top={top} />
      {notes.map(([lv, a, b]) => {
        const y = top + lv * TOWER_GAP + TOWER_H / 2
        return <g key={lv}>
          <line x1={x0 + w + 24} y1={y} x2={tx - 8} y2={y} stroke="var(--color-border)" />
          <text x={tx} y={y - 10} textAnchor="start" className="fk-strong">{a}</text>
          <text x={tx} y={y + 10} textAnchor="start" className="fk-muted">{b}</text>
        </g>
      })}
    </>}
  </Figure>
}

// ---------------------------------------------------------------- Hopper 的线程层级：上篇的四层之间新增 warpgroup 与 cluster
export function HierarchyHopperFigure({ focus }: { focus: 'warpgroup' | 'cluster' }) {
  const cx = (i: number) => 64 + i * 126, top = 34, mid = 92
  const isNew = (i: number) => i === 2 || i === 4
  const dim = (i: number) => focus === 'warpgroup' && i === 4 ? 0.3 : 1
  const hot = (i: number) => (focus === 'warpgroup' && i === 2) || (focus === 'cluster' && i === 4)
  const cols: [string, string, string][] = [
    ['thread', '一个线程', ''],
    ['warp', '32 个线程', 'SIMT 同步执行'],
    ['warpgroup', '4 个 warp，128 个线程', '共同发出 wgmma'],
    ['block（CTA）', '驻留在一个 SM', '共用 shared memory'],
    ['cluster', '同一 GPC 的几个 block', '可访问彼此的 shared memory'],
    ['grid', '一次 kernel 启动', ''],
  ]
  const mult = ['× 32', '× 4', '× 1 或 3', '× 2', '× 多个']
  const warpDots = (x: number, y: number, op = 1) => Array.from({ length: 32 }, (_, k) =>
    <circle key={k} cx={x + (k % 8) * 5.5} cy={y + Math.floor(k / 8) * 5.5} r={1.8} className="fk-fill-blue" opacity={op} />)
  return <Figure title="Hopper 的线程层级" desc="thread、warp、block、grid 四层之外，Hopper 新增 warpgroup（4 个 warp）与 cluster（同一 GPC 的几个 block）。" height={226}
    caption="青色为 Hopper 新增的两层；本文的 warp specialization 版每个 block 有 3 个 warpgroup，cluster 版每个 cluster 有 2 个 block。">
    {() => <>
      {cols.map(([name, a, b], i) => {
        const x = cx(i), op = dim(i), col = isNew(i) ? 'var(--fk-teal)' : 'var(--fk-blue)'
        return <g key={name} opacity={op}>
          <text x={x} y={top - 18} className="fk-strong" style={{ fill: isNew(i) ? 'var(--fk-teal)' : undefined }}>{name}</text>
          {isNew(i) && <g><rect x={x - 34} y={top - 8} width={68} height={16} rx={8} className="fk-fill-teal" opacity={0.18} />
            <text x={x} y={top} style={{ fill: 'var(--fk-teal)', fontSize: 10.5, fontWeight: 600 }}>Hopper 新增</text></g>}
          <text x={x} y={mid + 70} className="fk-small">{a}</text>
          {b && <text x={x} y={mid + 88} className="fk-small fk-muted">{b}</text>}
          {hot(i) && <rect x={x - 56} y={mid - 44} width={112} height={88} rx={8} fill="none" stroke={col} strokeWidth={2} />}
        </g>
      })}
      {/* thread */}
      <circle cx={cx(0)} cy={mid} r={5} className="fk-fill-blue" />
      {/* warp */}
      <g>{warpDots(cx(1) - 19, mid - 8)}</g>
      {/* warpgroup：4 个 warp */}
      <g opacity={dim(2)}>
        <rect x={cx(2) - 34} y={mid - 34} width={68} height={68} rx={6} className="fk-fill-teal" opacity={0.15} />
        {[0, 1, 2, 3].map(w => <rect key={w} x={cx(2) - 26} y={mid - 27 + w * 14} width={52} height={10} rx={2} className="fk-fill-blue" opacity={0.7} />)}
      </g>
      {/* block：3 个 warpgroup + shared memory */}
      <g>
        <rect x={cx(3) - 46} y={mid - 36} width={92} height={72} rx={6} className="fk-fill-gray" opacity={0.6} />
        {[0, 1, 2].map(w => <rect key={w} x={cx(3) - 38 + w * 26} y={mid - 28} width={22} height={34} rx={3} className="fk-fill-teal" opacity={0.75} />)}
        <rect x={cx(3) - 38} y={mid + 12} width={76} height={14} rx={2} className="fk-fill-green" opacity={0.45} />
        <text x={cx(3)} y={mid + 19} style={{ fontSize: 9.5, fill: '#2f6b4a' }}>shared memory</text>
      </g>
      {/* cluster：2 个 block，shared memory 互通 */}
      <g opacity={dim(4)}>
        <rect x={cx(4) - 50} y={mid - 38} width={100} height={76} rx={6} fill="none" stroke="var(--fk-teal)" strokeDasharray="4 3" strokeWidth={1.4} />
        {[0, 1].map(bk => <g key={bk}>
          <rect x={cx(4) - 42 + bk * 46} y={mid - 30} width={38} height={60} rx={4} className="fk-fill-gray" opacity={0.7} />
          <rect x={cx(4) - 38 + bk * 46} y={mid + 10} width={30} height={14} rx={2} className="fk-fill-green" opacity={0.5} />
        </g>)}
        <line x1={cx(4) - 6} y1={mid + 17} x2={cx(4) + 6} y2={mid + 17} stroke="var(--fk-teal)" strokeWidth={2} />
      </g>
      {/* grid：许多 cluster */}
      <g>{Array.from({ length: 24 }, (_, k) => <rect key={k} x={cx(5) - 40 + (k % 6) * 14} y={mid - 26 + Math.floor(k / 6) * 14} width={11} height={11} rx={2} className="fk-fill-gray" opacity={0.85} />)}</g>
      {mult.map((m, i) => <text key={m} x={(cx(i) + cx(i + 1)) / 2} y={mid + 50} className="fk-muted fk-small" opacity={i === 3 ? dim(4) : 1}>{m}</text>)}
    </>}
  </Figure>
}
