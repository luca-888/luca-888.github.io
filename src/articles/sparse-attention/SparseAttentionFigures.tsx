import type { ReactNode } from 'react'
import { Cell, Figure, Legend, Link, useCompact } from '../../figure-kit'

// 语义色：blue 为被读取的未压缩 KV，purple 为被读取的压缩 entry，teal 为 indexer key，
// gray 为留在缓存中但这一步不读的 entry，empty 为已释放的位置，red 为 sink。

// 窄版只改总宽与左侧标签栏，格子按比例收窄。
const useFrame = () => {
  const compact = useCompact()
  return compact ? { W: 520, X0: 0, SW: 470, compact } : { W: 760, X0: 150, SW: 540, compact }
}

// 图 1：decode 一步在一层中读取哪些历史 token，48 个历史 token
const N = 48
const PICKED = [2, 9, 10, 17, 26, 31, 38, 45] // top-k 示例：位置由内容决定，分散在整个上下文中

export function SparseReadSet() {
  const { W, X0, SW, compact } = useFrame()
  const P = SW / N, CW = P - 2, CH = 26, LAB = compact ? 22 : 0
  const rows: { name: string; note: string; read: string; fill: (i: number) => 'blue' | 'gray' | 'empty' }[] = [
    { name: 'Full attention', note: '读取全部', read: '读 48', fill: () => 'blue' },
    { name: 'Sliding window', note: '最近 8 个', read: '读 8', fill: i => (i >= N - 8 ? 'blue' : 'empty') },
    { name: '学习式选择', note: '分数最高的 8 个', read: '读 8', fill: i => (PICKED.includes(i) ? 'blue' : 'gray') },
  ]
  const PITCH = 70 + LAB
  return (
    <Figure
      title="三种方式在 decode 一步中读取的历史 token"
      desc="48 个历史 token。Full attention 读取全部 48 个；sliding window 只读最近 8 个，其余位置的 KV 已释放；学习式选择读取分散在上下文各处的 8 个，其余 40 个仍留在缓存中。"
      width={W}
      height={rows.length * PITCH + 4}
      below={<Legend items={[['blue', '这一步读取'], ['gray', '在缓存中，这一步不读'], ['empty', '已释放'], ['ink', '当前 token']]} />}
      caption="每行是同一层的 48 个历史 token，最右侧为当前 token。Sliding window 与学习式选择每步都只读 8 个，区别在于其余 token 的 KV 能否释放。"
    >{() => rows.map((r, n) => {
      const y = n * PITCH + 16 + LAB
      return <g key={r.name}>
        {compact
          ? <>
              <text x={0} y={y - 16} className="fk-strong" textAnchor="start">{r.name}</text>
              <text x={130} y={y - 16} className="fk-muted fk-small" textAnchor="start">{r.note}</text>
            </>
          : <>
              <text x={X0 - 16} y={y + 6} className="fk-strong" textAnchor="end">{r.name}</text>
              <text x={X0 - 16} y={y + 24} className="fk-muted fk-small" textAnchor="end">{r.note}</text>
            </>}
        {Array.from({ length: N }, (_, i) => <rect key={i} x={X0 + i * P} y={y} width={CW} height={CH} rx={2} className={`fk-fill-${r.fill(i)}`} />)}
        <rect x={X0 + N * P + 6} y={y} width={CW} height={CH} rx={2} className="fk-fill-ink" />
        <text x={X0 + N * P + 6 + CW + 10} y={y + CH / 2} className="fk-stat" textAnchor="start">{r.read}</text>
        {n === 0 && <>
          <text x={X0} y={y + CH + 12} className="fk-muted fk-small" textAnchor="start">最早</text>
          <text x={X0 + N * P - 2} y={y + CH + 12} className="fk-muted fk-small" textAnchor="end">最近</text>
        </>}
      </g>
    })}</Figure>
  )
}

// 图 2：sink 滑出窗口前后，一个 head 的 attention 权重如何分配；长度即权重，数值为示意
const SINK = 0.8
const REST = [0.07, 0.05, 0.04, 0.04] // 窗口内 4 个 token 的原始权重，合计 0.2

export function SinkSoftmax() {
  const { W, X0, SW, compact } = useFrame()
  const BH = 30, LAB = compact ? 22 : 0, PITCH = 78 + LAB
  const bar = (y: number, sink: 'red' | 'void' | null, scale: number) => {
    let x = X0
    const parts: ReactNode[] = []
    if (sink) {
      const w = SINK * SW
      parts.push(<Cell key="s" x={x} y={y} w={w - 2} h={BH} fill={sink} label={sink === 'red' ? '第一个 token 0.80' : 'sink logit 0.80，不贡献输出'} small />)
      x += w
    }
    REST.forEach((p, i) => {
      const w = p * scale * SW
      parts.push(<Cell key={i} x={x} y={y} w={w - 2} h={BH} fill="blue" label={(p * scale).toFixed(2)} small />)
      x += w
    })
    return parts
  }
  const rows: { name: string; note: string; sink: 'red' | 'void' | null; scale: number; tail: string }[] = [
    { name: '完整上下文', note: '第一个 token 在缓存中', sink: 'red', scale: 1, tail: '窗口内 4 个 token 合计 0.20' },
    { name: '只留窗口', note: '第一个 token 已滑出', sink: null, scale: 1 / (1 - SINK), tail: '同样的 4 个 token，权重各放大到 5 倍' },
    { name: '窗口加 sink logit', note: '分母中多一项', sink: 'void', scale: 1, tail: '窗口内 4 个 token 合计仍为 0.20' },
  ]
  return (
    <Figure
      title="sink 滑出窗口前后一个 head 的 attention 权重"
      desc="完整上下文时第一个 token 占 0.80，窗口内 4 个 token 合计 0.20。只保留窗口时 softmax 重新归一，4 个 token 的权重放大到 5 倍，合计为 1。加入 sink logit 后，0.80 的权重落在 sink 上，4 个 token 合计仍为 0.20。"
      width={W}
      height={rows.length * PITCH + 6}
      below={<Legend items={[['red', '作为 sink 的第一个 token'], ['blue', '窗口内的 token'], ['void', 'sink logit']]} />}
      caption="每条的总长为 1，各段长度为权重，数值为示意。第二行中 4 个 token 的 logits 没有变化，权重变为 5 倍是因为分母少了最大的一项；第三行的 sink logit 补上了这一项。"
    >{() => rows.map((r, n) => {
      const y = n * PITCH + 4 + LAB
      return <g key={r.name}>
        {compact
          ? <>
              <text x={0} y={y - 12} className="fk-strong" textAnchor="start">{r.name}</text>
              <text x={130} y={y - 12} className="fk-muted fk-small" textAnchor="start">{r.note}</text>
            </>
          : <>
              <text x={X0 - 16} y={y + 8} className="fk-strong" textAnchor="end">{r.name}</text>
              <text x={X0 - 16} y={y + 25} className="fk-muted fk-small" textAnchor="end">{r.note}</text>
            </>}
        {bar(y, r.sink, r.scale)}
        <text x={X0 + SW} y={y + BH + 14} className="fk-muted fk-small" textAnchor="end">{r.tail}</text>
      </g>
    })}</Figure>
  )
}

// 图 3：DSA 的两步读取，上下文 128K、top-k 2048。上排为 indexer key（全部读取），下排为主 attention 的 MLA entry（只读 top-k）；
// 两排高度与每个 token 的元素数成正比（128 : 576）。下方放大其中 64 个 token，平均只有 1 个被选中。
const DL = 131072, DK = 2048
const DZ = 64 // 放大的 token 数
const DPICK = 41 // 放大段中被选中的 token
const DPOS = 0.37 // 放大段在上下文中的位置

export function DsaTwoStage() {
  const compact = useCompact()
  const W = compact ? 520 : 760, X0 = compact ? 0 : 200, SW = compact ? 470 : 490
  const LAB = compact ? 22 : 0
  const H1 = 12, H2 = 54
  const Y1 = 8 + LAB, Y2 = Y1 + H1 + 44 + LAB, YZ = Y2 + H2 + 46, YZ2 = YZ + H1 + 30
  const ZW = compact ? 300 : 320, ZX = X0 + 20, ZP = ZW / DZ
  const SX = X0 + DPOS * SW, SWD = 6 // 主行中被放大的位置；64 个 token 不到 1 像素，框画成 6 像素宽
  const label = (y: number, name: string, ...notes: string[]) => compact
    ? <>
        <text x={0} y={y - 12} className="fk-strong" textAnchor="start">{name}</text>
        <text x={130} y={y - 12} className="fk-muted fk-small" textAnchor="start">{notes.join('，')}</text>
      </>
    : <>
        <text x={X0 - 16} y={y + 5} className="fk-strong" textAnchor="end">{name}</text>
        {notes.map((t, k) => <text key={k} x={X0 - 16} y={y + 21 + k * 15} className="fk-muted fk-small" textAnchor="end">{t}</text>)}
      </>
  return (
    <Figure
      title="DSA 在 decode 一步中的两步读取"
      desc="上下文为 128K。第一步，lightning indexer 读取全部 128K 个 indexer key，每个 128 维，得到 index score 并取 top-2048。第二步，主 attention 只读取被选中的 2048 个 token 的 MLA entry，每个 576 维，占全部 token 的 1/64。放大其中 64 个 token：indexer key 全部读取，MLA entry 只有 1 个被读取。"
      width={W}
      height={YZ2 + H2 + 24}
      below={<Legend items={[['teal', 'indexer key，全部读取'], ['purple', '被选中的 MLA entry'], ['gray', '未选中，留在缓存中']]} />}
      caption="上下文为 128K，top-k 为 2048。两排的高度与每个 token 的元素数成正比：indexer key 为 128 维，MLA entry 为 576 维。下方放大其中 64 个 token。"
    >{arrow => <>
      {label(Y1, 'Lightning indexer', '上下文 128K', '① 给全部 token 打分')}
      <rect x={X0} y={Y1} width={SW - 2} height={H1} rx={2} className="fk-fill-teal" />
      <text x={X0 + SW + 8} y={Y1 + H1 / 2} className="fk-stat" textAnchor="start">{compact ? '128K' : '读 128K'}</text>

      {label(Y2, '主 attention', '② 只读分数最高的 2048 个', '2048 / 128K = 1/64')}
      <rect x={X0} y={Y2} width={SW - 2} height={H2} rx={2} className="fk-fill-gray" />
      <text x={X0 + SW + 8} y={Y2 + H2 / 2} className="fk-stat" textAnchor="start">{compact ? '2048' : '读 2048'}</text>
      <rect x={SX - 1} y={Y1 - 3} width={SWD} height={Y2 + H2 + 3 - (Y1 - 3)} rx={2} fill="none" stroke="#141414" strokeWidth={1.5} />

      <line x1={SX - 1} y1={Y2 + H2 + 4} x2={ZX - 6} y2={YZ - 8} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <line x1={SX + SWD - 1} y1={Y2 + H2 + 4} x2={ZX + ZW + 6} y2={YZ - 8} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <rect x={ZX - 6} y={YZ - 8} width={ZW + 12} height={YZ2 + H2 + 8 - (YZ - 8)} rx={6} fill="none" className="fk-rule" />
      {Array.from({ length: DZ }, (_, i) => <rect key={i} x={ZX + i * ZP} y={YZ} width={ZP - 1} height={H1} rx={1} className="fk-fill-teal" />)}
      <Link d={`M${ZX + (DPICK + 0.5) * ZP} ${YZ + H1 + 3} V${YZ2 - 2}`} hue="purple" arrow={arrow} />
      {Array.from({ length: DZ }, (_, i) => <rect key={i} x={ZX + i * ZP} y={YZ2} width={ZP - 1} height={H2} rx={1} className={`fk-fill-${i === DPICK ? 'purple' : 'gray'}`} />)}
      <text x={ZX + ZW + 16} y={YZ + H1 / 2} className="fk-small" textAnchor="start">其中 64 个 token：indexer key 全部读取</text>
      <text x={ZX + ZW + 16} y={YZ2 + H2 / 2} className="fk-small" textAnchor="start">MLA entry 平均只读 1 个</text>
    </>}</Figure>
  )
}

// 图：NSA 的三个分支在 decode 一步中各读取什么。上下文 4K，配置与 NSA 论文相同：
// compression 块长 32、步长 16，selection 的 block 为 64 个 token、选 16 个，窗口 512
const NN = 4096
const CL = 32, CS = 16 // compression 块长与步长
const NB = 64 // selection 的 block 长度
const NPICK = [0, 5, 9, 13, 18, 22, 27, 31, 34, 38, 43, 47, 51, 56, 62, 63] // 16 个选中的 block，含第一个与最近两个
const NWIN = 512

export function NsaReadSet() {
  const compact = useCompact()
  // 左侧标签要放下算式，这张图的标签栏比其他图宽
  const W = compact ? 520 : 760, X0 = compact ? 0 : 200, SW = compact ? 470 : 490
  const P = SW / NN, LAB = compact ? 22 : 0
  const EH = 14, TH = 20
  const Y0 = 8 + LAB, YC = Y0 + 16 + 48 + LAB, YZ = YC + 2 * EH + 4 + 40, ZH = 16
  const YB = YZ + 2 * ZH + 4 + 8, BH = 20 // 放大图中 block 的位置
  const YS = YB + BH + 56 + LAB, YW = YS + TH + 44 + LAB
  const GX = X0 + SW + 14, GW = W - GX - 2
  const label = (y: number, name: string, ...notes: string[]) => compact
    ? <>
        <text x={0} y={y - 12} className="fk-strong" textAnchor="start">{name}</text>
        <text x={110} y={y - 12} className="fk-muted fk-small" textAnchor="start">{notes.join('，')}</text>
      </>
    : <>
        <text x={X0 - 16} y={y + 5} className="fk-strong" textAnchor="end">{name}</text>
        {notes.map((t, k) => <text key={k} x={X0 - 16} y={y + 21 + k * 15} className="fk-muted fk-small" textAnchor="end">{t}</text>)}
      </>
  // 放大第 5 个 block（第 320 到 383 个 token），与它重叠的是第 19 到 23 个压缩 entry
  const ZB = 5, Z0 = ZB * NB - 32, ZW = compact ? 260 : 320, ZX = X0 + 20, ZS = ZW / (NB + 64)
  const zx = (t: number) => ZX + (t - Z0) * ZS
  const over = [19, 20, 21, 22, 23]
  const mid = [YC + EH + 2, YS + TH / 2, YW + TH / 2]
  const ax1 = zx(ZB * NB + NB / 2), ax2 = X0 + (ZB * NB + NB / 2) * P
  return (
    <Figure
      title="NSA 的三个分支在 decode 一步中读取的 KV"
      desc="上下文为 4K，历史 token 全部缓存。Compression 分支每 32 个 token 压成一个 entry，起点相隔 16 个 token，共 255 个，全部读取。Selection 分支把 4K 个 token 切成 64 个 block，每个 block 的分数是与它重叠的约 5 个压缩 entry 的分数之和，取最高的 16 个 block，读取其中 1024 个原始 token。Sliding window 分支读取最近 512 个 token。三个分支的输出由 gate 加权相加。"
      width={W}
      height={YW + TH + 22}
      below={<Legend items={[['purple', '读取的压缩 entry'], ['blue', '读取的原始 KV'], ['gray', '在缓存中，这一分支不读']]} />}
      caption="上下文为 4K。中间放大 block 5（block 从 0 编号，token 320–383）及与它重叠的 5 个压缩 entry，数字为 entry 覆盖的 token 范围。"
    >{arrow => <>
      {label(Y0, '原始 token', '4K，全部缓存')}
      <rect x={X0} y={Y0} width={SW - 2} height={16} rx={2} className="fk-fill-gray" />

      {label(YC, 'Compression', '(4K − 32) / 16 + 1 = 255 个 entry', '全部读取')}
      <Cell x={X0} y={YC} w={SW - 2} h={2 * EH + 4} fill="purple" label="255 个 entry，每个 32 个 token，起点相隔 16 个" small />

      <rect x={X0 + ZB * NB * P - 2} y={YC - 3} width={NB * P + 4} height={2 * EH + 10} rx={2} fill="none" stroke="#141414" strokeWidth={1.5} />
      <line x1={X0 + ZB * NB * P - 2} y1={YC + 2 * EH + 7} x2={zx(Z0) - 6} y2={YZ - 6} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <line x1={X0 + (ZB + 1) * NB * P + 2} y1={YC + 2 * EH + 7} x2={zx(Z0 + NB + 64) + 6} y2={YZ - 6} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <rect x={zx(Z0) - 6} y={YZ - 6} width={ZW + 12} height={YB + BH - YZ + 12} rx={6} fill="none" className="fk-rule" />
      {over.map(i => <Cell key={i} x={zx(i * CS)} y={YZ + (i % 2) * (ZH + 4)} w={CL * ZS - 2} h={ZH} fill="purple" label={`${i * CS}–${i * CS + CL - 1}`} small fs={9} />)}
      {[0, 1].map(r => <g key={r}>
        <text x={ZX + 6} y={YZ + r * (ZH + 4) + ZH / 2 + 1} className="fk-muted" textAnchor="start">…</text>
        <text x={ZX + ZW - 6} y={YZ + r * (ZH + 4) + ZH / 2 + 1} className="fk-muted" textAnchor="end">…</text>
      </g>)}
      <Cell x={zx(ZB * NB)} y={YB} w={NB * ZS - 2} h={BH} fill="blue" label="block 5，64 个 token" small />
      <text x={zx(Z0 + NB + 64) + 16} y={YZ + ZH + 2} className="fk-small" textAnchor="start">与它重叠的 5 个 entry</text>
      <text x={zx(Z0 + NB + 64) + 16} y={YB + BH / 2} className="fk-small" textAnchor="start">分数相加，即 block 的分数</text>

      <Link d={`M${ax1} ${YB + BH + 8} L${ax2} ${YS - 2}`} arrow={arrow} />
      <text x={Math.max(ax1, ax2) + 14} y={(YB + BH + YS) / 2 + 4} className="fk-muted fk-small" textAnchor="start">64 个 block 各得一个分数，取最高的 16 个</text>

      {label(YS, 'Selection', '4K / 64 = 64 个 block', '选 16 个：16 × 64 = 1024 个 token')}
      {Array.from({ length: NN / NB }, (_, b) => <rect key={b} x={X0 + b * NB * P} y={YS} width={NB * P - 1.5} height={TH} rx={1.5} className={`fk-fill-${NPICK.includes(b) ? 'blue' : 'gray'}`} />)}

      {label(YW, 'Sliding window', '读最近 512 个 token')}
      <rect x={X0} y={YW} width={(NN - NWIN) * P - 1.5} height={TH} rx={2} className="fk-fill-gray" />
      <rect x={X0 + (NN - NWIN) * P} y={YW} width={NWIN * P - 2} height={TH} rx={2} className="fk-fill-blue" />

      <text x={X0} y={YW + TH + 14} className="fk-muted fk-small" textAnchor="start">最早</text>
      <text x={X0 + SW - 2} y={YW + TH + 14} className="fk-muted fk-small" textAnchor="end">最近</text>

      {mid.map((y, n) => <Link key={n} d={`M${X0 + SW + 1} ${y} H${GX}`} arrow={arrow} />)}
      <Cell x={GX} y={YC} w={GW} h={YW + TH - YC} fill="ink" label="gate" small />
    </>}</Figure>
  )
}

// 图 4：DeepSeek-V4-Pro 的 CSA 层与 HCA 层在 16K 上下文下的压缩与读取。CSA 每 4 个 token 产生一个 entry（每个由 8 个 token 压成），
// top-k 1024；HCA 每 128 个 token 一个 entry，全部读取；两种层都另读最近 128 个 token。下方放大 CSA 中的一段。
const VL = 16384, VWIN = 128
const VZ0 = 0, VZ1 = 48 // 放大开头 48 个 token，含第 0 到 11 个 CSA entry
const VPICK = [2, 6, 9] // 放大段中被选中的 entry，约占 1/4

export function V4ReadSet() {
  const compact = useCompact()
  const W = compact ? 520 : 760, X0 = compact ? 0 : 200, SW = compact ? 470 : 490
  const P = SW / VL, LAB = compact ? 22 : 0
  const EH = 28, TH = 12, ZH = 16
  const Y0 = 8 + LAB, YC = Y0 + 16 + 48 + LAB, YZ = YC + EH + TH + 6 + 44, YH = YZ + 2 * ZH + 4 + 54 + LAB
  const ZW = compact ? 360 : 400, ZX = X0 + 20, ZP = ZW / (VZ1 - VZ0)
  const label = (y: number, name: string, ...notes: string[]) => compact
    ? <>
        <text x={0} y={y - 12} className="fk-strong" textAnchor="start">{name}</text>
        <text x={90} y={y - 12} className="fk-muted fk-small" textAnchor="start">{notes.join('，')}</text>
      </>
    : <>
        <text x={X0 - 16} y={y + 5} className="fk-strong" textAnchor="end">{name}</text>
        {notes.map((t, k) => <text key={k} x={X0 - 16} y={y + 21 + k * 15} className="fk-muted fk-small" textAnchor="end">{t}</text>)}
      </>
  const windowRow = (y: number) => <>
    <rect x={X0} y={y} width={(VL - VWIN) * P - 1} height={TH} rx={2} className="fk-fill-empty" />
    <rect x={X0 + (VL - VWIN) * P} y={y} width={VWIN * P - 1} height={TH} rx={1} className="fk-fill-blue" />
  </>
  const SX = X0 + VZ0 * P, SWD = Math.max(6, (VZ1 - VZ0) * P)
  const entries = Array.from({ length: 12 }, (_, k) => k)
  return (
    <Figure
      title="CSA 层与 HCA 层在 16K 上下文下的压缩与读取"
      desc="上下文为 16K，原始 token 只缓存最近 128 个。CSA 层每 4 个 token 产生一个 entry，每个 entry 由相邻 8 个 token 压成，共 4K 个，indexer 选出其中 1024 个读取，另读窗口内 128 个 token。HCA 层每 128 个 token 压成一个 entry，共 128 个，全部读取，另读窗口内 128 个 token。放大 CSA 开头的 48 个 token：12 个相互重叠的 entry 中有 3 个被选中。"
      width={W}
      height={YH + EH + TH + 26}
      below={<Legend items={[['purple', '读取的压缩 entry'], ['gray', '未选中的压缩 entry'], ['blue', '窗口内的未压缩 KV'], ['empty', '窗口外的未压缩 KV，已释放'], ['void', '原始 token，不缓存']]} />}
      caption="上下文为 16K，配置与 DeepSeek-V4-Pro 相同。放大框中的数字为 entry 覆盖的 token 范围，相邻两个 entry 重叠 4 个 token。"
    >{() => <>
      {label(Y0, '原始 token', '16K，不缓存')}
      <rect x={X0} y={Y0} width={SW - 2} height={16} rx={2} className="fk-fill-void" />

      {label(YC, 'CSA 层', '16K / 4 = 4K 个 entry', '选 1024 个，加窗口 128', '1024 + 128 = 1152')}
      <Cell x={X0} y={YC} w={SW - 2} h={EH} fill="gray" label="4K 个 entry，选中的 1024 个分散在各处" small />
      {windowRow(YC + EH + 6)}
      <text x={X0 + SW + 8} y={YC + EH / 2} className="fk-stat" textAnchor="start">读 1152</text>
      <rect x={SX - 1} y={YC - 3} width={SWD} height={EH + 6} rx={2} fill="none" stroke="#141414" strokeWidth={1.5} />

      <line x1={SX - 1} y1={YC + EH + 3} x2={ZX - 6} y2={YZ - 8} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <line x1={SX + SWD - 1} y1={YC + EH + 3} x2={ZX + ZW + 6} y2={YZ - 8} stroke="#141414" strokeWidth={1} strokeDasharray="3 3" />
      <rect x={ZX - 6} y={YZ - 8} width={ZW + 12} height={2 * ZH + 4 + 16} rx={6} fill="none" className="fk-rule" />
      {entries.map(i => { const a = Math.max(0, 4 * i - 4), b = 4 * i + 3
        return <Cell key={i} x={ZX + (a - VZ0) * ZP} y={YZ + (i % 2) * (ZH + 4)} w={(b - a + 1) * ZP - 2} h={ZH} fill={VPICK.includes(i) ? 'purple' : 'gray'} label={`${a}–${b}`} small fs={9} /> })}
      {[0, 1].map(r => <g key={r}>
        <text x={ZX + ZW + 1} y={YZ + r * (ZH + 4) + ZH / 2 + 1} className="fk-muted" textAnchor="start">…</text>
      </g>)}
      {!compact && <text x={ZX + ZW + 20} y={YZ + ZH + 2} className="fk-small" textAnchor="start">约 1/4 被选中</text>}

      {label(YH, 'HCA 层', '16K / 128 = 128 个 entry', '全部读取，加窗口 128', '128 + 128 = 256')}
      {Array.from({ length: VL / 128 }, (_, e) => <rect key={e} x={X0 + e * 128 * P} y={YH} width={128 * P - 0.8} height={EH} className="fk-fill-purple" />)}
      {windowRow(YH + EH + 6)}
      <text x={X0 + SW + 8} y={YH + EH / 2} className="fk-stat" textAnchor="start">读 256</text>
    </>}</Figure>
  )
}

// 图：每步读取的 entry 数随上下文长度的变化，双对数坐标；曲线按各方案的公式计算
const K = 1024
const CURVES: { name: string; color: string; f: (L: number) => number; dy?: number }[] = [
  { name: 'full attention', color: '#141414', f: L => L },
  { name: 'NSA', color: 'var(--fk-amber)', f: L => Math.min(L, L / 16 + 1536) },
  { name: 'HCA', color: 'var(--fk-purple)', f: L => Math.min(L, L / 128 + 128) },
  { name: 'DSA，top-k 2048', color: 'var(--fk-teal)', f: L => Math.min(L, 2048) },
  { name: 'CSA，top-k 1024', color: 'var(--fk-green)', f: L => Math.min(L, Math.min(L / 4, 1024) + 128), dy: -7 },
  { name: 'sliding window 1024', color: 'var(--fk-blue)', f: L => Math.min(L, 1024), dy: 7 },
]
const TICKS: [number, string][] = [[K, '1K'], [4 * K, '4K'], [16 * K, '16K'], [64 * K, '64K'], [256 * K, '256K'], [1024 * K, '1M']]

export function ReadVsContext() {
  const compact = useCompact()
  // 横纵轴同为 128 到 1M、每个倍频程同样的像素数，full attention 是 45° 的对角线
  const W = compact ? 520 : 760, PW = compact ? 300 : 338, PH = PW, PX = compact ? 56 : 170, PY = 14
  const lg = Math.log2
  const x = (L: number) => PX + (lg(L) - 7) / 13 * PW
  const y = (n: number) => PY + PH - (lg(n) - 7) / 13 * PH
  const xs = Array.from({ length: 131 }, (_, i) => 2 ** (7 + i / 10))
  return (
    <Figure
      title="每步读取的 KV entry 数随上下文长度的变化"
      desc="双对数坐标。Full attention 的读取量等于上下文长度；sliding window 在 1024 之后不再增长，DSA 在 2048 之后不再增长；NSA 与 HCA 仍随上下文增长，斜率分别为 1/16 与 1/128。上下文为 1M 时，full attention 读取 1M 个，NSA 约 66K 个，HCA 约 8K 个，DSA 2048 个，CSA 1152 个，sliding window 1024 个。"
      width={W}
      height={PY + PH + 44}
      caption="一层中 decode 一步读取的 KV entry 数，按各方案的配置计算；双对数坐标，横纵轴比例相同，full attention 为对角线。上下文不超过窗口或 top-k 时，sliding window、DSA、NSA 的曲线与 full attention 重合。HCA 的窗口只有 128 个 token，短上下文时读取量最小，超过约 115K 后高于 sliding window。CSA 在上下文超过约 4K 后固定为 1152，略高于 sliding window 的 1024。"
    >{() => <>
      {[[256, '256'] as [number, string], ...TICKS].map(([v, t]) => <g key={`x${t}`}>
        <line x1={x(v)} y1={PY} x2={x(v)} y2={PY + PH} className="fk-rule" />
        <text x={x(v)} y={PY + PH + 14} className="fk-muted fk-small">{t}</text>
      </g>)}
      {[[256, '256'] as [number, string], ...TICKS].map(([v, t]) => <g key={t}>
        <line x1={PX} y1={y(v)} x2={PX + PW} y2={y(v)} className="fk-rule" />
        <text x={PX - 8} y={y(v)} className="fk-muted fk-small" textAnchor="end">{t}</text>
      </g>)}
      <text x={PX + PW / 2} y={PY + PH + 34} className="fk-muted">上下文长度（token）</text>
      <text x={PX - 44} y={PY + PH / 2} className="fk-muted" transform={`rotate(-90 ${PX - 44} ${PY + PH / 2})`}>每步读取的 entry</text>
      {CURVES.map(c => <g key={c.name}>
        <path d={xs.map((L, i) => `${i ? 'L' : 'M'}${x(L).toFixed(1)} ${y(c.f(L)).toFixed(1)}`).join(' ')} fill="none" style={{ stroke: c.color }} strokeWidth={2.5} strokeLinejoin="round" />
        <text x={PX + PW + 10} y={y(c.f(1024 * K)) + (c.dy ?? 0)} className="fk-strong" textAnchor="start" style={{ fill: c.color, fontSize: 12 }}>{c.name}</text>
      </g>)}
    </>}</Figure>
  )
}

// 图：Gemma 3 27B 的 62 层在 128K 上下文下各缓存多少 entry。一列一层，列高即该层缓存的 entry 数。
const LAYERS = 62
const isGlobal = (i: number) => i % 6 === 5
const CTX = 131072, WINDOW = 1024

export function LocalGlobalLayers() {
  const { W, X0, SW, compact } = useFrame()
  const P = SW / LAYERS, CW = P - 1.5
  const Y0 = 30, H = 150, YB = Y0 + H
  const localH = 6 // 按比例只有约 1.2 像素，放大画出并在图中注明
  const nGlobal = Array.from({ length: LAYERS }, (_, i) => i).filter(isGlobal).length
  const g0 = X0 + 5 * P + CW / 2 // 第一个 global 层的中线
  return (
    <Figure
      title="Gemma 3 27B 各层在 128K 上下文下缓存的 KV entry"
      desc="62 层中每 6 层的最后一层为 global 层，共 10 层，每层缓存 128K 个 entry；其余 52 层为 local 层，每层只缓存 1024 个。全部缓存合计为 62 层都使用 full attention 时的 16.8%，约等于 global 层的占比 10/62。"
      width={W}
      height={YB + 30}
      below={<Legend items={[['purple', 'global 层缓存的 entry'], ['blue', 'local 层缓存的 entry'], ['void', '改用 sliding window 后不再缓存']]} />}
      caption="一列是一层，列高是这一层在 128K 上下文下缓存的 entry 数，灰色是使用 full attention 时原本要缓存的部分。"
    >{() => <>
      {!compact && <>
        <text x={X0 - 12} y={Y0 + 4} className="fk-muted fk-small" textAnchor="end">128K 个 entry</text>
        <text x={X0 - 12} y={YB - 4} className="fk-muted fk-small" textAnchor="end">0</text>
        <text x={X0 - 12} y={Y0 + H / 2} className="fk-strong" textAnchor="end">每层的 KV cache</text>
      </>}
      {Array.from({ length: LAYERS }, (_, i) => isGlobal(i)
        ? <rect key={i} x={X0 + i * P} y={Y0} width={CW} height={H} rx={1.5} className="fk-fill-purple" />
        : <g key={i}>
            <rect x={X0 + i * P} y={Y0} width={CW} height={H - localH - 2} rx={1.5} className="fk-fill-void" />
            <rect x={X0 + i * P} y={YB - localH} width={CW} height={localH} rx={1} className="fk-fill-blue" />
          </g>)}
      <text x={g0} y={Y0 - 14} className="fk-strong fk-text-purple">global 层，共 10 层</text>
      <path d={`M${g0} ${Y0 - 6} V${Y0 - 1}`} className="fk-link fk-line-purple" />
      <text x={X0} y={YB + 14} className="fk-muted fk-small" textAnchor="start">第 1 层</text>
      <text x={X0 + SW / 2} y={YB + 14} className="fk-strong fk-text-blue">local 层，共 52 层，每层 1024 个 entry（高度放大 5 倍画出）</text>
      <text x={X0 + SW - 2} y={YB + 14} className="fk-muted fk-small" textAnchor="end">第 62 层</text>
      <text x={X0 + SW + 10} y={Y0 + H / 2 - 9} className="fk-stat" textAnchor="start" style={{ fontSize: 18 }}>≈ {nGlobal}/{LAYERS}</text>
      {!compact && <text x={X0 + SW + 10} y={Y0 + H / 2 + 11} className="fk-muted fk-small" textAnchor="start">相对全部 full attention</text>}
    </>}</Figure>
  )
}

// 章首的规格卡：读哪些、每步读多少、缓存多少、代表模型
export function SpecCard({ items }: { items: [string, ReactNode][] }) {
  return <dl className="sa-spec">{items.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
}
