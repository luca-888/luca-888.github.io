import type { ReactNode } from 'react'
import { Cell, Figure, Legend, Link, tex, useCompact } from '../../figure-kit'

// 语义色：teal 为线性层与它的 state，purple 为 full attention 层与它的 KV cache，
// void 为改用线性层后不再缓存的部分。图 1 中 amber / blue / green 分别是旧 value、新 value 与无关的另一条关联。

const KiB = 1024, MiB = KiB * KiB, GiB = MiB * KiB

// 章首规格卡：更新式、写入、遗忘、代表模型。
export function SpecCard({ items }: { items: [string, ReactNode][] }) {
  return <dl className="la-spec">{items.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
}

export const Tex = ({ src }: { src: string }) => <span dangerouslySetInnerHTML={tex(src)} />

// 图 1：三种写入方式之后，用 k 与 k' 从 state 中读出的内容。长度为向量的大小，数值为示意（α = 0.5，β = 1，k 与 k' 正交）。
const U = 96 // 一个单位向量画多长

type Seg = { fill: 'amber' | 'blue' | 'green'; len: number; label: string }
const ROWS: { name: string; rule: string; k: Seg[]; kRead: string; k2: Seg[]; k2Read: string; focus?: boolean }[] = [
  { name: '写入前', rule: '已有两条关联', k: [{ fill: 'amber', len: 1, label: 'v₁' }], kRead: 'v₁', k2: [{ fill: 'green', len: 1, label: 'u' }], k2Read: 'u' },
  { name: '直接相加', rule: 'S + v₂kᵀ', k: [{ fill: 'amber', len: 1, label: 'v₁' }, { fill: 'blue', len: 1, label: 'v₂' }], kRead: 'v₁ + v₂', k2: [{ fill: 'green', len: 1, label: 'u' }], k2Read: 'u' },
  { name: '先 decay 再相加', rule: 'αS + v₂kᵀ', k: [{ fill: 'amber', len: 0.5, label: '' }, { fill: 'blue', len: 1, label: 'v₂' }], kRead: '0.5 v₁ + v₂', k2: [{ fill: 'green', len: 0.5, label: '' }], k2Read: '0.5 u' },
  { name: 'Delta rule', rule: 'S + (v₂ − Sk)kᵀ', k: [{ fill: 'blue', len: 1, label: 'v₂' }], kRead: 'v₂', k2: [{ fill: 'green', len: 1, label: 'u' }], k2Read: 'u', focus: true },
]

export function DeltaWrite() {
  const compact = useCompact()
  const W = compact ? 520 : 760, X0 = compact ? 0 : 200, C1 = X0 + (compact ? 10 : 30), C2 = C1 + 2 * U + (compact ? 70 : 110)
  const BH = 26, LAB = compact ? 22 : 0, PITCH = 56 + LAB, Y0 = 34
  const bar = (x: number, y: number, segs: Seg[], focus?: boolean) => {
    let cx = x
    const out: ReactNode[] = segs.map((s, i) => {
      const w = s.len * U
      const c = <Cell key={i} x={cx} y={y} w={w - 2} h={BH} fill={s.fill} label={s.label || undefined} />
      cx += w
      return c
    })
    if (focus) out.push(<rect key="f" x={x - 3} y={y - 3} width={cx - x + 4} height={BH + 6} rx={4} className="fk-focus" />)
    return { out, end: cx }
  }
  return (
    <Figure
      title="三种写入方式之后用 k 与 k′ 从 state 读出的内容"
      desc="写入前 state 中 k 对应 v₁，k′ 对应 u。新 token 要让 k 对应 v₂。直接相加后 k 读出 v₁ + v₂，k′ 仍读出 u；先 decay 再相加后 k 读出 0.5 v₁ + v₂，k′ 读出 0.5 u；delta rule 写入后 k 读出 v₂，k′ 仍读出 u。"
      width={W}
      height={Y0 + ROWS.length * PITCH}
      below={<Legend items={[['amber', '旧 value v₁'], ['blue', '新 value v₂'], ['green', '无关的另一条关联 u']]} />}
      caption="每行是一种写入之后的 state，条的长度是读出向量的大小。数值为示意：α = 0.5，β = 1，k 与 k′ 正交。"
    >{() => <>
      <text x={C1} y={12} className="fk-muted" textAnchor="start">用 k 读出</text>
      <text x={C2} y={12} className="fk-muted" textAnchor="start">用 k′ 读出</text>
      {ROWS.map((r, n) => {
        const y = Y0 + n * PITCH + LAB
        const a = bar(C1, y, r.k, r.focus), b = bar(C2, y, r.k2, r.focus)
        return <g key={r.name}>
          {compact
            ? <>
                <text x={0} y={y - 13} className="fk-strong" textAnchor="start">{r.name}</text>
                <text x={130} y={y - 13} className="fk-muted fk-mono" textAnchor="start">{r.rule}</text>
              </>
            : <>
                <text x={X0 - 10} y={y + 5} className="fk-strong" textAnchor="end">{r.name}</text>
                <text x={X0 - 10} y={y + 22} className="fk-muted fk-mono" textAnchor="end">{r.rule}</text>
              </>}
          {a.out}
          <text x={a.end + 8} y={y + BH / 2} className="fk-stat" textAnchor="start">{r.kRead}</text>
          {b.out}
          <text x={b.end + 8} y={y + BH / 2} className="fk-stat" textAnchor="start">{r.k2Read}</text>
          {n === 0 && <line x1={X0 - 120} y1={y + BH + 14} x2={W} y2={y + BH + 14} className="fk-rule" />}
        </g>
      })}
    </>}</Figure>
  )
}

// 图 2：4K token 的 prefill 按 C = 64 切成 64 个 chunk；放大其中两个。chunk 内为 64 × 64 下三角的矩阵乘，chunk 之间串行传递 state。
const NCH = 64, ZA = 31 // 放大第 31、32 个 chunk（从 0 计）

export function Chunkwise() {
  const compact = useCompact()
  const W = compact ? 520 : 760, X0 = compact ? 0 : 150, SW = compact ? 500 : 560
  const P = SW / NCH, CH = 18, Y0 = 26
  const YS = 132, SS = 56 // state 方块
  const YC = 196, CS = 100 // chunk 方块
  const xs = compact ? [20, 225, 430] : [X0 + 10, X0 + 250, X0 + 490] // 三个 state 的左边
  const cx = (i: number) => (xs[i] + SS + xs[i + 1]) / 2 - CS / 2 // 两个 state 之间的 chunk
  const tri = (x: number, y: number) => `M${x} ${y} L${x} ${y + CS} L${x + CS} ${y + CS} Z`
  return (
    <Figure
      title="Chunkwise 计算：chunk 内并行，chunk 之间传递 state"
      desc="4K token 的 prefill 按每 64 个 token 切成 64 个 chunk。放大其中两个：chunk n 读入口 state S[n]，与 chunk 内 64 × 64 的下三角矩阵乘一起得到 64 个输出，再把 state 更新为 S[n+1] 交给下一个 chunk。串行的只有 64 次 state 传递。"
      width={W}
      height={YC + CS + 64}
      below={<Legend items={[['teal', 'state，每个 head 128 × 128'], ['blue', 'chunk 内的计算，矩阵乘']]} />}
      caption="上排是整段序列的 64 个 chunk，下排放大其中两个。斜向下的箭头是读入口 state（Q Sᵀ），斜向上的箭头是用本 chunk 的 K、V 更新 state。"
    >{arrow => <>
      {!compact && <>
        <text x={X0 - 14} y={Y0 + 3} className="fk-strong" textAnchor="end">4K token</text>
        <text x={X0 - 14} y={Y0 + 19} className="fk-muted fk-small" textAnchor="end">C = 64，共 64 个 chunk</text>
      </>}
      {Array.from({ length: NCH }, (_, i) => <rect key={i} x={X0 + i * P} y={Y0} width={P - 1.5} height={CH} rx={1.5}
        className={i === ZA || i === ZA + 1 ? 'fk-fill-blue' : 'fk-fill-gray'} />)}
      <text x={X0} y={Y0 + CH + 12} className="fk-muted fk-small" textAnchor="start">第 0 个 chunk</text>
      <text x={X0 + SW} y={Y0 + CH + 12} className="fk-muted fk-small" textAnchor="end">第 63 个</text>
      {/* 放大引线 */}
      <path d={`M${X0 + ZA * P} ${Y0 + CH + 2} L${xs[0]} ${YS - 26}`} className="fk-link fk-line-muted" strokeDasharray="3 3" strokeWidth={1} />
      <path d={`M${X0 + (ZA + 2) * P} ${Y0 + CH + 2} L${xs[2] + SS} ${YS - 26}`} className="fk-link fk-line-muted" strokeDasharray="3 3" strokeWidth={1} />
      {/* state 与 chunk */}
      {xs.map((x, i) => <Cell key={i} x={x} y={YS} w={SS} h={SS} fill="teal" label={['S[n]', 'S[n+1]', 'S[n+2]'][i]} small />)}
      {[0, 1].map(i => {
        const x = cx(i)
        return <g key={i}>
          <rect x={x} y={YC} width={CS} height={CS} rx={3} className="fk-fill-free" />
          <path d={tri(x, YC)} className="fk-fill-blue" />
          <text x={x + CS * 0.68} y={YC + 28} className="fk-muted fk-small">64 × 64</text>
          <text x={x + CS * 0.36} y={YC + CS - 18} className="fk-cell-text">{`chunk ${i ? 'n+1' : 'n'}`}</text>
          <Link d={`M${xs[i] + SS / 2} ${YS + SS} L${x + 12} ${YC}`} hue="teal" arrow={arrow} />
          <Link d={`M${x + CS - 12} ${YC} L${xs[i + 1] + SS / 2} ${YS + SS}`} hue="teal" arrow={arrow} />
          <Link d={`M${x + CS / 2} ${YC + CS} V${YC + CS + 26}`} hue="blue" arrow={arrow} />
          <text x={x + CS / 2} y={YC + CS + 38} className="fk-muted fk-small">{`64 个输出 O[${i ? 'n+1' : 'n'}]`}</text>
        </g>
      })}
      <text x={(xs[0] + xs[2] + SS) / 2} y={YS - 14} className="fk-strong fk-text-teal">串行：4K / 64 = 64 次 state 传递</text>
      {!compact && <>
        <text x={X0 - 14} y={YC + CS / 2 - 8} className="fk-strong" textAnchor="end">chunk 内并行</text>
        <text x={X0 - 14} y={YC + CS / 2 + 9} className="fk-muted fk-small" textAnchor="end">QKᵀ ⊙ M，Tensor Core</text>
      </>}
    </>}</Figure>
  )
}

// 图 3：Qwen3.8-27B 每层在 256K 上下文下的缓存。full attention 层 1 GiB（4 KV head × 256 × 2 × BF16 × 256K），
// GDN 层 3 MiB（48 head × 128 × 128 × FP32），按比例约 0.4 像素，放大画出。
const QL = 64
const isFull = (i: number) => i % 4 === 3

export function HybridLayers() {
  const compact = useCompact()
  const W = compact ? 520 : 760, X0 = compact ? 0 : 150, SW = compact ? 440 : 520
  const P = SW / QL, CW = P - 1.5
  const Y0 = 30, H = 150, YB = Y0 + H, LIN = 5
  const f0 = X0 + 3 * P + CW / 2
  return (
    <Figure
      title="Qwen3.8-27B 各层在 256K 上下文下的缓存"
      desc="64 层中每 4 层的最后一层为 full attention，共 16 层，每层缓存 1 GiB 的 KV；其余 48 层为 Gated DeltaNet，每层只有 3 MiB 的 state。合计约 16.1 GiB，64 层全部使用 full attention 时为 64 GiB。"
      width={W}
      height={YB + 30}
      below={<Legend items={[['purple', 'full attention 层的 KV cache'], ['teal', 'GDN 层的 state'], ['void', '改用 GDN 后不再缓存']]} />}
      caption="一列是一层，列高是这一层在 256K 上下文下的缓存大小；state 按 FP32、KV 按 BF16 计。"
    >{() => <>
      {!compact && <>
        <text x={X0 - 12} y={Y0 + 4} className="fk-muted fk-small" textAnchor="end">1 GiB</text>
        <text x={X0 - 12} y={YB - 4} className="fk-muted fk-small" textAnchor="end">0</text>
        <text x={X0 - 12} y={Y0 + H / 2} className="fk-strong" textAnchor="end">每层的缓存</text>
      </>}
      {Array.from({ length: QL }, (_, i) => isFull(i)
        ? <rect key={i} x={X0 + i * P} y={Y0} width={CW} height={H} rx={1.5} className="fk-fill-purple" />
        : <g key={i}>
            <rect x={X0 + i * P} y={Y0} width={CW} height={H - LIN - 2} rx={1.5} className="fk-fill-void" />
            <rect x={X0 + i * P} y={YB - LIN} width={CW} height={LIN} rx={1} className="fk-fill-teal" />
          </g>)}
      <text x={f0} y={Y0 - 14} className="fk-strong fk-text-purple" textAnchor="start" dx={-6}>full attention 层，共 16 层，每层 1 GiB</text>
      <path d={`M${f0} ${Y0 - 6} V${Y0 - 1}`} className="fk-link fk-line-purple" />
      <text x={X0} y={YB + 14} className="fk-muted fk-small" textAnchor="start">第 1 层</text>
      <text x={X0 + SW / 2} y={YB + 14} className="fk-strong fk-text-teal">GDN 层，共 48 层，每层 3 MiB（高度放大画出）</text>
      <text x={X0 + SW - 2} y={YB + 14} className="fk-muted fk-small" textAnchor="end">第 64 层</text>
      <text x={X0 + SW + 12} y={Y0 + H / 2 - 18} className="fk-stat" textAnchor="start" style={{ fontSize: 18 }}>16.1 GiB</text>
      <text x={X0 + SW + 12} y={Y0 + H / 2 + 2} className="fk-muted fk-small" textAnchor="start">全部 full attention</text>
      <text x={X0 + SW + 12} y={Y0 + H / 2 + 17} className="fk-muted fk-small" textAnchor="start">时为 64 GiB</text>
    </>}</Figure>
  )
}

// 图 4：线性层的 state 与同样层数改用 full attention 时的 KV，随上下文长度的变化。双对数坐标。
// Qwen3.8-27B：48 层 GDN，state 144 MiB；改用 GQA（4 KV head，head dim 256）每 token 192 KiB，交叉点 768。
// Kimi K3：69 层 KDA，state 414 MiB；改用 MLA（576 维）每 token 77.625 KiB，交叉点约 5461。
const QS = 144 * MiB, QKV = 192 * KiB, KS = 414 * MiB, KKV = 69 * 576 * 2

export function StateVsKv() {
  const compact = useCompact()
  const W = compact ? 520 : 760, PX = compact ? 60 : 150, PW = compact ? 330 : 440, PH = 270, PY = 12
  const lg = Math.log2
  const XL = 8, XR = 20, YL = 25, YR = 37 // 横轴 256 到 1M，纵轴 32 MiB 到 128 GiB
  const x = (L: number) => PX + (lg(L) - XL) / (XR - XL) * PW
  const y = (b: number) => PY + PH - (lg(b) - YL) / (YR - YL) * PH
  const line = (f: (L: number) => number, L1: number) => {
    const pts = Array.from({ length: 61 }, (_, i) => 2 ** (XL + (lg(L1) - XL) * i / 60))
    return pts.map((L, i) => `${i ? 'L' : 'M'}${x(L).toFixed(1)} ${y(f(L)).toFixed(1)}`).join(' ')
  }
  const xt: [number, string][] = [[256, '256'], [1024, '1K'], [4096, '4K'], [16384, '16K'], [65536, '64K'], [262144, '256K'], [1048576, '1M']]
  const yt: [number, string][] = [[32 * MiB, '32 MiB'], [128 * MiB, '128 MiB'], [512 * MiB, '512 MiB'], [2 * GiB, '2 GiB'], [8 * GiB, '8 GiB'], [32 * GiB, '32 GiB'], [128 * GiB, '128 GiB']]
  const curves = [
    { d: line(() => QS, 262144), cls: 'fk-line-teal', dash: undefined },
    { d: line(L => QKV * L, 262144), cls: 'fk-line-purple', dash: undefined },
    { d: line(() => KS, 1048576), cls: 'fk-line-teal', dash: '6 4' },
    { d: line(L => KKV * L, 1048576), cls: 'fk-line-purple', dash: '6 4' },
  ]
  const lab = (L: number, b: number, t: string, hue: 'teal' | 'purple', dy = 0) =>
    <text x={x(L) + 8} y={y(b) + dy} className={`fk-strong fk-text-${hue}`} textAnchor="start" style={{ fontSize: 12 }}>{t}</text>
  const qx = QS / QKV, kx = KS / KKV
  return (
    <Figure
      title="线性层的 state 与同样层数改用 full attention 时的 KV"
      desc="双对数坐标。Qwen3.8-27B 的 48 层 GDN 共有 144 MiB 的 state，同样 48 层改用 GQA 时每个 token 的 KV 为 192 KiB，两者在 768 个 token 处相等；Kimi K3 的 69 层 KDA 共有 414 MiB 的 state，改用 MLA 时每个 token 的 KV 为 77.6 KiB，两者在约 5.5K 个 token 处相等。"
      width={W}
      height={PY + PH + 44}
      below={<Legend items={[['teal', '线性层的 state'], ['purple', '同样层数改用 full attention 的 KV']]} />}
      caption="实线为 Qwen3.8-27B（画到它的最大上下文 256K），虚线为 Kimi K3（画到 1M），圆点为交叉点。"
    >{() => <>
      {xt.map(([v, t]) => <g key={t}>
        <line x1={x(v)} y1={PY} x2={x(v)} y2={PY + PH} className="fk-rule" />
        <text x={x(v)} y={PY + PH + 14} className="fk-muted fk-small">{t}</text>
      </g>)}
      {yt.map(([v, t]) => <g key={t}>
        <line x1={PX} y1={y(v)} x2={PX + PW} y2={y(v)} className="fk-rule" />
        <text x={PX - 8} y={y(v)} className="fk-muted fk-small" textAnchor="end">{t}</text>
      </g>)}
      <text x={PX + PW / 2} y={PY + PH + 34} className="fk-muted">上下文长度（token）</text>
      {!compact && <text x={PX - 78} y={PY + PH / 2} className="fk-muted" transform={`rotate(-90 ${PX - 78} ${PY + PH / 2})`}>每个请求的缓存</text>}
      {curves.map((c, i) => <path key={i} d={c.d} className={`fk-link ${c.cls}`} strokeWidth={2.5} strokeDasharray={c.dash} />)}
      <circle cx={x(qx)} cy={y(QS)} r={5} className="fk-fill-ink" stroke="#fff" strokeWidth={1.5} />
      <circle cx={x(kx)} cy={y(KS)} r={5} className="fk-fill-ink" stroke="#fff" strokeWidth={1.5} />
      <text x={x(qx)} y={y(QS) + 18} className="fk-strong" style={{ fontSize: 12 }}>768</text>
      <text x={x(kx)} y={y(KS) - 16} className="fk-strong" style={{ fontSize: 12 }}>约 5.5K</text>
      {lab(262144, QS, 'Qwen 144 MiB', 'teal', 10)}
      <text x={x(262144) - 8} y={y(QKV * 262144) - 4} className="fk-strong fk-text-purple" textAnchor="end" style={{ fontSize: 12 }}>Qwen 48 GiB</text>
      {lab(1048576, KS, 'K3 414 MiB', 'teal', -8)}
      {lab(1048576, KKV * 1048576, 'K3 78 GiB', 'purple')}
    </>}</Figure>
  )
}


// 图 0：线性 attention 的一步。写入：S_t = S_{t-1} + v_t k_tᵀ；读取：o_t = S_t q_t。按 head dim 128 画，向量长度与矩阵边长相同。
export function StateStep() {
  const W = 760, Y = 64, N = 112, T = 12 // 矩阵边长与向量粗细
  const sx0 = 24, ox = 196, sx1 = 396, qx = sx1 + N + 14, rx = qx + T + 52
  const op = (x: number, t: string) => <text x={x} y={Y + N / 2} className="fk-stat" style={{ fontSize: 20 }}>{t}</text>
  return (
    <Figure
      title="线性 attention 的一步：写入外积，读取矩阵–向量乘"
      desc="写入：上一步的 state S_{t-1}（128 × 128）加上当前 token 的外积 v_t k_tᵀ，得到 S_t。读取：S_t 乘以 q_t，得到 128 维的输出 o_t。每个 head 的 state 为 128 × 128 × 4 B = 64 KiB。"
      width={W}
      height={Y + N + 54}
      caption="一个 head，head dim 为 128。写入与读取的开销都只取决于 state 的大小，与上下文长度无关。"
    >{() => <>
      <text x={sx0} y={18} className="fk-strong fk-text-teal" textAnchor="start">① 写入</text>
      <text x={sx1} y={18} className="fk-strong fk-text-purple" textAnchor="start">② 读取</text>
      <Cell x={sx0} y={Y} w={N} h={N} fill="teal" label="Sₜ₋₁" />
      {op(sx0 + N + 22, '+')}
      {/* 外积：v_t 竖条 × k_tᵀ 横条 */}
      <rect x={ox - T - 6} y={Y} width={T} height={N} rx={2} className="fk-fill-blue" />
      <rect x={ox} y={Y - T - 6} width={N} height={T} rx={2} className="fk-fill-ink" />
      <rect x={ox} y={Y} width={N} height={N} rx={3} className="fk-fill-blue" opacity={0.45} />
      <text x={ox - T / 2 - 6} y={Y + N + 14} className="fk-strong" style={{ fontSize: 12 }}>vₜ</text>
      <text x={ox + N + 14} y={Y - T / 2 - 6} className="fk-strong" style={{ fontSize: 12 }}>kₜᵀ</text>
      <text x={ox + N / 2} y={Y + N / 2} className="fk-strong" style={{ fill: '#fff' }}>vₜ kₜᵀ</text>
      {op(ox + N + 38, '=')}
      <Cell x={sx1} y={Y} w={N} h={N} fill="teal" label="Sₜ" focus />
      <rect x={qx} y={Y} width={T} height={N} rx={2} className="fk-fill-amber" />
      <text x={qx + T / 2} y={Y + N + 14} className="fk-strong" style={{ fontSize: 12 }}>qₜ</text>
      {op(qx + T + 24, '=')}
      <rect x={rx} y={Y} width={T} height={N} rx={2} className="fk-fill-green" />
      <text x={rx + T / 2} y={Y + N + 14} className="fk-strong" style={{ fontSize: 12 }}>oₜ</text>
      <text x={sx0 + N / 2} y={Y + N + 14} className="fk-muted fk-small">128 × 128</text>
      <text x={sx1 + N / 2} y={Y + N + 14} className="fk-muted fk-small">128 × 128</text>
      <text x={rx + T + 22} y={Y + 30} className="fk-stat" textAnchor="start" style={{ fontSize: 18 }}>64 KiB</text>
      <text x={rx + T + 22} y={Y + 50} className="fk-muted fk-small" textAnchor="start">每个 head 的 state，FP32</text>
      <text x={rx + T + 22} y={Y + 78} className="fk-stat" textAnchor="start" style={{ fontSize: 18 }}>3 MiB</text>
      <text x={rx + T + 22} y={Y + 98} className="fk-muted fk-small" textAnchor="start">Qwen3.8-27B 一层 48 个 head</text>
    </>}</Figure>
  )
}

// 图：S-NIAH 检索准确率（Gated DeltaNet 论文，1.3B 模型、100B token）。条长为准确率，0 到 100。
const NIAH: { task: string; ctx: string; scores: [number, number, number]; note: string }[] = [
  { task: '重复填充文本中找 pass-key', ctx: '8K', scores: [98.8, 30.4, 91.8], note: '只有 decay：pass-key 也被衰减' },
  { task: '真实文章中找数字', ctx: '4K', scores: [18.6, 56.2, 92.2], note: '只有 delta rule：持续写入的内容无法清理' },
  { task: '真实文章中找 UUID', ctx: '1K', scores: [85.2, 64.4, 86.6], note: '只有 decay：UUID 无法精确写入' },
]
const NIAH_MODELS: { name: string; fill: 'blue' | 'amber' | 'teal' }[] = [
  { name: 'DeltaNet', fill: 'blue' }, { name: 'Mamba2', fill: 'amber' }, { name: 'Gated DeltaNet', fill: 'teal' },
]

export function NiahBars() {
  const W = 760, X0 = 180, BW = 260, BH = 15, GAP = 4, GP = 3 * (BH + GAP) + 26
  return (
    <Figure
      title="S-NIAH 检索准确率：DeltaNet、Mamba2 与 Gated DeltaNet"
      desc="重复填充文本中找 pass-key（8K）：DeltaNet 98.8，Mamba2 30.4，Gated DeltaNet 91.8。真实文章中找数字（4K）：18.6、56.2、92.2。真实文章中找 UUID（1K）：85.2、64.4、86.6。"
      width={W}
      height={NIAH.length * GP + 4}
      below={<Legend items={[['blue', 'DeltaNet：只有 delta rule'], ['amber', 'Mamba2：只有 decay'], ['teal', 'Gated DeltaNet：两者都有']]} />}
      caption="准确率（%），条长从 0 到 100。右侧为分数低的一方缺少的机制。"
    >{() => NIAH.map((g, n) => {
      const y0 = n * GP + 4
      return <g key={g.task}>
        <text x={X0 - 14} y={y0 + BH + 2} className="fk-strong" textAnchor="end">{g.task}</text>
        <text x={X0 - 14} y={y0 + BH + 20} className="fk-muted fk-small" textAnchor="end">{`上下文 ${g.ctx}`}</text>
        {g.scores.map((s, i) => {
          const y = y0 + i * (BH + GAP)
          return <g key={i}>
            <rect x={X0} y={y} width={BW} height={BH} rx={2} className="fk-fill-void" />
            <rect x={X0} y={y} width={BW * s / 100} height={BH} rx={2} className={`fk-fill-${NIAH_MODELS[i].fill}`} />
            <text x={X0 + BW + 8} y={y + BH / 2} className="fk-stat" textAnchor="start" style={{ fontSize: 12 }}>{s.toFixed(1)}</text>
          </g>
        })}
        <text x={X0 + BW + 56} y={y0 + BH + 2} className="fk-muted" textAnchor="start">{g.note}</text>
      </g>
    })}</Figure>
  )
}

// 图：各混合模型的层排布，一格一层，按 config.json 的层类型着色。
const STRIPS: { name: string; sub: string; n: number; full: (i: number) => boolean; sparse?: boolean; ratio: string }[] = [
  { name: 'MiniMax-Text-01', sub: 'lightning attention', n: 80, full: i => i % 8 === 7, ratio: '70 : 10' },
  { name: 'Qwen3-Next-80B-A3B', sub: 'Gated DeltaNet', n: 48, full: i => i % 4 === 3, ratio: '36 : 12' },
  { name: 'Qwen3.8-27B', sub: 'Gated DeltaNet', n: 64, full: i => i % 4 === 3, ratio: '48 : 16' },
  { name: 'Qwen3.8-Flash-Next', sub: 'Gated DeltaNet + QSA', n: 48, full: i => i % 4 === 3, sparse: true, ratio: '36 : 12' },
  { name: 'Kimi Linear 48B-A3B', sub: 'KDA', n: 27, full: i => i % 4 === 3 || i === 26, ratio: '20 : 7' },
  { name: 'Kimi K3', sub: 'KDA', n: 93, full: i => i % 4 === 3 || i === 92, ratio: '69 : 24' },
]

export function LayerStrips() {
  const W = 760, X0 = 170, P = 5.6, CH = 22, PITCH = 40
  return (
    <Figure
      title="各混合模型的层排布"
      desc="一格一层。MiniMax-Text-01 共 80 层，每 8 层的最后一层为 full attention；Qwen3-Next、Qwen3.8-27B 与 Qwen3.8-Flash-Next 每 4 层的最后一层为 full attention，Flash-Next 的这些层为 QSA；Kimi Linear 共 27 层、Kimi K3 共 93 层，每 4 层的最后一层为 MLA，最后一层也是 MLA。"
      width={W}
      height={STRIPS.length * PITCH + 6}
      below={<Legend items={[['teal', '线性层'], ['purple', 'full attention 层'], ['blue', '稀疏 attention 层（QSA）']]} />}
      caption="一格一层，从左到右为第 1 层到最后一层；右侧为线性层与其余层的层数。"
    >{() => STRIPS.map((r, n) => {
      const y = n * PITCH + 6
      return <g key={r.name}>
        <text x={X0 - 14} y={y + 6} className="fk-strong" textAnchor="end">{r.name}</text>
        <text x={X0 - 14} y={y + 22} className="fk-muted fk-small" textAnchor="end">{r.sub}</text>
        {Array.from({ length: r.n }, (_, i) => <rect key={i} x={X0 + i * P} y={y} width={P - 1.2} height={CH} rx={1}
          className={r.full(i) ? (r.sparse ? 'fk-fill-blue' : 'fk-fill-purple') : 'fk-fill-teal'} />)}
        <text x={X0 + r.n * P + 10} y={y + CH / 2} className="fk-stat" textAnchor="start">{r.ratio}</text>
      </g>
    })}</Figure>
  )
}
