import { Cell, Figure, Legend, useCompact, type Fill } from './figure-kit'

// 语义色（两张图一致）：draft 分布 q purple，target 分布 p teal，接受 green，
// 拒绝 red，拒绝后从残差分布重新采样 amber。

const TOKENS = ['A', 'B', 'C', 'D']
const BASE = 168 // 柱基线
const SCALE = 150 // 概率 1.0 的柱高

type Position = { p: number[]; q: number[]; draft: number; u: number }

// 构造示例：词表只有 4 个 token，三个位置的 p、q 是手工设定的。
const POSITIONS: Position[] = [
  { p: [0.5, 0.3, 0.1, 0.1], q: [0.3, 0.4, 0.2, 0.1], draft: 0, u: 0.62 },
  { p: [0.2, 0.5, 0.2, 0.1], q: [0.1, 0.3, 0.5, 0.1], draft: 2, u: 0.3 },
  { p: [0.1, 0.6, 0.2, 0.1], q: [0.4, 0.2, 0.3, 0.1], draft: 0, u: 0.6 },
]
const OUTPUT: [string, Fill][] = [['A', 'green'], ['C', 'green'], ['B', 'amber']]

// 图 1：一次验证的完整路径。紧凑模式（约 520 宽）下三个位置竖排，判定写在柱状图右侧。
export function SpecVerifyPath() {
  const compact = useCompact()
  const oy = compact ? 620 : 332
  return (
    <Figure
      title="一次验证：三个 draft token 的逐位判定"
      desc="词表只有 A 到 D 四个 token。位置 1，draft 采到 A，target 概率 0.5 高于 draft 概率 0.3，必然接受。位置 2，draft 采到 C，接受概率 0.2 除以 0.5 等于 0.4，随机数 0.30 小于 0.4，接受。位置 3，draft 采到 A，接受概率 0.1 除以 0.4 等于 0.25，随机数 0.60 大于 0.25，拒绝；残差分布 max(p−q,0) 只剩 B，输出 B，本轮结束。最终一轮输出 A、C、B 三个 token。"
      width={compact ? 520 : 760}
      height={compact ? 706 : 372}
      below={<Legend items={[['purple', 'draft 分布 q'], ['teal', 'target 分布 p'], ['green', '接受'], ['red', '拒绝'], ['amber', '残差采样']]} />}
      caption="图 1　构造示例：p、q 为设定值，u 为一次均匀随机数。柱上数字是概率，圈出的是 draft 采到的 token。"
    >{() => <>
      {POSITIONS.map((pos, i) => {
        const x0 = compact ? 8 : 12 + i * 250
        const y0 = compact ? i * 204 : 0
        const base = BASE + y0
        const dx = compact ? 392 : x0 + 115 // 判定文字的中线
        const dy = compact ? y0 - 150 : 0
        const t = pos.draft
        const ratio = pos.p[t] / pos.q[t]
        const accept = pos.u <= Math.min(1, ratio)
        const rejected = !accept
        return <g key={i}>
          <text x={x0 + 115} y={16 + y0} className="fk-strong">位置 {i + 1}</text>
          {TOKENS.map((tok, j) => {
            const bx = x0 + 22 + j * 52
            return <g key={tok}>
              <rect x={bx} y={base - SCALE * pos.q[j]} width={20} height={SCALE * pos.q[j]} rx={2} className="fk-fill-purple" />
              <rect x={bx + 22} y={base - SCALE * pos.p[j]} width={20} height={SCALE * pos.p[j]} rx={2} className="fk-fill-teal" />
              <text x={bx + 10} y={base - SCALE * pos.q[j] - 9} className="fk-muted fk-small">{pos.q[j].toFixed(1)}</text>
              <text x={bx + 32} y={base - SCALE * pos.p[j] - 9} className="fk-muted fk-small">{pos.p[j].toFixed(1)}</text>
              <text x={bx + 21} y={base + 14} className="fk-strong">{tok}</text>
            </g>
          })}
          <rect x={x0 + 22 + t * 52 - 3} y={base - 100} width={48} height={124} rx={4} className="fk-focus" />
          <text x={x0 + 115} y={38 + y0} className="fk-muted">draft 采到 {TOKENS[t]}</text>
          {!compact && <line x1={x0 + 10} x2={x0 + 232} y1={214} y2={214} className="fk-rule" />}
          <text x={dx} y={234 + dy} className="fk-mono">min(1, {pos.p[t].toFixed(1)}/{pos.q[t].toFixed(1)}) = {Math.min(1, ratio).toFixed(2)}</text>
          <text x={dx} y={254 + dy} className="fk-mono">u = {pos.u.toFixed(2)} {accept ? '≤' : '>'} {Math.min(1, ratio).toFixed(2)}</text>
          <Cell x={dx - 50} y={270 + dy} w={100} h={26} fill={accept ? 'green' : 'red'} label={accept ? '接受' : '拒绝'} rx={4} />
          {rejected && (compact
            ? <><text x={dx} y={312 + dy} className="fk-muted">残差 max(p−q, 0) 只剩 B</text><text x={dx} y={330 + dy} className="fk-muted">→ 输出 B</text></>
            : <text x={dx} y="312" className="fk-muted">残差 max(p−q, 0) 只剩 B → 输出 B</text>)}
        </g>
      })}
      <text x="12" y={oy + 14} className="fk-strong" textAnchor="start">本轮输出</text>
      {OUTPUT.map(([tok, fill], i) => <Cell key={i} x={96 + i * 36} y={oy} w={30} h={28} fill={fill} label={tok} />)}
      {compact
        ? <><text x="12" y={oy + 56} className="fk-muted" textAnchor="start">一次 target forward 产出 3 个 token；</text><text x="12" y={oy + 76} className="fk-muted" textAnchor="start">三个全被接受时，再从位置 4 的 p 采一个 bonus token</text></>
        : <text x="216" y={oy + 14} className="fk-muted" textAnchor="start">一次 target forward 产出 3 个 token；三个全被接受时，再从位置 4 的 p 采一个 bonus token</text>}
    </>}</Figure>
  )
}

// 图 2：p、q 的重叠面积 = 接受概率 α
const P2 = [0.1, 0.6, 0.2, 0.1]
const Q2 = [0.4, 0.2, 0.3, 0.1]

export function SpecOverlap() {
  const compact = useCompact()
  const notes: [string, string][] = [['被接受的质量', 'Σ min(p, q) = 0.6'], ['被拒绝的质量', 'Σ max(q−p, 0) = 0.4'], ['重采样补上的质量', 'Σ max(p−q, 0) = 0.4']]
  const bw = 40
  const gx = 92
  const base = 200
  const sc = 220
  return (
    <Figure
      title="p 与 q 的重叠面积等于接受概率"
      desc="沿用位置 3 的分布。每个 token 有两根柱：左侧 q，右侧 p。两根柱各自被分成公共部分 min(p,q) 与多出来的部分。公共部分之和为 0.6，即 α；q 多出来的部分之和 0.4 是被拒绝的概率质量；p 多出来的部分之和 0.4 是残差，恰好由拒绝后的重新采样补上。"
      width={compact ? 520 : 760}
      height={compact ? 344 : 274}
      below={<Legend items={[['blue', 'min(p, q)：直接接受'], ['purple', 'q 多出的部分：被拒绝'], ['teal', 'p 多出的部分：残差，由重采样补上']]} />}
      caption="图 2　沿用图 1 位置 3 的分布。α = Σ min(p, q) = 0.6：draft 的一次采样有 0.6 的概率被接受。"
    >{() => <>
      {TOKENS.map((tok, j) => {
        const x = 24 + j * (2 * bw + gx / 2)
        const m = Math.min(P2[j], Q2[j])
        const qx = Q2[j] - m
        const px = P2[j] - m
        return <g key={tok}>
          <rect x={x} y={base - sc * m} width={bw} height={sc * m} className="fk-fill-blue" />
          {qx > 0 && <rect x={x} y={base - sc * Q2[j]} width={bw} height={sc * qx} className="fk-fill-purple" />}
          <rect x={x + bw + 4} y={base - sc * m} width={bw} height={sc * m} className="fk-fill-blue" />
          {px > 0 && <rect x={x + bw + 4} y={base - sc * P2[j]} width={bw} height={sc * px} className="fk-fill-teal" />}
          <text x={x + bw / 2} y={base + 14} className="fk-muted">q</text>
          <text x={x + bw + 4 + bw / 2} y={base + 14} className="fk-muted">p</text>
          <text x={x + bw + 2} y={base + 34} className="fk-strong">{tok}</text>
          <text x={x + bw / 2} y={base - sc * Q2[j] - 9} className="fk-muted fk-small">{Q2[j].toFixed(1)}</text>
          <text x={x + bw + 4 + bw / 2} y={base - sc * P2[j] - 9} className="fk-muted fk-small">{P2[j].toFixed(1)}</text>
        </g>
      })}
      <line x1="16" x2="480" y1={base} y2={base} className="fk-rule" />
      {notes.map(([name, value], k) => compact
        ? <g key={name}><text x="24" y={base + 72 + k * 30} className="fk-strong" textAnchor="start">{name}</text><text x="250" y={base + 72 + k * 30} className="fk-mono" textAnchor="start">{value}</text></g>
        : <g key={name}><text x="520" y={52 + k * 60} className="fk-strong" textAnchor="start">{name}</text><text x="520" y={76 + k * 60} className="fk-mono" textAnchor="start">{value}</text></g>)}
    </>}</Figure>
  )
}
