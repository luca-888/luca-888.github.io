import { Cell, Figure, Legend, Link, type Fill } from './figure-kit'

// 语义色：decode 蓝、prefill 紫、被 prefill 挡住的 decode 红、排队灰、已完成仍在计算琥珀。
type Kind = 'decode' | 'prefill' | 'stall' | 'wait' | 'waste'
type Seg = { start: number; end: number; kind: Kind; label?: string }
type Row = 'A' | 'B' | 'C'
type Timeline = { rows: Record<Row, Seg[]>; ttft: number; gap: number }

const FILL: Record<Kind, Fill> = { decode: 'blue', prefill: 'purple', stall: 'red', wait: 'gray', waste: 'amber' }

// 构造示例：A、B 已在 decode，还需生成 6、3 个 token；C 在 t=1 到达，prompt 16 个 token，共生成 3 个 token。
// 一次迭代的耗时 = max(1, tokens / 4)：不足 4 个 token 时受权重读取限制，超过后随 token 数线性增长。
const NEED = { A: 6, B: 3, C: 3 }
const PROMPT = 16
const ARRIVAL = 1
const BUDGET = 8
const cost = (tokens: number) => Math.max(1, tokens / 4)

function simulate(policy: 'static' | 'prefill-first' | 'budget'): Timeline {
  const rows: Record<Row, Seg[]> = { A: [], B: [], C: [] }
  const left = { ...NEED }
  const stamps: Record<Row, number[]> = { A: [], B: [], C: [] }
  const push = (row: Row, start: number, end: number, kind: Kind, label?: string) => rows[row].push({ start, end, kind, label })
  const decode = (row: Row, start: number, end: number) => { push(row, start, end, 'decode'); left[row]--; stamps[row].push(end) }
  let t = 0
  let prefillLeft = PROMPT

  if (policy === 'static') {
    // 请求级：{A, B} 跑到全部结束，C 排队；结束的请求仍占着 batch 里的位置。
    while (left.A > 0 || left.B > 0) {
      for (const row of ['A', 'B'] as const) left[row] > 0 ? decode(row, t, t + 1) : push(row, t, t + 1, 'waste')
      t += 1
    }
    push('C', ARRIVAL, t, 'wait')
    push('C', t, t + cost(PROMPT), 'prefill', String(PROMPT)); t += cost(PROMPT); left.C--; stamps.C.push(t)
    while (left.C > 0) { decode('C', t, t + 1); t += 1 }
  } else {
    while (left.A > 0 || left.B > 0 || left.C > 0 || prefillLeft > 0) {
      const running = (['A', 'B'] as const).filter(row => left[row] > 0)
      const cReady = t >= ARRIVAL && prefillLeft > 0
      const cDecoding = prefillLeft === 0 && left.C > 0
      if (t < ARRIVAL || !cReady) {
        const decoding: Row[] = [...running, ...(cDecoding ? ['C' as const] : [])]
        const dt = cost(decoding.length)
        for (const row of decoding) decode(row, t, t + dt)
        t += dt
      } else if (policy === 'prefill-first') {
        const dt = cost(prefillLeft)
        for (const row of running) push(row, t, t + dt, 'stall')
        push('C', t, t + dt, 'prefill', String(prefillLeft))
        t += dt; prefillLeft = 0; left.C--; stamps.C.push(t)
      } else {
        const chunk = Math.min(prefillLeft, BUDGET - running.length)
        const dt = cost(running.length + chunk)
        for (const row of running) decode(row, t, t + dt)
        push('C', t, t + dt, 'prefill', String(chunk))
        t += dt; prefillLeft -= chunk
        if (prefillLeft === 0) { left.C--; stamps.C.push(t) }
      }
    }
  }
  const gaps = stamps.A.slice(1).map((end, i) => end - stamps.A[i])
  return { rows, ttft: stamps.C[0] - ARRIVAL, gap: Math.max(...gaps) }
}

const SCALE = 44
const X0 = 76
const TIME_END = 12

export function VllmSchedulingTimelines() {
  const panels: [string, string, Timeline][] = [
    ['请求级 batch', '{A, B} 跑完才接新请求', simulate('static')],
    ['迭代级，prefill 优先', 'C 一到就整段 prefill', simulate('prefill-first')],
    [`迭代级，token budget = ${BUDGET}`, 'C 的 prefill 切块，与 decode 同批', simulate('budget')],
  ]
  const panelH = 118
  const num = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0$/, ''))
  return (
    <Figure
      title="三种调度下同一组请求的时间线"
      desc="A、B 在 decode 时 C 到达。请求级 batch 下 C 排队等待 A、B 全部结束，B 结束后仍被计算。迭代级、prefill 优先时 C 的整段 prefill 让 A、B 停顿。迭代级、token budget 为 8 时，C 的 prefill 分成三块，与 A、B 的 decode 同批执行。"
      height={panelH * 3 + 34}
      below={<Legend items={[['blue', 'decode 一个 token'], ['purple', 'prefill（数字为 token 数）'], ['red', 'decode 被 prefill 挡住'], ['gray', '排队等待'], ['amber', '已完成仍在计算']]} />}
      caption="构造示例：色块宽度为迭代耗时，一次纯 decode 迭代记为 1；耗时 = max(1, tokens / 4)。A、B 还需生成 6、3 个 token，C 在 t=1 到达，prompt 16 个 token。TTFT 从 C 到达算起，TPOT 间隔取 A 相邻两个 token 的最大时间差。"
    >{() => <>
      {panels.map(([title, sub, timeline], panel) => {
        const y0 = panel * panelH
        return <g key={title}>
          <text x="0" y={y0 + 12} className="fk-title" textAnchor="start">{title}</text>
          <text x={title.length * 14 + 12} y={y0 + 12} className="fk-muted" textAnchor="start">{sub}</text>
          {(['A', 'B', 'C'] as const).map((row, index) => {
            const y = y0 + 28 + index * 24
            return <g key={row}>
              <text x="0" y={y + 10} className="fk-strong" textAnchor="start">{row}</text>
              {timeline.rows[row].map((seg, i) => <Cell key={i} x={X0 + seg.start * SCALE + 0.5} y={y} w={(seg.end - seg.start) * SCALE - 1} h={20} rx={2}
                fill={FILL[seg.kind]} label={seg.label} small />)}
            </g>
          })}
          <path d={`M${X0} ${y0 + 100} H${X0 + TIME_END * SCALE}`} className="fk-link fk-line-muted" />
          {Array.from({ length: TIME_END / 2 + 1 }, (_, i) => <text key={i} x={X0 + i * 2 * SCALE} y={y0 + 110} className="fk-muted fk-small">{i * 2}</text>)}
          <text x={X0 + TIME_END * SCALE + 12} y={y0 + 46} className="fk-stat fk-small" textAnchor="start">C 的 TTFT：{num(timeline.ttft)}</text>
          <text x={X0 + TIME_END * SCALE + 12} y={y0 + 66} className="fk-stat fk-small" textAnchor="start">A 的最大间隔：{num(timeline.gap)}</text>
        </g>
      })}
    </>}</Figure>
  )
}

// 单步 token budget 的分配：先 running，再 waiting，耗尽即停。
export function VllmStepBudget() {
  const cell = 40
  const x0 = 24
  const budget = 12
  const groups: { name: string; tokens: number; fill: Fill; note: string; from: number }[] = [
    { name: 'R1', tokens: 1, fill: 'blue', note: 'decode', from: 0 },
    { name: 'R2', tokens: 1, fill: 'blue', note: 'decode', from: 1 },
    { name: 'R3', tokens: 6, fill: 'purple', note: 'prefill 的后续块（还剩 6）', from: 2 },
    { name: 'W1', tokens: 4, fill: 'purple', note: '新请求，只分到 4 / 22', from: 8 },
  ]
  return (
    <Figure
      title="一次 schedule() 中 token budget 的分配"
      desc="token budget 为 12。running 队列中 R1、R2 各一个 decode token，R3 是未完成的 prefill，还剩 6 个 token，全部放入；剩余 4 个 token 分给 waiting 队首的新请求 W1，W1 还有 22 个 token 要算，只分到 4 个；W2 因 budget 耗尽本步不被调度。第二行：KV block 不足时，running 队尾的 R4 被抢占，回到 waiting 队首。"
      height={280}
      caption="构造示例，budget = 12。W1 的 prompt 共 30 个 token，prefix cache 命中 8 个，命中的部分不占 budget，未命中的 22 个本步只分到 4 个。"
    >{arrow => <>
      <text x={x0} y="18" className="fk-title" textAnchor="start">一步的 token budget</text>
      <text x={x0 + 150} y="18" className="fk-muted" textAnchor="start">每格 1 个 token，最多 {budget} 个</text>
      <path d={`M${x0} 42 H${x0 + 8 * cell - 4}`} className="fk-link fk-line-muted" />
      <text x={x0} y="32" className="fk-muted" textAnchor="start">① running：decode 与未完成的 prefill，先调度</text>
      <path d={`M${x0 + 8 * cell} 42 H${x0 + budget * cell}`} className="fk-link fk-line-muted" />
      <text x={x0 + 8 * cell} y="32" className="fk-muted" textAnchor="start">② waiting</text>
      {Array.from({ length: budget }, (_, i) => {
        const group = groups.find(g => i >= g.from && i < g.from + g.tokens)!
        return <Cell key={i} x={x0 + i * cell + 1} y={52} w={cell - 2} h={34} rx={3} fill={group.fill} />
      })}
      <text x={x0 + cell} y="106" className="fk-strong">R1, R2</text>
      <text x={x0 + cell} y="124" className="fk-muted fk-small">decode，各 1 个</text>
      <text x={x0 + 5 * cell} y="106" className="fk-strong">R3</text>
      <text x={x0 + 5 * cell} y="124" className="fk-muted fk-small">prefill 的后续块，还剩 6 个</text>
      <text x={x0 + 10 * cell} y="106" className="fk-strong">W1</text>
      <text x={x0 + 10 * cell} y="124" className="fk-muted fk-small">新请求，只分到 4 / 22</text>
      <rect x={x0 + budget * cell + 14} y="52" width="74" height="34" rx="3" className="fk-fill-gray" />
      <text x={x0 + budget * cell + 51} y="69" className="fk-cell-text fk-small">W2</text>
      <text x={x0 + budget * cell + 51} y="106" className="fk-muted fk-small">budget 耗尽，等下一步</text>

      <text x={x0} y="168" className="fk-title" textAnchor="start">KV block 不足时</text>
      <text x={x0 + 120} y="168" className="fk-muted" textAnchor="start">allocate_slots 失败，抢占 running 队尾</text>
      {['R1', 'R2', 'R3', 'R4'].map((name, i) => <Cell key={name} x={x0 + i * 64} y={188} w={58} h={34} rx={3} fill={name === 'R4' ? 'red' : 'blue'} label={name} />)}
      <Link d={`M${x0 + 4 * 64 + 4} 205 H${x0 + 340 - 8}`} arrow={arrow} />
      {['R4', 'W1', 'W2'].map((name, i) => <Cell key={name} x={x0 + 340 + i * 64} y={188} w={58} h={34} rx={3} fill={name === 'R4' ? 'red' : 'empty'} label={name} />)}
      <text x={x0 + 2 * 64 + 29} y="240" className="fk-muted fk-small">running</text>
      <text x={x0 + 340 + 96} y="240" className="fk-muted fk-small">waiting，R4 回到队首</text>
      <text x={x0 + 5 * 64 + 160} y="264" className="fk-strong fk-small">R4 释放全部 block，num_computed_tokens = 0，之后从头重算</text>
    </>}</Figure>
  )
}
