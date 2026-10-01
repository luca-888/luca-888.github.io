import { Cell, Figure, Legend, Link, useCompact, type Fill } from './figure-kit'

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

const TIMELINE_LEGEND: [Fill, string][] = [['blue', 'decode 一个 token'], ['purple', 'prefill（数字为 token 数）'], ['red', 'decode 被 prefill 挡住'], ['gray', '排队等待'], ['amber', '已完成仍在计算']]
const TIMELINE_DESC = 'A、B 在 decode 时 C 到达。请求级 batch 下 C 排队等待 A、B 全部结束，B 结束后仍被计算。迭代级、prefill 优先时 C 的整段 prefill 让 A、B 停顿。迭代级、token budget 为 8 时，C 的 prefill 分成三块，与 A、B 的 decode 同批执行。'
const TIMELINE_CAPTION = '构造示例：色块宽度为迭代耗时，一次纯 decode 迭代记为 1；耗时 = max(1, tokens / 4)。A、B 还需生成 6、3 个 token，C 在 t=1 到达，prompt 16 个 token。TTFT 从 C 到达算起，A 的最大 TBT 取其相邻两个 token 的最大时间差。'
const fmt = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0$/, ''))

export function VllmSchedulingTimelines() {
  const compact = useCompact()
  const panels: [string, string, Timeline][] = [
    ['请求级 batch', '{A, B} 跑完才接新请求', simulate('static')],
    ['迭代级，prefill 优先', 'C 一到就整段 prefill', simulate('prefill-first')],
    [`迭代级，token budget = ${BUDGET}`, 'C 的 prefill 切块，与 decode 同批', simulate('budget')],
  ]
  if (compact) {
    // 窄版：统计数字移到标题下方，时间轴占满宽度。
    const scale = 38, x0 = 30, panelH = 146
    return (
      <Figure title="三种调度下同一组请求的时间线" desc={TIMELINE_DESC} width={520} height={panelH * 3 - 4}
        below={<Legend items={TIMELINE_LEGEND} />} caption={TIMELINE_CAPTION}>{() => <>
        {panels.map(([title, sub, timeline], panel) => {
          const y0 = panel * panelH
          return <g key={title}>
            <text x="0" y={y0 + 10} className="fk-title" textAnchor="start">{title}</text>
            <text x="0" y={y0 + 32} className="fk-muted" textAnchor="start">{sub}</text>
            <text x="490" y={y0 + 32} className="fk-stat" textAnchor="end">C 的 TTFT {fmt(timeline.ttft)} · A 的最大 TBT {fmt(timeline.gap)}</text>
            {(['A', 'B', 'C'] as const).map((row, index) => {
              const y = y0 + 46 + index * 26
              return <g key={row}>
                <text x="0" y={y + 12} className="fk-strong" textAnchor="start">{row}</text>
                {timeline.rows[row].map((seg, i) => <Cell key={i} x={x0 + seg.start * scale + 0.5} y={y} w={(seg.end - seg.start) * scale - 1} h={22} rx={2}
                  fill={FILL[seg.kind]} label={seg.label} small />)}
              </g>
            })}
            <path d={`M${x0} ${y0 + 124} H${x0 + TIME_END * scale}`} className="fk-link fk-line-muted" />
            {Array.from({ length: TIME_END / 2 + 1 }, (_, i) => <text key={i} x={x0 + i * 2 * scale} y={y0 + 136} className="fk-muted fk-small">{i * 2}</text>)}
          </g>
        })}
      </>}</Figure>
    )
  }
  const panelH = 118
  const num = fmt
  return (
    <Figure
      title="三种调度下同一组请求的时间线"
      desc={TIMELINE_DESC}
      height={panelH * 3 + 34}
      below={<Legend items={TIMELINE_LEGEND} />}
      caption={TIMELINE_CAPTION}
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
          <text x={X0 + TIME_END * SCALE + 12} y={y0 + 66} className="fk-stat fk-small" textAnchor="start">A 的最大 TBT：{num(timeline.gap)}</text>
        </g>
      })}
    </>}</Figure>
  )
}

// 3 个槽位 × 8 步：静态 batch 与 iteration-level 调度对比。请求的长度（步数）：A 8、B 2、C 4、D 3、E 4、F 3。
const LENGTHS: Record<string, number> = { A: 8, B: 2, C: 4, D: 3, E: 4, F: 3 }
const STEPS = 8

function slotPlans() {
  const initial = ['A', 'B', 'C']
  const queue = ['D', 'E', 'F']
  const statics: (string | null)[][] = initial.map(name => Array.from({ length: STEPS }, (_, t) => (t < LENGTHS[name] ? name : null)))
  const continuous: (string | null)[][] = initial.map(() => [])
  const pending = [...queue]
  const current: [string, number][] = initial.map(name => [name, 0])
  for (let t = 0; t < STEPS; t++) {
    current.forEach((state, slot) => {
      if (state[1] >= LENGTHS[state[0]] && pending.length) current[slot] = [pending.shift()!, 0]
      const [name, done] = current[slot]
      continuous[slot].push(done < LENGTHS[name] ? name : null)
      current[slot] = [name, done + 1]
    })
  }
  return { statics, continuous }
}

const SLOTS_DESC = '3 个槽位，8 个迭代步。静态 batch 中 A、B、C 分别占满 8 步，B 在第 2 步结束、C 在第 4 步结束后槽位空转，D、E、F 要等到第 9 步以后。iteration-level 调度中 B 结束后 D 立即补位，C 结束后 E 补位，D 结束后 F 补位，槽位始终有请求在算。'
const SLOTS_LEGEND: [Fill, string][] = [['blue', '请求正在生成 token（字母为请求）'], ['amber', '已结束，仍占着位置空转']]
const SLOTS_CAPTION = '构造示例：3 个槽位，A、B、C 生成 8、2、4 个 token，D、E、F 在队列中，需要 3、4、3 步。每格是一次迭代。'
const REQ_FILL: Record<string, Fill> = { A: 'blue', B: 'purple', C: 'teal', D: 'green', E: 'blue', F: 'purple' }

export function VllmBatchingSlots() {
  const compact = useCompact()
  const { statics, continuous } = slotPlans()
  if (compact) {
    // 窄版：两个面板上下排。
    const pitch = 56, x0 = 44, panelH = 168
    const rows: [string, string, (string | null)[][]][] = [['静态 batch', '{A, B, C} 跑完才接新请求', statics], ['iteration-level', '每步都可换人', continuous]]
    return (
      <Figure title="静态 batch 与 iteration-level 调度的槽位占用" desc={SLOTS_DESC} width={520} height={panelH * 2 - 6}
        below={<Legend items={SLOTS_LEGEND} />} caption={SLOTS_CAPTION}>{() => <>
        {rows.map(([title, sub, plan], panel) => {
          const y0 = panel * panelH
          return <g key={title}>
            <text x="0" y={y0 + 10} className="fk-title" textAnchor="start">{title}</text>
            <text x="500" y={y0 + 10} className="fk-muted" textAnchor="end">{sub}</text>
            {plan.map((row, slot) => <g key={slot}>
              <text x="0" y={y0 + 46 + slot * 34} className="fk-muted" textAnchor="start">槽 {slot + 1}</text>
              {row.map((name, t) => <Cell key={t} x={x0 + t * pitch} y={y0 + 32 + slot * 34} w={pitch - 6} h={28} rx={2}
                fill={name ? REQ_FILL[name] : 'amber'} label={name ?? undefined} />)}
            </g>)}
            {Array.from({ length: STEPS }, (_, t) => <text key={t} x={x0 + t * pitch + (pitch - 6) / 2} y={y0 + 146} className="fk-muted fk-small">{t + 1}</text>)}
          </g>
        })}
      </>}</Figure>
    )
  }
  const cell = 38
  const panels: [string, string, (string | null)[][], number][] = [
    ['静态 batch', '{A, B, C} 跑完才接新请求', statics, 0],
    ['iteration-level', '每步都可换人', continuous, 400],
  ]
  return (
    <Figure
      title="静态 batch 与 iteration-level 调度的槽位占用"
      desc={SLOTS_DESC}
      height={176}
      below={<Legend items={SLOTS_LEGEND} />}
      caption={SLOTS_CAPTION}
    >{() => <>
      {panels.map(([title, sub, plan, x0]) => <g key={title}>
        <text x={x0} y="12" className="fk-title" textAnchor="start">{title}</text>
        <text x={x0 + title.length * 14 + 10} y="12" className="fk-muted" textAnchor="start">{sub}</text>
        {plan.map((row, slot) => <g key={slot}>
          <text x={x0} y={46 + slot * 28} className="fk-muted fk-small" textAnchor="start">槽 {slot + 1}</text>
          {row.map((name, t) => <Cell key={t} x={x0 + 36 + t * (cell - 8) + 0.5} y={34 + slot * 28} w={cell - 9} h={24} rx={2}
            fill={name ? REQ_FILL[name] : 'amber'} label={name ?? undefined} small />)}
        </g>)}
        {Array.from({ length: STEPS }, (_, t) => <text key={t} x={x0 + 36 + t * (cell - 8) + (cell - 9) / 2} y="128" className="fk-muted fk-small">{t + 1}</text>)}
        <text x={x0 + 36 + (STEPS * (cell - 8)) / 2} y="146" className="fk-muted fk-small">迭代步</text>
      </g>)}
    </>}</Figure>
  )
}

// selective batching：按请求 pad 的 [3, 5, H] 与展平后的 [7, H]；attention 按 cu_seqlens 分段。颜色沿用槽位图的请求色。
export function VllmFlattenedBatch() {
  const compact = useCompact()
  const reqs: { name: string; len: number }[] = [{ name: 'A', len: 5 }, { name: 'B', len: 1 }, { name: 'C', len: 1 }]
  const pad = 34
  const step = 42
  const gap = 14
  const x1 = 250
  const tokens = reqs.flatMap(r => Array.from({ length: r.len }, () => r.name))
  // attention 行里每个 token 的横坐标：段与段之间留出间隔。
  const segX = tokens.map((_, i) => x1 + i * step + (i >= 5 ? gap : 0) + (i >= 6 ? gap : 0))
  const bounds = [0, 5, 6, 7].map(b => b === 0 ? x1 - 1 : b === 7 ? segX[6] + step - 1 : segX[b] - gap / 2 - 1)
  const legend: [Fill, string][] = [['blue', 'A：prefill，本步 5 个 token'], ['purple', 'B：decode，1 个'], ['teal', 'C：decode，1 个'], ['empty', 'padding']]
  const desc = '上：三个请求按 [batch, seq_len, H] 组织，A 有 5 个 token，B、C 各 1 个，B、C 各 pad 4 格。中：7 个 token 首尾相接成 [7, H]，Linear、LayerNorm、MLP 整条一起算。下：attention 按 cu_seqlens_q = [0, 5, 6, 7] 分成 A、B、C 三段，各自计算。'
  if (compact) {
    // 窄版：pad、展平、分段三行上下排。
    const p = 62, g = 18
    const xs = tokens.map((_, i) => i * p + (i >= 5 ? g : 0) + (i >= 6 ? g : 0))
    const bx = [0, xs[5] - g / 2, xs[6] - g / 2, xs[6] + p - 6]
    return (
      <Figure title="展平后的 batch 与 attention 的分段" desc={desc} width={520} height={352}
        below={<Legend items={legend} />} caption="构造示例。每格是一个 token，对应张量中长度为 H 的一行。">{() => <>
        <text x="0" y="10" className="fk-title" textAnchor="start">按请求 pad：[3, 5, H]</text>
        {reqs.map((r, row) => <g key={r.name}>
          <text x="0" y={42 + row * 34} className="fk-strong" textAnchor="start">{r.name}</text>
          {Array.from({ length: 5 }, (_, i) => <Cell key={i} x={26 + i * p} y={28 + row * 34} w={p - 6} h={28} rx={2}
            fill={i < r.len ? REQ_FILL[r.name] : 'empty'} label={i < r.len ? r.name : undefined} />)}
        </g>)}
        <text x={26 + 5 * p + 6} y="76" className="fk-muted" textAnchor="start">B、C 各 4 格</text>
        <text x={26 + 5 * p + 6} y="96" className="fk-muted" textAnchor="start">padding 白算</text>

        <text x="0" y="148" className="fk-title" textAnchor="start">展平：[∑L, H] = [7, H]</text>
        <text x="0" y="172" className="fk-muted" textAnchor="start">Linear、LayerNorm、MLP：整条一起算</text>
        {tokens.map((name, i) => <Cell key={i} x={i * p} y={184} w={p - 6} h={28} rx={2} fill={REQ_FILL[name]} label={name} />)}
        <text x="0" y="238" className="fk-muted" textAnchor="start">Attention：按段各自计算，段间不交互</text>
        {tokens.map((name, i) => <Cell key={i} x={xs[i]} y={250} w={p - 6} h={28} rx={2} fill={REQ_FILL[name]} label={name} />)}
        {bx.map((x, i) => <g key={i}>
          <path d={`M${x} 284 V292`} className="fk-link fk-line-muted" />
          <text x={x} y="304" className="fk-strong">{[0, 5, 6, 7][i]}</text>
        </g>)}
        <text x="0" y="336" className="fk-strong" textAnchor="start">cu_seqlens_q = [0, 5, 6, 7]</text>
      </>}</Figure>
    )
  }
  return (
    <Figure
      title="展平后的 batch 与 attention 的分段"
      desc={desc.replace('上：', '左：').replace('中：', '右上：').replace('下：', '右下：')}
      height={190}
      below={<Legend items={legend} />}
      caption="构造示例。每格是一个 token，对应张量中长度为 H 的一行。"
    >{() => <>
      <text x="0" y="12" className="fk-title" textAnchor="start">按请求 pad</text>
      <text x="90" y="12" className="fk-muted" textAnchor="start">[3, 5, H]</text>
      {reqs.map((r, row) => <g key={r.name}>
        <text x="0" y={52 + row * 34} className="fk-strong" textAnchor="start">{r.name}</text>
        {Array.from({ length: 5 }, (_, i) => <Cell key={i} x={20 + i * pad} y={36 + row * 34} w={pad - 4} h={26} rx={2}
          fill={i < r.len ? REQ_FILL[r.name] : 'empty'} label={i < r.len ? r.name : undefined} small />)}
      </g>)}
      <text x="20" y="152" className="fk-muted fk-small" textAnchor="start">B、C 各有 4 格 padding 在白算</text>

      <text x={x1} y="12" className="fk-title" textAnchor="start">展平</text>
      <text x={x1 + 40} y="12" className="fk-muted" textAnchor="start">[∑L, H] = [7, H]</text>
      <text x={x1} y="36" className="fk-muted fk-small" textAnchor="start">Linear、LayerNorm、MLP：整条一起算</text>
      {tokens.map((name, i) => <Cell key={i} x={x1 + i * step} y={44} w={step - 4} h={28} rx={2} fill={REQ_FILL[name]} label={name} small />)}
      <text x={x1} y="98" className="fk-muted fk-small" textAnchor="start">Attention：按段各自计算，段与段之间不交互</text>
      {tokens.map((name, i) => <Cell key={i} x={segX[i]} y={106} w={step - 4} h={28} rx={2} fill={REQ_FILL[name]} label={name} small />)}
      {bounds.map((x, i) => <g key={i}>
        <path d={`M${x} 140 V148`} className="fk-link fk-line-muted" />
        <text x={x} y="162" className="fk-strong fk-small">{[0, 5, 6, 7][i]}</text>
      </g>)}
      <text x={bounds[3] + 24} y="124" className="fk-strong fk-small" textAnchor="start">cu_seqlens_q</text>
      <text x={bounds[3] + 24} y="142" className="fk-muted fk-small" textAnchor="start">= [0, 5, 6, 7]</text>
    </>}</Figure>
  )
}

// 单步 token budget 的分配：先 running，再 waiting，耗尽即停。
export function VllmStepBudget() {
  const compact = useCompact()
  const cell = 40
  const x0 = 24
  const budget = 12
  const groups: { name: string; tokens: number; fill: Fill; note: string; from: number }[] = [
    { name: 'R1', tokens: 1, fill: 'blue', note: 'decode', from: 0 },
    { name: 'R2', tokens: 1, fill: 'blue', note: 'decode', from: 1 },
    { name: 'R3', tokens: 6, fill: 'purple', note: 'prefill 的后续块（还剩 6）', from: 2 },
    { name: 'W1', tokens: 4, fill: 'purple', note: '新请求，只分到 4 / 22', from: 8 },
  ]
  const desc = 'token budget 为 12。running 队列中 R1、R2 各一个 decode token，R3 是未完成的 prefill，还剩 6 个 token，全部放入；剩余 4 个 token 分给 waiting 队首的新请求 W1，W1 还有 22 个 token 要算，只分到 4 个；W2 因 budget 耗尽本步不被调度。第二部分：KV block 不足时，running 队尾的 R4 被抢占，回到 waiting 队首。'
  const caption = '构造示例，budget = 12。R 为 running 队列中的请求，W 为 waiting 队列中的请求。W1 的 prompt 共 30 个 token，prefix cache 命中 8 个，命中的部分不占 budget，未命中的 22 个本步只分到 4 个。'
  if (compact) {
    // 窄版：12 格占满宽度，W2 与抢占示意各占一行；正文已有小标题，图内不再重复。
    const c = 42, x0 = 8
    const mid = (from: number, n: number) => x0 + (from + n / 2) * c - 1
    return (
      <Figure title="一次 schedule() 中 token budget 的分配" desc={desc} width={520} height={352} caption={`${caption}每格 1 个 token。`}>{arrow => <>
        <text x={x0} y="10" className="fk-muted" textAnchor="start">① running 先调度</text>
        <path d={`M${x0} 24 H${x0 + 8 * c - 4}`} className="fk-link fk-line-muted" />
        <text x={x0 + 8 * c} y="10" className="fk-muted" textAnchor="start">② waiting</text>
        <path d={`M${x0 + 8 * c} 24 H${x0 + budget * c - 2}`} className="fk-link fk-line-muted" />
        {Array.from({ length: budget }, (_, i) => {
          const group = groups.find(g => i >= g.from && i < g.from + g.tokens)!
          return <Cell key={i} x={x0 + i * c} y={34} w={c - 4} h={36} rx={3} fill={group.fill} />
        })}
        <text x={mid(0, 2)} y="90" className="fk-strong">R1 R2</text>
        <text x={mid(0, 2)} y="112" className="fk-muted fk-small">decode</text>
        <text x={mid(2, 6)} y="90" className="fk-strong">R3</text>
        <text x={mid(2, 6)} y="112" className="fk-muted fk-small">未完成的 prefill，剩 6 个</text>
        <text x={mid(8, 4)} y="90" className="fk-strong">W1</text>
        <text x={mid(8, 4)} y="112" className="fk-muted fk-small">新请求，分到 4 / 22</text>
        <Cell x={x0} y={132} w={c * 2 - 4} h={32} rx={3} fill="gray" label="W2" />
        <text x={x0 + 2 * c + 8} y="148" className="fk-muted" textAnchor="start">budget 已耗尽，W2 等下一步</text>

        <text x="0" y="204" className="fk-title" textAnchor="start">KV block 不足时</text>
        <text x="500" y="204" className="fk-muted" textAnchor="end">抢占 running 队尾</text>
        {['R1', 'R2', 'R3', 'R4'].map((name, i) => <Cell key={name} x={x0 + i * 56} y={226} w={50} h={36} rx={3} fill={name === 'R4' ? 'red' : 'blue'} label={name} />)}
        <Link d={`M${x0 + 4 * 56 + 4} 244 H${x0 + 282}`} arrow={arrow} />
        {['R4', 'W1', 'W2'].map((name, i) => <Cell key={name} x={x0 + 292 + i * 56} y={226} w={50} h={36} rx={3} fill={name === 'R4' ? 'red' : 'gray'} label={name} />)}
        <text x={x0 + 2 * 56 - 3} y="280" className="fk-muted fk-small">running</text>
        <text x={x0 + 292 + 81} y="280" className="fk-muted fk-small">waiting，R4 回到队首</text>
        <text x="0" y="316" className="fk-strong" textAnchor="start">R4 释放全部 block，</text>
        <text x="0" y="340" className="fk-strong" textAnchor="start">num_computed_tokens = 0，之后从头重算</text>
      </>}</Figure>
    )
  }
  return (
    <Figure
      title="一次 schedule() 中 token budget 的分配"
      desc={desc}
      height={280}
      caption={caption}
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
      {['R4', 'W1', 'W2'].map((name, i) => <Cell key={name} x={x0 + 340 + i * 64} y={188} w={58} h={34} rx={3} fill={name === 'R4' ? 'red' : 'gray'} label={name} />)}
      <text x={x0 + 2 * 64 + 29} y="240" className="fk-muted fk-small">running</text>
      <text x={x0 + 340 + 96} y="240" className="fk-muted fk-small">waiting，R4 回到队首</text>
      <text x={x0 + 5 * 64 + 160} y="264" className="fk-strong fk-small">R4 释放全部 block，num_computed_tokens = 0，之后从头重算</text>
    </>}</Figure>
  )
}
