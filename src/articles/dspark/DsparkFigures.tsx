import { useId, type ReactNode } from 'react'
import { Cell, Figure, Legend, Link } from '../../figure-kit'

// 语义色与视频（videos/dspark）一致：drafter 与 draft token purple，target 与送去验证的 token blue，
// 接受 teal，拒绝 red，已确定的上文 ink，不验证或被丢弃 void（斜纹）。
// 示例也沿用视频：target 会回答“当然可以”或“没问题”。

// 斜纹填充：每张图各自的 pattern
function useHatch() {
  const id = useId()
  const defs = <defs><pattern id={id} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="7" height="7" className="fk-fill-void" /><rect width="3" height="7" className="fk-hatch-line" />
  </pattern></defs>
  return [defs, `url(#${id})`] as const
}

type SlotState = 'on' | 'soft' | 'void'
// 一个 draft token 的位置：on 送去验证，soft 仅空闲时验证，void 不验证或丢弃
function Slot({ x, y, w, h, state, hatch, label, sub }: { x: number; y: number; w: number; h: number; state: SlotState; hatch: string; label: ReactNode; sub?: ReactNode }) {
  if (state === 'on') return <Cell x={x} y={y} w={w} h={h} fill="blue" label={label} sub={sub} />
  const cy = y + h / 2 + 0.5 - (sub !== undefined ? 8 : 0), cls = state === 'soft' ? 'fk-text-blue' : 'fk-muted'
  return <g>
    {state === 'soft' ? <rect x={x} y={y} width={w} height={h} rx="3" className="fk-fill-blue" fillOpacity=".16" /> : <rect x={x} y={y} width={w} height={h} rx="3" fill={hatch} />}
    <text x={x + w / 2} y={cy} className={cls} style={{ fontSize: 12, fontWeight: 600 }}>{label}</text>
    {sub !== undefined && <text x={x + w / 2} y={cy + 18} className={cls} style={{ fontSize: 11 }}>{sub}</text>}
  </g>
}

// 图 0：MTP-1 的一轮与两种结果
export function DsparkMtp() {
  const bx = 104, dw = 26, vw = 250, by = 45, bh = 30, ox = 452
  const end = bx + dw + 2 + vw
  const row = (y: number, ok: boolean, note: string) => <g>
    <text x={ox} y={y - 10} textAnchor="start" className="fk-muted">{note}</text>
    <Cell x={ox} y={y} w={64} h={28} fill="ink" label="…当然" />
    <Cell x={ox + 68} y={y} w={64} h={28} fill={ok ? 'teal' : 'red'} label={ok ? '可以 ✓' : '可以 ✗'} />
    <Cell x={ox + 136} y={y} w={40} h={28} fill="blue" label={ok ? '，' : '是'} />
    <text x={ox + 186} y={y + 14} textAnchor="start" className="fk-stat">{ok ? '产出 2 个' : '产出 1 个'}</text>
  </g>
  return (
    <Figure
      title="MTP-1 的一轮与两种结果"
      desc="MTP-1 的一轮：drafter 生成 1 个 draft token，target 用一次 forward 验证。draft 被接受时，这次 forward 算出的下一位也可用，一轮产出 2 个 token；被拒绝时只有 target 在该位补的 1 个。"
      height={126}
      below={<Legend items={[['purple', 'MTP 生成 draft'], ['blue', 'target 的 forward 与它给出的 token'], ['teal', '接受'], ['red', '拒绝'], ['ink', '已确定的上文']]} />}
      caption="示意，时间条不按实测比例。target 的一次 forward 同时验证 draft token 并给出再下一位：draft 被接受时一轮产出 2 个 token，被拒绝时产出 target 在该位补的 1 个。"
    >{arrow => <>
      <text x="0" y={by + bh / 2} className="fk-strong" textAnchor="start">MTP-1 的一轮</text>
      <text x={bx + dw / 2} y={by - 12} className="fk-muted fk-text-purple">draft 1 个</text>
      <Cell x={bx} y={by} w={dw} h={bh} fill="purple" />
      <Cell x={bx + dw + 2} y={by} w={vw} h={bh} fill="blue" label="验证：target 一次 forward" />
      <Link d={`M${end + 6} ${by + bh / 2} H${end + 34} V36 H${ox - 6}`} arrow={arrow} />
      <Link d={`M${end + 6} ${by + bh / 2} H${end + 34} V100 H${ox - 6}`} arrow={arrow} />
      {row(22, true, 'draft 被接受：加上 forward 算出的下一位')}
      {row(86, false, 'draft 被拒绝：只有 target 补的 1 个')}
    </>}</Figure>
  )
}

// 图 1：一轮 DSpark 的完整路径（5 个 draft token 的小例子）
export function DsparkCycle() {
  const [defs, hatch] = useHatch()
  const X0 = 196, STEP = 88, W = 78, H = 34, CX = 112, CW = 70, NX = 642
  const cx = (k: number) => X0 + k * STEP, mid = (k: number) => cx(k) + W / 2
  const tok = ['可以', '，', '我', '来', '看'], surv = [0.92, 0.81, 0.66, 0.41, 0.25]
  const rows: [number, string, string, string, string][] = [
    [8, 'target', 'fk-text-blue', '输出 hidden state', '模型不变'],
    [60, '并行 backbone', 'fk-text-purple', '一次 forward', '5 个位置的预测'],
    [112, 'Markov head', 'fk-text-purple', '逐位采样', '参考前一个 token'],
    [164, 'confidence head', 'fk-text-purple', '估计每个 token', '的存活概率'],
    [216, '调度器', 'fk-text-blue', '按存活概率与负载', '选出验证哪些'],
    [268, 'target 验证', 'fk-text-blue', '接受 2 个', '第 3 位由 target 补'],
  ]
  return (
    <Figure
      title="DSpark 的一轮：drafter 的三个部件加调度器"
      desc="target 给出 hidden state。并行 backbone 一次 forward 给出 5 个位置的预测，Markov head 从左到右采样出 5 个 draft token，confidence head 估计每个 token 的存活概率。调度器选前 3 个送去验证，后 2 个不验证。target 接受前 2 个，拒绝第 3 个并补上自己的 token。"
      height={310}
      below={<Legend items={[['ink', '已确定的上文'], ['purple', 'drafter 与 draft token'], ['blue', 'target 与送去验证的 token'], ['teal', '接受'], ['red', '拒绝'], ['void', '不验证']]} />}
      caption="示意，数值为举例。这一轮生成 5 个 draft token、验证 3 个、产出 3 个：前 2 个被接受，第 3 个被拒绝后由 target 补上自己的 token。"
    >{arrow => <>
      {defs}
      <rect x={-10} y={50} width={NX - 4} height={158} rx="10" className="fk-band-purple" />
      {rows.map(([y, name, cls, n1, n2], i) => <g key={name}>
        <text x="0" y={y + H / 2} className={`fk-strong ${cls}`} textAnchor="start">{name}</text>
        <text x={NX} y={y + 9} className="fk-muted" textAnchor="start">{n1}</text>
        <text x={NX} y={y + 26} className="fk-muted" textAnchor="start">{n2}</text>
        {i < 5 && <Link d={`M${CX + CW / 2} ${y + H + 2} V${y + 51}`} arrow={arrow} />}
      </g>)}
      <Cell x={CX} y={8} w={cx(4) + W - CX} h={H} fill="blue" label="DeepSeek-V4：算完上文" />
      <Cell x={CX} y={60} w={CW} h={H} fill="ink" label="当然" fs={13} />
      {tok.map((t, k) => <g key={k}>
        <rect x={cx(k) + 1} y={61} width={W - 2} height={H - 2} rx="3" className="fk-outline fk-line-purple" />
        <text x={mid(k)} y={60 + H / 2} className="fk-muted fk-text-purple">mask</text>

        <Cell x={cx(k)} y={112} w={W} h={H} fill="purple" label={t} fs={13} />
        {k < 4 && <path d={`M${mid(k)} 109 Q${mid(k) + STEP / 2} 92 ${mid(k + 1)} 109`} className="fk-link fk-line-purple" />}

        <rect x={cx(k)} y={164} width={W} height={H} rx="3" className="fk-fill-void" />
        <rect x={cx(k)} y={164} width={W * surv[k]} height={H} rx="3" className="fk-fill-purple" fillOpacity=".4" />
        <text x={mid(k)} y={164 + H / 2} className="fk-stat">{surv[k].toFixed(2)}</text>

        {k < 3 ? <Cell x={cx(k)} y={216} w={W} h={H} fill="blue" label={t} fs={13} /> : <Slot x={cx(k)} y={216} w={W} h={H} state="void" hatch={hatch} label="不验证" />}
      </g>)}
      <Cell x={cx(0)} y={268} w={W} h={H} fill="teal" label="可以 ✓" fs={13} />
      <Cell x={cx(1)} y={268} w={W} h={H} fill="teal" label="， ✓" fs={13} />
      <Cell x={cx(2)} y={268} w={W / 2 - 2} h={H} fill="red" label="我 ✗" />
      <Cell x={cx(2) + W / 2 + 2} y={268} w={W / 2 - 2} h={H} fill="blue" label="帮" fs={13} />
    </>}</Figure>
  )
}

// 图 2：自回归与并行生成 5 个 draft token 的一轮耗时，以及第 1 位的条件接受率
export function DsparkDraft() {
  const bx = 92, U = 32, H = 34, vw = 230, ax = 536, AW = 150
  const rows: [number, string, string, string, number][] = [
    [26, '自回归', 'EAGLE-3', '1 层 × 5 次：γ 增大，draft 段变长', 0.53],
    [98, '并行', 'DFlash', '5 层 × 1 次：γ 增大，draft 段不变', 0.72],
  ]
  return (
    <Figure
      title="自回归与并行生成 draft 的一轮耗时与第 1 位的条件接受率"
      desc="自回归 drafter 一层的网络运行 5 次，并行 drafter 五层的网络运行 1 次，draft 段的耗时相近；第 1 位的条件接受率自回归为 0.53，并行为 0.72。"
      height={162}
      below={<Legend items={[['purple', 'drafter 生成 draft'], ['blue', 'target 验证']]} />}
      caption={<>示意：draft 段的长度按层数 × 运行次数画，不按实测。并行生成只运行一次，同样的耗时可以换成更深的网络。条件接受率为聊天任务、target 为 Qwen3-4B，来源：<a href="https://arxiv.org/abs/2607.05147">DSpark</a>。</>}
    >{() => <>
      <text x={ax} y="10" className="fk-muted" textAnchor="start">第 1 位的条件接受率</text>
      {rows.map(([y, name, sub, note, p], r) => <g key={name}>
        <text x="0" y={y + 9} className="fk-strong" textAnchor="start">{name}</text>
        <text x="0" y={y + 27} className="fk-muted" textAnchor="start">{sub}</text>
        {r === 0
          ? [0, 1, 2, 3, 4].map(k => <Cell key={k} x={bx + k * U} y={y} w={U - 2} h={H} fill="purple" label="1 层" small />)
          : [0, 1, 2, 3, 4].map(k => <rect key={k} x={bx} y={y + k * 7} width={5 * U - 2} height={6} rx="1.5" className="fk-fill-purple" />)}
        <Cell x={bx + 5 * U + 2} y={y} w={vw} h={H} fill="blue" label="验证：target 一次 forward" />
        <text x={bx} y={y + H + 14} className="fk-muted" textAnchor="start">{note}</text>
        <rect x={ax} y={y + 6} width={AW} height={22} rx="3" className="fk-fill-void" />
        <rect x={ax} y={y + 6} width={AW * p} height={22} rx="3" className="fk-fill-purple" fillOpacity={r === 0 ? 0.45 : 1} />
        <text x={ax + AW + 10} y={y + 17} className="fk-stat" textAnchor="start">{p.toFixed(2)}</text>
      </g>)}
    </>}</Figure>
  )
}

// 图 3：并行生成的独立采样，以及 Markov head 怎样修正第 2 位
export function DsparkMarkov() {
  const [defs, hatch] = useHatch()
  const dist = (x: number, y: number, word: string, p: number, hue: 'blue' | 'amber') => <g key={`${x}-${y}`}>
    <text x={x} y={y + 11} textAnchor="start" className="fk-stat">{word}</text>
    <rect x={x + 44} y={y} width={p * 150} height={22} rx="3" className={`fk-fill-${hue}`} />
    <text x={x + 52 + p * 150} y={y + 11} textAnchor="start" className="fk-stat">{Math.round(p * 100)}%</text>
  </g>
  const R = 408, tx = R + 60
  return (
    <Figure
      title="并行生成的独立采样与 Markov head 的修正"
      desc="target 会回答“当然可以”或“没问题”，各占一半。左边并行生成：第 1 位在“当然”和“没”之间各半，第 2 位在“可以”和“问题”之间各半，两位独立采样，可能得到“当然问题”，验证在第 2 位停止，其后丢弃。右边加了 Markov head：第 1 位采到“当然”后，取“当然”一行的偏置加到第 2 位的 logits 上，“可以”变为 90%。"
      height={246}
      below={<Legend items={[['blue', '回答“当然可以”'], ['amber', '回答“没问题”'], ['purple', 'Markov head 的偏置'], ['teal', '接受'], ['red', '拒绝'], ['void', '丢弃']]} />}
      caption="示意，概率与偏置为举例。两个位置各自的分布都正确，独立采样时第 2 位有一半可能接不上；加上“当然”一行的偏置后，第 2 位变为 90% 对 10%。"
    >{arrow => <>
      {defs}
      <text x="0" y="12" className="fk-strong" textAnchor="start">并行：两个位置独立采样</text>
      <text x="0" y="62" className="fk-muted" textAnchor="start">第 1 位</text>
      {dist(56, 38, '当然', 0.5, 'blue')}{dist(56, 64, '没', 0.5, 'amber')}
      <text x="0" y="128" className="fk-muted" textAnchor="start">第 2 位</text>
      {dist(56, 104, '可以', 0.5, 'blue')}{dist(56, 130, '问题', 0.5, 'amber')}
      <text x="0" y="191" className="fk-muted" textAnchor="start">各采一次</text>
      <Cell x={56} y={176} w={54} h={30} fill="teal" label="当然 ✓" />
      <Cell x={114} y={176} w={54} h={30} fill="red" label="问题 ✗" />
      {[0, 1, 2].map(k => <Slot key={k} x={172 + k * 58} y={176} w={54} h={30} state="void" hatch={hatch} label="丢弃" />)}
      <text x="56" y="226" className="fk-muted" textAnchor="start">验证在第一个被拒的位置停止，其后全部丢弃</text>

      <line x1="384" y1="0" x2="384" y2="236" className="fk-rule" />
      <text x={R} y="12" className="fk-strong" textAnchor="start">Markov head：加上前一位对应的一行</text>
      <text x={R} y="34" className="fk-muted" textAnchor="start">前一位</text>
      <text x={tx + 35} y="34" className="fk-muted">可以</text>
      <text x={tx + 109} y="34" className="fk-muted">问题</text>
      {[['当然', '+1.1', '−1.1'], ['没', '−1.1', '+1.1']].map(([w, a, b], i) => <g key={w} opacity={i ? 0.35 : 1}>
        <text x={R} y={59 + i * 30} className="fk-stat" textAnchor="start">{w}</text>
        <Cell x={tx} y={46 + i * 30} w={70} h={26} fill={i ? 'gray' : 'purple'} label={a} />
        <Cell x={tx + 74} y={46 + i * 30} w={70} h={26} fill={i ? 'gray' : 'purple'} label={b} />
      </g>)}
      <rect x={tx - 4} y={42} width={152} height={34} rx="5" className="fk-focus" />
      <Cell x={R + 236} y={46} w={104} h={26} fill="ink" label="第 1 位采到 当然" />
      <Link d={`M${R + 236} 59 H${tx + 150}`} hue="purple" arrow={arrow} />
      <text x={R} y="128" className="fk-muted" textAnchor="start">第 2 位</text>
      {dist(R + 56, 104, '可以', 0.9, 'blue')}{dist(R + 56, 130, '问题', 0.1, 'amber')}
      <Cell x={R + 56} y={176} w={54} h={30} fill="teal" label="当然 ✓" />
      <Cell x={R + 114} y={176} w={54} h={30} fill="teal" label="可以 ✓" />
      <text x={R + 180} y="191" className="fk-muted" textAnchor="start">“当然问题”降到 10%</text>
    </>}</Figure>
  )
}

// 图 4：调度器在两种负载下选中的 draft token。三个请求各 4 个 draft token，confidence 与代价曲线为示意。
const CONF = [[.95, .93, .92, .90], [.80, .70, .65, .60], [.50, .50, .50, .50]]
const SURV = CONF.map(r => { let p = 1; return r.map(c => (p *= c)) })
const QUEUE = SURV.flatMap((r, i) => r.map((a, j) => ({ a, i, j }))).sort((x, y) => y.a - x.a)
// 贪心：所有 draft token 按存活概率排队，逐个加入，取 τ / stepMs(N) 最大处；curve[n] 为加入 n 个时的吞吐
function schedule(stepMs: (b: number) => number) {
  let tau = SURV.length
  const curve = [tau / stepMs(tau)]
  QUEUE.forEach((s, k) => { tau += s.a; curve.push(tau / stepMs(SURV.length + k + 1)) })
  const n = curve.indexOf(Math.max(...curve))
  const len = SURV.map(() => 0)
  QUEUE.slice(0, n).forEach(s => len[s.i]++)
  return { len, n, curve }
}

export function DsparkSchedule() {
  const [defs, hatch] = useHatch()
  const idle = schedule(b => 20 + 0.4 * b), busy = schedule(b => 5 + 2 * b)
  const state = (i: number, j: number): SlotState => j < busy.len[i] ? 'on' : j < idle.len[i] ? 'soft' : 'void'
  const reqs = [['请求 A', '代码'], ['请求 B', '聊天'], ['请求 C', '难预测']], names = 'ABC'
  const gx = 430, gw = 316, top = 50, bot = 140
  const px = (n: number) => gx + n * gw / QUEUE.length
  const py = (c: number[], n: number) => { const lo = Math.min(...c), hi = Math.max(...c); return bot - (c[n] - lo) / (hi - lo) * (bot - top) }
  const plot = (c: number[], peak: number, op: number, label: string, lx: number, anchor: 'middle' | 'end') => <g>
    <polyline points={c.map((_, n) => `${px(n)},${py(c, n)}`).join(' ')} className="fk-link fk-line-blue" strokeOpacity={op} />
    {c.map((_, n) => <circle key={n} cx={px(n)} cy={py(c, n)} r={n === peak ? 5.5 : 2.5} className="fk-fill-blue" fillOpacity={n === peak ? 1 : op} />)}
    {op < 1 && <circle cx={px(peak)} cy={py(c, peak)} r="5.5" fill="#fff" fillOpacity={1 - op} />}
    <text x={lx} y={top - 16} textAnchor={anchor} className="fk-text-blue" style={{ fontSize: 12, fontWeight: 600 }} fillOpacity={op < 1 ? 0.7 : 1}>{label}</text>
  </g>
  const QW = 58, qx = (k: number) => k * 63.8
  const sum = (r: { len: number[]; n: number }) => `验证 ${r.n} 个（${r.len.map((l, i) => `${names[i]} ${l}`).join('、')}）`
  return (
    <Figure
      title="同样三个请求，调度器在空闲与繁忙时选中的 draft token"
      desc="三个请求各生成四个 draft token，格子里是存活概率。全部十二个 token 按存活概率排序后逐个加入，吞吐先升后降，在峰值处停止：空闲时验证十个，繁忙时只验证五个，请求 A 四个，请求 B 一个，请求 C 不验证。"
      height={316}
      below={<div className="fk-legend">
        <span><i className="fk-swatch-blue" />繁忙与空闲时都验证</span>
        <span><i className="fk-swatch-blue" style={{ opacity: 0.25 }} />只在空闲时验证</span>
        <span><i className="fk-swatch-void" />不验证</span>
      </div>}
      caption="示意：格子里的数字是存活概率，即 confidence 的累乘；一步耗时空闲时取 20 + 0.4 × token 数，繁忙时取 5 + 2 × token 数（ms）。两种负载下存活概率相同，区别只在一步耗时随 token 数涨得多快，吞吐的峰值因此落在不同位置。两条曲线各自缩放到自身的取值范围。"
    >{() => <>
      {defs}
      {SURV.map((row, i) => <g key={i}>
        <text x="0" y={36 + i * 44} className="fk-strong" textAnchor="start">{reqs[i][0]}</text>
        <text x="0" y={54 + i * 44} className="fk-muted" textAnchor="start">{reqs[i][1]}</text>
        {row.map((a, j) => <Slot key={j} x={84 + j * 62} y={28 + i * 44} w={56} h={34} state={state(i, j)} hatch={hatch} label={a.toFixed(2)} />)}
      </g>)}

      <text x={gx} y="8" className="fk-muted" textAnchor="start">期望产出 ÷ 一步耗时</text>
      <line x1={gx} y1={bot + 12} x2={gx + gw} y2={bot + 12} className="fk-rule" />
      {[busy.n, idle.n].map(n => <text key={n} x={px(n)} y={bot + 24} className="fk-muted">{n}</text>)}
      <text x={gx + gw} y={bot + 40} className="fk-muted" textAnchor="end">加入的 draft token 数</text>
      {plot(idle.curve, idle.n, 0.4, `空闲：峰值在 ${idle.n} 个`, gx + gw, 'end')}
      {plot(busy.curve, busy.n, 1, `繁忙：峰值在 ${busy.n} 个`, px(busy.n), 'middle')}

      <text x="0" y="196" className="fk-strong" textAnchor="start">按存活概率排序</text>
      {QUEUE.map((s, k) => <Slot key={k} x={qx(k)} y={210} w={QW} h={42} state={state(s.i, s.j)} hatch={hatch} label={s.a.toFixed(2)} sub={`请求 ${names[s.i]}`} />)}
      <rect x="0" y="260" width={qx(busy.n - 1) + QW} height="3" className="fk-fill-blue" />
      <text x="0" y="276" className="fk-stat" textAnchor="start">繁忙：{sum(busy)}</text>
      <rect x="0" y="290" width={qx(idle.n - 1) + QW} height="3" className="fk-fill-blue" fillOpacity=".35" />
      <text x="0" y="306" className="fk-stat" textAnchor="start">空闲：{sum(idle)}</text>
    </>}</Figure>
  )
}
