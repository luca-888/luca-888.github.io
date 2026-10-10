import { Band, Cell, Figure, Legend, Link, type Fill } from '../../figure-kit'

// 配色约定：green = 返回缓存且正确，red = 返回了意思相反的缓存，blue = 上游模型生成，
// amber = 校验（以及本可命中、改为上游生成的请求），teal = 缓存检索，purple = 示例路径。

// 图 1：#2691 中反义句与同义句的相似度，以及三个阈值
export function ScvSimilarity() {
  const x0 = 250, w = 480, lo = 0.62, hi = 0.95
  const X = (s: number) => x0 + ((s - lo) / (hi - lo)) * w
  const rows: [string, number][] = [
    ['turn on / off dark mode', 0.915],
    ['port open / closed', 0.900],
    ['commit / not commit', 0.869],
    ['require / not require license', 0.853],
    ['caching enabled / disabled', 0.848],
    ['rate limit active / inactive', 0.847],
    ['increase / decrease quota', 0.838],
    ['enable / disable 2FA', 0.823],
  ]
  const y0 = 46, pitch = 24
  const halo = { paintOrder: 'stroke', stroke: 'var(--color-page)', strokeWidth: 4 } as const
  const synY = y0 + rows.length * pitch + 14
  const axisY = synY + 34
  return (
    <Figure
      title="反义句与同义句的相似度"
      desc="8 对反义句的相似度在 0.823 到 0.915 之间，全部高于阈值 0.80；同义改写 enable 2FA 与 turn on 2FA 只有 0.673。阈值 0.86 时仍有 3 对误命中，0.92 时没有误命中，但同义改写也不命中。"
      height={axisY + 34}
      below={<Legend items={[['red', '意思相反的一对'], ['green', '意思相同的一对']]} />}
      caption="数据来自 Semantic Router 的 issue #2691：路由器自带的 mmbert embedding（384 维）计算的余弦相似度，图中为 issue 列出的 8 对反义句。三条竖线是仓库配置中出现过的阈值，竖线右侧的点都会命中。"
    >{() => <>
      {[[0.80, '0.80'], [0.86, '0.86'], [0.92, '0.92']].map(([t, l]) => <g key={l}>
        <path d={`M${X(+t)} ${y0 - 18} V${axisY}`} className="fk-link" style={{ stroke: 'var(--fk-amber)', strokeDasharray: '4 3', strokeWidth: 1.6 }} />
        <text x={X(+t)} y={y0 - 28} className="fk-strong">阈值 {l}</text>
      </g>)}
      {rows.map(([name, s], i) => {
        const y = y0 + i * pitch
        return <g key={name}>
          <text x={0} y={y} textAnchor="start" className="fk-mono">{name}</text>
          <line x1={x0} x2={X(s)} y1={y} y2={y} className="fk-rule" />
          <circle cx={X(s)} cy={y} r={6.5} className="fk-fill-red" />
          <text x={X(s) + 12} y={y} textAnchor="start" className="fk-muted" style={halo}>{s.toFixed(3)}</text>
        </g>
      })}
      <line x1={0} x2={760} y1={synY - 14} y2={synY - 14} className="fk-rule" />
      <text x={0} y={synY + 4} textAnchor="start" className="fk-mono">enable 2FA / turn on 2FA</text>
      <line x1={x0} x2={X(0.673)} y1={synY + 4} y2={synY + 4} className="fk-rule" />
      <circle cx={X(0.673)} cy={synY + 4} r={6.5} className="fk-fill-green" />
      <text x={X(0.673) + 12} y={synY + 4} textAnchor="start" className="fk-muted">0.673（同义）</text>
      <line x1={x0} x2={x0 + w} y1={axisY} y2={axisY} className="fk-rule" />
      {[0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95].map(t => <text key={t} x={X(t)} y={axisY + 14} className="fk-muted fk-small">{t.toFixed(2)}</text>)}
      <text x={x0 - 12} y={axisY + 14} textAnchor="end" className="fk-muted fk-small">余弦相似度</text>
    </>}</Figure>
  )
}

// 图 2：一次查询从检索到命中或转发上游的路径
export function ScvLookupPath() {
  const bw = 104, by = 96, bh = 52
  const xs = [0, 131, 262, 393, 524, 656]
  const cx = (i: number) => xs[i] + bw / 2
  const laneY = 268
  const notes: [string, string][] = [
    ['turn on', 'the lights'],
    ['最近的缓存：', 'add a contact'],
    ['0.83', '高于阈值'],
    ['差异超过', '2 个 token，放行'],
    ['判为矛盾', '拒绝'],
  ]
  return (
    <Figure
      title="一次查询的路径"
      desc="新问题经 embedding 检索找到最近的缓存问题，相似度达到阈值后依次经过词法层与 NLI 层，全部通过才返回缓存；任何一步不通过都转为 miss，请求转发给上游模型。示例 turn on the lights 与 add a contact 相似度 0.83，词法层放行，NLI 层判为矛盾，转发上游。"
      height={330}
      below={<Legend items={[['teal', '缓存检索'], ['amber', '命中校验'], ['green', '返回缓存'], ['blue', '上游模型生成'], ['purple', '示例路径']]} />}
      caption="紫色是 PR #3075 端到端测试中的一次查询：“turn on the lights”与缓存中的“add a contact”相似度为 0.83。两句相差多于 2 个 token，不在词法层的判断范围内，由 NLI 层拒绝。三条向下的箭头是三种 miss；NLI 层出故障时，fail-closed 走同一条箭头。"
    >{arrow => <>
      <Band x={xs[3] - 12} y={by - 34} w={xs[4] + bw - xs[3] + 24} h={bh + 46} hue="gray" title="命中校验" />
      <Cell x={xs[0]} y={by} w={bw} h={bh} rx={8} fill="gray" label="新问题" />
      <Cell x={xs[1]} y={by} w={bw} h={bh} rx={8} fill="teal" label="embedding" sub="最近的候选" />
      <Cell x={xs[2]} y={by} w={bw} h={bh} rx={8} fill="teal" label="相似度" sub="≥ 阈值？" />
      <Cell x={xs[3]} y={by} w={bw} h={bh} rx={8} fill="amber" label="词法层" sub="否定词 / 反义词" />
      <Cell x={xs[4]} y={by} w={bw} h={bh} rx={8} fill="amber" label="NLI 层" sub="contradiction" />
      <Cell x={xs[5]} y={by} w={bw} h={bh} rx={8} fill="green" label="返回缓存" />

      {notes.map(([a, b], i) => <g key={i}>
        <text x={cx(i)} y={by + bh + 18} className="fk-strong fk-text-purple fk-small">{a}</text>
        <text x={cx(i)} y={by + bh + 33} className="fk-text-purple fk-small">{b}</text>
      </g>)}

      {[0, 1, 2, 3].map(i => <Link key={i} d={`M${xs[i] + bw} ${by + bh / 2} H${xs[i + 1]}`} hue="purple" arrow={arrow} focus />)}
      <Link d={`M${xs[4] + bw} ${by + bh / 2} H${xs[5]}`} arrow={arrow} />

      <Cell x={xs[2]} y={laneY} w={xs[5] + bw - xs[2]} h={44} rx={8} fill="blue" label="miss：转发给上游模型生成" />
      {[2, 3].map(i => <Link key={i} d={`M${cx(i) + 34} ${by + bh} V${laneY}`} arrow={arrow} />)}
      <Link d={`M${cx(4) + 34} ${by + bh} V${laneY}`} hue="purple" arrow={arrow} focus />
      <text x={cx(2) + 42} y={laneY - 22} textAnchor="start" className="fk-muted fk-small">低于阈值</text>
      <text x={cx(3) + 42} y={laneY - 22} textAnchor="start" className="fk-muted fk-small">否定或反义</text>
      <text x={cx(4) + 42} y={laneY - 30} textAnchor="start" className="fk-muted fk-small">矛盾</text>
      <text x={cx(4) + 42} y={laneY - 15} textAnchor="start" className="fk-muted fk-small">或校验器故障</text>

    </>}</Figure>
  )
}

// 图 3：校验器故障期间，同样 100 个请求在两种策略下的去向（演示数据）
export function ScvFailurePolicy() {
  const cell = 18, pitch = 21, cols = 10
  const wrong = new Set([6, 19])
  const grid = (x: number, open: boolean) => Array.from({ length: 100 }, (_, i) => {
    const candidate = i < 30
    const fill: Fill = candidate ? (open ? (wrong.has(i) ? 'red' : 'green') : 'amber') : 'blue'
    return <rect key={i} x={x + (i % cols) * pitch} y={44 + Math.floor(i / cols) * pitch} width={cell} height={cell} rx={2} className={`fk-fill-${fill}`} />
  })
  const panel = (x: number, open: boolean) => <g>
    <text x={x} y={16} textAnchor="start" className="fk-strong">{open ? 'fail-open：照常返回缓存' : 'fail-closed：候选按 miss 处理'}</text>
    {grid(x, open)}
    {(open
      ? [['上游生成', '70 次'], ['意思相反的回答', '2 个'], ['命中率', '30%，与平时相同']]
      : [['上游生成', '100 次，多 30 次'], ['意思相反的回答', '0 个'], ['命中率', '从 30% 降到 0']]
    ).map(([k, v], j) => <g key={k}>
      <text x={x} y={274 + j * 20} textAnchor="start" className="fk-muted">{k}</text>
      <text x={x + 112} y={274 + j * 20} textAnchor="start" className={j === (open ? 1 : 0) ? 'fk-strong fk-text-purple' : 'fk-strong'}>{v}</text>
    </g>)}
  </g>
  return (
    <Figure
      title="校验器故障时两种策略的代价"
      desc="同样 100 个请求，其中 30 个有达到阈值的候选、2 个候选与新问题意思相反。fail-open 时上游生成 70 次，返回 2 个意思相反的回答，命中率不变；fail-closed 时上游生成 100 次，没有错误回答，命中率降到 0。"
      height={330}
      below={<Legend items={[['green', '返回缓存，意思正确'], ['red', '返回缓存，意思相反'], ['amber', '本可命中，改为上游生成'], ['blue', '上游生成']]} />}
      caption="演示数据：校验器故障期间的 100 个请求，每格一个。30 个请求有达到阈值的候选，其中 2 个候选与新问题意思相反，正常情况下会被校验拒绝。fail-open 的错误只存在于返回给用户的回答里，监控上的命中率不变；fail-closed 的代价是多出的 30 次生成，在命中率与上游负载上都能看到。"
    >{() => <>
      {panel(40, true)}
      {panel(430, false)}
    </>}</Figure>
  )
}

// 实测占位：方向确认后再补测
export function ScvMeasurePlaceholder() {
  return (
    <div style={{ border: '1.5px dashed var(--color-border)', borderRadius: 8, padding: '16px 20px', color: 'var(--color-muted)', margin: '1.5em 0' }}>
      <strong>实测待补。</strong>
      <ol style={{ margin: '8px 0 0', paddingLeft: '1.4em' }}>
        <li>同一个问题命中缓存与转发上游的端到端延迟对比，给出 fail-closed 在校验器故障期间每个请求多等的时间。需要一张能运行上游模型的 GPU（如 RTX 4090，vLLM 加 8B 模型），以及 #3162 合并时、带 native binding 构建的 Semantic Router。</li>
        <li>回放一段真实请求，测量校验器故障期间上游请求量的增幅。需要同样的环境与一份带重复问题的请求日志。</li>
      </ol>
    </div>
  )
}
