import { Badge, Band, Cell, Figure, Legend, Link, useCompact } from './figure-kit'
import measured from './data/vllm-cpu-overhead-measured.json'

// 语义色：GPU 计算 blue，GPU 空转 red，EngineCore 关键路径 purple，不在关键路径上的 CPU 工作 teal，
// IO 线程 gray，示例请求的路径用 purple 箭头；token 格中 decode 为 blue、prefill 为 amber。
const ms = (us: number) => us / 1000
const fmt = (v: number) => v.toFixed(1)
type StepKey = keyof typeof measured.steps

// 图 1：同步与 async 的 step timeline（按实测中位数等比例绘制）
export function EngineStepTimeline() {
  const compact = useCompact()
  const PX = 17 // px / ms
  const X0 = compact ? 0 : 150
  const rows: { label: string; key: StepKey }[] = [
    { label: '关 async scheduling', key: 'no_async/c8' },
    { label: '默认（async）', key: 'default/c8' },
  ]
  return (
    <Figure
      title="decode 连续三步的 GPU 时间线：关闭与开启 async scheduling"
      desc="按实测中位数等比例绘制。关闭 async scheduling 时每步 GPU 计算约 6.8 ms，步间空转约 2.9 ms；开启后空转缩到约 0.3 ms。"
      width={compact ? 520 : 760}
      height={compact ? 232 : 196}
      below={<Legend items={[['blue', 'GPU 计算（kernel）'], ['red', 'GPU 空转']]} />}
      caption="H100，Qwen3-8B，并发 8，torch profiler 实测。空转指 GPU 上没有任何 kernel 在跑的时间。"
    >{() => <>
      {rows.map((r, i) => {
        const s = measured.steps[r.key]
        const busy = ms(s.gpu_busy) * PX
        const idle = ms(s.period - s.gpu_busy) * PX
        // 窄版：配置名与每步耗时放在色条上方一行。
        const y = compact ? 58 + i * 92 : 36 + i * 70
        const idleText = fmt(ms(s.period - s.gpu_busy))
        return <g key={r.key}>
          <text x="0" y={compact ? y - 14 : y + 18} className="fk-strong" textAnchor="start">{r.label}</text>
          {[0, 1, 2].map(n => {
            const x = X0 + n * (busy + idle)
            return <g key={n}>
              <rect x={x} y={y} width={busy - 2} height="36" rx="4" className="fk-fill-blue" />
              <rect x={x + busy} y={y} width={idle - 2} height="36" rx="4" className="fk-fill-red" />
            </g>
          })}
          <text x={X0 + busy / 2} y={y + 18} className="fk-cell-text">{fmt(ms(s.gpu_busy))} ms</text>
          {idle > 30
            ? <text x={X0 + busy + idle / 2 - 1} y={y + 18} className="fk-cell-text">{idleText}</text>
            : <text x={X0 + busy + idle / 2 - 1} y={y + 48} className="fk-muted fk-small">{idleText}</text>}
          {compact
            ? <text x="520" y={y - 14} className="fk-stat" textAnchor="end">一步 {fmt(ms(s.period))} ms</text>
            : <text x={X0 + 3 * (busy + idle) + 8} y={y + 18} className="fk-stat" textAnchor="start">一步 {fmt(ms(s.period))} ms</text>}
        </g>
      })}
      <text x="0" y="14" className="fk-muted" textAnchor="start">时间 →（等比例）</text>
    </>}</Figure>
  )
}

// 图 2：一个请求穿过的进程与线程（单卡）
export function EngineProcessPath() {
  const compact = useCompact()
  const legend = <Legend items={[['teal', '不在 GPU 关键路径上的 CPU 工作'], ['purple', '关键路径上的 CPU 工作'], ['gray', 'IO 线程'], ['blue', 'GPU']]} />
  const title = '一个请求穿过的进程与线程'
  const desc = '单卡时有两个进程。API server 负责 tokenize 和 detokenize；EngineCore 的输入线程、主线程 busy loop、输出线程分工，主线程把 batch 提交给 GPU。示例请求依次经过六步。'
  const caption = '单卡部署：API server 与 EngineCore 两个进程，之间用 ZMQ 通信。紫色描边是示例请求的路径：1–2 进入，3 提交执行，4–5 返回，6 流式输出。'
  if (compact) {
    // 窄版：三个分组自上而下排列，请求沿左侧下行、右侧上行。
    return (
      <Figure title={title} desc={desc} width={520} height={474} below={legend} caption={caption}>{arrow => <>
        <Band x={0} y={0} w={520} h={110} hue="teal" title="进程 1 · API server" />
        <Band x={0} y={130} w={520} h={220} hue="purple" title="进程 2 · EngineCore" />
        <Band x={0} y={370} w={520} h={104} hue="blue" title="GPU" />

        <Cell x={16} y={38} w={230} h={58} rx={8} fill="teal" label="HTTP · tokenize" fs={14} />
        <Cell x={274} y={38} w={230} h={58} rx={8} fill="teal" label="增量 detokenize" sub="流式返回" fs={14} />
        <Cell x={16} y={170} w={200} h={44} rx={8} fill="gray" label="输入线程" fs={14} />
        <Cell x={304} y={170} w={200} h={44} rx={8} fill="gray" label="输出线程" fs={14} />
        <Cell x={16} y={240} w={440} h={64} rx={8} fill="purple" label="主线程 busy loop" sub="schedule → 准备输入 → 提交" fs={15} />
        <Cell x={40} y={402} w={260} h={58} rx={8} fill="blue" label="forward" sub="+ sample" fs={14} />

        <Link d="M196 96 V170" hue="purple" arrow={arrow} focus />
        <Link d="M196 214 V240" hue="purple" arrow={arrow} focus />
        <Link d="M100 304 V402" hue="purple" arrow={arrow} focus />
        <Link d="M300 431 H484 V214" hue="purple" arrow={arrow} focus />
        <Link d="M424 170 V96" hue="purple" arrow={arrow} focus />
        <text x="212" y="133" className="fk-muted fk-small" textAnchor="start">ZMQ</text>
        <text x="408" y="133" className="fk-muted fk-small" textAnchor="end">ZMQ</text>
        {[[178, 133], [178, 227], [82, 366], [400, 417], [442, 133], [500, 24]].map(([x, y], i) => <Badge key={i} x={x} y={y} n={i + 1} />)}
        <text x="124" y="328" className="fk-muted fk-small" textAnchor="start">tokenize、detokenize 与网络 IO 都不占主线程</text>
      </>}</Figure>
    )
  }
  return (
    <Figure
      title={title}
      desc={desc}
      height={314}
      below={legend}
      caption={caption}
    >{arrow => <>
      <Band x={0} y={0} w={222} h={312} hue="teal" title="进程 1 · API server" />
      <Band x={240} y={0} w={340} h={312} hue="purple" title="进程 2 · EngineCore" />
      <Band x={598} y={0} w={162} h={312} hue="blue" title="GPU" />

      <Cell x={18} y={52} w={186} h={54} rx={8} fill="teal" label="HTTP · tokenize" fs={13} />
      <Cell x={18} y={223} w={186} h={60} rx={8} fill="teal" label="增量 detokenize" sub="流式返回" fs={13} />
      <Cell x={258} y={58} w={150} h={42} rx={8} fill="gray" label="输入线程" fs={13} />
      <Cell x={258} y={124} w={304} h={76} rx={8} fill="purple" label="主线程 busy loop" sub="schedule → 准备输入 → 提交" fs={15} />
      <Cell x={412} y={226} w={150} h={54} rx={8} fill="gray" label="输出线程" fs={13} />
      <Cell x={616} y={124} w={126} h={76} rx={8} fill="blue" label="forward" sub="+ sample" fs={13} />

      <Link d="M204 79 H258" hue="purple" arrow={arrow} focus />
      <Link d="M333 100 V124" hue="purple" arrow={arrow} focus />
      <Link d="M562 162 H616" hue="purple" arrow={arrow} focus />
      <Link d="M679 200 V253 H562" hue="purple" arrow={arrow} focus />
      <Link d="M412 253 H204" hue="purple" arrow={arrow} focus />
      <text x="231" y="93" className="fk-muted fk-small">ZMQ</text>
      <text x="300" y="268" className="fk-muted fk-small">ZMQ</text>
      {[[231, 64], [349, 112], [589, 146], [695, 226], [372, 238], [111, 298]].map(([x, y], i) => <Badge key={i} x={x} y={y} n={i + 1} />)}
      <text x="18" y="146" className="fk-muted" textAnchor="start">tokenize、detokenize 与</text>
      <text x="18" y="164" className="fk-muted" textAnchor="start">网络 IO 都不占主线程</text>
    </>}</Figure>
  )
}

// 图 3：MRV1 persistent batch 与 MRV2 固定行状态表
export function EnginePersistentBatch() {
  const compact = useCompact()
  const AX = compact ? 300 : 330 // 右侧注释的起点
  const CW = 46
  const CH = 30
  const row = (x: number, y: number, ids: (string | null)[], fill: 'blue' | 'purple') =>
    ids.map((id, i) => id
      ? <Cell key={i} x={x + i * (CW + 4)} y={y} w={CW} h={CH} fill={fill} label={id} />
      : <Cell key={i} x={x + i * (CW + 4)} y={y} w={CW} h={CH} fill="empty" />)
  return (
    <Figure
      title="persistent batch 与固定行状态表"
      desc="左：MRV1 中持久状态的行序就是输入的行序，请求 B 结束后 C、D 要整体上移。右：MRV2 中每个请求占固定的一行，结束后该行留空，每步再按需要的顺序 gather 出输入。"
      width={compact ? 520 : 760}
      height={310}
      below={<Legend items={[['blue', '持久状态（MRV1）'], ['purple', '状态表（MRV2）'], ['teal', '本步输入']]} />}
      caption="示意图，请求 A–D 各有一行状态。MRV1：状态即输入，请求增删要整表重排。MRV2：状态与输入分离，状态行位置固定，输入由 gather 在 GPU 上生成。"
    >{arrow => <>
      <text x="0" y="12" className="fk-strong" textAnchor="start">MRV1</text>
      <text x="0" y="60" className="fk-muted" textAnchor="start">B 结束前</text>
      {row(90, 46, ['A', 'B', 'C', 'D'], 'blue')}
      <text x="0" y="112" className="fk-muted" textAnchor="start">B 结束后</text>
      {row(90, 98, ['A', 'C', 'D', null], 'blue')}
      <Link d="M213 76 L163 98" arrow={arrow} />
      <Link d="M263 76 L213 98" arrow={arrow} />
      <text x={AX} y="113" className="fk-muted" textAnchor="start">C、D 上移一行，整表重排</text>
      <text x="0" y="146" className="fk-muted fk-small" textAnchor="start">状态的行序 = 输入的行序，异步下还要 barrier</text>

      <line x1="0" y1="164" x2={compact ? 520 : 760} y2="164" className="fk-rule" />

      <text x="0" y="186" className="fk-strong" textAnchor="start">MRV2</text>
      <text x="0" y="216" className="fk-muted" textAnchor="start">状态表</text>
      {row(90, 202, ['A', null, 'C', 'D'], 'purple')}
      <text x={AX} y="217" className="fk-muted" textAnchor="start">B 的行留空，其他请求不移动</text>
      <Link d="M113 232 V264" hue="teal" arrow={arrow} />
      <Link d="M213 232 L163 264" hue="teal" arrow={arrow} />
      <Link d="M263 232 L213 264" hue="teal" arrow={arrow} />
      <text x="0" y="277" className="fk-muted" textAnchor="start">本步输入</text>
      <Cell x={90} y={264} w={CW} h={CH} fill="teal" label="A" />
      <Cell x={140} y={264} w={CW} h={CH} fill="teal" label="C" />
      <Cell x={190} y={264} w={CW} h={CH} fill="teal" label="D" />
      {compact
        ? <><text x="256" y="270" className="fk-muted" textAnchor="start">gather：按 attention 需要的顺序，</text>
          <text x="256" y="289" className="fk-muted" textAnchor="start">在 GPU 上生成</text></>
        : <text x="330" y="279" className="fk-muted" textAnchor="start">gather：按 attention 需要的顺序，在 GPU 上生成</text>}
    </>}</Figure>
  )
}

// 图 4：CUDA Graph 分派
export function EngineCudaGraphDispatch() {
  const compact = useCompact()
  const title = '一个 batch 最终走整图、分段图还是 eager'
  const desc = '13 个 token 的混合 batch 补齐到 16。纯 decode 且有对应 size 走整图；prefill 或混合 batch 走分段图，attention 仍逐 kernel 发射；没有可用的 capture size 则 eager。'
  const legend = <Legend items={[['blue', '进图的部分'], ['red', 'attention（仍逐 kernel launch）'], ['gray', 'eager 逐个 launch']]} />
  const caption = '示意图。默认 capture sizes 为 1、2、4，8 至 248 步长 8，之后步长 16，上限 512。示例 batch：3 个 decode token 加 10 个 prefill token，补齐到 16。'
  if (compact) {
    // 窄版：补齐说明移到 token 行下方，三栏收窄，说明文字分行。
    const BY = 178
    const small = (x: number, y: number, t: string) => <text x={x} y={y} className="fk-muted fk-small" textAnchor="start">{t}</text>
    return (
      <Figure title={title} desc={desc} width={520} height={BY + 190} below={legend} caption={caption}>{arrow => <>
        <text x="0" y="12" className="fk-strong" textAnchor="start">本步 batch：3 个 decode + 1 个 prefill chunk = 13 个 token</text>
        {Array.from({ length: 16 }, (_, i) => <Cell key={i} x={i * 30} y={32} w={26} h={26} rx={4} fill={i < 3 ? 'blue' : i < 13 ? 'amber' : 'empty'} label={i < 3 ? 'd' : i < 13 ? 'p' : undefined} />)}
        <text x="476" y="74" className="fk-muted" textAnchor="end">↑ 补齐 3 个空位到 16（最近的 capture size）</text>
        <Link d="M128 58 V96" arrow={arrow} />
        <Cell x={95} y={96} w={330} h={44} rx={8} fill="green" label="dispatcher：token 数、是否 uniform decode" fs={13} />
        <Link d="M82 158 H438" />
        <Link d="M82 158 V178" arrow={arrow} />
        <Link d="M438 158 V178" arrow={arrow} />
        <Link d="M260 140 V178" hue="purple" arrow={arrow} focus />
        {([[0, 'FULL 整图', ['纯 uniform decode，', '且有对应的 size'], 'blue'], [178, 'PIECEWISE 分段图', ['prefill 或混合 batch', '（本例）'], 'purple'], [356, 'eager', ['没有可用的', 'capture size'], 'gray']] as const).map(([x, t, c, hue]) =>
          <g key={t}><Band x={x} y={BY} w={164} h={190} hue={hue} title={t} />
            {small(x + 14, BY + 44, c[0])}{small(x + 14, BY + 61, c[1])}</g>)}
        <Cell x={14} y={BY + 80} w={136} h={44} rx={4} fill="blue" label="整条 forward" sub="含 attention" />
        {small(14, BY + 146, 'replay 一次 = 一次 launch')}
        {[192, 240, 288].map(x => <rect key={x} x={x} y={BY + 87} width={30} height={30} rx="4" className="fk-fill-blue" />)}
        {[224, 272].map(x => <rect key={x} x={x} y={BY + 87} width={14} height={30} rx="4" className="fk-fill-red" />)}
        {small(192, BY + 136, 'attention 形状随请求')}
        {small(192, BY + 153, '变化，留在图外')}
        {small(192, BY + 170, '逐 kernel launch')}
        {Array.from({ length: 7 }, (_, i) => <rect key={i} x={370 + i * 19} y={BY + 87} width={14} height={30} rx="3" className="fk-fill-gray" />)}
        {small(370, BY + 146, '每个 kernel 单独 launch')}
      </>}</Figure>
    )
  }
  return (
    <Figure
      title={title}
      desc={desc}
      height={348}
      below={legend}
      caption={caption}
    >{arrow => <>
      <text x="0" y="12" className="fk-strong" textAnchor="start">本步 batch：3 个 decode + 1 个 prefill chunk = 13 个 token</text>
      {Array.from({ length: 16 }, (_, i) => <Cell key={i} x={i * 30} y={32} w={26} h={26} rx={4} fill={i < 3 ? 'blue' : i < 13 ? 'amber' : 'empty'} label={i < 3 ? 'd' : i < 13 ? 'p' : undefined} />)}
      <text x="492" y="45" className="fk-muted" textAnchor="start">← 补齐 3 个空位到 16（最近的 capture size）</text>
      <Link d="M253 58 V90" arrow={arrow} />
      <Cell x={225} y={90} w={330} h={44} rx={8} fill="green" label="dispatcher：token 数、是否 uniform decode" fs={13} />
      <Link d="M100 158 H660" />
      <Link d="M100 158 V178" arrow={arrow} />
      <Link d="M660 158 V178" arrow={arrow} />
      <Link d="M390 134 V178" hue="purple" arrow={arrow} focus />
      {([[0, 'FULL 整图', '纯 uniform decode，且有对应的 size', 'blue'], [290, 'PIECEWISE 分段图', 'prefill 或混合 batch（本例）', 'purple'], [560, 'eager', '没有可用的 capture size', 'gray']] as const).map(([x, t, c, hue]) =>
        <g key={t}><Band x={x} y={178} w={200} h={166} hue={hue} title={t} />
          <text x={x + 14} y={220} className="fk-muted fk-small" textAnchor="start">{c}</text></g>)}
      <Cell x={14} y={244} w={172} h={30} rx={4} fill="blue" label="整条 forward，含 attention" />
      <text x="14" y="300" className="fk-muted fk-small" textAnchor="start">replay 一次 = 一次 launch</text>
      {[304, 372, 440].map(x => <rect key={x} x={x} y={244} width={44} height={30} rx="4" className="fk-fill-blue" />)}
      {[350, 418].map(x => <rect key={x} x={x} y={244} width={20} height={30} rx="4" className="fk-fill-red" />)}
      <text x="304" y="296" className="fk-muted fk-small" textAnchor="start">attention 的形状随请求变化，</text>
      <text x="304" y="314" className="fk-muted fk-small" textAnchor="start">留在图外逐 kernel launch</text>
      {Array.from({ length: 10 }, (_, i) => <rect key={i} x={574 + i * 17.5} y={244} width={13} height={30} rx="3" className="fk-fill-gray" />)}
      <text x="574" y="300" className="fk-muted fk-small" textAnchor="start">每个 kernel 单独 launch</text>
    </>}</Figure>
  )
}

// 图 5：实测：每步 GPU 空转与端到端 ITL
const CFG = [
  ['default', '默认'],
  ['no_async', '关 async'],
  ['mrv1', 'MRV1'],
  ['piecewise', 'PIECEWISE'],
  ['graph_none', '无 CUDA Graph'],
] as const

export function EngineMeasured() {
  const compact = useCompact()
  const maxItl = 16
  const X0 = 120
  const PX = compact ? 22 : 34 // px per ms
  return (
    <Figure
      title="不同配置下的 decode ITL"
      desc="H100 上 Qwen3-8B 的中位数 ITL：默认 7.03 ms，关 async 9.02 ms，MRV1 7.03 ms，PIECEWISE 7.57 ms，无 CUDA Graph 13.95 ms（并发 8）。"
      width={compact ? 520 : 760}
      height={330}
      below={<Legend items={[['blue', '并发 8'], ['amber', '并发 64']]} />}
      caption="H100，Qwen3-8B，输入 128 / 输出 256。ITL 越低越好。"
    >{() => <>
      {[0, 4, 8, 12, 16].map(v => <g key={v}>
        <line x1={X0 + v * PX} y1="10" x2={X0 + v * PX} y2="290" className="fk-rule" />
        <text x={X0 + v * PX} y="304" className="fk-muted">{v}</text>
      </g>)}
      <text x={X0 + maxItl * PX} y="322" className="fk-muted" textAnchor="end">ITL（ms）</text>
      {CFG.map(([k, label], i) => {
        const y = 18 + i * 54
        const a = measured.bench[k]['8'].itl
        const b = measured.bench[k]['64'].itl
        return <g key={k}>
          <text x={X0 - 10} y={y + 19} className="fk-strong" textAnchor="end">{label}</text>
          <rect x={X0} y={y} width={a * PX} height="18" rx="2" className="fk-fill-blue" />
          <text x={X0 + a * PX + 6} y={y + 9} className="fk-stat" textAnchor="start">{a.toFixed(2)}</text>
          <rect x={X0} y={y + 22} width={b * PX} height="18" rx="2" className="fk-fill-amber" />
          <text x={X0 + b * PX + 6} y={y + 31} className="fk-stat" textAnchor="start">{b.toFixed(2)}</text>
        </g>
      })}
    </>}</Figure>
  )
}
