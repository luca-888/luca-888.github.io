import { Band, Badge, Cell, Figure, Legend, Link, type Fill } from '../../figure-kit'

// 配色约定：blue = P 与 prefill 计算，purple = D 与示例路径，green = 已缓存的 KV，
// amber = 本请求的 KV（传输中或占着），void = 排队与空闲，teal = router 与合并部署。

const amber = { stroke: 'var(--fk-amber)' }
const red = { stroke: 'var(--fk-red)' }

// 一排小格子：显存 block、缓存 block 或排队的请求
function Blocks({ x, y, fills, w = 11, h = 14, pitch = 13 }: { x: number; y: number; fills: Fill[]; w?: number; h?: number; pitch?: number }) {
  return <>{fills.map((f, i) => <rect key={i} x={x + i * pitch} y={y} width={w} height={h} rx={2} className={`fk-fill-${f}`} />)}</>
}
const run = (...parts: [Fill, number][]) => parts.flatMap(([f, n]) => Array<Fill>(n).fill(f))

// 图 1：分离部署下一个请求经过的四次决策
export function PdDecisions() {
  const ps = [
    { name: 'P1', hit: 8, queue: 2 },
    { name: 'P2', hit: 0, queue: 0 },
    { name: 'P3', hit: 2, queue: 1 },
  ]
  return (
    <Figure
      title="PD 分离下的四次决策"
      desc="router 在三个 P 中选择缓存命中 80%、排队 2 个的 P1，KV 传到显存还有空余的 D1；planner 调整 P 与 D 的数量。"
      height={340}
      below={<Legend items={[['green', '这个请求的前缀在该 P 上已缓存'], ['gray', '排队的请求'], ['purple', 'D 上已占用的 KV'], ['amber', '这个请求的 KV'], ['free', '空闲显存']]} />}
      caption="紫色是一个 6000 token 请求的路径，每格代表 10% 的 prompt 或一段显存。① router 选 P：P1 命中 80% 但有 2 个请求在排队，P2 空闲但要从头算。② router 决定这个请求要不要 P；D 上已缓存大部分 prompt 时直接发给 D。③ D1 为这个请求预留 4 格显存，接收 P1 算出的 KV；D2 只剩 1 格，接纳不了。④ planner 按负载增减 P 与 D 的数量。"
    >{arrow => <>
      <Band x={0} y={0} w={140} h={340} hue="teal" title="router" />
      <Band x={160} y={0} w={430} h={150} hue="blue" title="Prefill 实例（P）" />
      <Band x={160} y={195} w={430} h={145} hue="purple" title="" />
      {/* 标题放右侧，避开 P1 → D1 的传输线 */}
      <text x={576} y={215} textAnchor="end" className="fk-strong fk-text-purple">Decode 实例（D）</text>
      <Band x={610} y={0} w={150} h={340} hue="gray" title="planner" />

      <Cell x={12} y={140} w={116} h={60} rx={8} fill="teal" label="请求" sub="prompt 6000 token" fs={14} />

      {ps.map((p, i) => {
        const gx = 175 + i * 140
        return <g key={p.name}>
          <Cell x={gx} y={40} w={130} h={26} rx={6} fill="blue" label={p.name} />
          <Blocks x={gx} y={74} fills={run(['green', p.hit], ['void', 10 - p.hit])} />
          <text x={gx} y={98} textAnchor="start" className="fk-muted fk-small">命中 {p.hit * 10}%</text>
          <Blocks x={gx} y={112} fills={run(['gray', p.queue])} w={14} pitch={17} />
          <text x={gx + p.queue * 17} y={119} textAnchor="start" className="fk-muted fk-small">{p.queue ? `排队 ${p.queue}` : '空闲'}</text>
          {i === 0 && <rect x={gx - 6} y={34} width={142} height={104} rx={8} className="fk-focus" />}
        </g>
      })}

      <Cell x={200} y={160} w={80} h={26} rx={6} fill="amber" label="KV 传输" small />

      <Cell x={175} y={236} w={250} h={26} rx={6} fill="purple" label="D1" />
      {[0, 1].map(r => <Blocks key={r} x={175} y={270 + r * 16} pitch={15.6} w={13} h={13}
        fills={r === 0 ? run(['purple', 14], ['amber', 2]) : run(['amber', 2], ['free', 14])} />)}
      <text x={175} y={318} textAnchor="start" className="fk-muted fk-small">已占用 14 · 为这个请求预留 4 · 空闲 14</text>
      <rect x={169} y={230} width={262} height={98} rx={8} className="fk-focus" />

      <Cell x={445} y={236} w={130} h={26} rx={6} fill="purple" label="D2" />
      {[0, 1].map(r => <Blocks key={r} x={445} y={270 + r * 16} pitch={16.25} w={13} h={13}
        fills={r === 0 ? run(['purple', 8]) : run(['purple', 7], ['free', 1])} />)}
      <text x={445} y={318} textAnchor="start" className="fk-muted fk-small">已占用 15 / 16</text>

      <Blocks x={630} y={56} fills={run(['blue', 3])} w={22} h={22} pitch={28} />
      <text x={630} y={96} textAnchor="start" className="fk-muted">P × 3</text>
      <Badge x={685} y={170} n="4" />
      <text x={685} y={196} className="fk-muted">按负载增减</text>
      <Blocks x={630} y={258} fills={run(['purple', 2])} w={22} h={22} pitch={28} />
      <text x={630} y={298} textAnchor="start" className="fk-muted">D × 2</text>
      <Link d="M620 67 H592" arrow={arrow} />
      <Link d="M620 269 H592" arrow={arrow} />

      <Link d="M128 160 H148 V53 H175" hue="purple" arrow={arrow} focus />
      <Link d="M128 180 H148 V249 H175" hue="purple" arrow={arrow} focus />
      <Link d="M240 138 V160" hue="purple" focus />
      <Link d="M240 186 V230" hue="purple" arrow={arrow} focus />

      <Badge x={136} y={100} n="1" />
      <Badge x={136} y={226} n="2" />
      <Badge x={298} y={173} n="3" />
    </>}</Figure>
  )
}

// 图 2：同一个请求在三个 P 上的预计 TTFT（演示数据）
export function PdPickP() {
  const x0 = 150, s = 0.6
  const rows = [
    { name: 'P1', note: '命中 80% · 排队 2', q: 400, pre: 120, y: 24, best: true },
    { name: 'P2', note: '命中 0% · 空闲', q: 0, pre: 600, y: 66 },
    { name: 'P3', note: '命中 20% · 排队 1', q: 200, pre: 480, y: 108 },
    { name: 'P1', note: '若排队 4 个', q: 800, pre: 120, y: 164, ghost: true },
  ]
  return (
    <Figure
      title="三个 P 上的预计 TTFT"
      desc="P1 排队 400 ms、prefill 120 ms，共 520 ms，最短；P2 不排队但要算全部 600 ms；P3 共 680 ms。P1 的队列再长 2 个时变为 920 ms，不如 P2。"
      height={236}
      below={<Legend items={[['void', '排队'], ['blue', 'prefill（只算未命中的部分）']]} />}
      caption="演示数据：prompt 6000 token，每个未命中的 token 0.1 ms，队列中每个请求 200 ms。条形长度按时间绘制，即 Mooncake 的估计式 TTFT = 排队时间 + prefill 时间。最后一行是 P1 的队列从 2 个变为 4 个时的结果。"
    >{() => <>
      {rows.map(r => <g key={r.y} opacity={r.ghost ? 0.55 : 1}>
        <text x={0} y={r.y + 7} textAnchor="start" className="fk-strong">{r.name}</text>
        <text x={0} y={r.y + 24} textAnchor="start" className="fk-muted fk-small">{r.note}</text>
        {r.q > 0 && <Cell x={x0} y={r.y} w={r.q * s} h={30} fill="void" label={`排队 ${r.q}`} />}
        <Cell x={x0 + r.q * s} y={r.y} w={r.pre * s} h={30} fill="blue" label={r.pre >= 200 ? `prefill ${r.pre}` : r.pre} />
        <text x={x0 + (r.q + r.pre) * s + 8} y={r.y + 15} textAnchor="start" className={r.best ? 'fk-strong fk-text-purple' : 'fk-strong'}>{r.q + r.pre} ms</text>
        {r.best && <rect x={x0 - 2} y={r.y - 2} width={(r.q + r.pre) * s + 4} height={34} rx={4} className="fk-focus" />}
      </g>)}
      <line x1={0} x2={760} y1={150} y2={150} className="fk-rule" />
      {[0, 200, 400, 600, 800, 1000].map(t => <text key={t} x={x0 + t * s} y={216} className="fk-muted fk-small">{t}</text>)}
      <text x={x0 + 1000 * s} y={232} className="fk-muted fk-small">ms</text>
    </>}</Figure>
  )
}

// 图 3：同样 6000 token 的 prompt，D 上缓存的多少决定本地完成还是远端 prefill
export function PdLocalOrRemote() {
  const x0 = 150, pitch = 35, perBlock = 500, thr = 2048
  const row = (y: number, cached: number, label: string, note: string, verdict: string, sub: string, remote: boolean) => {
    const start = x0 + (cached / perBlock) * pitch
    const end = start + (thr / perBlock) * pitch
    return <g>
      <text x={0} y={y + 14} textAnchor="start" className="fk-strong">{label}</text>
      <text x={0} y={y + 32} textAnchor="start" className="fk-muted">{note}</text>
      <Blocks x={x0} y={y} fills={run(['green', cached / perBlock], ['blue', 12 - cached / perBlock])} w={32} h={44} pitch={pitch} />
      <path d={`M${start} ${y - 6} V${y - 13} H${end} V${y - 6}`} className="fk-link" style={amber} />
      <text x={start} y={y - 24} textAnchor="start" className="fk-muted fk-small">阈值 2048</text>
      <Cell x={610} y={y} w={150} h={44} rx={8} fill={remote ? 'amber' : 'purple'} label={verdict} sub={sub} />
    </g>
  }
  return (
    <Figure
      title="有效 prompt 长度决定是否远端 prefill"
      desc="多轮对话第 5 轮的 6000 个 token 中 5500 个已缓存在 D 上，只需算 500 个，短于 2048 的阈值，在 D 本地完成；新上传文档的 6000 个 token 全部未缓存，远超阈值，交给 P。"
      height={190}
      below={<Legend items={[['green', 'D 上已缓存'], ['blue', '需要计算（有效 prompt 长度）']]} />}
      caption="每格 500 token，两条 prompt 都是 6000 个 token。橙色括号是 Dynamo conditional disaggregation 的默认阈值 2048 token，从需要计算的部分开头量起：蓝色没有超出括号时在 D 本地完成。Dynamo 还要求有效长度占 prompt 的比例小于 0.7。"
    >{() => <>
      {row(36, 5500, '多轮对话第 5 轮', '前 4 轮在 D1 上生成', 'D 本地完成', '500 < 2048', false)}
      {row(130, 0, '新上传的长文档', '没有可复用的前缀', '远端 prefill', '6000 ≥ 2048', true)}
    </>}</Figure>
  )
}

// 图 4：D 暂时放不下时，先计算与先预留两种顺序各自占着哪一侧的显存（示意，时间不按比例）
export function PdAdmissionOrder() {
  const lane = (y: number, name: string) => <text x={96} y={y + 17} textAnchor="end" className="fk-strong">{name}</text>
  return (
    <Figure
      title="先计算与先预留"
      desc="同样是 D 在 t1 时刻才有空闲显存：先计算时 P 立即 prefill，KV 留在 P 上等 D，首个 token 较早；先预留时 P 等 D 预留好 block 再开始，D 上的空 block 在 prefill 期间空等，首个 token 较晚。"
      height={290}
      below={<Legend items={[['blue', 'GPU 计算'], ['amber', '占着的 KV'], ['empty', '预留的空 block / lease 余量'], ['void', '等待，不占显存']]} />}
      caption="示意图，时间不按比例。两种情形中 D 都在 t1 才有空闲显存。先计算时 P 的显存从 prefill 结束一直占到 D 读走；如果 D 在 lease 到期前仍放不下，KV 被释放，这次 prefill 作废。先预留时 P 在等待期间可以服务别的请求，代价是 D 的预留在 prefill 期间空着，首个 token 也更晚。"
    >{arrow => <>
      <text x={460} y={8} className="fk-muted">t1：D 有空闲显存</text>
      <Link d="M460 20 V262" />

      <text x={0} y={28} textAnchor="start" className="fk-strong fk-text-blue">先计算（vLLM 默认）</text>
      {lane(44, 'P')}
      <Cell x={110} y={44} w={190} h={34} fill="blue" label="prefill" />
      <Cell x={300} y={44} w={240} h={34} fill="amber" label="KV 留在 P 上" />
      <Cell x={540} y={44} w={100} h={34} fill="empty" label="lease 余量" />
      <path d="M640 38 V84" className="fk-link" style={red} />
      <text x={648} y={52} textAnchor="start" className="fk-muted fk-small">到期释放</text>
      <text x={648} y={68} textAnchor="start" className="fk-muted fk-small">prefill 作废</text>
      {lane(88, 'D')}
      <Cell x={110} y={88} w={350} h={34} fill="void" label="放不下，排队" />
      <Cell x={460} y={88} w={80} h={34} fill="amber" label="读 KV" />
      <Cell x={540} y={88} w={220} h={34} fill="blue" label="decode" />
      <Link d="M540 78 V88" hue="purple" arrow={arrow} />

      <text x={0} y={164} textAnchor="start" className="fk-strong fk-text-purple">先预留（SGLang 默认）</text>
      {lane(180, 'P')}
      <Cell x={110} y={180} w={350} h={34} fill="void" label="等 D 预留" />
      <Cell x={460} y={180} w={190} h={34} fill="blue" label="prefill" />
      {lane(224, 'D')}
      <Cell x={110} y={224} w={350} h={34} fill="void" label="放不下，排队" />
      <Cell x={460} y={224} w={240} h={34} fill="empty" label="预留的空 block" />
      <Cell x={700} y={224} w={60} h={34} fill="blue" label="decode" />
      <Link d="M650 214 H700 V224" hue="purple" arrow={arrow} />

      <Badge x={540} y={136} n="★" />
      <text x={554} y={136} textAnchor="start" className="fk-muted fk-small">首个 token</text>
      <Badge x={700} y={272} n="★" />
      <text x={686} y={272} textAnchor="end" className="fk-muted fk-small">首个 token</text>
    </>}</Figure>
  )
}

// 图 5：P 与 D 的能力之比决定 goodput（按 DistServe 的单卡数字推算）
export function PdRatio() {
  const x0 = 150, s = 21
  const rows = [
    { label: '2P + 1D', p: 2, d: 1, y: 30, best: true },
    { label: '1P + 1D', p: 1, d: 1, y: 92 },
    { label: '2P + 2D', p: 2, d: 2, y: 154 },
  ]
  const bar = (y: number, cap: number, g: number, fill: Fill, name: string) => <>
    <Cell x={x0} y={y} w={g * s} h={20} fill={fill} label={`${name} ${cap.toFixed(1)}`} small />
    {cap > g && <Cell x={x0 + g * s} y={y} w={(cap - g) * s} h={20} fill="void" label={(cap - g) * s > 40 ? '空闲' : undefined} small />}
  </>
  return (
    <Figure
      title="P 与 D 的比例决定 goodput"
      desc="2P+1D 时 P 的能力 11.2、D 的能力 10，goodput 10，每卡 3.3；1P+1D 受 P 限制，每卡 2.8；2P+2D 受 P 限制，D 空闲一半，每卡 2.8；合并部署 3 张卡每卡 1.6。"
      height={300}
      below={<Legend items={[['blue', 'P 的能力'], ['purple', 'D 的能力'], ['void', '多出、空闲的部分'], ['teal', '合并部署']]} />}
      caption="DistServe 的单卡数字：13B 模型、A100，同时满足 TTFT 与 ITL 要求时，一张卡只做 prefill 为 5.6 请求/秒，只做 decode 为 10，合并部署为 1.6。一个配置的 goodput 取 P、D 两池能力中较小的一个，即竖线位置。2P+1D 是 DistServe 论文中的例子，其余各行按同样方法推算。"
    >{() => <>
      {rows.map(r => {
        const P = r.p * 5.6, D = r.d * 10, g = Math.min(P, D)
        return <g key={r.label}>
          <text x={0} y={r.y + 20} textAnchor="start" className="fk-strong">{r.label}</text>
          {bar(r.y, P, g, 'blue', 'P')}
          {bar(r.y + 24, D, g, 'purple', 'D')}
          <path d={`M${x0 + g * s} ${r.y - 6} V${r.y + 50}`} className="fk-link fk-link-focus" style={{ stroke: 'var(--color-heading)' }} />
          <text x={630} y={r.y + 12} textAnchor="start" className={r.best ? 'fk-strong fk-text-purple' : 'fk-strong'}>每卡 {(g / (r.p + r.d)).toFixed(1)}</text>
          <text x={630} y={r.y + 30} textAnchor="start" className="fk-muted fk-small">{g.toFixed(1)} ÷ {r.p + r.d} 张卡</text>
        </g>
      })}
      <text x={0} y={236} textAnchor="start" className="fk-strong">合并部署 × 3</text>
      <Cell x={x0} y={226} w={4.8 * s} h={20} fill="teal" label="4.8" small />
      <text x={630} y={232} textAnchor="start" className="fk-strong">每卡 1.6</text>
      <text x={630} y={250} textAnchor="start" className="fk-muted fk-small">4.8 ÷ 3 张卡</text>
      {[0, 5, 10, 15, 20].map(t => <text key={t} x={x0 + t * s} y={278} className="fk-muted fk-small">{t}</text>)}
      <text x={x0 + 20 * s + 30} y={278} textAnchor="start" className="fk-muted fk-small">请求/秒</text>
    </>}</Figure>
  )
}

// 实测占位：方向确认后在 2 × H100 上补测
export function PdMeasurePlaceholder() {
  return (
    <div style={{ border: '1.5px dashed var(--color-border)', borderRadius: 8, padding: '16px 20px', color: 'var(--color-muted)', margin: '1.5em 0' }}>
      <strong>实测待补。</strong>2 × H100，同一模型，合并部署（2 个实例，chunked prefill）与 1P1D（NixlConnector）对比：prompt 长度变化时 ITL 的 p99 与 goodput。这里将放一张 ITL p99 随 prompt 长度变化的曲线和一个代表性数字。
    </div>
  )
}
