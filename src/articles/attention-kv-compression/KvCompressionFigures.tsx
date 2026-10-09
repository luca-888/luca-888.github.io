import type { ReactNode } from 'react'
import { Band, Cell, Figure, Legend, Link, useCompact } from '../../figure-kit'

// 语义色：blue 为 K，amber 为 V，purple 为 MLA 的 latent（同时充当 K 与 V），red 为 RoPE key，gray 为 query head。
const sub = (base: string, s: string): ReactNode => <>{base}<tspan dy="4" style={{ fontSize: '0.75em' }}>{s}</tspan></>

// MHA decode 一步的信息传递：3 个 head、4 个历史 token，示例路径为 head 1
export function MhaFlow() {
  const compact = useCompact()
  // 窄版：各列收窄，输出格只写“输出”。
  const g = compact
    ? { W: 520, QX: 4, QW: 100, TX: 146, TP: 59, TW: 53, PX: 388, PW: 60, OX: 460, OW: 60, B1: [0, 110], B2: [116, 266], DX: 131 }
    : { W: 760, QX: 24, QW: 118, TX: 232, TP: 76, TW: 68, PX: 556, PW: 64, OX: 644, OW: 108, B1: [8, 150], B2: [170, 372], DX: 196 }
  const { TX, TP, TW } = g
  const RH = 22, GE = TX + 3 * TP + TW
  const HEADS: { name: string; y: number }[] = [{ name: 'head 1', y: 54 }, { name: 'head 2', y: 126 }, { name: 'head h', y: 206 }]
  return (
    <Figure
      title="MHA 在 decode 一步中的信息传递"
      desc="当前 token 在每个 head 中有一个 query，每个历史 token 在每个 head 中各缓存一个 key 与一个 value。head 1 的 query 与 head 1 的 4 个 key 点积，softmax 后对 head 1 的 4 个 value 加权求和，得到 head 1 的输出；其余 head 用各自的 key 与 value 做同样的计算。"
      width={g.W}
      height={280}
      below={<Legend items={[['gray', 'query 与输出'], ['blue', 'key'], ['amber', 'value']]} />}
      caption="示例为 3 个 head、4 个历史 token，描出的路径是 head 1：query 与这个 head 的 4 个 key 点积，softmax 后对 4 个 value 加权求和。其余 head 读取各自的 key 与 value。"
    >{arrow => <>
      <Band x={g.B1[0]} y={4} w={g.B1[1]} h={270} hue="gray" title="当前 token" />
      <Band x={g.B2[0]} y={4} w={g.B2[1]} h={270} hue="blue" title="历史 token 的 KV cache" />
      {[1, 2, 3, 4].map(n => <text key={n} x={TX + (n - 1) * TP + TW / 2} y={42} className="fk-muted fk-small">token {n}</text>)}
      <text x={g.QX + g.QW / 2} y={186} className="fk-muted">⋮</text>
      <text x={TX + 1.5 * TP + TW / 2} y={186} className="fk-muted">⋮</text>
      {HEADS.map((h, r) => <g key={h.name}>
        <Cell x={g.QX} y={h.y} w={g.QW} h={RH} fill="gray" label={`${h.name} 的 query`} small focus={r === 0} />
        {[0, 1, 2, 3].map(n => <g key={n}>
          <Cell x={TX + n * TP} y={h.y} w={TW} h={RH} fill="blue" label="key" small />
          <Cell x={TX + n * TP} y={h.y + RH + 4} w={TW} h={RH} fill="amber" label="value" small />
        </g>)}
        <Link d={`M${g.QX + g.QW} ${h.y + RH / 2} H${TX}`} arrow={arrow} hue={r === 0 ? 'purple' : undefined} focus={r === 0} />
        <Cell x={g.OX} y={h.y + RH + 4} w={g.OW} h={RH} fill="gray" label={compact ? '输出' : `${h.name} 的输出`} small focus={r === 0} />
        {r === 0
          ? <>
              <text x={g.DX} y={h.y - 2} className="fk-muted fk-small">点积</text>
              <Cell x={g.PX} y={h.y} w={g.PW} h={RH} rx={11} fill="ink" label="softmax" small />
              <Cell x={g.PX} y={h.y + RH + 4} w={g.PW} h={RH} rx={11} fill="ink" label="加权求和" small />
              <Link d={`M${GE} ${h.y + RH / 2} H${g.PX}`} arrow={arrow} hue="purple" focus />
              <Link d={`M${GE} ${h.y + RH * 1.5 + 4} H${g.PX}`} arrow={arrow} hue="purple" focus />
              <Link d={`M${g.PX + g.PW / 2} ${h.y + RH} V${h.y + RH + 4}`} hue="purple" focus />
              <Link d={`M${g.PX + g.PW} ${h.y + RH * 1.5 + 4} H${g.OX}`} arrow={arrow} hue="purple" focus />
            </>
          : <Link d={`M${GE} ${h.y + RH * 1.5 + 4} H${g.OX}`} arrow={arrow} />}
      </g>)}
    </>}</Figure>
  )
}

// 图 1：128 个 query head、head dim 128 时，四种形式每层每 token 缓存的元素数，面积按真实比例
const H = 128
const D = 128
const RH = 15 // K、V 各一行的高度；MHA 的 K 行恰好铺满 W
const GAP = 2

type Scheme = { name: string; note: string; groups: number; cache: number; mla?: boolean }
const SCHEMES: Scheme[] = [
  { name: 'MHA', note: '每个 query head 一组 K/V', groups: H, cache: 2 * H * D },
  { name: 'GQA-8', note: '每 16 个 query head 共享一组', groups: 8, cache: 2 * 8 * D },
  { name: 'MQA', note: '全部 query head 共享一组', groups: 1, cache: 2 * D },
  { name: 'MLA', note: '全部 query head 共享一个 latent', groups: 1, cache: 512 + 64, mla: true },
]

export function KvCachePerLayer() {
  const compact = useCompact()
  // 窄版：名称与说明移到每行上方，缓存块从左边缘起
  const X0 = compact ? 0 : 184, W = compact ? 440 : 516, LAB = compact ? 24 : 0
  const PXR = W / (H * D) // 每行每个元素的宽度
  const TOPS = compact ? [0, 110, 260, 394] : [0, 100, 234, 346] // 各行起点：MHA 的 K/V 紧贴 query head，不需要汇聚区
  return (
    <Figure
      title="四种 attention 形式每层每 token 的 KV cache"
      desc="128 个 query head、head dim 128 时，每层每 token 的缓存元素数：MHA 32768，GQA-8 2048，MQA 256，MLA 576（512 维 latent 加 64 维 RoPE key）。"
      width={compact ? 520 : 760}
      height={TOPS[3] + 110 + LAB}
      below={<Legend items={[['gray', 'query head'], ['blue', 'K'], ['amber', 'V'], ['purple', 'latent，同时充当 K 与 V'], ['red', 'RoPE key']]} />}
      caption="缓存块的面积与元素数成正比，四行同一比例：MHA 的 K、V 各铺满一行，MQA 只有 MHA 的 1/128。MLA 只缓存一个低维向量（latent）和一个 RoPE key，维度取 DeepSeek-V3 的配置：latent 512 维，RoPE key 64 维；为便于比较，四行都按每个 head 的 key、value 为 128 维计算。"
    >{() => SCHEMES.map((s, r) => {
      const y = TOPS[r]
      const qy = y + 4 + LAB, qh = 20, cy = y + 64 + LAB
      const span = W / s.groups
      const kvW = D * PXR // 一组 K（或 V）的宽度
      return <g key={s.name}>
        {compact
          ? <>
              <text x={0} y={y + 12} className="fk-strong" textAnchor="start">{s.name}</text>
              <text x={58} y={y + 12} className="fk-muted fk-small" textAnchor="start">{s.note}</text>
            </>
          : <>
              <text x={X0 - 16} y={qy + 10} className="fk-strong" textAnchor="end">{s.name}</text>
              <text x={X0 - 16} y={qy + 30} className="fk-muted fk-small" textAnchor="end">{s.note}</text>
            </>}
        {s.groups === 8
          ? Array.from({ length: 8 }, (_, g) => <Cell key={g} x={X0 + g * span + GAP} y={qy} w={span - 2 * GAP} h={qh} fill="gray" label="16" small />)
          : <Cell x={X0} y={qy} w={W} h={qh} fill="gray" label="128 个 query head" small />}
        {s.groups === H
          ? <>
              <Cell x={X0} y={cy - 26} w={W} h={RH} fill="blue" label="K：128 个 head × 128 维" small />
              <Cell x={X0} y={cy - 26 + RH + GAP} w={W} h={RH} fill="amber" label="V：128 个 head × 128 维" small />
            </>
          : Array.from({ length: s.groups }, (_, g) => {
              const mid = X0 + (g + 0.5) * span
              const bw = s.mla ? 512 / 2 * PXR : kvW
              const bx = mid - bw / 2
              return <g key={g}>
                <polygon points={`${X0 + g * span + GAP},${qy + qh + 2} ${X0 + (g + 1) * span - GAP},${qy + qh + 2} ${bx + bw + 1},${cy - 3} ${bx - 1},${cy - 3}`} className="fk-band-gray" />
                {s.mla
                  ? <>
                      <rect x={bx} y={cy} width={bw} height={2 * RH + GAP} className="fk-fill-purple" />
                      <rect x={bx + bw} y={cy} width={64 / 2 * PXR} height={2 * RH + GAP} className="fk-fill-red" />
                    </>
                  : <>
                      <rect x={bx} y={cy} width={bw} height={RH} className="fk-fill-blue" />
                      <rect x={bx} y={cy + RH + GAP} width={bw} height={RH} className="fk-fill-amber" />
                    </>}
              </g>
            })}
        {s.groups === 1 && (s.mla
          ? <>
              <text x={X0 + W / 2 + 16} y={cy + 7} className="fk-muted fk-text-purple" textAnchor="start">latent 512 维</text>
              <text x={X0 + W / 2 + 16} y={cy + RH + 9} className="fk-muted" textAnchor="start">+ RoPE key 64 维</text>
            </>
          : <>
              <text x={X0 + W / 2 + 12} y={cy + RH / 2} className="fk-muted" textAnchor="start">K 128 维</text>
              <text x={X0 + W / 2 + 12} y={cy + RH * 1.5 + GAP} className="fk-muted" textAnchor="start">V 128 维</text>
            </>)}
        {s.groups === 8 && <text x={X0 + W / 2} y={cy + 2 * RH + 16} className="fk-muted">每组 K、V 各 128 维，共 8 组</text>}
        <text x={X0 + W + 14} y={(s.groups === H ? cy - 26 : cy) + RH + 1} className="fk-stat" textAnchor="start">{s.cache}</text>
      </g>
    })}</Figure>
  )
}

// MHA 与 MLA 从 hidden state 到 key、value 的投影路径，描边的块写入 KV cache
export function MlaLatentPath() {
  const compact = useCompact()
  const CH = 44, Y1 = 30, Y2 = 142
  const sup = (base: string, s: string): ReactNode => <>{base}<tspan dy="-5" style={{ fontSize: '0.7em' }}>{s}</tspan></>
  const kv = (x: number, y: number, w: number, dims?: boolean) => <>
    <Cell x={x} y={y} w={w} h={CH / 2 - 1} fill="blue" label={dims ? '128 个 key · 16384 维' : '128 个 key'} small />
    <Cell x={x} y={y + CH / 2 + 1} w={w} h={CH / 2 - 1} fill="amber" label={dims ? '128 个 value · 16384 维' : '128 个 value'} small />
  </>
  const title = 'MHA 与 MLA 的投影路径'
  const desc = 'MHA 把 hidden state 经每个 head 的投影矩阵直接投影为 128 个 key 与 128 个 value，共 32768 维，全部缓存。MLA 先用所有 head 共用的下投影得到 512 维的 latent 并缓存，计算时再用每个 head 的上投影还原出 key 与 value，还原结果不缓存。'
  const caption = '描边的块写入 KV cache。维度取 DeepSeek-V3 的配置，为每层每 token 的数值。'
  if (compact) {
    // 窄版：MLA 一行折成两行，latent 向下接上投影，还原出的 key、value 在左侧。
    const A = 30, B = 150, C = 228
    return (
      <Figure title={title} desc={desc} width={520} height={300} caption={caption}>{arrow => <>
        <text x={0} y={12} className="fk-strong" textAnchor="start">MHA</text>
        <Cell x={0} y={A} w={104} h={CH} rx={6} fill="gray" label="hidden state" />
        <Cell x={130} y={A} w={124} h={CH} rx={6} fill="gray" label="投影" sub="每个 head 一份" />
        {kv(282, A, 236, true)}
        <rect x={280} y={A - 2} width={240} height={CH + 4} rx={4} className="fk-focus" />
        <Link d={`M104 ${A + CH / 2} H130`} arrow={arrow} />
        <Link d={`M254 ${A + CH / 2} H280`} arrow={arrow} />
        <text x={400} y={A + CH + 20} className="fk-stat">缓存 32768 维</text>

        <text x={0} y={B - 18} className="fk-strong" textAnchor="start">MLA</text>
        <Cell x={0} y={B} w={104} h={CH} rx={6} fill="gray" label="hidden state" />
        <Cell x={130} y={B} w={124} h={CH} rx={6} fill="gray" label={sup('下投影 W', 'DKV')} sub="所有 head 共用" />
        <Cell x={282} y={B} w={110} h={CH} rx={6} fill="purple" label="latent" sub="512 维" focus />
        <text x={406} y={B + CH / 2} className="fk-stat" textAnchor="start">缓存 512 维</text>
        <Cell x={282} y={C} w={110} h={CH} rx={6} fill="gray" label="上投影" sub="每个 head 一份" />
        <g opacity={0.45}>{kv(100, C, 150)}</g>
        <Link d={`M104 ${B + CH / 2} H130`} arrow={arrow} />
        <Link d={`M254 ${B + CH / 2} H280`} arrow={arrow} hue="purple" focus />
        <Link d={`M337 ${B + CH + 2} V${C}`} arrow={arrow} />
        <Link d={`M282 ${C + CH / 2} H250`} arrow={arrow} />
        <text x={175} y={C + CH + 20} className="fk-muted">用时还原，不缓存</text>
      </>}</Figure>
    )
  }
  return (
    <Figure
      title={title}
      desc={desc}
      height={222}
      caption={caption}
    >{arrow => <>
      <text x={84} y={Y1 + CH / 2} className="fk-strong" textAnchor="end">MHA</text>
      <Cell x={100} y={Y1} w={104} h={CH} rx={6} fill="gray" label="hidden state" />
      <Cell x={236} y={Y1} w={130} h={CH} rx={6} fill="gray" label="投影" sub="每个 head 一份" />
      {kv(398, Y1, 200, true)}
      <rect x={396} y={Y1 - 2} width={204} height={CH + 4} rx={4} className="fk-focus" />
      <Link d={`M204 ${Y1 + CH / 2} H236`} arrow={arrow} />
      <Link d={`M366 ${Y1 + CH / 2} H396`} arrow={arrow} />
      <text x={612} y={Y1 + CH / 2} className="fk-stat" textAnchor="start">缓存 32768 维</text>

      <text x={84} y={Y2 + CH / 2} className="fk-strong" textAnchor="end">MLA</text>
      <Cell x={100} y={Y2} w={104} h={CH} rx={6} fill="gray" label="hidden state" />
      <Cell x={236} y={Y2} w={110} h={CH} rx={6} fill="gray" label={sup('下投影 W', 'DKV')} sub="所有 head 共用" />
      <Cell x={378} y={Y2} w={90} h={CH} rx={6} fill="purple" label="latent" sub="512 维" focus />
      <Cell x={500} y={Y2} w={110} h={CH} rx={6} fill="gray" label="上投影" sub="每个 head 一份" />
      <g opacity={0.45}>{kv(642, Y2, 104)}</g>
      <Link d={`M204 ${Y2 + CH / 2} H236`} arrow={arrow} />
      <Link d={`M346 ${Y2 + CH / 2} H376`} arrow={arrow} hue="purple" focus />
      <Link d={`M470 ${Y2 + CH / 2} H500`} arrow={arrow} />
      <Link d={`M610 ${Y2 + CH / 2} H642`} arrow={arrow} />
      <text x={423} y={Y2 + CH + 20} className="fk-stat">缓存 512 维</text>
      <text x={694} y={Y2 + CH + 20} className="fk-muted">用时还原，不缓存</text>
    </>}</Figure>
  )
}

// 图 2：一个 query head 与一个历史 token 的分数，两种算法的计算顺序
export function MlaAbsorb() {
  const compact = useCompact()
  // 窄版：两个分组上下排列，各占整行。
  const CH = 44
  const CW = compact ? 100 : 84, BW = compact ? 520 : 364, BH = compact ? 248 : 266
  const XS = compact ? [44, 210, 376] : [22, 142, 262] // 三列相对分组左边缘的位置
  const L = { x: compact ? 0 : 8, y: compact ? 0 : 4 }
  const R = { x: compact ? 0 : 388, y: compact ? BH + 12 : 4 }
  const wuk = (t: string) => <>× W<tspan dy="-5" style={{ fontSize: '0.7em' }}>{t}</tspan></>
  const note = (o: { x: number; y: number }, a: string, b: string) => compact
    ? <text x={o.x + BW / 2} y={o.y + 226} className="fk-muted">{a}{b}</text>
    : <>
        <text x={o.x + BW / 2} y={o.y + 232} className="fk-muted">{a}</text>
        <text x={o.x + BW / 2} y={o.y + 250} className="fk-muted">{b}</text>
      </>
  return (
    <Figure
      title="MLA 的矩阵吸收"
      desc="左侧先用上投影矩阵把缓存的 512 维 latent 还原成 128 维 key，再与 query 点积；右侧先把 query 乘以上投影矩阵的转置，得到 512 维向量，再直接与缓存的 latent 点积。两者结果相同。"
      width={compact ? 520 : 760}
      height={compact ? 2 * BH + 12 : 318}
      caption="i 是 query head，j 是历史 token；W 的上标 UK 表示 key 的上投影，每个 head 一份，形状为 128 × 512。"
    >{arrow => {
      const [l1, l2, l3] = XS.map(x => L.x + x), [r1, r2, r3] = XS.map(x => R.x + x)
      const ly = L.y + 52, ry = R.y + 52 // 上排格子的 y；下排为 +100
      return <>
        <Band x={L.x} y={L.y} w={BW} h={BH} hue="gray" title="展开：先还原 key" />
        <Band x={R.x} y={R.y} w={BW} h={BH} hue="purple" title="吸收：先变换 query" />

        {/* 展开：c_j → W^UK → k → · q */}
        <Cell x={l1} y={ly} w={CW} h={CH} rx={6} fill="purple" label={sub('c', 'j')} sub="512 · 缓存" focus />
        <Cell x={l2} y={ly} w={CW} h={CH} rx={6} fill="gray" label={wuk('UK')} sub="每个 token 一次" />
        <Cell x={l3} y={ly} w={CW} h={CH} rx={6} fill="blue" label={sub('k', 'j,i')} sub="128" />
        <Link d={`M${l1 + CW} ${ly + 22} H${l2}`} arrow={arrow} hue="purple" focus />
        <Link d={`M${l2 + CW} ${ly + 22} H${l3}`} arrow={arrow} hue="purple" focus />
        <Cell x={l2} y={ly + 100} w={CW} h={CH} rx={6} fill="gray" label={sub('q', 'i')} sub="128" />
        <Cell x={l3 + CW / 2 - 22} y={ly + 108} w={44} h={28} rx={14} fill="ink" label="点积" small />
        <Link d={`M${l3 + CW / 2} ${ly + CH} V${ly + 108}`} arrow={arrow} hue="purple" focus />
        <Link d={`M${l2 + CW} ${ly + 122} H${l3 + CW / 2 - 22}`} arrow={arrow} />
        {note(L, '点积在 128 维上；历史有 L 个 token，', '就要做 L 次 128 × 512 的还原')}

        {/* 吸收：q_i → W^UK⊤ → q̃ ; c_j 直接参与 */}
        <Cell x={r1} y={ry + 100} w={CW} h={CH} rx={6} fill="gray" label={sub('q', 'i')} sub="128" />
        <Cell x={r2} y={ry + 100} w={CW} h={CH} rx={6} fill="gray" label={wuk('UK⊤')} sub="每步一次" />
        <Cell x={r3} y={ry + 100} w={CW} h={CH} rx={6} fill="purple" label={<>q̃<tspan dy="4" style={{ fontSize: '0.75em' }}>i</tspan></>} sub="512" />
        <Link d={`M${r1 + CW} ${ry + 122} H${r2}`} arrow={arrow} />
        <Link d={`M${r2 + CW} ${ry + 122} H${r3}`} arrow={arrow} />
        <Cell x={r2} y={ry} w={CW} h={CH} rx={6} fill="purple" label={sub('c', 'j')} sub="512 · 缓存" focus />
        <Cell x={r3 + CW / 2 - 22} y={ry + 8} w={44} h={28} rx={14} fill="ink" label="点积" small />
        <Link d={`M${r2 + CW} ${ry + 22} H${r3 + CW / 2 - 22}`} arrow={arrow} hue="purple" focus />
        <Link d={`M${r3 + CW / 2} ${ry + 100} V${ry + 36}`} arrow={arrow} />
        {note(R, '点积在 512 维上；不生成 key，', '缓存的 latent 直接读入')}
      </>
    }}</Figure>
  )
}

// 图 3：RoPE 与吸收的冲突，以及 decoupled RoPE 的分数；下半部分条的长度与维度成正比
export function DecoupledRope() {
  const compact = useCompact()
  // 窄版：两行的名称移到条的上方与下方，两个分数的说明放在两个点积之间
  const FW = compact ? 520 : 760, PAD = compact ? 0 : 8
  const CW = 96, CH = 42, G = 28, CX = (FW - 4 * 96 - 3 * 28) / 2, CY = 38
  const BX = compact ? 24 : 190, LW = compact ? 400 : 384, RW = compact ? 50 : 48, RX = BX + LW + 4, BH = 36, QY = compact ? 198 : 190, KY = compact ? 288 : 280
  const MY = (QY + BH + KY) / 2, LC = BX + LW / 2, RC = RX + RW / 2
  const sup = (base: string, s: string): ReactNode => <>{base}<tspan dy="-5" style={{ fontSize: '0.7em' }}>{s}</tspan></>
  const xs = [0, 1, 2, 3].map(n => CX + n * (CW + G))
  const dot = (x: number) => <g>
    <Link d={`M${x} ${QY + BH} V${MY - 12}`} />
    <Link d={`M${x} ${KY} V${MY + 12}`} />
    <Cell x={x - 22} y={MY - 12} w={44} h={24} rx={12} fill="ink" label="点积" small />
  </g>
  return (
    <Figure
      title="RoPE 与吸收的冲突，以及 decoupled RoPE"
      desc="上半部分：对还原出的 key 施加 RoPE 时，分数是 query、相对位置旋转 R、上投影矩阵 W 与 latent 四者的乘积，R 位于 query 与 W 之间。下半部分：decoupled RoPE 的 query 与缓存各分为 512 维与 64 维两段，分数是两段点积之和。"
      width={FW}
      height={compact ? 362 : 352}
      below={<Legend items={[['gray', 'query'], ['purple', 'latent'], ['red', 'RoPE']]} />}
      caption="上半部分是一个 query head 与一个历史 token 的分数中依次相乘的四项。下半部分两行的长度与维度成正比，上下对齐的两段分别做点积。"
    >{() => <>
      <Band x={PAD} y={4} w={FW - 2 * PAD} h={126} hue="gray" title="对还原出的 key 施加 RoPE" />
      <Cell x={xs[0]} y={CY} w={CW} h={CH} rx={6} fill="gray" label="query" sub="128" />
      <Cell x={xs[1]} y={CY} w={CW} h={CH} rx={6} fill="red" label={sub('R', 'j−t')} sub="随 t、j 变化" />
      <Cell x={xs[2]} y={CY} w={CW} h={CH} rx={6} fill="gray" label={sup('W', 'UK')} sub="128 × 512" />
      <Cell x={xs[3]} y={CY} w={CW} h={CH} rx={6} fill="purple" label="latent" sub="512 · 缓存" focus />
      {[0, 1, 2].map(n => <text key={n} x={xs[n] + CW + G / 2} y={CY + CH / 2} className="fk-strong">×</text>)}
      <text x={FW / 2} y={CY + CH + 28} className="fk-muted">R 随 token 对变化，位于 query 与 W 之间，W 不能预先乘到 query 上</text>

      <Band x={PAD} y={142} w={FW - 2 * PAD} h={compact ? 216 : 206} hue="purple" title="Decoupled RoPE：位置放到单独的 64 维" />
      {compact
        ? <>
            <text x={BX} y={QY - 12} className="fk-strong" textAnchor="start">head i 的 query<tspan className="fk-muted fk-small" dx="10" style={{ fontWeight: 400 }}>每步计算一次</tspan></text>
            <text x={BX} y={KY + BH + 16} className="fk-strong" textAnchor="start">token j 的缓存<tspan className="fk-muted fk-small" dx="10" style={{ fontWeight: 400 }}>576 维，所有 head 共享</tspan></text>
          </>
        : <>
            <text x={BX - 12} y={QY + 11} className="fk-strong" textAnchor="end">head i 的 query</text>
            <text x={BX - 12} y={QY + 28} className="fk-muted fk-small" textAnchor="end">每步计算一次</text>
            <text x={BX - 12} y={KY + 11} className="fk-strong" textAnchor="end">token j 的缓存</text>
            <text x={BX - 12} y={KY + 28} className="fk-muted fk-small" textAnchor="end">576 维，所有 head 共享</text>
          </>}
      <Cell x={BX} y={QY} w={LW} h={BH} fill="gray" label="已乘上投影矩阵的 query · 512 维" />
      <Cell x={RX} y={QY} w={RW} h={BH} fill="gray" label={sup('q', 'R')} />
      <text x={RC} y={QY - 10} className="fk-muted">64 维，带 RoPE</text>

      <Cell x={BX} y={KY} w={LW} h={BH} fill="purple" label="latent · 512 维，不带位置" focus />
      <Cell x={RX} y={KY} w={RW} h={BH} fill="red" label={sup('k', 'R')} />
      <text x={RC} y={KY + BH + 16} className="fk-muted">64 维，带 RoPE</text>

      {dot(LC)}{dot(RC)}
      {compact
        ? <>
            <text x={LC + 30} y={MY} className="fk-muted" textAnchor="start">内容分数，可吸收</text>
            <text x={RC - 92} y={MY} className="fk-strong">+</text>
            <text x={RC - 30} y={MY} className="fk-muted" textAnchor="end">位置分数</text>
          </>
        : <>
            <text x={LC - 30} y={MY} className="fk-muted" textAnchor="end">内容分数，可吸收</text>
            <text x={(LC + RC) / 2} y={MY} className="fk-strong">+</text>
            <text x={RC + 30} y={MY} className="fk-muted" textAnchor="start">位置分数</text>
          </>}
    </>}</Figure>
  )
}

// 图 4：真实模型每 token 的 KV cache（BF16，只计 attention 层的 KV；数据为官方 config.json 计算，2026-10-08 读取）
// DeepSeek-V4-Pro：61 层中 30 层每 4 个 token 一个 512 维 entry，31 层每 128 个 token 一个，平均到每个 token 为 7928 B。
type ModelRow = { name: string; kind: string; kib: number; fill: 'teal' | 'purple' | 'ink'; note?: string }
const MODELS: ModelRow[] = [
  { name: 'Llama-3.1-70B', kind: 'GQA · 80 层 × 2048', kib: 320, fill: 'teal' },
  { name: 'Qwen3-235B-A22B', kind: 'GQA · 94 层 × 1024', kib: 188, fill: 'teal' },
  { name: 'MiniMax-M3', kind: 'GQA · 60 层 × 1024', kib: 120, fill: 'teal' },
  { name: 'GLM-5', kind: 'MLA · 78 层 × 576', kib: 87.75, fill: 'purple' },
  { name: 'DeepSeek-V3', kind: 'MLA · 61 层 × 576', kib: 68.625, fill: 'purple' },
  { name: 'Kimi-K2', kind: 'MLA · 61 层 × 576', kib: 68.625, fill: 'purple' },
  { name: 'Kimi-K3', kind: 'MLA · 24 层 × 576', kib: 27, fill: 'purple', note: '另 69 层为线性 attention' },
  { name: 'DeepSeek-V4-Pro', kind: 'shared-KV MQA · 61 层 × 512', kib: 7928 / 1024, fill: 'ink', note: '每 4 或 128 个 token 一个 entry' },
]

export function KvCacheModels() {
  const compact = useCompact()
  // 窄版：模型名与配置写在条的上方
  const X0 = compact ? 0 : 210, PX = compact ? 1.0 : 1.2, BH = compact ? 24 : 26, PITCH = compact ? 58 : 40
  const rows = MODELS.length
  const fmt = (v: number) => (v >= 10 ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : v.toFixed(1))
  return (
    <Figure
      title="真实模型每 token 的 KV cache"
      desc={`BF16 下每 token 的 KV cache：${MODELS.map(m => `${m.name} ${fmt(m.kib)} KiB`).join('，')}。`}
      width={compact ? 520 : 760}
      height={rows * PITCH + (compact ? 0 : 10)}
      below={<Legend items={[['teal', 'GQA'], ['purple', 'MLA'], ['ink', 'shared-KV MQA']]} />}
      caption="由各模型官方 config.json 计算，统一按 BF16（每个元素 2 字节），不含 DSA indexer 的 key 与线性 attention 的固定大小状态。DeepSeek-V4-Pro 按压缩后的 entry 平均到每个 token，不含每层固定 128 个 token 的 sliding window；它实际以 FP8 与 BF16 混合存储，约为图中的 56%。右侧为一条 128K token 的请求所需的 KV cache。"
    >{() => <>
      {MODELS.map((m, i) => {
        const y = 4 + i * PITCH + (compact ? 20 : 0)
        return <g key={m.name}>
          {compact
            ? <text x={0} y={y - 12} className="fk-strong" textAnchor="start">{m.name}<tspan className="fk-muted fk-small" dx="10" style={{ fontWeight: 400 }}>{m.kind}</tspan></text>
            : <>
                <text x={X0 - 12} y={y + BH / 2 - 7} className="fk-strong" textAnchor="end">{m.name}</text>
                <text x={X0 - 12} y={y + BH / 2 + 9} className="fk-muted fk-small" textAnchor="end">{m.kind}</text>
              </>}
          <Cell x={X0} y={y} w={Math.max(m.kib * PX, 3)} h={BH} rx={3} fill={m.fill} />
          <text x={X0 + Math.max(m.kib * PX, 3) + 10} y={y + BH / 2} className="fk-stat" textAnchor="start">
            {fmt(m.kib)} KiB<tspan className="fk-muted" dx="10">128K：{fmt(m.kib / 8)} GiB{m.note ? `，${m.note}` : ''}</tspan>
          </text>
        </g>
      })}
    </>}</Figure>
  )
}
