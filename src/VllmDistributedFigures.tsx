import { Badge, Band, Cell, Figure, Legend, Link, useCompact } from './figure-kit'

// 语义色：权重与计算 blue，expert teal，KV cache green，重复的 KV red，
// 卡间与实例间通信 amber，hidden state 与 router gray，示例路径用 purple 箭头。

// 图 1：TP 下一层 Transformer 的切法与两次 all-reduce（两张卡、8 个 head 的小例子）
export function DistTpLayer() {
  const compact = useCompact()
  const gpus = [
    { y: 0, name: 'GPU 0', heads: 'head 0–3' },
    { y: 128, name: 'GPU 1', heads: 'head 4–7' },
  ]
  const title = 'TP 下一层 Transformer 的切法'
  const desc = '两张卡各持有每个权重矩阵的一半。attention 按 head 切，o_proj 之后做第一次 all-reduce；MLP 按中间维度切，down 之后做第二次 all-reduce。'
  const below = <Legend items={[['blue', '权重（每张卡一半）'], ['amber', 'all-reduce：两张卡的部分和逐元素相加（不是拼接）'], ['gray', 'hidden state（每张卡一份完整的）']]} />
  const caption = 'TP = 2、8 个 head 的普通 Transformer 层示例，MLP 的切法与一个 expert 相同。o_proj 与 down 只接本卡算出的那一半中间结果，输出是完整长度的部分和，all-reduce 相加后才是这一子层的输出。'
  // 紧凑布局：两张卡左右并排，数据自上而下流，两条 all-reduce 横跨两张卡
  if (compact) return (
    <Figure title={title} desc={desc} width={520} height={470} below={below} caption={caption}>{arrow => <>
      {gpus.map((g, i) => {
        const bx = i * 270, cx = bx + 125
        return <g key={g.name}>
          <Band x={bx} y={0} w={250} h={470} hue="gray" title={g.name} />
          <Cell x={bx + 85} y={34} w={80} h={32} rx={6} fill="gray" label="x" fs={14} />
          <Cell x={bx + 18} y={86} w={214} h={52} rx={6} fill="blue" label="QKV + attention" sub={g.heads} />
          <Cell x={bx + 18} y={158} w={214} h={52} rx={6} fill="blue" label="o_proj" sub="只接本卡的 head" />
          <Cell x={bx + 18} y={290} w={214} h={52} rx={6} fill="blue" label="up / gate" sub="算出中间结果的一半" />
          <Cell x={bx + 18} y={362} w={214} h={52} rx={6} fill="blue" label="down" sub="只接本卡那一半" />
          {[[66, 86], [138, 158], [210, 230], [266, 290], [342, 362], [414, 426]].map(([a, b]) =>
            <Link key={a} d={`M${cx} ${a} V${b}`} arrow={arrow} />)}
        </g>
      })}
      <Cell x={10} y={230} w={500} h={36} rx={6} fill="amber" label="all-reduce：attention 的输出相加" />
      <Cell x={10} y={426} w={500} h={36} rx={6} fill="amber" label="all-reduce：MLP 的输出相加" />
    </>}</Figure>
  )
  return (
    <Figure
      title="TP 下一层 Transformer 的切法"
      desc="两张卡各持有每个权重矩阵的一半。attention 按 head 切，o_proj 之后做第一次 all-reduce；MLP 按中间维度切，down 之后做第二次 all-reduce。"
      width={800}
      height={286}
      below={<Legend items={[['blue', '权重（每张卡一半）'], ['amber', 'all-reduce：两张卡的部分和逐元素相加（不是拼接）'], ['gray', 'hidden state（每张卡一份完整的）']]} />}
      caption="TP = 2、8 个 head 的普通 Transformer 层示例，MLP 的切法与一个 expert 相同。o_proj 与 down 只接本卡算出的那一半中间结果，输出是完整长度的部分和，all-reduce 相加后才是这一子层的输出。"
    >{arrow => <>
      {gpus.map(g => {
        const y = g.y + 44
        const mid = y + 26
        return <g key={g.name}>
          <Band x={0} y={g.y} w={800} h={112} hue="gray" title={g.name} />
          <Cell x={16} y={y} w={60} h={52} rx={6} fill="gray" label="x" fs={14} />
          <Cell x={100} y={y} w={134} h={52} rx={6} fill="blue" label="QKV + attention" sub={g.heads} />
          <Cell x={258} y={y} w={100} h={52} rx={6} fill="blue" label="o_proj" sub="只接本卡的 head" />
          <Cell x={470} y={y} w={110} h={52} rx={6} fill="blue" label="up / gate" sub="算出中间结果的一半" />
          <Cell x={600} y={y} w={90} h={52} rx={6} fill="blue" label="down" sub="只接本卡那一半" />
          {[[76, 100], [234, 258], [358, 378], [450, 470], [580, 600], [690, 710]].map(([a, b]) =>
            <Link key={a} d={`M${a} ${mid} H${b}`} arrow={arrow} />)}
        </g>
      })}
      <Cell x={378} y={44} w={72} h={180} rx={6} fill="amber" label="all-reduce" />
      <Cell x={710} y={44} w={72} h={180} rx={6} fill="amber" label="all-reduce" />
      <text x="229" y="262" className="fk-strong">attention：按 head 切</text>
      <text x="580" y="262" className="fk-strong">MLP：按中间维度切</text>
    </>}</Figure>
  )
}

// 图 2：只有一个 KV head 时，TP 与 DP attention 各存了什么
export function DistKvTpDp() {
  const compact = useCompact()
  const BW = 175
  const rows = [
    { y: 0, label: 'TP = 4：每张卡都要存全部请求的 KV', stat: '4 张卡装 4 个请求', reqs: (_: number) => ['A', 'B', 'C', 'D'], dup: (g: number) => g > 0 },
    { y: 128, label: 'DP = 4：每张卡只存自己那批请求的 KV', stat: '4 张卡装 16 个请求', reqs: (g: number) => 'ABCDEFGHIJKLMNOP'.slice(g * 4, g * 4 + 4).split(''), dup: () => false },
  ]
  // 紧凑布局：每张卡收窄，4 个请求排成 2 × 2
  if (compact) return (
    <Figure
      title="只有 1 个 KV head 时，KV cache 在 TP 与 DP attention 下的存放"
      desc="每张卡的 KV 空间能放 4 个请求。TP 下四张卡存的是同样的四个请求，后三张卡上都是重复的副本；DP 下每张卡存不同的四个请求，共 16 个。"
      width={520}
      height={352}
      below={<Legend items={[['green', '一个请求的 KV'], ['red', '重复的副本']]} />}
      caption="示意图：每张卡的 KV 空间按 4 个请求画。DeepSeek-V4 每个 token 在每层只存一份 KV，只有 1 个 KV head，TP 没法按 head 把它分到各卡。"
    >{() => <>
      {rows.map((r, k) => {
        const y0 = k * 186
        return <g key={r.label}>
          <text x="0" y={y0 + 12} className="fk-strong" textAnchor="start">{r.label}</text>
          <text x="0" y={y0 + 34} className="fk-stat" textAnchor="start">{r.stat}</text>
          {[0, 1, 2, 3].map(g => {
            const bx = g * 132.67
            return <g key={g}>
              <Band x={bx} y={y0 + 48} w={122} h={118} hue="gray" title={`GPU ${g}`} />
              {r.reqs(g).map((id, i) =>
                <Cell key={i} x={bx + 11 + (i % 2) * 54} y={y0 + 80 + Math.floor(i / 2) * 40} w={46} h={34} rx={4} fill={r.dup(g) ? 'red' : 'green'} label={id} fs={14} />)}
            </g>
          })}
        </g>
      })}
    </>}</Figure>
  )
  return (
    <Figure
      title="只有 1 个 KV head 时，KV cache 在 TP 与 DP attention 下的存放"
      desc="每张卡的 KV 空间能放 4 个请求。TP 下四张卡存的是同样的四个请求，后三张卡上都是重复的副本；DP 下每张卡存不同的四个请求，共 16 个。"
      height={236}
      below={<Legend items={[['green', '一个请求的 KV'], ['red', '重复的副本']]} />}
      caption="示意图：每张卡的 KV 空间按 4 个请求画。DeepSeek-V4 每个 token 在每层只存一份 KV，只有 1 个 KV head，TP 没法按 head 把它分到各卡。"
    >{() => <>
      {rows.map(r => <g key={r.label}>
        <text x="0" y={r.y + 12} className="fk-strong" textAnchor="start">{r.label}</text>
        <text x="760" y={r.y + 12} className="fk-stat" textAnchor="end">{r.stat}</text>
        {[0, 1, 2, 3].map(g => {
          const bx = g * 195
          return <g key={g}>
            <Band x={bx} y={r.y + 26} w={BW} h={82} hue="gray" title={`GPU ${g}`} />
            {r.reqs(g).map((id, i) =>
              <Cell key={i} x={bx + 12 + i * 39} y={r.y + 60} w={34} h={34} rx={4} fill={r.dup(g) ? 'red' : 'green'} label={id} />)}
          </g>
        })}
      </g>)}
    </>}</Figure>
  )
}

// 图 3：DP attention + EP 下，一个 token 穿过一层 MoE
export function DistEpPath() {
  const compact = useCompact()
  // 紧凑布局只收窄各列，纵向结构不变
  const P = compact ? 132.67 : 195, BW = compact ? 122 : 175, CW = BW - 20, EW = compact ? 48 : 72
  const FW = compact ? 520 : 760, MID = BW / 2
  const bx = (g: number) => g * P
  const ex = (e: number) => bx(Math.floor(e / 2)) + (e % 2 ? BW - 10 - EW : 10) // expert 格子的左边
  const ec = (e: number) => ex(e) + EW / 2 // expert 格子的中线
  const picked = [3, 6]
  return (
    <Figure
      title="DP attention 加 EP：一个 token 穿过一层 MoE"
      desc="四张卡各有完整的 attention 权重和两个 expert。GPU 0 上的一个 token 被 router 分给 E3 与 E6，hidden state 经 dispatch 发到 GPU 1 和 GPU 3，算完经 combine 回到 GPU 0 加权求和。"
      width={FW}
      height={368}
      below={<Legend items={[['blue', 'attention（每张卡一份完整权重）'], ['teal', 'expert（每个只在一张卡上）'], ['amber', '卡间通信']]} />}
      caption="4 张卡、8 个 expert、每个 token 选 2 个的示例。紫色是 GPU 0 上一个 token 的路径：1 router 选出 E3 与 E6，2 dispatch 把 hidden state 发到它们所在的卡，3 expert 计算，4 combine 把结果送回 GPU 0。"
    >{arrow => <>
      {[0, 1, 2, 3].map(g => <Band key={g} x={bx(g)} y={0} w={BW} h={366} hue="gray" title={compact ? `GPU ${g}` : `GPU ${g} · DP rank ${g}`} />)}
      <rect x="6" y="146" width={FW - 12} height="28" rx="6" className="fk-fill-amber" />
      <rect x="6" y="268" width={FW - 12} height="28" rx="6" className="fk-fill-amber" />
      <text x={FW - 16} y="160" className="fk-cell-text fk-on-amber" textAnchor="end">dispatch</text>
      <text x={FW - 16} y="282" className="fk-cell-text fk-on-amber" textAnchor="end">combine</text>
      {[0, 1, 2, 3].map(g => <g key={g}>
        <Cell x={bx(g) + 10} y={34} w={CW} h={48} rx={6} fill="blue" label="attention" sub="本卡请求的 KV" />
        <Link d={`M${bx(g) + MID} 82 V96`} arrow={arrow} />
        <Cell x={bx(g) + 10} y={96} w={CW} h={28} rx={6} fill="gray" label="router" />
        <Cell x={bx(g) + 10} y={318} w={CW} h={36} rx={6} fill="gray" label="加权求和" />
      </g>)}
      {Array.from({ length: 8 }, (_, e) =>
        <Cell key={e} x={ex(e)} y={198} w={EW} h={48} rx={6} fill="teal" label={`E${e}`} fs={14} focus={picked.includes(e)} />)}
      <Link d={`M${MID} 124 V160 H${ec(6)} V196`} hue="purple" arrow={arrow} focus />
      <Link d={`M${ec(3)} 160 V196`} hue="purple" arrow={arrow} focus />
      <Link d={`M${ec(3)} 248 V282`} hue="purple" focus />
      <Link d={`M${ec(6)} 248 V282 H${MID} V318`} hue="purple" arrow={arrow} focus />
      {[[MID - 21, 135], [bx(2) + MID + (compact ? 0 : 0.5), 134], [ec(3) + (compact ? -22 : 22), 186], [bx(2) + MID + (compact ? 0 : 0.5), 308]].map(([x, y], i) => <Badge key={i} x={x} y={y} n={i + 1} />)}
    </>}</Figure>
  )
}

// 图 4：PD 分离下一个请求的路径
export function DistPdFlow() {
  const compact = useCompact()
  const kv = (y: number, x0 = 474) => Array.from({ length: 6 }, (_, i) =>
    <Cell key={i} x={x0 + i * 36} y={y} w={30} h={30} rx={4} fill="green" />)
  // 紧凑布局：router、P、D 自上而下三条带，路径沿左右两侧绕行
  if (compact) return (
    <Figure
      title="PD 分离下一个请求的路径"
      desc="router 先把请求发给 prefill 实例算出 prompt 的 KV，再把 decode 请求连同 KV 的位置发给 decode 实例；decode 实例把 KV 读到自己的显存后逐 token 生成，流式返回。"
      width={520}
      height={444}
      below={<Legend items={[['blue', 'GPU 计算'], ['green', 'prompt 的 KV block'], ['amber', '实例间通信']]} />}
      caption="紫色是一个请求的路径：1 prefill 请求（只要 1 个输出 token），2 P 算出 prompt 的 KV，3 decode 请求带上 KV 的位置，4 D 把 KV 读进自己的 KV cache，5 D 接着生成，6 流式返回。"
    >{arrow => <>
      <Band x={0} y={0} w={520} h={92} hue="teal" title="router" />
      <Band x={0} y={112} w={520} h={128} hue="blue" title="Prefill 实例（P）" />
      <Band x={0} y={304} w={520} h={140} hue="purple" title="Decode 实例（D）" />

      <Cell x={160} y={28} w={200} h={52} rx={8} fill="teal" label="一个请求" sub="拆成两段转发" fs={13} />
      <Cell x={24} y={150} w={200} h={60} rx={8} fill="blue" label="prefill" sub="一次算完整个 prompt" fs={14} />
      {kv(165, 286)}
      <text x="391" y="150" className="fk-muted fk-small">prompt 的 KV block</text>
      <Cell x={281} y={254} w={220} h={36} rx={6} fill="amber" label="KV 传输（NIXL · RDMA）" />
      {kv(356, 286)}
      <text x="391" y="407" className="fk-muted fk-small">写进 D 的 KV cache</text>
      <Cell x={24} y={342} w={200} h={60} rx={8} fill="blue" label="decode" sub="逐 token 生成" fs={14} />

      <Link d="M190 80 V150" hue="purple" arrow={arrow} focus />
      <Link d="M224 180 H286" hue="purple" arrow={arrow} focus />
      <Link d="M160 54 H6 V372 H24" hue="purple" arrow={arrow} focus />
      <Link d="M391 195 V254" hue="purple" arrow={arrow} focus />
      <Link d="M391 290 V356" hue="purple" arrow={arrow} focus />
      <Link d="M286 371 H224" hue="purple" arrow={arrow} focus />
      <Link d="M124 402 V430 H514 V54 H360" hue="purple" arrow={arrow} focus />
      {[[208, 102], [255, 164], [24, 272], [409, 226], [255, 355], [496, 318]].map(([x, y], i) => <Badge key={i} x={x} y={y} n={i + 1} />)}
    </>}</Figure>
  )
  return (
    <Figure
      title="PD 分离下一个请求的路径"
      desc="router 先把请求发给 prefill 实例算出 prompt 的 KV，再把 decode 请求连同 KV 的位置发给 decode 实例；decode 实例把 KV 读到自己的显存后逐 token 生成，流式返回。"
      height={312}
      below={<Legend items={[['blue', 'GPU 计算'], ['green', 'prompt 的 KV block'], ['amber', '实例间通信']]} />}
      caption="紫色是一个请求的路径：1 prefill 请求（只要 1 个输出 token），2 P 算出 prompt 的 KV，3 decode 请求带上 KV 的位置，4 D 把 KV 读进自己的 KV cache，5 D 接着生成，6 流式返回。"
    >{arrow => <>
      <Band x={0} y={0} w={160} h={310} hue="teal" title="router" />
      <Band x={180} y={0} w={580} h={120} hue="blue" title="Prefill 实例（P）" />
      <Band x={180} y={180} w={580} h={130} hue="purple" title="Decode 实例（D）" />

      <Cell x={16} y={100} w={128} h={100} rx={8} fill="teal" label="一个请求" sub="拆成两段转发" fs={13} />
      <Cell x={200} y={47} w={220} h={60} rx={8} fill="blue" label="prefill" sub="一次算完整个 prompt" fs={14} />
      {kv(62)}
      <text x="579" y="44" className="fk-muted fk-small">prompt 的 KV block</text>
      <Cell x={469} y={132} w={220} h={36} rx={6} fill="amber" label="KV 传输（NIXL · RDMA）" />
      {kv(227)}
      <text x="579" y="274" className="fk-muted fk-small">写进 D 的 KV cache</text>
      <Cell x={200} y={214} w={220} h={56} rx={8} fill="blue" label="decode" sub="逐 token 生成" fs={14} />

      <Link d="M144 120 H170 V77 H200" hue="purple" arrow={arrow} focus />
      <Link d="M420 77 H474" hue="purple" arrow={arrow} focus />
      <Link d="M144 180 H170 V242 H200" hue="purple" arrow={arrow} focus />
      <Link d="M579 96 V132" hue="purple" arrow={arrow} focus />
      <Link d="M579 168 V227" hue="purple" arrow={arrow} focus />
      <Link d="M474 242 H420" hue="purple" arrow={arrow} focus />
      <Link d="M310 270 V292 H80 V200" hue="purple" arrow={arrow} focus />
      {[[156, 96], [447, 61], [156, 222], [597, 114], [447, 226], [120, 276]].map(([x, y], i) => <Badge key={i} x={x} y={y} n={i + 1} />)}
    </>}</Figure>
  )
}

// 图 5：DeepSeek-V4-Pro 的结构，上面是整条路径，下面拉开其中一层
export function DistV4Anatomy() {
  const compact = useCompact()
  const picked = [70, 133, 201, 260, 318, 371]
  const gx = 330, gy = 176
  // 紧凑布局：attention 在上、MoE 在下，expert 格子放大
  if (compact) return (
    <Figure
      title="DeepSeek-V4-Pro 的结构"
      desc="一个 token 依次穿过 embedding、61 个结构相同的 Transformer 层和输出层。每层先做 attention，读前面 token 的 KV 并写下自己的一份；再进 MoE，router 从 384 个 expert 里选 6 个，加上 shared expert，结果加权求和。"
      width={520}
      height={650}
      below={<Legend items={[['blue', 'attention 等'], ['teal', 'expert'], ['purple', '这个 token 选中的 6 个'], ['green', 'KV cache']]} />}
      caption="上：一个 token 的完整路径。下：拉开其中一层。每层的 attention 约 0.3 GB，384 个 expert 共 13.5 GB，权重几乎都在 expert 里。"
    >{arrow => <>
      <text x="308" y="12" className="fk-strong" textAnchor="middle">61 个 Transformer 层，结构都一样</text>
      <Cell x={0} y={32} w={48} h={44} rx={6} fill="gray" label="token" />
      <Cell x={62} y={24} w={84} h={60} rx={6} fill="blue" label="embedding" />
      {Array.from({ length: 61 }, (_, k) => <rect key={k} x={160 + k * 5} y={24} width={3.6} height={60} rx={1} className={k === 30 ? 'fk-fill-purple' : 'fk-fill-teal'} />)}
      <Cell x={476} y={24} w={44} h={60} rx={6} fill="gray" label="输出" />
      <Link d="M48 54 H62" arrow={arrow} />
      <Link d="M146 54 H160" arrow={arrow} />
      <Link d="M464 54 H476" arrow={arrow} />
      <Link d="M310 84 L2 122" />
      <Link d="M314 84 L518 122" />

      <Band x={0} y={122} w={520} h={478} hue="gray" title="其中一层" />
      <Cell x={36} y={158} w={200} h={64} rx={8} fill="blue" label="attention" sub="读前面 token 的 KV" fs={14} />
      <Link d="M236 184 H270" arrow={arrow} />
      {Array.from({ length: 6 }, (_, i) => <Cell key={i} x={270 + i * 26} y={172} w={22} h={24} rx={3} fill="green" />)}
      <text x="270" y="212" className="fk-muted fk-small" textAnchor="start">每个 token 写下一份 KV</text>
      <Link d="M136 222 V274" arrow={arrow} />

      <Band x={12} y={244} w={496} h={344} hue="teal" title="MoE" />
      <Cell x={81} y={274} w={110} h={46} rx={6} fill="gray" label="router" sub="选 6 个" />
      <Link d="M136 320 V338" arrow={arrow} />
      {Array.from({ length: 384 }, (_, k) => <rect key={k} x={36 + (k % 32) * 14} y={338 + Math.floor(k / 32) * 14} width={12} height={12} rx={2} className={picked.includes(k) ? 'fk-fill-purple' : 'fk-fill-teal'} />)}
      <Cell x={36} y={510} w={446} h={20} rx={3} fill="teal" label="shared expert：每个 token 都用" small />
      <Link d="M259 530 V544" arrow={arrow} />
      <Cell x={236} y={544} w={46} h={36} rx={6} fill="gray" label="Σ" fs={16} />
      <text x="0" y="616" className="fk-strong" textAnchor="start">attention：每层约 0.3 GB</text>
      <text x="0" y="638" className="fk-strong fk-text-teal" textAnchor="start">expert：每个约 35 MB，每层 384 个共 13.5 GB</text>
    </>}</Figure>
  )
  return (
    <Figure
      title="DeepSeek-V4-Pro 的结构"
      desc="一个 token 依次穿过 embedding、61 个结构相同的 Transformer 层和输出层。每层先做 attention，读前面 token 的 KV 并写下自己的一份；再进 MoE，router 从 384 个 expert 里选 6 个，加上 shared expert，结果加权求和。"
      width={800}
      height={388}
      below={<Legend items={[['blue', 'attention 等'], ['teal', 'expert'], ['purple', '这个 token 选中的 6 个'], ['green', 'KV cache']]} />}
      caption="上：一个 token 的完整路径。下：拉开其中一层。每层的 attention 约 0.3 GB，384 个 expert 共 13.5 GB，权重几乎都在 expert 里。"
    >{arrow => <>
      <text x="456" y="12" className="fk-strong" textAnchor="middle">61 个 Transformer 层，结构都一样</text>
      <Cell x={0} y={32} w={64} h={44} rx={6} fill="gray" label="token" />
      <Cell x={88} y={24} w={104} h={60} rx={6} fill="blue" label="embedding" />
      {Array.from({ length: 61 }, (_, k) => <rect key={k} x={212 + k * 8} y={24} width={6} height={60} rx={1} className={k === 30 ? 'fk-fill-purple' : 'fk-fill-teal'} />)}
      <Cell x={712} y={24} w={88} h={60} rx={6} fill="gray" label="输出" />
      <Link d="M64 54 H88" arrow={arrow} />
      <Link d="M192 54 H212" arrow={arrow} />
      <Link d="M698 54 H712" arrow={arrow} />
      <Link d="M452 84 L2 122" />
      <Link d="M458 84 L798 122" />

      <Band x={0} y={122} w={800} h={236} hue="gray" title="其中一层" />
      <Cell x={20} y={176} w={150} h={84} rx={8} fill="blue" label="attention" sub="读前面 token 的 KV" fs={14} />
      <Link d="M95 260 V298" arrow={arrow} />
      {Array.from({ length: 6 }, (_, i) => <Cell key={i} x={20 + i * 25} y={298} w={21} h={24} rx={3} fill="green" />)}
      <text x="20" y="342" className="fk-muted fk-small" textAnchor="start">每个 token 写下一份 KV</text>
      <Link d="M170 218 H196" arrow={arrow} />

      <Band x={196} y={150} w={584} h={196} hue="teal" title="MoE" />
      <Cell x={212} y={214} w={92} h={52} rx={6} fill="gray" label="router" sub="选 6 个" />
      <Link d="M304 240 H330" arrow={arrow} />
      {Array.from({ length: 384 }, (_, k) => <rect key={k} x={gx + (k % 32) * 12} y={gy + Math.floor(k / 32) * 12} width={10} height={10} rx={1.5} className={picked.includes(k) ? 'fk-fill-purple' : 'fk-fill-teal'} />)}
      <Cell x={gx} y={324} w={382} h={18} rx={3} fill="teal" label="shared expert：每个 token 都用" small />
      <Link d="M712 240 H730" arrow={arrow} />
      <Cell x={730} y={218} w={40} h={44} rx={6} fill="gray" label="Σ" fs={16} />
      <text x="20" y="378" className="fk-strong" textAnchor="start">attention：每层约 0.3 GB</text>
      <text x="196" y="378" className="fk-strong fk-text-teal" textAnchor="start">expert：每个约 35 MB，每层 384 个共 13.5 GB</text>
    </>}</Figure>
  )
}

// 图 6：两种切法下，一张 180 GB 的卡里装了什么（高度按显存画）
export function DistLedger() {
  const compact = useCompact()
  const S = 1.4, TOP = 34, W = compact ? 70 : 130
  type Seg = { gb: number; fill: 'teal' | 'blue' | 'green' | 'red'; note: string; sub: string }
  const cards: { x: number; title: string; stat: string; segs: Seg[] }[] = [
    { x: compact ? 0 : 30, title: 'TP = 8：一张卡', stat: '8 张卡不重复的 KV：72 GB', segs: [
      { gb: 104, fill: 'teal', note: '每个 expert 的 ⅛', sub: '836 ÷ 8 = 104 GB' },
      { gb: 4, fill: 'blue', note: 'attention 等的 ⅛', sub: '29 ÷ 8 ≈ 4 GB' },
      { gb: 72, fill: 'red', note: 'KV cache', sub: '8 张卡存的是同一份' }] },
    { x: compact ? 268 : 430, title: 'DP attention + EP：一张卡', stat: '8 张卡不重复的 KV：8 × 46.5 = 372 GB', segs: [
      { gb: 104, fill: 'teal', note: '每层 48 个完整的 expert', sub: '384 ÷ 8 = 48 个，共 104 GB' },
      { gb: 29, fill: 'blue', note: 'attention 等，完整一份', sub: '29 GB，比 TP 多占 25 GB' },
      { gb: 47, fill: 'green', note: 'KV cache', sub: '只放自己这批请求的' }] },
  ]
  return (
    <Figure
      title="两种切法下一张卡的显存"
      desc="一张 180 GB 的卡。TP 下是 104 GB 的 expert 切片、4 GB 的 attention 切片和 72 GB 的 KV，但 8 张卡的 KV 是同一份。DP attention 加 EP 下是 104 GB 的完整 expert、29 GB 的完整 attention 和 47 GB 只属于本卡请求的 KV。"
      width={compact ? 520 : 800}
      height={TOP + 180 * S + 40}
      below={<Legend items={[['teal', 'expert'], ['blue', 'attention 等'], ['green', '本卡请求的 KV'], ['red', '与其他卡重复的 KV']]} />}
      caption="高度按显存画，一张卡 180 GB，各部分按权重文件的大小估算。DP attention + EP 每张卡留给 KV 的显存更少，但 8 张卡存的内容不重复。"
    >{() => <>
      {cards.map(c => {
        let y = TOP
        return <g key={c.title}>
          <text x={c.x} y={16} className="fk-strong" textAnchor="start">{c.title}</text>
          {c.segs.map(s => {
            const h = s.gb * S, y0 = y; y += h
            return <g key={s.note}>
              <Cell x={c.x} y={y0} w={W} h={h - 1.5} rx={2} fill={s.fill} label={h > 30 ? String(s.gb) : undefined} fs={16} />
              <text x={c.x + W + (compact ? 10 : 14)} y={y0 + h / 2 - 2} className="fk-strong" textAnchor="start">{s.note}</text>
              <text x={c.x + W + (compact ? 10 : 14)} y={y0 + h / 2 + 15} className="fk-muted fk-small" textAnchor="start">{s.sub}</text>
            </g>
          })}
          <text x={c.x} y={TOP + 180 * S + 26} className="fk-stat" textAnchor="start">{c.stat}</text>
        </g>
      })}
    </>}</Figure>
  )
}
