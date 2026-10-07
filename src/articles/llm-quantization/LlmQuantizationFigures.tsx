import { Badge, Cell, Figure, Legend, Link, Steps, TexLabel } from '../../figure-kit'

// 语义色：权重 blue，activation teal，示例路径（正在量化的列、被追踪的 4-bit 权重）purple，
// 尚未量化的权重 gray，KV cache green（BF16）与 amber（FP8）。

// 图 0：同一段数值（−2 … 2）在 16 bit、8 bit、4 bit 下各有多少个刻度，以及权重 0.37 落到哪一个
export function QuantRulers() {
  const X0 = 110, UNIT = 150, H = 40 // 150 px / 单位，三把尺子各宽 600
  const x = (v: number) => X0 + (v + 2) * UNIT
  const W = 0.37
  const rows = [
    { bits: 16, n: 65536, y: 44, err: '误差小于 0.0001' },
    { bits: 8, n: 256, y: 124, err: '误差 0.005' },
    { bits: 4, n: 16, y: 204, err: '误差 0.12' },
  ]
  return (
    <Figure
      title="同一段数值在 16 bit、8 bit、4 bit 下的刻度"
      desc="三组刻度覆盖同一区间 −2 到 2。16 bit 有 65536 个刻度，在图中连成一片；8 bit 有 256 个；4 bit 只有 16 个，scale 为 0.25。权重 0.37 在 16 bit 与 8 bit 下误差不超过 0.005，在 4 bit 下舍入为 0.25，误差为 0.12。"
      height={278}
      caption="示意数值。虚线为权重 0.37 的原值，紫色为舍入后的刻度。"
    >{() => <>
      {rows.map(r => {
        const step = 4 / r.n, q = Math.round(W / step) * step
        return <g key={r.bits}>
          <text x={X0 - 14} y={r.y + H / 2 - 8} className="fk-strong" textAnchor="end">{r.bits} bit</text>
          <text x={X0 - 14} y={r.y + H / 2 + 10} className="fk-muted fk-small" textAnchor="end">{r.n.toLocaleString('en-US')} 个刻度</text>
          {r.n > 600
            ? <rect x={X0} y={r.y} width={4 * UNIT} height={H} className="fk-fill-green" />
            : Array.from({ length: r.n }, (_, k) => <rect key={k} x={x(-2 + k * step) - (r.n > 16 ? 0.5 : 2)} y={r.y} width={r.n > 16 ? 1 : 4} height={H} className="fk-fill-green" />)}
          <rect x={x(q) - 2} y={r.y - 4} width="4" height={H + 8} className="fk-fill-purple" />
          <text x={X0 + 4 * UNIT + 14} y={r.y + H / 2} className={r.n === 16 ? 'fk-strong' : 'fk-muted'} textAnchor="start">{r.err}</text>
        </g>
      })}
      <line x1={x(W)} y1="24" x2={x(W)} y2="252" className="fk-line-purple" strokeWidth="1.5" strokeDasharray="4 3" />
      <text x={x(W) + 8} y="16" className="fk-strong fk-text-purple" textAnchor="start">w = 0.37</text>
      <rect x={x(0.25) + 2} y="254" width={x(W) - x(0.25) - 2} height="4" className="fk-fill-red" />
      {[-2, -1, 0, 1, 2].map(v => <text key={v} x={x(v)} y="270" className="fk-muted fk-small">{String(v).replace('-', '−')}</text>)}
    </>}</Figure>
  )
}

// 图 0a：4 bit 的 16 个编号乘以 s 变成刻度；s 由最大值定，换一个 s 范围就变
export function QuantCodesToTicks() {
  const X0 = 110, P = 40 // 每个编号占 40 px；s = 0.25 时 1 个单位 = 160 px
  const c = (q: number) => X0 + (q + 8) * P + P / 2
  const v = (w: number) => c(0) + w * 160
  const codes = Array.from({ length: 16 }, (_, k) => k - 8)
  const bin = (q: number) => (q & 15).toString(2).padStart(4, '0')
  const fmt = (x: number) => String(+x.toFixed(2)).replace('-', '−')
  const W = 0.37, RY = 112, RH = 30, SY = 214
  return (
    <Figure
      title="4 bit 编号乘以 scale 变成刻度"
      desc="上排是 4 bit 能存的 16 个编号 −8 到 7。每个编号乘以 s 得到下方对应的刻度：s = 1.75 / 7 = 0.25 时刻度从 −2 到 1.75，绝对值最大的数 1.75 对应编号 7。权重 0.37 除以 0.25 得 1.48，取整为编号 1，存储为 0001，还原为 0.25。最下排是同样 16 个编号在 s = 0.1 时的刻度，范围为 −0.8 到 0.7。"
      height={262}
      below={<>
        <Steps items={[
          ['1 · 除以 s，取最近的整数', '0.37 / 0.25 = 1.48 \\to 1'],
          ['2 · 存储编号', '1 \\to \\texttt{0001}'],
          ['3 · 还原：编号乘以 s', '1 \\times 0.25 = 0.25'],
          ['误差', '0.37 - 0.25 = 0.12'],
        ]} />
        <Legend items={[['blue', '编号（存储在显存中）'], ['green', '刻度（编号 × s）'], ['purple', '示例：0.37 的量化与还原']]} />
      </>}
      caption="示意数值，设待量化的数绝对值最大为 1.75。"
    >{arrow => <>
      <text x={X0 - 14} y="40" className="fk-strong" textAnchor="end">编号</text>
      <text x={X0 - 14} y="58" className="fk-muted fk-small" textAnchor="end">4 bit</text>
      {codes.map(q => <Cell key={q} x={c(q) - 19} y={26} w={38} h={44} fill="blue" label={fmt(q)} sub={bin(q)} focus={q === 1} />)}

      <text x={X0 - 14} y="94" className="fk-muted" textAnchor="end">× s ↓</text>

      <text x={X0 - 14} y={RY + 10} className="fk-strong" textAnchor="end">s = 0.25</text>
      <text x={X0 - 14} y={RY + 28} className="fk-muted fk-small" textAnchor="end">= 1.75 / 7</text>
      <line x1={c(-8)} y1={RY + RH} x2={c(7)} y2={RY + RH} className="fk-line-muted" />
      {codes.map(q => <g key={q}>
        <rect x={c(q) - 2} y={RY} width="4" height={RH} className="fk-fill-green" />
        <text x={c(q)} y={RY + RH + 16} className={q === 1 || q === 7 ? 'fk-strong fk-small' : 'fk-muted fk-small'}>{fmt(q * 0.25)}</text>
      </g>)}
      <text x={c(7)} y={RY + RH + 34} className="fk-muted fk-small" textAnchor="end">max |w| 对应编号 7</text>

      <line x1={v(W)} y1={RY - 6} x2={v(W)} y2={RY + RH + 4} className="fk-line-purple" strokeWidth="1.5" strokeDasharray="4 3" />
      <rect x={c(1)} y={RY + RH + 2} width={v(W) - c(1)} height="4" className="fk-fill-red" />
      <text x={v(W) + 6} y={RY + RH + 34} className="fk-strong fk-text-purple" textAnchor="start">w = 0.37</text>
      <Link d={`M${v(W)} ${RY - 6} V90 H${c(1) + 10} V70`} hue="purple" arrow={arrow} focus />
      <Link d={`M${c(1) - 8} 72 V${RY}`} hue="purple" arrow={arrow} focus />
      <Badge x={v(W) + 16} y={96} n={1} />
      <Badge x={c(1) + 34} y={14} n={2} />
      <Badge x={c(1) - 24} y={96} n={3} />

      <text x={X0 - 14} y={SY + 10} className="fk-strong" textAnchor="end">s = 0.1</text>
      <text x={X0 - 14} y={SY + 28} className="fk-muted fk-small" textAnchor="end">同样 16 个编号</text>
      {codes.map(q => <rect key={q} x={c(0) + q * 16 - 1.5} y={SY} width="3" height={24} className="fk-fill-green" />)}
      {[-8, 0, 7].map(q => <text key={q} x={c(0) + q * 16} y={SY + 40} className="fk-muted fk-small">{fmt(q * 0.1)}</text>)}
    </>}</Figure>
  )
}

// 图 1：GPTQ 量化一列时，误差分摊给同一行中尚未量化的权重
export function QuantGptqColumns() {
  const ROWS = 6, COLS = 12, CUR = 5, FOCUS_ROW = 2
  const X0 = 150, Y0 = 56, PX = 40, PY = 32
  const SX = 150, SY = 300 // 放大的一行：正在量化的格子加宽，写得下舍入前后的值
  const stripX = (i: number) => SX + i * 56 + (i > 3 ? 48 : 0)
  const strip: { fill: 'blue' | 'purple' | 'gray'; label: string; sub: string }[] = [
    { fill: 'blue', label: '0.50', sub: '已定' },
    { fill: 'blue', label: '−0.25', sub: '已定' },
    { fill: 'blue', label: '0.25', sub: '已定' },
    { fill: 'purple', label: '0.37 → 0.25', sub: '误差 0.12' },
    { fill: 'gray', label: '+0.05', sub: '调整' },
    { fill: 'gray', label: '−0.02', sub: '调整' },
    { fill: 'gray', label: '+0.03', sub: '调整' },
    { fill: 'gray', label: '+0.01', sub: '调整' },
  ]
  const curX = stripX(3) + 50
  return (
    <Figure
      title="GPTQ 逐列量化与误差补偿"
      desc="权重矩阵从左到右逐列量化。左侧五列已量化不再改动，第六列正在量化，右侧六列尚未量化。正在量化的权重产生的舍入误差，分摊给同一行里右侧尚未量化的权重。"
      height={424}
      below={<Legend items={[['blue', '已量化'], ['purple', '正在量化'], ['gray', '尚未量化（仍是 FP16）']]} />}
      caption="示意数值。每量化一列，同一行右侧尚未量化的权重做相应调整，使这一层的输出误差最小。"
    >{arrow => <>
      <text x="0" y="12" className="fk-strong" textAnchor="start">一层的权重矩阵（每行一个输出 channel，每列一个输入 channel）</text>
      <text x={X0 + 2.5 * PX - 2} y="40" className="fk-muted">已量化，不再改动</text>
      <text x={X0 + CUR * PX + 18} y="40" className="fk-strong fk-text-purple">本列</text>
      <text x={X0 + 9 * PX - 2} y="40" className="fk-muted">尚未量化</text>
      <text x={X0 + COLS * PX + 10} y="40" className="fk-muted" textAnchor="start">量化顺序 →</text>
      {Array.from({ length: ROWS * COLS }, (_, i) => {
        const r = Math.floor(i / COLS), c = i % COLS
        return <Cell key={i} x={X0 + c * PX} y={Y0 + r * PY} w={36} h={28} fill={c < CUR ? 'blue' : c === CUR ? 'purple' : 'gray'} />
      })}
      <rect x={X0 - 4} y={Y0 + FOCUS_ROW * PY - 4} width={COLS * PX + 4} height={36} rx="5" className="fk-focus" />
      <text x={X0 - 14} y={Y0 + FOCUS_ROW * PY + 14} className="fk-muted" textAnchor="end">放大这一行 ↓</text>

      {strip.map((s, i) => <Cell key={i} x={stripX(i)} y={SY} w={i === 3 ? 100 : 52} h={44} rx={5} fill={s.fill} label={s.label} sub={s.sub} focus={i === 3} />)}
      {[4, 5, 6, 7].map(i => <Link key={i} d={`M${curX} ${SY + 46} V${SY + 64} H${stripX(i) + 26} V${SY + 46}`} hue="purple" arrow={arrow} focus={i === 7} />)}
      <text x={SX - 14} y={SY + 22} className="fk-muted" textAnchor="end">这一行（节选）</text>
      <TexLabel x={398} y={SY + 102} w={520} h={44} source={'\\delta_{j+1:} = -\\,\\dfrac{w_j - \\hat w_j}{[H^{-1}]_{jj}}\\; H^{-1}_{j,\\,j+1:}'} />
    </>}</Figure>
  )
}

// 图 2：SmoothQuant 的 per-channel 缩放（α = 0.5）
export function QuantSmoothChannels() {
  const UNIT = 10 // px / 单位幅度
  const BASE = 236
  const panels = [
    { x: 0, title: '变换前', act: [16, 1, 4, 1], w: [1, 1, 1, 1], note: 'c1 是 outlier：scale 被它撑大，其余 channel 只用到很少的刻度' },
    { x: 400, title: '变换后', act: [4, 1, 2, 1], w: [4, 1, 2, 1], note: 'activation 与权重的最大值相等，各自的量化范围都被用满' },
  ]
  const bars = (x: number, vals: number[], hue: 'teal' | 'blue') => {
    const top = BASE - Math.max(...vals) * UNIT
    return <g>
      {vals.map((v, i) => <g key={i}>
        <rect x={x + i * 38} y={BASE - v * UNIT} width="30" height={v * UNIT} rx="3" className={`fk-fill-${hue}`} />
        <text x={x + i * 38 + 15} y={BASE - v * UNIT - 10} className="fk-stat">{v}</text>
        <text x={x + i * 38 + 15} y={BASE + 14} className="fk-muted fk-small">c{i + 1}</text>
      </g>)}
      <line x1={x - 6} y1={top} x2={x + 150} y2={top} className="fk-line-muted" strokeWidth="1.2" strokeDasharray="4 3" />
    </g>
  }
  return (
    <Figure
      title="SmoothQuant 的 per-channel 缩放"
      desc="变换前 activation 四个 channel 的最大值为 16、1、4、1，权重均为 1。按 s 等于 4、1、2、1 缩放后，activation 变为 4、1、2、1，权重变为 4、1、2、1。"
      height={316}
      below={<Steps items={[
        ['c1（outlier）', 's_1 = \\sqrt{16 / 1} = 4'],
        ['c3', 's_3 = \\sqrt{4 / 1} = 2'],
        ['c2、c4', 's = \\sqrt{1 / 1} = 1'],
      ]} />}
      caption="示意数值，α = 0.5。柱高是每个 channel 的最大绝对值，虚线是整个 tensor 共用的量化范围。乘积 X·W 在变换前后不变。"
    >{() => <>
      {panels.map(p => <g key={p.title}>
        <text x={p.x} y="12" className="fk-strong" textAnchor="start">{p.title}</text>
        <text x={p.x} y="34" className="fk-muted fk-small" textAnchor="start">{p.note}</text>
        {bars(p.x + 16, p.act, 'teal')}
        {bars(p.x + 206, p.w, 'blue')}
        <TexLabel x={p.x + 16 + 72} y={BASE + 44} w={150} source={'\\max |X_j|'} hue="teal" />
        <TexLabel x={p.x + 206 + 72} y={BASE + 44} w={150} source={'\\max |W_j|'} hue="blue" />
      </g>)}
      <line x1="380" y1="50" x2="380" y2={BASE + 60} className="fk-rule" />
    </>}</Figure>
  )
}

// 图 3：一个 4-bit 权重如何通过位运算变成 FP16（vLLM dequant.h，n = 11）
export function QuantMarlinDequant() {
  const X0 = 76, P = 38, Y = 112
  const bits = '0110010000001011'.split('')
  const cx = (i: number) => X0 + i * P + 17
  return (
    <Figure
      title="Marlin 的还原：把 4 bit 编号写入 FP16 的尾数"
      desc="4 bit 编号 1011 即 11，被写入一个 FP16 的尾数低 4 位。该 FP16 的符号位为 0，指数位来自常数 0x6400，表示 2 的 10 次方，所以整体的值是 1024 加 11。再减去 1032 得到 3。"
      height={212}
      below={<Steps items={[
        ['① lop3 拼出一个 FP16', '\\texttt{0x640B} = 2^{10}\\left(1 + \\tfrac{11}{1024}\\right) = 1035'],
        ['② 减去常数 0x6408', '1035 - 1032 = 3'],
        ['③ 乘以这一组的 scale', '\\hat w = 3\\,s'],
      ]} />}
      caption="INT4 以 0 到 15 存储，代表 −8 到 7，所以 n = 11 代表 3。指数取 2^10 时，FP16 尾数的最低位恰好等于 1，整数到浮点的转换因此只需一次按位或和一次减法。"
    >{arrow => <>
      <text x={cx(12) - 30} y="35" className="fk-strong" textAnchor="end">从显存读取的 4 bit 编号：n = 11</text>
      {[12, 13, 14, 15].map(i => <g key={i}>
        <Cell x={X0 + i * P} y={18} w={34} h={34} fill="purple" label={bits[i]} fs={14} />
        <Link d={`M${cx(i)} 52 V${Y}`} hue="purple" arrow={arrow} focus />
      </g>)}
      {bits.map((b, i) => <Cell key={i} x={X0 + i * P} y={Y} w={34} h={34} fill={i === 0 ? 'gray' : i < 6 ? 'teal' : i < 12 ? 'free' : 'purple'} label={i >= 6 && i < 12 ? undefined : b} fs={14} focus={i >= 12} />)}
      {[6, 7, 8, 9, 10, 11].map(i => <text key={i} x={cx(i)} y={Y + 17} className="fk-muted" style={{ fontSize: 14 }}>0</text>)}
      <text x={X0 - 14} y={Y + 17} className="fk-strong" textAnchor="end">FP16</text>
      <text x={cx(3)} y={Y - 16} className="fk-muted fk-small">符号与指数来自常数 0x6400</text>
      <text x={cx(0)} y={Y + 52} className="fk-muted fk-small">符号</text>
      <TexLabel x={cx(3)} y={Y + 54} w={170} source={'\\text{指数：}2^{10}'} hue="teal" />
      <text x={cx(8.5)} y={Y + 52} className="fk-muted fk-small">尾数高 6 位为 0</text>
      <text x={cx(13.5)} y={Y + 52} className="fk-strong fk-text-purple">低 4 位 = n</text>
      <text x={cx(13.5)} y={Y + 76} className="fk-stat">值 = 1024 + 11</text>
    </>}</Figure>
  )
}

// 图 4：Marlin 端到端加速比随 batch 的变化（论文 Table 2，Llama-2-7B，A10）
const SPEEDUP: [number, number][] = [[1, 2.93], [2, 3.19], [4, 3.02], [8, 2.9], [16, 2.74], [32, 2.26], [64, 1.78], [128, 1.2]]

export function QuantMarlinSpeedup() {
  const X0 = 70, PITCH = 80, BASE = 246, UNIT = 52 // px / 1×
  const y = (v: number) => BASE - v * UNIT
  return (
    <Figure
      title="INT4 加 Marlin 相对 FP16 的端到端加速比"
      desc="Llama-2-7B 在 A10 上，batch 为 1、2、4、8、16、32、64、128 时加速比分别为 2.93、3.19、3.02、2.90、2.74、2.26、1.78、1.20 倍。单层理想值为 3.87 倍。"
      height={300}
      caption="数据来自 Marlin 论文 Table 2：Llama-2-7B，A10，vLLM 中 INT4 相对 FP16 的生成加速比。batch 16 以内受带宽限制，加速稳定；之后转为受算力限制，加速比下降。"
    >{() => <>
      {[0, 1, 2, 3].map(v => <g key={v}>
        <line x1={X0 - 10} y1={y(v)} x2={X0 + 8 * PITCH - 10} y2={y(v)} className="fk-rule" />
        <text x={X0 - 18} y={y(v)} className="fk-muted" textAnchor="end">{v}×</text>
      </g>)}
      <line x1={X0 - 10} y1={y(3.87)} x2={X0 + 8 * PITCH - 10} y2={y(3.87)} className="fk-line-purple" strokeWidth="1.5" strokeDasharray="5 4" />
      <text x={X0 + 8 * PITCH - 10} y={y(3.87) - 12} className="fk-strong fk-text-purple" textAnchor="end">单层 kernel 的理想值 3.87×</text>
      {SPEEDUP.map(([b, v], i) => <g key={b}>
        <rect x={X0 + i * PITCH} y={y(v)} width="56" height={v * UNIT} rx="3" className="fk-fill-blue" />
        <text x={X0 + i * PITCH + 28} y={y(v) - 11} className="fk-stat">{v.toFixed(2)}×</text>
        <text x={X0 + i * PITCH + 28} y={BASE + 16} className="fk-muted">{b}</text>
      </g>)}
      <text x={X0 + 8 * PITCH - 10} y={BASE + 40} className="fk-muted" textAnchor="end">batch size</text>
    </>}</Figure>
  )
}

// 图 5：Qwen3.8-27B 各 checkpoint 与一张 32 GiB 显卡（体积取 Hugging Face 的 safetensors 元数据，2026-10-02 读取）
const CARD = 32
const CHECKPOINTS: { label: string; from: string; gib: number }[] = [
  { label: 'BF16', from: 'Qwen', gib: 51.75 },
  { label: 'FP8', from: 'Qwen', gib: 28.75 },
  { label: 'NVFP4', from: 'NVIDIA', gib: 20.42 },
  { label: 'INT4', from: 'RedHatAI', gib: 18.12 },
]

export function QuantMemoryBudget() {
  const X0 = 130, PX = 11.5, H = 44, PITCH = 58 // 11.5 px / GiB
  const edge = X0 + CARD * PX
  return (
    <Figure
      title="Qwen3.8-27B 四种 checkpoint 的显存占用"
      desc="BF16 的权重 51.8 GiB，超出 32 GiB 显卡 19.8 GiB；FP8 为 28.8 GiB，剩余 3.3 GiB；NVFP4 为 20.4 GiB，剩余 11.6 GiB；INT4 为 18.1 GiB，剩余 13.9 GiB。"
      height={CHECKPOINTS.length * PITCH + 46}
      below={<Legend items={[['blue', '权重'], ['amber', '余下的显存，留给 KV cache 与运行时'], ['red', '超出显卡的部分']]} />}
      caption="体积取自 Hugging Face 上各 checkpoint 的文件大小，长度与体积成正比。竖线是 RTX 5090 的 32 GiB。"
    >{() => <>
      {CHECKPOINTS.map((c, i) => {
        const y = 8 + i * PITCH, fit = c.gib <= CARD
        return <g key={c.label}>
          <text x={X0 - 12} y={y + H / 2 - 8} className="fk-strong" textAnchor="end">{c.label}</text>
          <text x={X0 - 12} y={y + H / 2 + 10} className="fk-muted fk-small" textAnchor="end">{c.from}</text>
          <Cell x={X0} y={y} w={Math.min(c.gib, CARD) * PX - 2} h={H} rx={4} fill="blue" label={`权重 ${c.gib.toFixed(1)} GiB`} />
          {fit
            ? <Cell x={X0 + c.gib * PX} y={y} w={(CARD - c.gib) * PX} h={H} rx={4} fill="amber" label={CARD - c.gib > 6 ? `余 ${(CARD - c.gib).toFixed(1)}` : undefined} />
            : <Cell x={edge + 2} y={y} w={(c.gib - CARD) * PX - 2} h={H} rx={4} fill="red" label={`超出 ${(c.gib - CARD).toFixed(1)} GiB`} />}
          {fit && CARD - c.gib <= 6 && <text x={edge + 12} y={y + H / 2} className="fk-strong" textAnchor="start">余 {(CARD - c.gib).toFixed(1)}</text>}
        </g>
      })}
      <line x1={edge} y1="0" x2={edge} y2={CHECKPOINTS.length * PITCH} className="fk-line-muted" strokeWidth="1.5" strokeDasharray="5 4" />
      {[0, 16, 32, 48].map(v => <text key={v} x={X0 + v * PX} y={CHECKPOINTS.length * PITCH + 14} className="fk-muted">{v}</text>)}
      <text x={edge} y={CHECKPOINTS.length * PITCH + 34} className="fk-strong">RTX 5090 · 32 GiB</text>
    </>}</Figure>
  )
}

// 图 6：INT4 与 NVFP4（E2M1）能表示的值，都归一到各自的最大值
const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

export function QuantNvfp4Levels() {
  const CX = 380, HALF = 320, Y1 = 60, Y2 = 150 // 两条数轴：−1 … 1
  const x = (v: number) => CX + v * HALF
  return (
    <Figure
      title="INT4 与 NVFP4 的 16 个编码能表示的值"
      desc="INT4 的 16 个值间隔相等。NVFP4 的每个值是 E2M1 浮点数，绝对值为 0、0.5、1、1.5、2、3、4、6，零附近间隔 0.5，远处间隔 2。每 16 个值共用一个 FP8 的 scale。"
      height={300}
      caption="两条数轴各自归一到最大值。NVFP4 的正零与负零相同，实际有 15 个不同的值。"
    >{() => <>
      <text x="60" y={Y1 - 34} className="fk-strong fk-text-blue" textAnchor="start">INT4：间隔相等</text>
      <line x1="48" y1={Y1} x2="712" y2={Y1} className="fk-rule" />
      {Array.from({ length: 16 }, (_, i) => <rect key={i} x={x((i - 8) / 8) - 1.5} y={Y1 - 16} width="3" height="32" className="fk-fill-blue" />)}
      <text x="60" y={Y2 - 34} className="fk-strong fk-text-teal" textAnchor="start">NVFP4：零附近密，远处疏</text>
      <line x1="48" y1={Y2} x2="712" y2={Y2} className="fk-rule" />
      {E2M1.flatMap(v => (v ? [v, -v] : [v])).map(v => <g key={v}>
        <rect x={x(v / 6) - 1.5} y={Y2 - 16} width="3" height="32" className="fk-fill-teal" />
        <text x={x(v / 6)} y={Y2 + 32} className="fk-muted fk-small">{String(v).replace('-', '−')}</text>
      </g>)}
      <text x="60" y="232" className="fk-strong" textAnchor="start">一个 block</text>
      {Array.from({ length: 16 }, (_, i) => <Cell key={i} x={150 + i * 26} y={214} w={23} h={28} fill="teal" label="4" />)}
      <text x="576" y="228" className="fk-strong">+</text>
      <Cell x={590} y={214} w={90} h={28} fill="purple" label="FP8 scale" />
      <text x="150" y="270" className="fk-muted" textAnchor="start">16 个 4 bit 的值共用一个 8 bit 的 scale，平均每个值 4.5 bit</text>
    </>}</Figure>
  )
}

// 图 7：Qwen3.8-27B 第 0 层 down_proj 的真实权重，和整层只用一个 scale 时 INT4 的 16 个刻度
// 直方图是 ±8 倍标准差内 161 个 bin 的计数（归一到最大值），本站用 scripts/llm-quantization/weight_hist.py 计算，数据在 public/measurements/llm-quantization/weights.json。
const WH = { std: 0.010634, max: 0.9805, hist: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0001,0.0,0.0,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0002,0.0001,0.0002,0.0003,0.0003,0.0004,0.0007,0.0007,0.0009,0.0016,0.0018,0.0024,0.0031,0.0054,0.0058,0.0075,0.0121,0.0154,0.018,0.0261,0.0342,0.0389,0.0551,0.0702,0.0779,0.108,0.1341,0.1445,0.1954,0.2351,0.2458,0.314,0.377,0.4083,0.4638,0.5538,0.5838,0.6421,0.7454,0.7498,0.8373,0.887,0.9094,0.9511,0.986,0.9937,1.0,0.9937,0.9862,0.9507,0.9105,0.8865,0.8374,0.7502,0.7445,0.6433,0.5835,0.5546,0.464,0.4093,0.3756,0.3134,0.2459,0.2351,0.1954,0.1442,0.1342,0.1083,0.0781,0.0702,0.0554,0.0387,0.0341,0.0261,0.0178,0.0153,0.012,0.0076,0.0058,0.0054,0.0031,0.0023,0.0018,0.0016,0.0009,0.0007,0.0007,0.0004,0.0003,0.0003,0.0002,0.0001,0.0002,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0,0.0,0.0001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] }

export function QuantWeightHist() {
  const CX = 380, U = 300, BASE = 196 // 300 px / 1.0
  const x = (v: number) => CX + v * U, s = WH.max / 7, bw = 16 * WH.std / 161
  const bell = 'M' + WH.hist.map((h, k) => `${x((k - 80) * bw).toFixed(1)} ${(BASE - h * 150).toFixed(1)}`).join(' L') + ` L${x(8 * WH.std)} ${BASE} L${x(-8 * WH.std)} ${BASE} Z`
  return (
    <Figure
      title="一层真实权重的分布，与整层共用一个 scale 时的 16 个刻度"
      desc="Qwen3.8-27B 第 0 层 MLP 的 down_proj 有 8913 万个权重，标准差 0.0106，几乎全部集中在零附近，形成很窄的尖峰；绝对值最大为 0.98。整层共用一个 scale 时，scale 等于 0.98 除以 7，约 0.14，16 个刻度间隔 0.14，尖峰全部位于 0 这一个刻度附近。"
      height={292}
      caption="本站计算：Qwen3.8-27B 的 layers.0.mlp.down_proj（5120 × 17408），横轴为权重的值。蓝色是分布的形状（高度归一），竖线是 INT4 的 16 个刻度。"
    >{() => <>
      <path d={bell} className="fk-fill-blue" />
      <line x1={x(-1.2)} y1={BASE} x2={x(1.2)} y2={BASE} className="fk-rule" />
      {Array.from({ length: 16 }, (_, k) => <rect key={k} x={x((k - 8) * s) - 1.5} y={BASE - 22} width="3" height="44" className="fk-fill-gray" />)}
      <circle cx={x(WH.max)} cy={BASE} r="6" className="fk-fill-red" />
      <text x={x(WH.max)} y={BASE - 60} className="fk-strong" style={{ fill: 'var(--fk-red)' }}>最大值：0.98</text>
      <text x={x(WH.max)} y={BASE - 40} className="fk-muted fk-small">标准差的 92 倍</text>
      <text x={x(0) + 24} y="40" className="fk-strong" textAnchor="start">99.99% 的权重绝对值小于 0.044</text>
      <line x1={x(0) + 20} y1="36" x2={x(0) + 6} y2="50" className="fk-line-muted" strokeWidth="1.2" />
      <rect x={x(0)} y={BASE + 34} width={s * U} height="3" className="fk-fill-purple" />
      <text x={x(s / 2)} y={BASE + 54} className="fk-strong fk-text-purple">scale = 0.98 ÷ 7 ≈ 0.14</text>
      {[-1, -0.5, 0, 0.5, 1].map(v => <text key={v} x={x(v)} y={BASE + 80} className="fk-muted fk-small">{String(v).replace('-', '−')}</text>)}
    </>}</Figure>
  )
}
