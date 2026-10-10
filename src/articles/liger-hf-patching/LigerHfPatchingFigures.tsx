import { Badge, Band, Cell, Figure, Legend, Link, type Fill } from '../../figure-kit'

// 配色约定：blue = Liger 实现，gray = 原实现，red = 应替换却保留原实现（漏掉），
// purple = 示例路径与替换动作，teal = hidden states，amber = 生成出来的 logits。

const placeholder = { border: '1.5px dashed var(--color-border)', borderRadius: 8, padding: '16px 20px', color: 'var(--color-muted)', margin: '1.5em 0' }

// 图 1：三种替换各改了哪个引用，影响哪些对象
export function LigerBindingScope() {
  return (
    <Figure
      title="三种替换的作用范围"
      desc="模块中的名字 Qwen3RMSNorm 改指 LigerRMSNorm 后，只有之后构建的 q_norm 是 Liger 类型；改原类的 forward 影响它的全部实例；写入实例 __dict__ 的 forward 只影响这一个实例。"
      height={318}
      below={<Legend items={[['blue', 'Liger 实现'], ['gray', '原实现'], ['purple', '替换动作']]} />}
      caption="名字只在构建 q_norm 时查找一次，决定实例的类型；forward 在每次调用时先查实例的 __dict__，再查类。灰色箭头是 ① 之前名字的指向，紫色是 ① 之后。"
    >{arrow => <>
      <Band x={0} y={0} w={190} h={318} hue="gray" title="modeling_qwen3 模块" />
      <Band x={210} y={0} w={260} h={318} hue="gray" title="类" />
      <Band x={490} y={0} w={270} h={318} hue="gray" title="已构建的实例" />

      <Cell x={20} y={120} w={150} h={40} rx={6} fill="void" label="名字 Qwen3RMSNorm" fs={12} />
      <Badge x={184} y={186} n="1" />

      <Cell x={230} y={44} w={220} h={36} rx={6} fill="gray" label="Qwen3RMSNorm（原类）" />
      <Cell x={230} y={84} w={220} h={28} rx={5} fill="void" label="forward" small />
      <Badge x={462} y={98} n="2" />
      <Cell x={230} y={196} w={220} h={36} rx={6} fill="blue" label="LigerRMSNorm" />
      <Cell x={230} y={236} w={220} h={28} rx={5} fill="blue" label="forward" small />

      <Cell x={510} y={44} w={230} h={36} rx={6} fill="gray" label="A：① 之前构建的 q_norm" />
      <Cell x={510} y={84} w={230} h={28} rx={5} fill="blue" label="__dict__['forward'] = Liger 的" small />
      <Badge x={500} y={98} n="3" />
      <Cell x={510} y={196} w={230} h={36} rx={6} fill="blue" label="B：① 之后构建的 q_norm" />

      <Link d="M170 130 H200 V62 H230" arrow={arrow} />
      <text x={200} y={44} className="fk-muted fk-small">① 之前</text>
      <Link d="M170 150 H200 V214 H230" hue="purple" arrow={arrow} focus />
      <text x={200} y={258} className="fk-small fk-text-purple">① 之后</text>

      <Link d="M510 62 H450" arrow={arrow} />
      <text x={480} y={36} className="fk-muted fk-small">类型</text>
      <Link d="M510 214 H450" arrow={arrow} />

      <text x={95} y={296} className="fk-strong fk-text-purple">① 之后构建的实例</text>
      <text x={340} y={296} className="fk-strong fk-text-purple">② 这个类的全部实例</text>
      <text x={625} y={296} className="fk-strong fk-text-purple">③ 只有这一个实例</text>
    </>}</Figure>
  )
}

// 图 2：Qwen3.5-9B 32 层中，各条接入路径换上的 norm
const LAYERS = Array.from({ length: 32 }, (_, i) => (i + 1) % 4 === 0)
type Path = { title: string; note: string; qk: Fill }
const PATHS: Path[] = [
  { title: '类路径', note: '80 个换上', qk: 'blue' },
  { title: '实例路径，#1470 之前', note: '64 个换上，16 个漏掉', qk: 'red' },
  { title: '实例路径，#1470 之后', note: '80 个换上', qk: 'blue' },
]

export function LigerCoverage() {
  const x0 = 178, pitch = 18, sw = 14, sh = 9, rp = 11
  return (
    <Figure
      title="Qwen3.5-9B 各路径换上的 norm"
      desc="32 层中每 4 层一个 full attention 层。类路径与修复后的实例路径换上全部 80 个 Qwen3.5 RMSNorm；修复前的实例路径漏掉 8 个 full attention 层的 16 个 q_norm 与 k_norm；线性 attention 层带 gate 的 norm 在三条路径下都是原实现。"
      height={286}
      below={<Legend items={[['blue', 'Liger 实现'], ['red', '漏掉，保留原实现'], ['gray', '原实现（Liger 无对应 kernel）']]} />}
      caption="每列一层，自上而下为 input_layernorm、q_norm、k_norm、post_attention_layernorm；线性 attention 层没有 q_norm 与 k_norm，第二格是带 gate 的输出 norm。模型最后的 norm 在三条路径下都换上，图中未画。"
    >{() => <>
      {LAYERS.map((full, i) => full && <g key={i}>
        <rect x={x0 + i * pitch - 2} y={20} width={sw + 4} height={6} rx={2} className="fk-fill-purple" />
        <text x={x0 + i * pitch + sw / 2} y={10} className="fk-muted fk-small">{i + 1}</text>
      </g>)}
      <text x={x0 - 10} y={23} textAnchor="end" className="fk-muted fk-small">full attention 层</text>

      {PATHS.map((p, g) => {
        const gy = 44 + g * 80
        return <g key={p.title}>
          <text x={0} y={gy + 12} textAnchor="start" className="fk-strong">{p.title}</text>
          <text x={0} y={gy + 32} textAnchor="start" className={g === 1 ? 'fk-small' : 'fk-muted fk-small'} style={g === 1 ? { fill: 'var(--fk-red)' } : undefined}>{p.note}</text>
          {LAYERS.map((full, i) => {
            const slots: Fill[] = full ? ['blue', p.qk, p.qk, 'blue'] : ['blue', 'gray', 'void', 'blue']
            return slots.map((f, r) => <rect key={`${i}-${r}`} x={x0 + i * pitch} y={gy + r * rp} width={sw} height={sh} rx={1.5} className={`fk-fill-${f}`} />)
          })}
          {g === 1 && <rect x={x0 - 4} y={gy - 4} width={32 * pitch + 4} height={4 * rp + 6} rx={5} className="fk-focus" />}
        </g>
      })}
    </>}</Figure>
  )
}

// 图 3：一次 forward 生成多少 logits
export function LigerLogits() {
  const V = 200, d = V * 4096 / 151936, H = 180, top = 46
  const panels = [
    { title: '生成：签名中没有 logits_to_keep', note: '10955 × 151936，BF16 下 3.3 GB' },
    { title: '生成：logits_to_keep = 1', note: '1 × 151936，0.3 MB' },
    { title: '训练：FLCE', note: '按 token 分块，每块算完 loss 即释放' },
  ]
  return (
    <Figure
      title="三种情形下生成的 logits"
      desc="Qwen3-VL-8B，10955 个 token。签名缺 logits_to_keep 时生成完整的 10955 × 151936 logits；logits_to_keep=1 时只生成最后一行；训练时 FLCE 每次只生成一块。"
      height={300}
      below={<Legend items={[['teal', 'hidden states'], ['amber', '生成出来的 logits'], ['empty', '不生成']]} />}
      caption="Qwen3-VL-8B，prompt 为 10955 个 token。窄条是 hidden states（宽 4096），宽块是 logits（宽 151936），宽度按真实比例，高度为 10955 个位置。左：patched forward 不声明 logits_to_keep，generate() 不传这个参数，prefill 对每个位置都经过 lm_head。中：只截取最后一个位置。右：训练时 lm_head 与 cross-entropy 按 token 分块融合，同一时刻只有一块 logits。"
    >{arrow => <>
      {panels.map((p, k) => {
        const px = 8 + k * 252, lx = px + d + 18
        return <g key={p.title}>
          <text x={px + (d + 18 + V) / 2} y={14} className="fk-strong fk-small">{p.title}</text>
          <rect x={px} y={top} width={d} height={H} rx={1} className="fk-fill-teal" />
          <Link d={`M${px + d + 3} ${top + H / 2} H${lx}`} arrow={arrow} />
          {k === 0 && <rect x={lx} y={top} width={V} height={H} rx={2} className="fk-fill-amber" />}
          {k === 1 && <>
            <rect x={lx} y={top} width={V} height={H} rx={2} className="fk-fill-empty" />
            <rect x={lx} y={top + H - 3} width={V} height={3} className="fk-fill-amber" />
            <rect x={px} y={top + H - 3} width={d} height={3} className="fk-fill-purple" />
          </>}
          {k === 2 && <>
            <rect x={lx} y={top} width={V} height={H} rx={2} className="fk-fill-empty" />
            {[0, 1, 2, 3, 4, 5, 6, 7, 8].map(c => <line key={c} x1={lx} x2={lx + V} y1={top + (c + 1) * H / 10} y2={top + (c + 1) * H / 10} className="fk-rule" />)}
            <rect x={lx} y={top + 2 * H / 10} width={V} height={H / 10} className="fk-fill-amber" />
            <text x={lx + V / 2} y={top + 2.5 * H / 10} className="fk-cell-text fk-on-amber fk-small">当前一块</text>
          </>}
          <text x={px + (d + 18 + V) / 2} y={top + H + 22} className={k === 0 ? 'fk-small' : 'fk-muted fk-small'} style={k === 0 ? { fill: 'var(--fk-red)' } : undefined}>{p.note}</text>
        </g>
      })}
      <text x={8} y={top - 12} textAnchor="start" className="fk-muted fk-small">10955 个位置</text>
    </>}</Figure>
  )
}

// 图 4：切分序列前先生成 shift_labels
export function LigerShiftLabels() {
  const x0 = [168, 466], pitch = 68, w = 60
  const rows: { y: number; label: string; cells: (r: number, i: number) => [Fill, string, boolean?] }[] = [
    { y: 34, label: '输入 token', cells: (r, i) => ['gray', `t${r * 4 + i + 1}`] },
    { y: 86, label: '切分前左移', cells: (r, i) => {
      const n = r * 4 + i + 2
      return n > 8 ? ['void', 'pad'] : ['blue', `t${n}`, r === 0 && i === 3]
    } },
    { y: 138, label: '切分后本地左移', cells: (r, i) => i === 3 ? (r === 0 ? ['red', 'pad', true] : ['void', 'pad']) : ['blue', `t${r * 4 + i + 2}`] },
  ]
  return (
    <Figure
      title="shift_labels 与序列切分"
      desc="8 个 token 切到两个 rank。切分前左移时，rank 0 第 4 个位置的 label 是 t5；切分后在本地左移时，这个位置变成 padding，t5 不再被预测。"
      height={180}
      below={<Legend items={[['blue', '有效 label'], ['red', '本应有 label，却成为 padding'], ['void', 'padding']]} />}
      caption="每格是一个位置，位置 i 的 label 是第 i+1 个 token。切分前左移得到 shift_labels，rank 0 的最后一个位置拿到下一段的 t5；切分后在本地左移，这个位置只能补 padding。"
    >{() => <>
      <Band x={156} y={0} w={286} h={180} hue="blue" title="rank 0" />
      <Band x={454} y={0} w={286} h={180} hue="purple" title="rank 1" />
      {rows.map(row => <g key={row.label}>
        <text x={0} y={row.y + 16} textAnchor="start" className="fk-strong fk-small">{row.label}</text>
        {[0, 1].map(r => [0, 1, 2, 3].map(i => {
          const [fill, label, focus] = row.cells(r, i)
          return <Cell key={`${r}-${i}`} x={x0[r] + i * pitch} y={row.y} w={w} h={32} rx={5} fill={fill} label={label} focus={focus} />
        }))}
      </g>)}
    </>}</Figure>
  )
}

export function LigerMeasureTrain() {
  return (
    <div style={placeholder}>
      <strong>实测待补。</strong>1 × H100，Qwen3-8B，HF Trainer 开启 <code>use_liger_kernel=True</code>，在 #1470 之前与之后各训练若干 step，记录每 step 耗时与峰值显存，并与类路径对比。这里将放一个代表性数字，说明 Q/K norm 在一整个训练 step 中的占比。
    </div>
  )
}

export function LigerMeasureAudit() {
  return (
    <div style={placeholder}>
      <strong>实测待补。</strong>CPU 环境即可：对 Liger 支持的每种模型用小配置构建，分别走类路径与实例路径，列出实例路径下仍为原实现的模块，在 #1470 之前与当前版本各做一次。这里将放一张覆盖表。
    </div>
  )
}
