import type { ReactNode } from 'react'
import { Band, Cell, Figure, Legend, Link } from '../../figure-kit'

// 语义色：w（BF16 存储）blue，1 + w 的精确值（FP32 相加）teal，BF16 相加的舍入 red，
// 临时向量 amber，RMSNorm kernel 内的计算 blue。

// 图 1：w 附近与 1 附近的 BF16 刻度，四个 w 加 1 后舍入到哪里
const D0 = -0.0045
const D1 = 0.0092
const X0 = 70
const X1 = 730
const S = (X1 - X0) / (D1 - D0)
const xw = (v: number) => X0 + (v - D0) * S
const bf16 = (v: number) => {
  const e = Math.floor(Math.log2(Math.abs(v)))
  const q = 2 ** (e - 7)
  const n = v / q
  const r = Math.floor(n)
  const f = n - r
  return (f > 0.5 || (f === 0.5 && r % 2 !== 0) ? r + 1 : r) * q
}

// 范围内 |v| ≥ 2^-8 的 BF16 刻度；更小的部分刻度间隔不到 1 px，画成实心带。
function denseTicks() {
  const out: number[] = []
  for (let e = -8; e <= -7; e++) {
    for (let m = 128; m < 256; m++) {
      const v = m * 2 ** (e - 7)
      if (v <= D1) out.push(v)
      if (-v >= D0) out.push(-v)
    }
  }
  return out
}

export function OffsetRulers() {
  const TOP = 80
  const BOT = 170
  const solidL = xw(-(2 ** -8))
  const solidR = xw(2 ** -8)
  const grid = [1 - 2 ** -8, 1, 1 + 2 ** -7]
  const samples: { w: number; label: string; row: number; anchor: 'start' | 'middle' | 'end'; dy: number; tie?: boolean }[] = [
    { w: -0.003, label: '−0.003', row: 204, anchor: 'middle', dy: 0 },
    { w: 0.003, label: '0.003', row: 226, anchor: 'end', dy: 0 },
    { w: 0.00390625, label: '0.00390625', row: 248, tie: true, anchor: 'middle', dy: -16 },
    { w: 0.005, label: '0.005', row: 270, anchor: 'start', dy: 0 },
  ]
  return (
    <Figure
      title="w 附近与 1 附近的 BF16 刻度"
      desc="上方数轴是 w 在 −0.0045 到 0.0092 之间的 BF16 刻度，密到连成一片；下方是 1 + w 在同一宽度内的 BF16 刻度，只有 0.99609375、1、1.0078125 三个。四个 w 加 1 后的精确值分别舍入到最近的刻度：−0.003 到 0.99609375，0.003 与 0.00390625 到 1，0.005 到 1.0078125。"
      height={290}
      below={<Legend items={[['blue', 'w（BF16 存储）'], ['teal', '1 + w 的精确值（FP32 相加）'], ['red', 'BF16 相加的结果']]} />}
      caption="两条数轴的比例尺相同，下方整体平移 1。0.00390625 加 1 恰在 1 与 1.0078125 正中，取尾数为偶数的 1。"
    >{() => <>
      <text x="0" y={TOP - 7} textAnchor="start" className="fk-strong">w</text>
      <text x="0" y={BOT - 7} textAnchor="start" className="fk-strong">1 + w</text>

      {/* 上方数轴：w 附近的刻度 */}
      <line x1={X0} x2={X1} y1={TOP} y2={TOP} className="fk-rule" />
      <rect x={solidL} y={TOP - 14} width={solidR - solidL} height={14} className="fk-hatch-line" />
      {denseTicks().map(v => <line key={v} x1={xw(v)} x2={xw(v)} y1={TOP - 14} y2={TOP} stroke="#cdd2da" strokeWidth={0.6} />)}
      {[-0.004, 0, 0.004, 0.008].map(v => <text key={v} x={xw(v)} y={TOP + 12} className="fk-muted fk-small">{v === 0 ? '0' : v.toString().replace('-', '−')}</text>)}
      <text x={X1} y={TOP + 30} textAnchor="end" className="fk-muted fk-small">BF16 刻度间隔：|w| &lt; 0.0039 时不超过 1.5e-5，0.0039–0.0078 为 3.1e-5</text>

      {/* 下方数轴：1 附近的刻度 */}
      <line x1={X0} x2={X1} y1={BOT} y2={BOT} className="fk-rule" />
      {grid.map(v => <g key={v}>
        <line x1={xw(v - 1)} x2={xw(v - 1)} y1={BOT - 16} y2={BOT} stroke="#6b7280" strokeWidth={2} />
        <line x1={xw(v - 1)} x2={xw(v - 1)} y1={BOT + 22} y2={282} className="fk-rule" strokeDasharray="2 3" />
        <text x={xw(v - 1)} y={BOT + 12} className="fk-muted fk-small">{v === 1 ? '1' : v.toString()}</text>
      </g>)}
      <text x={X1} y={BOT - 28} textAnchor="end" className="fk-muted fk-small">BF16 刻度间隔：1 以下 0.0039，1 以上 0.0078</text>

      {samples.map(s => {
        const wb = bf16(s.w)
        const x = xw(wb)
        const r = bf16(1 + wb)
        const xr = xw(r - 1)
        const left = xr < x
        return <g key={s.w}>
          <line x1={x} x2={x} y1={TOP - 22} y2={s.row} className="fk-rule" strokeDasharray="3 3" />
          <circle cx={x} cy={TOP - 22} r={5} className="fk-fill-blue" />
          <text x={x + (s.anchor === 'end' ? 4 : s.anchor === 'start' ? -4 : 0)} y={TOP - 36 + s.dy} textAnchor={s.anchor} className="fk-text-blue fk-small">{s.label}</text>
          <circle cx={x} cy={BOT} r={4.5} className="fk-fill-teal" />
          <path d={`M${x} ${s.row} H${xr}`} stroke="var(--fk-red)" strokeWidth={2} fill="none" />
          <path d={left ? `M${xr} ${s.row} l8 -4.5 v9 Z` : `M${xr} ${s.row} l-8 -4.5 v9 Z`} className="fk-fill-red" />
          <text x={left ? xr - 6 : xr + 6} y={s.row} textAnchor={left ? 'end' : 'start'} className="fk-small" style={{ fill: 'var(--fk-red)', fontWeight: 600 }}>
            {r}{s.tie ? '（正中，取偶）' : ''}
          </text>
        </g>
      })}
    </>}</Figure>
  )
}

// 图 2：修复前后 offset 在哪里相加
export function OffsetDataflow() {
  const row = (y: number) => y + 50
  return (
    <Figure
      title="修复前后 offset 的相加位置"
      desc="修复前，一个单独的 elementwise kernel 在 BF16 中计算 w + 1 并写出临时向量 W_eff，RMSNorm kernel 读入后转成 FP32 再乘。修复后，RMSNorm kernel 直接读 w，转成 FP32 后加 1 再乘。"
      height={290}
      below={<Legend items={[['red', 'BF16 舍入的位置'], ['teal', 'FP32 中加 offset'], ['amber', '临时向量'], ['blue', 'RMSNorm kernel 内的计算']]} />}
      caption="forward 中的一行。backward 计算 dY ⊙ (1 + w) 时，修复前读同一个 W_eff，修复后读 w 并同样在 FP32 中加 1。"
    >{arrow => <>
      <text x="0" y={row(20) + 22} textAnchor="start" className="fk-strong">修复前</text>
      <Cell x={64} y={row(20)} w={76} h={46} rx={6} fill="gray" label="w" sub="BF16" />
      <Band x={156} y={18} w={176} h={104} hue="gray" title="elementwise kernel" />
      <Cell x={170} y={row(20)} w={148} h={46} rx={6} fill="red" label="w + 1" sub="BF16 加法，舍入" />
      <Cell x={348} y={row(20)} w={108} h={46} rx={6} fill="amber" label="W_eff" sub="N × 2 字节" />
      <Band x={472} y={18} w={288} h={104} hue="blue" title="RMSNorm kernel" />
      <Cell x={486} y={row(20)} w={104} h={46} rx={6} fill="blue" label="转 FP32" />
      <Cell x={610} y={row(20)} w={136} h={46} rx={6} fill="blue" label="x̂ ⊙ W_eff" sub="写回 Y（BF16）" />
      <Link d={`M140 ${row(20) + 23} H170`} arrow={arrow} />
      <Link d={`M318 ${row(20) + 23} H348`} arrow={arrow} />
      <Link d={`M456 ${row(20) + 23} H486`} arrow={arrow} />
      <Link d={`M590 ${row(20) + 23} H610`} arrow={arrow} />

      <text x="0" y={row(168) + 22} textAnchor="start" className="fk-strong">修复后</text>
      <Cell x={64} y={row(168)} w={76} h={46} rx={6} fill="gray" label="w" sub="BF16" />
      <Band x={156} y={166} w={604} h={104} hue="blue" title="RMSNorm kernel" />
      <Cell x={170} y={row(168)} w={150} h={46} rx={6} fill="blue" label="转 FP32" />
      <Cell x={348} y={row(168)} w={150} h={46} rx={6} fill="teal" label="+ 1" sub="FP32 加法" />
      <Cell x={526} y={row(168)} w={220} h={46} rx={6} fill="blue" label="x̂ ⊙ (1 + w)" sub="写回 Y（BF16）" />
      <Link d={`M140 ${row(168) + 23} H170`} arrow={arrow} />
      <Link d={`M320 ${row(168) + 23} H348`} arrow={arrow} />
      <Link d={`M498 ${row(168) + 23} H526`} arrow={arrow} />
    </>}</Figure>
  )
}

// 实测占位：方向确认后补数据，不填编造的数字
function Placeholder({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ border: '1.5px dashed var(--color-border)', borderRadius: 8, padding: '16px 20px', color: 'var(--color-muted)', margin: '1.5em 0' }}>
      <strong>{title}</strong>{children}
    </div>
  )
}

export function OffsetCheckpointPlaceholder() {
  return <Placeholder title="数据待补。">读取 Gemma 3 checkpoint 中各层 RMSNorm 的 weight（只需 CPU 与网络，按 safetensors 分段只下载 norm 张量），统计每层 1 + w 在 BF16 中被舍入改变的比例与最大相对误差。这里将放一张按层排列的统计图。</Placeholder>
}

export function OffsetTracePlaceholder() {
  return <Placeholder title="实测待补。">H100，Liger CuTe DSL 后端，BF16 Gemma RMSNorm forward（M = 128，N = 4096），修复前后各用 nsys 抓一条 trace。这里将放两条时间线：修复前为 elementwise add 与 RMSNorm 两个 kernel，修复后只有 RMSNorm 一个。</Placeholder>
}
