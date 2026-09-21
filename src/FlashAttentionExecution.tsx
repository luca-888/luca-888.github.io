import { useId } from 'react'
import { renderToString } from 'katex'
import './FlashAttentionExecution.css'

const formulas = {
  q0: String.raw`Q_0`, q1: String.raw`Q_1`, q31: String.raw`Q_{31}`,
  o0: String.raw`O_0`, o1: String.raw`O_1`, o31: String.raw`O_{31}`,
  tile: String.raw`[128,64]`,
  state: String.raw`m,\ell:[128],\quad U:[128,64]`,
  output: String.raw`O_0=U/\ell`,
}
const math = Object.fromEntries(Object.entries(formulas).map(([key, tex]) =>
  [key, { __html: renderToString(tex, { throwOnError: true }) }],
))
function Formula({ name, x, y, width = 100, height = 32 }: {
  name: keyof typeof formulas; x: number; y: number; width?: number; height?: number
}) {
  return <foreignObject x={x} y={y} width={width} height={height}>
    <div className="fa-execution-math" dangerouslySetInnerHTML={math[name]} />
  </foreignObject>
}

export function FlashAttentionExecution() {
  const id = useId()
  return <figure className="fa-execution">
    <svg viewBox="0 0 800 410" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>Q tile 独立执行，CTA 内固定 Q 并遍历 K/V</title>
      <desc id={`${id}-desc`}>选定一个 batch 和一个 head 后，Q 分为 32 个 tile，每个 tile shape 为 [128, 64]，分别交给 CTA 0 至 31，写出对应 O tile。放大 CTA 0：Q0 固定，依次读取 K/V tiles，更新并保留 m、指数和与 U，遍历完成后归一化输出。每个 query 的状态独立。</desc>
      <defs><marker id={`${id}-arrow`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#7c8992" strokeWidth="1.2" /></marker></defs>
      <text x="108" y="22" className="fa-execution-title">Q tile</text>
      <text x="400" y="22" textAnchor="middle" className="fa-execution-title">独立分配给 CTA</text>
      <text x="695" y="22" textAnchor="middle" className="fa-execution-title">O tile</text>
      {([0, 1, 31] as const).map((block, i) => {
        const y = [40, 85, 145][i]
        return <g key={block}>
          <rect x="65" y={y} width="130" height="36" rx="3" className={block === 0 ? 'fa-execution-q-selected' : 'fa-execution-tile'} />
          <Formula name={block === 0 ? 'q0' : block === 1 ? 'q1' : 'q31'} x={80} y={y + 2} />
          <path d={`M211 ${y + 18}H327 M473 ${y + 18}H605`} className="fa-execution-line" markerEnd={`url(#${id}-arrow)`} />
          <text x="400" y={y + 24} textAnchor="middle" className={block === 0 ? 'fa-execution-active' : ''}>CTA {block}</text>
          <rect x="625" y={y} width="130" height="36" rx="3" className={block === 0 ? 'fa-execution-o-selected' : 'fa-execution-tile'} />
          <Formula name={block === 0 ? 'o0' : block === 1 ? 'o1' : 'o31'} x={640} y={y + 2} />
        </g>
      })}
      <text x="130" y="139" textAnchor="middle">⋮</text><text x="400" y="139" textAnchor="middle">⋮</text><text x="690" y="139" textAnchor="middle">⋮</text>
      <path d="M40 201H770" className="fa-execution-divider" />
      <text x="40" y="230" className="fa-execution-title">放大 CTA 0：固定 Q，遍历 K/V</text>
      <text x="74" y="280" className="fa-execution-small">始终不变</text>
      <rect x="40" y="291" width="130" height="72" rx="3" className="fa-execution-q-selected" />
      <Formula name="q0" x={55} y={293} /><Formula name="tile" x={45} y={327} width={120} />
      <text x="420" y="259" textAnchor="middle" className="fa-execution-small">K/V tiles：依次送入，重复更新</text>
      {[0, 1, 2].map(i => <rect key={i} x={323 + i * 48} y="269" width="34" height="19" className="fa-execution-o-selected" />)}
      <text x="477" y="285">…</text>
      <path d="M185 327H258 M420 294V312 M582 340H655" className="fa-execution-line" markerEnd={`url(#${id}-arrow)`} />
      <rect x="274" y="319" width="292" height="53" rx="3" className="fa-execution-state" />
      <Formula name="state" x={279} y={325} width={282} height={40} />
      <text x="420" y="397" textAnchor="middle" className="fa-execution-small">跨 tile 保留；每个 query 各自一份状态</text>
      <text x="703" y="308" textAnchor="middle" className="fa-execution-small">全部处理完</text>
      <Formula name="output" x={656} y={325} width={120} height={40} />
    </svg>
    <figcaption>上：不同 Q tile 独立计算。下：一个 CTA 批量执行第二章的计算，最后写出对应的 O tile。</figcaption>
  </figure>
}
