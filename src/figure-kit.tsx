// 自绘技术图的共享图元：实心语义色块、直接标注、KaTeX 标签、示例路径描边。
// 颜色按色相命名（blue / purple / teal / green / amber / red / gray），语义由各篇文章自行约定。
import { useId, type ReactNode } from 'react'
import katex from 'katex'
import './figure-kit.css'

export type Hue = 'blue' | 'purple' | 'teal' | 'green' | 'amber' | 'red' | 'gray'
export type Fill = Hue | 'free' | 'empty'

export const tex = (source: string) => ({ __html: katex.renderToString(source, { throwOnError: true }) })

type FigureProps = {
  title: string
  desc: string
  width?: number
  height: number
  caption: ReactNode
  below?: ReactNode
  children: (arrow: string) => ReactNode
}

// SVG 图的外框：无障碍标题、箭头 marker、图下附加内容与图注。
export function Figure({ title, desc, width = 760, height, caption, below, children }: FigureProps) {
  const id = useId()
  return (
    <figure className="fk-figure">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>{title}</title>
        <desc id={`${id}-desc`}>{desc}</desc>
        <defs><marker id={`${id}-arrow`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M1 1 L7 4 L1 7" fill="none" stroke="currentColor" strokeWidth="1.3" />
        </marker></defs>
        {children(`url(#${id}-arrow)`)}
      </svg>
      {below}
      <figcaption>{caption}</figcaption>
    </figure>
  )
}

type CellProps = { x: number; y: number; w: number; h: number; fill: Fill; label?: ReactNode; small?: boolean; focus?: boolean; rx?: number }

// 一个格子：slot、token 或 block；focus 表示示例路径上的对象。
export function Cell({ x, y, w, h, fill, label, small, focus, rx = 3 }: CellProps) {
  return <g>
    <rect x={x} y={y} width={w} height={h} rx={rx} className={`fk-fill-${fill}`} />
    {label !== undefined && <text x={x + w / 2} y={y + h / 2 + 0.5} className={`fk-cell-text ${small ? 'fk-cell-small' : ''}`}>{label}</text>}
    {focus && <rect x={x - 2} y={y - 2} width={w + 4} height={h + 4} rx={rx + 1} className="fk-focus" />}
  </g>
}

export function Link({ d, hue, arrow, focus }: { d: string; hue?: Hue; arrow?: string; focus?: boolean }) {
  return <path d={d} className={`fk-link fk-line-${hue ?? 'muted'} ${focus ? 'fk-link-focus' : ''}`} markerEnd={arrow} />
}

// SVG 内的 KaTeX 标签，以 (x, y) 为中心。
export function TexLabel({ x, y, source, hue, w = 200, h = 30 }: { x: number; y: number; source: string; hue?: Hue; w?: number; h?: number }) {
  return <foreignObject x={x - w / 2} y={y - h / 2} width={w} height={h}>
    <div className={`fk-tex ${hue ? `fk-text-${hue}` : ''}`} dangerouslySetInnerHTML={tex(source)} />
  </foreignObject>
}

export function Legend({ items }: { items: [Fill, string][] }) {
  return <div className="fk-legend">{items.map(([fill, label]) => <span key={label}><i className={`fk-swatch-${fill}`} />{label}</span>)}</div>
}

// 图下的分步计算，每步一个说明加一个 KaTeX 表达式。
export function Steps({ items }: { items: [string, string][] }) {
  return <ol className="fk-steps">{items.map(([label, source]) => <li key={label}><span>{label}</span><b dangerouslySetInnerHTML={tex(source)} /></li>)}</ol>
}

// 引用原图：保留原样，夜间模式下衬白底。
export function PaperFigure({ src, alt, maxWidth = 560, caption }: { src: string; alt: string; maxWidth?: number; caption: ReactNode }) {
  return <figure className="fk-paper">
    <img src={src} alt={alt} style={{ maxWidth }} />
    <figcaption>{caption}</figcaption>
  </figure>
}
