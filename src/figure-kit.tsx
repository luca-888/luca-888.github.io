// 自绘技术图的共享图元：实心语义色块、直接标注、KaTeX 标签、示例路径描边。
// 颜色按色相命名（blue / purple / teal / green / amber / red / gray），语义由各篇文章自行约定。
import { createContext, useContext, useId, type ReactNode } from 'react'
import katex from 'katex'
import './figure-kit.css'

export type Hue = 'blue' | 'purple' | 'teal' | 'green' | 'amber' | 'red' | 'gray'
export type Fill = Hue | 'free' | 'empty' | 'ink' | 'void'

// 紧凑模式：小红书等窄屏导出时为 true，图按约 520 宽的竖排布局绘制，放大后图内文字接近正文字号。
export const CompactContext = createContext(false)
export const useCompact = () => useContext(CompactContext)

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

const MARKS = ['muted', 'blue', 'purple', 'teal'] as const
const HEAD = 8 // 箭头长度

// 线条在箭头底部截止，箭头尖端落在路径终点上；只处理以 L / H / V 结尾的绝对坐标路径。
function trimEnd(d: string) {
  const toks = d.match(/[MLHV]|-?\d*\.?\d+/g)
  if (!toks || /[^MLHV\d\s.,-]/.test(d)) return d
  let x = 0, y = 0, px = 0, py = 0, cmd = ''
  const nums: { i: number; axis: 'x' | 'y' }[] = []
  for (let i = 0; i < toks.length;) {
    if (/[MLHV]/.test(toks[i])) cmd = toks[i++]
    px = x; py = y; nums.length = 0
    if (cmd === 'H') { x = +toks[i]; nums.push({ i: i++, axis: 'x' }) }
    else if (cmd === 'V') { y = +toks[i]; nums.push({ i: i++, axis: 'y' }) }
    else { x = +toks[i]; y = +toks[i + 1]; nums.push({ i, axis: 'x' }, { i: i + 1, axis: 'y' }); i += 2 }
  }
  const len = Math.hypot(x - px, y - py)
  if (cmd === 'M' || len <= HEAD) return d
  const k = (len - HEAD) / len
  for (const n of nums) toks[n.i] = String(+(n.axis === 'x' ? px + (x - px) * k : py + (y - py) * k).toFixed(2))
  return toks.join(' ')
}

// SVG 图的外框：无障碍标题、箭头 marker、图下附加内容与图注。
export function Figure({ title, desc, width = 760, height, caption, below, children }: FigureProps) {
  const id = useId()
  return (
    <figure className="fk-figure">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>{title}</title>
        <desc id={`${id}-desc`}>{desc}</desc>
        <defs>{MARKS.map(m => <marker key={m} id={`${id}-arrow${m === 'muted' ? '' : `-${m}`}`} viewBox="0 0 10 10" refX="1" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M1 1.5 L9 5 L1 8.5 Z" className={`fk-mark-${m}`} />
        </marker>)}</defs>
        {children(`url(#${id}-arrow)`)}
      </svg>
      {below}
      <figcaption>{caption}</figcaption>
    </figure>
  )
}

type CellProps = { x: number; y: number; w: number; h: number; fill: Fill; label?: ReactNode; sub?: ReactNode; small?: boolean; focus?: boolean; rx?: number; fs?: number }

// 一个格子：slot、token 或 block；sub 为第二行小字，focus 表示示例路径上的对象。
export function Cell({ x, y, w, h, fill, label, sub, small, focus, rx = 3, fs }: CellProps) {
  const cy = y + h / 2 + 0.5 - (sub !== undefined ? 8 : 0)
  const cls = `fk-cell-text fk-on-${fill}`
  return <g>
    <rect x={x} y={y} width={w} height={h} rx={rx} className={`fk-fill-${fill}`} />
    {label !== undefined && <text x={x + w / 2} y={cy} className={`${cls} ${small ? 'fk-cell-small' : ''}`} style={fs ? { fontSize: fs } : undefined}>{label}</text>}
    {sub !== undefined && <text x={x + w / 2} y={cy + 18} className={`${cls} fk-cell-sub`}>{sub}</text>}
    {focus && <rect x={x - 2} y={y - 2} width={w + 4} height={h + 4} rx={rx + 1} className="fk-focus" />}
  </g>
}

// 连线；传入 arrow 时画实心箭头，路径终点写在目标边上。
export function Link({ d, hue, arrow, focus }: { d: string; hue?: Hue; arrow?: string; focus?: boolean }) {
  const mark = hue && (MARKS as readonly string[]).includes(hue) ? arrow?.replace(/\)$/, `-${hue})`) : arrow
  return <path d={arrow ? trimEnd(d) : d} className={`fk-link fk-line-${hue ?? 'muted'} ${focus ? 'fk-link-focus' : ''}`} markerEnd={mark} />
}

// 进程、分组等容器：淡色底带 + 左上角同色标题。
export function Band({ x, y, w, h, hue, title }: { x: number; y: number; w: number; h: number; hue: 'blue' | 'purple' | 'teal' | 'gray'; title: string }) {
  return <g>
    <rect x={x} y={y} width={w} height={h} rx="12" className={`fk-band-${hue}`} />
    <text x={x + 14} y={y + 20} textAnchor="start" className={`fk-strong ${hue === 'gray' ? '' : `fk-text-${hue}`}`}>{title}</text>
  </g>
}

// 示例路径上的步骤编号，放在线旁边而不是线上。
export function Badge({ x, y, n }: { x: number; y: number; n: number | string }) {
  return <g><circle cx={x} cy={y} r="10" className="fk-fill-purple" /><text x={x} y={y + 0.5} className="fk-cell-text" style={{ fontSize: 11 }}>{n}</text></g>
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
