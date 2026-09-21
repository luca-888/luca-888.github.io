import { useId, useState } from 'react'
import './RmsNormGraph.css'

type Path = 'all' | 'direct' | 'scale'
const views: { id: Path; label: string; explanation: string }[] = [
  { id: 'all', label: '完整计算图', explanation: '箭头表示 Forward 依赖。Backward 沿箭头反向传播，x 的两条路径贡献相加。' },
  { id: 'direct', label: '直接路径', explanation: 'x 直接参与 x_hat = x * r 的缩放，产生直接路径贡献。' },
  { id: 'scale', label: '经 r 的路径', explanation: 'x 还通过平方均值 s 改变 r，影响这一行的所有输出，产生经 r 路径贡献。' },
]
const colors = { direct: '#7899b5', scale: '#789888', shared: '#98a2aa', weight: '#b38b91' }
const gradients = {
  x: { title: 'dx', subtitle: '两路相加 [N]' },
  r: { title: 'dr', subtitle: 'Σᵢ dx_hatᵢ * xᵢ · 标量' },
  normalized: { title: 'dx_hat', subtitle: 'g * gamma [N]' },
  weight: { title: 'dgamma', subtitle: '沿 M 累加 [N]' },
  y: { title: 'g', subtitle: '上游梯度 [N]' },
}
const nodes = [
  { id: 'x', title: 'x', subtitle: '输入 [N]', branch: 'shared', wide: [12, 136, 90, 54], narrow: [12, 12, 82, 48] },
  { id: 'mean', title: 's = mean(x²)', subtitle: '沿 N 归约 · 标量', branch: 'scale', wide: [166, 22, 164, 54], narrow: [146, 12, 172, 48] },
  { id: 'r', title: 'r = rsqrt(s+eps)', subtitle: '归一化系数 · 标量', branch: 'scale', wide: [396, 22, 164, 54], narrow: [146, 88, 172, 48] },
  { id: 'normalized', title: 'x_hat = x * r', subtitle: '归一化结果 [N]', branch: 'shared', wide: [396, 136, 164, 54], narrow: [146, 164, 172, 48] },
  { id: 'weight', title: 'gamma', subtitle: '共享权重 [N]', branch: 'weight', wide: [666, 22, 172, 54], narrow: [12, 240, 82, 48] },
  { id: 'y', title: 'y = x_hat * gamma', subtitle: '输出 [N]', branch: 'shared', wide: [666, 136, 172, 54], narrow: [146, 240, 172, 48] },
] as const
const edges = [
  { branch: 'scale', wide: 'M57 136 V61 Q57 49 69 49 H162', narrow: 'M94 36 H142', gradient: '2xᵢ/N', derivative: '∂s/∂xᵢ = 2xᵢ/N', labelWide: [112, 38], labelNarrow: [118, 25] },
  { branch: 'scale', wide: 'M330 49 H392', narrow: 'M232 60 V84', gradient: '−r³/2', derivative: '∂r/∂s = −r³/2', labelWide: [362, 38], labelNarrow: [270, 76] },
  { branch: 'scale', wide: 'M478 76 V132', narrow: 'M232 136 V160', gradient: 'xᵢ', derivative: '∂x_hatᵢ/∂r = xᵢ', labelWide: [498, 108], labelNarrow: [254, 152] },
  { branch: 'direct', wide: 'M102 163 H392', narrow: 'M53 60 V176 Q53 188 65 188 H142', gradient: 'r', derivative: '∂x_hatᵢ/∂xᵢ = r（固定 r）', labelWide: [248, 152], labelNarrow: [68, 125] },
  { branch: 'shared', wide: 'M560 163 H662', narrow: 'M232 212 V236', gradient: 'gammaᵢ', derivative: '∂yᵢ/∂x_hatᵢ = gammaᵢ', labelWide: [612, 152], labelNarrow: [267, 228] },
  { branch: 'weight', wide: 'M752 76 V132', narrow: 'M94 264 H142', gradient: 'x_hatᵢ', derivative: '∂yᵢ/∂gammaᵢ = x_hatᵢ', labelWide: [786, 108], labelNarrow: [118, 253] },
] as const

export function RmsNormGraph() {
  const id = useId()
  const [view, setView] = useState<Path>('all')
  const [backward, setBackward] = useState(true)
  const muted = (branch: string) => view !== 'all' && branch !== view && branch !== 'shared'

  function diagram(compact: boolean) {
    const layout = compact ? 'narrow' : 'wide'
    const prefix = `${id}-${layout}`
    const visibleEdges = backward ? [
      { branch: 'scale' as const, wide: 'M57 136 V61 Q57 49 69 49 H392', narrow: 'M53 60 V100 Q53 112 65 112 H142', gradient: '−xᵢr³/N', derivative: '∂r/∂xᵢ = −xᵢr³/N', labelWide: [248, 38], labelNarrow: [100, 101] },
      ...edges.slice(2),
    ] : edges
    return (
      <svg className={`computation-graph-${layout}`} viewBox={compact ? '0 0 330 300' : '0 0 852 210'}
        role="img" aria-labelledby={`${prefix}-title ${prefix}-description`}>
        <title id={`${prefix}-title`}>RMSNorm {backward ? 'Backward 梯度流' : 'Forward 计算图'}</title>
        <desc id={`${prefix}-description`}>{backward
          ? '从上游梯度 g 出发，得到 dx_hat = g * gamma。直接路径贡献 r * dx_hat；另一条路径先沿 N 求和得到 dr，再乘 r 对 x 的导数，贡献 −x * r³ * dr / N。dx 合并两路贡献。dgamma 为所有 token 的 g * x_hat 之和。'
          : '输入 x 分为两路：一路直接参与缩放，另一路先求平方均值 s，再计算 r = rsqrt(s + eps)。两路在 x_hat = x * r 汇合，最后逐元素乘以共享权重 gamma，得到输出 y。'}</desc>
        <defs>{Object.entries(colors).map(([branch, color]) => (
          <marker key={branch} id={`${prefix}-${branch}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M1 1 L7 4 L1 7" fill="none" stroke={color} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
          </marker>
        ))}</defs>
        {visibleEdges.map((edge, i) => <path key={i} d={edge[layout]} className={muted(edge.branch) ? 'graph-edge is-muted' : 'graph-edge'}
          data-branch={edge.branch} stroke={colors[edge.branch]}
          markerStart={backward ? `url(#${prefix}-${edge.branch})` : undefined}
          markerEnd={backward ? undefined : `url(#${prefix}-${edge.branch})`} />)}
        {nodes.map(node => {
          if (backward && node.id === 'mean') return null
          const [x, y, width, height] = node[layout]
          const label = backward && node.id !== 'mean' ? gradients[node.id] : node
          return <g key={node.id} className={muted(node.branch) ? 'graph-node is-muted' : 'graph-node'} data-node={node.id} data-branch={node.branch}>
            <rect x={x} y={y} width={width} height={height} rx="4" />
            <text className="graph-node-title" x={x + width / 2} y={y + height / 2 - 3}>{label.title}</text>
            <text className="graph-node-subtitle" x={x + width / 2} y={y + height / 2 + 15}>{label.subtitle}</text>
          </g>
        })}
        {backward ? visibleEdges.map((edge, i) => {
          const [x, y] = compact ? edge.labelNarrow : edge.labelWide
          return <text key={i} className={`graph-gradient${muted(edge.branch) ? ' is-muted' : ''}`} data-branch={edge.branch} x={x} y={y}>
            <title>{edge.derivative}</title>{edge.gradient}
          </text>
        }) : <>
          <text className={`graph-edge-label${muted('direct') ? ' is-muted' : ''}`} fill="var(--color-accent, #567891)" x={compact ? 64 : 222} y={compact ? 125 : 151}>直接路径</text>
          <text className={`graph-edge-label${muted('scale') ? ' is-muted' : ''}`} fill="var(--color-secondary, #587767)" x={compact ? 244 : 490} y={compact ? 153 : 110}>广播 r</text>
        </>}
      </svg>
    )
  }

  return (
    <figure className="computation-graph" aria-labelledby={`${id}-heading`}>
      <div className="graph-toolbar">
        <header>
          <div className="graph-heading">
            <h4 id={`${id}-heading`}>{backward ? 'Backward 梯度流' : 'Forward 计算图'}</h4>
            <div className="graph-mode" role="group" aria-label="传播方向">
              <button type="button" aria-pressed={!backward} onClick={() => setBackward(false)}>Forward</button>
              <button type="button" aria-pressed={backward} onClick={() => setBackward(true)}>Backward</button>
            </div>
          </div>
          <span>{backward ? '节点显示梯度；dgamma 跨 token 汇总' : '单个 token · gamma 在所有 token 间共享'}</span>
        </header>
        <div className="graph-controls" role="group" aria-label="计算路径高亮">
          {views.map(item => <button key={item.id} type="button" data-path={item.id} aria-pressed={view === item.id} onClick={() => setView(item.id)}>{item.label}</button>)}
        </div>
      </div>
      {diagram(false)}
      {diagram(true)}
      {backward && <div className="graph-results" aria-label="梯度计算结果">
        <div className="graph-result graph-result-dx">
          <h5>输入梯度 dx <span>每行独立</span></h5>
          <div className="graph-result-formula">dx = <span className={muted('direct') ? 'result-direct is-muted' : 'result-direct'}>r * dx_hat</span> <span className={muted('scale') ? 'result-scale is-muted' : 'result-scale'}>− x * r³ * dr / N</span></div>
          <p>dx_hat = g * gamma，dr = sum(dx_hat * x, dim=1)</p>
        </div>
        <div className={`graph-result graph-result-dw${muted('weight') ? ' is-muted' : ''}`}>
          <h5>权重梯度 dgamma <span>沿 M 求和</span></h5>
          <div className="graph-result-formula">dgamma = ∑<sub>m</sub> g<sub>m</sub> * x_hat<sub>m</sub></div>
          <p>对 g * x_hat 执行 sum(dim=0)，得到 [N]。</p>
        </div>
      </div>}
      <figcaption aria-live="polite">{backward
        ? view === 'direct' ? '蓝色路径：dx_hat 直接乘 r，得到直接路径贡献。'
          : view === 'scale' ? '绿色路径：dr 乘以合并后的导数 −x * r³ / N，得到经 r 路径贡献。'
            : 'dx 合并蓝、绿两路贡献；dgamma 汇总所有 token。边上标注局部导数，* 表示逐元素相乘。'
        : views.find(item => item.id === view)!.explanation}</figcaption>
    </figure>
  )
}
