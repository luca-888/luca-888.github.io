import { useId } from 'react'
import './FlashAttentionCausalFigure.css'

const tileWidth = 76
const matrixX = 92
const matrixY = 54
const matrixHeight = 176
const maskedX = matrixX + 2 * tileWidth
const rowStep = matrixHeight / 128
const columnStep = tileWidth / 64
const causalBoundary = Array.from({ length: 128 }, (_, row) =>
  `H${maskedX + (row + 1) * columnStep}V${matrixY + (row + 1) * rowStep}`,
).join(' ')
const validRegion = `M${maskedX} ${matrixY} ${causalBoundary} H${maskedX}Z`
const diagonal = `M${maskedX} ${matrixY} ${causalBoundary}`

export function FlashAttentionCausalTiles() {
  const id = useId()

  return <figure id="fa-causal-tiles" className="fa-causal-figure">
    <div className="fa-causal-header">
      <p className="fa-causal-title">先处理边界，再向左遍历</p>
      <span className="fa-causal-meta">N = 384</span>
    </div>
      <svg width="600" height="296" viewBox="0 0 600 296" role="img" aria-labelledby={`${id}-title ${id}-description`}>
        <title id={`${id}-title`}>Causal attention 的 tile 可见性与倒序遍历</title>
        <desc id={`${id}-description`}>
          序列长度为 384，Q tile 大小为 128，K tile 大小为 64。
          固定 Q 的第 128 至 255 行，K tile 0 和 1 全部可见，tile 2 和 3 需要逐元素 causal mask，tile 4 和 5 整个 tile 跳过。
          阶梯边界包含 j 等于 i 的对角线元素。n_block_max 为 4，从强调标出的 tile 3 开始，依次向左遍历 2、1、0。
        </desc>
        <defs>
          <marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M1 1L6 3.5L1 6" fill="none" stroke="var(--color-accent, #567891)" strokeWidth="1.2" />
          </marker>
        </defs>

        {Array.from({ length: 6 }, (_, tile) => <g key={tile}>
          <text x={matrixX + (tile + 0.5) * tileWidth} y="19" textAnchor="middle" className={`fa-causal-tile-label ${tile === 2 || tile === 3 ? 'fa-causal-focus-label' : tile >= 4 ? 'fa-causal-muted-label' : ''}`}>tile {tile}</text>
          <text x={matrixX + (tile + 0.5) * tileWidth} y="40" textAnchor="middle" className="fa-causal-range">{tile * 64}–{tile * 64 + 63}</text>
          <rect x={matrixX + tile * tileWidth} y={matrixY} width={tileWidth} height={matrixHeight} className={tile < 2 ? 'fa-causal-visible-fill' : 'fa-causal-hidden-fill'} />
        </g>)}

        <path d={validRegion} className="fa-causal-partial-fill" />
        <path d={diagonal} className="fa-causal-diagonal" />

        {Array.from({ length: 6 }, (_, tile) => <rect
          key={tile}
          x={matrixX + tile * tileWidth}
          y={matrixY}
          width={tileWidth}
          height={matrixHeight}
          className={`fa-causal-tile-border ${tile < 2 ? 'fa-causal-visible-border' : tile < 4 ? 'fa-causal-partial-border' : 'fa-causal-hidden-border'}`}
        />)}

        <text x="43" y="137" textAnchor="middle" className="fa-causal-query">Q</text>
        <text x="43" y="160" textAnchor="middle" className="fa-causal-range">128–255</text>
        {[0, 1].map(tile => <text key={tile} x={matrixX + (tile + 0.5) * tileWidth} y="153" textAnchor="middle" className="fa-causal-visible-label">全可见</text>)}
        <text x="292" y="88" textAnchor="middle" className="fa-causal-mask-label">mask</text>
        <text x="358" y="88" textAnchor="middle" className="fa-causal-mask-label">mask</text>
        <text x="286" y="207" textAnchor="middle" className="fa-causal-valid-label">j ≤ i</text>
        {[4, 5].map(tile => <g key={tile}>
          <path d={`M${matrixX + (tile + 0.5) * tileWidth - 8} 126l16 16m-16 0l16-16`} className="fa-causal-skip-mark" />
          <text x={matrixX + (tile + 0.5) * tileWidth} y="165" textAnchor="middle" className="fa-causal-skip-label">跳过</text>
        </g>)}

        {[3, 2, 1].map(tile => <path
          key={tile}
          d={`M${matrixX + (tile + 0.5) * tileWidth - (tile === 3 ? 22 : 17)} 270H${matrixX + (tile - 0.5) * tileWidth + 17}`}
          className="fa-causal-arrow"
          markerEnd={`url(#${id}-arrow)`}
        />)}
        <text x="43" y="275" textAnchor="middle" className="fa-causal-order-label">逆序</text>
        {[0, 1, 2].map(tile => <text key={tile} x={matrixX + (tile + 0.5) * tileWidth} y="276" textAnchor="middle" className="fa-causal-step-number">{tile}</text>)}
        <circle cx="358" cy="270" r="16" className="fa-causal-start" />
        <text x="358" y="276" textAnchor="middle" className="fa-causal-start-number">3</text>
        <text x="389" y="275" className="fa-causal-start-label">从这里开始</text>
      </svg>
    <figcaption>Q tile 包含 128 个 query，K tile 包含 64 个 key，对角线跨过两个需要 mask 的 K tile。</figcaption>
  </figure>
}
