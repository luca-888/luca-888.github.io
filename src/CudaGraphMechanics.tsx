import { useId, useState } from 'react'
import structure from '../content/data/cuda-graph-structure.json'
import './CudaGraphMechanics.css'

const structureBase = `${import.meta.env.BASE_URL}measurements/cuda-graph/structure/`

function Arrow({ id }: { id: string }) {
  return <defs><marker id={id} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M1 1 L7 4 L1 7" fill="none" stroke="currentColor" strokeWidth="1.3" />
  </marker></defs>
}

export function CudaGraphLifecycle() {
  const id = useId()
  return (
    <figure className="cuda-mechanics">
      <svg viewBox="0 0 760 190" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>CUDA Graph 生命周期</title>
        <desc id={`${id}-desc`}>Definition 定义操作与依赖，本例通过 stream capture 得到图模板；Instantiation 验证并准备可执行图；Execution 阶段通过 replay 反复执行。图模板和可执行图是不同对象。</desc>
        <Arrow id={`${id}-arrow`} />
        <text x="258" y="18" className="cuda-muted">创建与准备 · 一次</text>
        <path d="M28 34 V29 H488 V34" className="cuda-guide" />
        <text x="624" y="18" className="cuda-green-text">每轮执行</text>
        <path d="M516 34 V29 H732 V34" className="cuda-guide" />
        {[
          { x: 28, title: 'Definition', action: '本例通过 capture 构图', object: 'cudaGraph_t' },
          { x: 272, title: 'Instantiation', action: '验证并准备执行', object: 'cudaGraphExec_t' },
          { x: 516, title: 'Execution', action: 'Replay：一次 launch', object: 'cudaGraphLaunch' },
        ].map(({ x, title, action, object }) => (
          <g key={title}>
            <rect x={x} y="47" width="216" height="84" rx="4" className={title === 'Execution' ? 'cuda-node-green' : 'cuda-node-blue'} />
            <text x={x + 108} y="69" className="cuda-emphasis">{title}</text>
            <text x={x + 108} y="93">{action}</text>
            <text x={x + 108} y="115" className="cuda-mono cuda-small">{object}</text>
          </g>
        ))}
        <path d="M246 89 H270" className="cuda-edge" markerEnd={`url(#${id}-arrow)`} />
        <path d="M490 89 H514" className="cuda-edge" markerEnd={`url(#${id}-arrow)`} />
        <path d="M705 134 V153 H546 V134" className="cuda-edge cuda-green-edge" markerEnd={`url(#${id}-arrow)`} />
        <text x="624" y="176" className="cuda-muted">复用同一个可执行图</text>
      </svg>
      <figcaption>生命周期示意：本例通过 stream capture 创建图模板，再实例化并反复执行。</figcaption>
    </figure>
  )
}

export function CudaGraphStructure() {
  const id = useId()
  const [selected, setSelected] = useState(0)
  const points = [{ x: 16, y: 90 }, { x: 284, y: 16 }, { x: 284, y: 164 }, { x: 552, y: 90 }]
  const positions = Object.fromEntries(structure.nodes.map((node, index) => [node.id, points[index]]))
  const active = structure.nodes[selected]
  return (
    <figure className="cuda-mechanics cuda-structure">
      <div className="cuda-mechanics-heading"><strong>本例捕获的真实 DAG</strong><span>{structure.nodes.length} 个 kernel · {structure.edges.length} 条依赖</span></div>
      <svg viewBox="0 0 760 248" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>从 CUDA 原始 DOT 提取的分叉与汇合图</title>
        <desc id={`${id}-desc`}>A 计算 x=2*static_input；B、C 分别计算 b=x+1、c=x+2，两者均依赖 A；D 计算 y=b+c，等待 B 和 C 都完成。B 与 C 之间没有依赖。</desc>
        <Arrow id={`${id}-arrow`} />
        {structure.edges.map(edge => {
          const from = positions[edge.from]
          const to = positions[edge.to]
          return <path key={`${edge.from}-${edge.to}`} d={`M${from.x + 192} ${from.y + 30} L${to.x - 4} ${to.y + 30}`} className="cuda-edge" markerEnd={`url(#${id}-arrow)`} />
        })}
        {structure.nodes.map((node, index) => {
          const { x, y } = positions[node.id]
          return <g key={node.id}>
            <rect x={x} y={y} width="192" height="60" rx="4" className={`${index === 2 ? 'cuda-node-green' : 'cuda-node-blue'} ${selected === index ? 'cuda-node-selected' : ''}`} />
            <text x={x + 96} y={y + 17} className="cuda-emphasis">{node.label} · NODE {node.node_id}</text>
            <text x={x + 96} y={y + 40} className="cuda-mono cuda-small">{node.operation}</text>
            <text x={x + 96} y={y + 73} className="cuda-muted">{index === 2 ? 'capture：side stream' : 'capture：origin stream'}</text>
          </g>
        })}
        <text x="380" y="120" className="cuda-muted">B、C 无相互依赖</text>
      </svg>
      <div className="cuda-node-picker" role="group" aria-label="查看节点原始信息">
        <span>原始节点信息</span>
        {structure.nodes.map((node, index) => <button key={node.id} type="button" aria-pressed={selected === index} onClick={() => setSelected(index)}>{node.label} · NODE {node.node_id}</button>)}
      </div>
      <div className="cuda-raw-node" aria-live="polite">
        <div className="cuda-raw-title">NODE {active.node_id}: KERNEL <span>{active.operation}</span></div>
        <dl>
          <div><dt>ID</dt><dd>{active.node_id} (topoId: {active.topo_id})</dd></div>
          <div><dt>Kernel</dt><dd className="cuda-raw-symbol">{active.kernel}</dd></div>
          <div><dt>Launch</dt><dd>{active.launch}</dd></div>
          <div><dt>Node handle</dt><dd>{active.node_handle}</dd></div>
          <div><dt>Func handle</dt><dd>{active.func_handle}</dd></div>
        </dl>
      </div>
      <figcaption>
        节点、依赖和原始字段取自与下方 Nsight trace 同一次捕获的 DOT；A–D 为正文标记。
        <a href={`${structureBase}README.md`} target="_blank" rel="noreferrer">字段说明</a> · <a href={`${structureBase}nodes.txt`} target="_blank" rel="noreferrer">全部节点文本</a> · <a href={`${structureBase}graph.dot`} download>原始 DOT</a> · <a href={`${structureBase}graph.svg`} target="_blank" rel="noreferrer">Graphviz 完整图</a>
      </figcaption>
    </figure>
  )
}
