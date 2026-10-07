import './CudaGraphFigures.css'

const traceBase = `${import.meta.env.BASE_URL}measurements/cuda-graph/profiling/`

export function CudaGraphTrace() {
  return (
    <div className="cuda-graph-traces">
      {[
        { file: 'diamond-ordinary.png', title: '多 stream eager', detail: '4 次 kernel launch', height: 351 },
        { file: 'diamond-replay.png', title: 'Graph replay · GPU 执行区间', detail: 'A → B / C → D', height: 202 },
      ].map(({ file, title, detail, height }) => (
        <figure className="cuda-graph-trace" key={file}>
          <figcaption><strong>{title}</strong><span>{detail}</span></figcaption>
          <img src={`${traceBase}${file}`} width="1106" height={height} loading="lazy" alt={`Nsight Systems diamond 原生时间轴：${title}，${detail}`} />
        </figure>
      ))}
      <p className="cuda-graph-trace-note">两图横轴比例不同。下图聚焦 GPU 执行，cudaGraphLaunch 位于该时间窗口之前。</p>
    </div>
  )
}
