import measured from '../content/data/rmsnorm-fused.json'
import './RmsNormTriton.css'

function median(values: number[]) {
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]
}

const rows = [
  { key: 'compile', label: 'torch.compile' },
  { key: 'handwritten', label: '手写 Triton' },
] as const

export function RmsNormTritonBenchmark() {
  return <section className="compile-benchmark triton-benchmark" aria-label="梯度融合性能与显存对比">
    <div className="table-scroll" tabIndex={0} role="region" aria-label="4096 × 8192 Backward 性能对比">
      <table className="benchmark-table triton-summary">
        <caption>Backward <span>· RTX 4090 · BF16 · 4096 × 8192</span></caption>
        <thead>
          <tr>
            <th rowSpan={2} scope="col">实现</th>
            <th colSpan={2} scope="colgroup" className="benchmark-time">耗时 <span>μs</span></th>
            <th rowSpan={2} scope="col" className="benchmark-memory">峰值显存增量 <span>MiB</span></th>
          </tr>
          <tr><th scope="col">GPU 执行</th><th scope="col">完整调用</th></tr>
        </thead>
        <tbody>{rows.map(({ key, label }) => <tr key={key}>
          <td>{label}</td>
          <td>{measured.selected_gpu_recheck[key].median_us.toFixed(1)}</td>
          <td>{median(measured.call_times[key]).toFixed(1)}</td>
          <td>{(measured.memory[key].peak_increment_bytes / 2 ** 20).toFixed(2)}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <p className="compile-note">本章 compile 基线来自本次融合实验，与第三章的独立测量略有波动。</p>
  </section>
}
