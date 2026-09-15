import { measurements, memory } from '../content/data/rmsnorm-eager.json'

const rows = measurements.filter(row => row.dtype === 'torch.bfloat16').map(row => ({
  ...row,
  memory: memory.measurements.find(item => item.M === row.M && item.N === row.N)!,
}))

export function RmsNormBenchmark() {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label="BF16 性能与显存基线">
      <table className="benchmark-table">
        <caption>RTX 4090 <span>· BF16</span></caption>
        <thead>
          <tr>
            <th rowSpan={2} scope="col">M</th>
            <th rowSpan={2} scope="col">N</th>
            <th colSpan={2} scope="colgroup" className="benchmark-time">耗时 <span>μs</span></th>
            <th colSpan={2} scope="colgroup" className="benchmark-memory">峰值显存增量 <span>MiB</span></th>
          </tr>
          <tr>
            <th scope="col">Forward</th>
            <th scope="col">Backward</th>
            <th scope="col">Forward</th>
            <th scope="col">Backward</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(row => (
            <tr key={`${row.M}-${row.N}`}>
              <td>{row.M}</td>
              <td>{row.N}</td>
              <td>{row.forward.median_us.toFixed(1)}</td>
              <td>{row.backward.median_us.toFixed(1)}</td>
              <td>{(row.memory.forward.peak_increment_bytes / 2 ** 20).toFixed(2)}</td>
              <td>{(row.memory.backward.peak_increment_bytes / 2 ** 20).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
