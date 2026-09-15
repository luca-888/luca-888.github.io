import { useState } from 'react'
import eager from '../content/data/rmsnorm-eager.json'
import compiled from '../content/data/rmsnorm-compile.json'
import recordUrl from '../content/data/rmsnorm-compile.json?url'
import './RmsNormCompile.css'

const rows = compiled.measurements.map(row => ({
  ...row,
  eager: eager.measurements.find(item => item.dtype === row.dtype && item.M === row.M && item.N === row.N)!,
  eagerMemory: eager.memory.measurements.find(item => item.M === row.M && item.N === row.N)!,
}))
const directions = ['forward', 'backward'] as const

export function RmsNormCompileBenchmark() {
  const [direction, setDirection] = useState<typeof directions[number]>('forward')
  const label = direction === 'forward' ? 'Forward' : 'Backward'

  return (
    <section className="compile-benchmark" aria-label="eager 与 torch.compile 性能对比">
      <div className="compile-toolbar">
        <span>RTX 4090 · BF16</span>
        <div className="compile-switch" role="group" aria-label="性能对比方向">
          {directions.map(value => <button type="button" key={value}
            aria-pressed={direction === value} onClick={() => setDirection(value)}>
            {value === 'forward' ? 'Forward' : 'Backward'}
          </button>)}
        </div>
      </div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label={`${label} 性能与显存对比`}>
        <table className="benchmark-table compile-table">
          <caption>{label} <span>· eager / compile</span></caption>
          <thead>
            <tr>
              <th rowSpan={2} scope="col">M</th>
              <th rowSpan={2} scope="col">N</th>
              <th colSpan={2} scope="colgroup" className="benchmark-time">耗时 <span>μs</span></th>
              <th rowSpan={2} scope="col">加速比</th>
              <th colSpan={2} scope="colgroup" className="benchmark-memory">峰值显存增量 <span>MiB</span></th>
            </tr>
            <tr>
              <th scope="col">eager</th><th scope="col">compile</th>
              <th scope="col">eager</th><th scope="col">compile</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => <tr key={`${row.M}-${row.N}`}>
              <td>{row.M}</td><td>{row.N}</td>
              <td>{row.eager[direction].median_us.toFixed(1)}</td>
              <td>{row[direction].median_us.toFixed(1)}</td>
              <td>{(row.eager[direction].median_us / row[direction].median_us).toFixed(2)}×</td>
              <td>{(row.eagerMemory[direction].peak_increment_bytes / 2 ** 20).toFixed(2)}</td>
              <td>{(row[direction].memory.peak_increment_bytes / 2 ** 20).toFixed(2)}</td>
            </tr>)}
          </tbody>
        </table>
      </div>
      <p className="compile-note">加速比 = eager / compile。<a href={recordUrl} download="rmsnorm-compile.json">原始测量记录</a></p>
    </section>
  )
}
