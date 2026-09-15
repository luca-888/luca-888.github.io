import { useState } from 'react'

const initialRows = [[3, -1, -2, 1], [5, 2, 0, -2], [9, 5, -4, -9]]
const initialWeight = [0.8, 1.2, 0.5, 1.5]
const format = (value: number) => Number(value.toFixed(4)).toString()
const meanSquare = (values: number[]) => values.reduce((sum, x) => sum + x * x, 0) / values.length

function ValueBar({ value, limit }: { value: number; limit: number }) {
  const width = Math.min(Math.abs(value) / limit, 1) * 50
  return <span className="rms-value-track" aria-hidden="true">
    <span className="rms-value-fill" style={{ left: `${value < 0 ? 50 - width : 50}%`, width: `${width}%` }} />
  </span>
}

function ValueSlider({ value, limit, step, label, prefix = '', onChange }: {
  value: number; limit: number; step: number; label: string; prefix?: string; onChange: (value: number) => void
}) {
  return <label className="rms-value rms-editable">
    <output>{prefix}{format(value)}</output>
    <span className="rms-slider">
      <ValueBar value={value} limit={limit} />
      <input type="range" min={-limit} max={limit} step={step} value={value}
        aria-label={label} onChange={event => onChange(Number(event.target.value))} />
    </span>
  </label>
}

export function RmsNormDemo() {
  const [rows, setRows] = useState(initialRows)
  const [weight, setWeight] = useState(initialWeight)
  const [eps, setEps] = useState(1e-6)
  const results = rows.map(row => {
    const divisor = Math.sqrt(meanSquare(row) + eps)
    const output = row.map((value, i) => value / divisor * weight[i])
    return { divisor, output, rms: Math.sqrt(meanSquare(output)) }
  })

  function reset() {
    setRows(initialRows)
    setWeight(initialWeight)
    setEps(1e-6)
  }

  return (
    <section className="rms-demo" id="rmsnorm-demo" aria-labelledby="demo-title">
      <header className="demo-header">
        <h3 id="demo-title">先归一化，再乘 γ</h3>
        <div className="demo-toolbar">
          <label>ε = <select aria-label="epsilon" value={eps} onChange={event => setEps(Number(event.target.value))}>
            <option value={1e-6}>10⁻⁶</option><option value={0.1}>0.1</option><option value={1}>1</option>
          </select></label>
          <button type="button" onClick={reset}>重置</button>
        </div>
      </header>
      <p className="demo-intro">X[3, 4]：每行一个 token。拖动输入或 γ 数字下方的滑条。</p>

      <div className="rms-matrix-scroll" tabIndex={0} role="region" aria-label="RMSNorm 逐行计算，可横向滚动">
        <div className="rms-matrix">
          <div className="rms-matrix-head">
            <span>m</span>
            <div className="rms-input-heading">输入 X<small>n = 0, 1, 2, 3</small></div>
            <div>除数<small>√(mean(x²) + ε)</small></div>
            <div className="rms-output-heading">输出 Y<small>x / 除数 × γ</small></div>
            <div>输出<small>RMS</small></div>
          </div>

          <div className="rms-weight-row">
            <div className="rms-weight-label">权重 γ<small>所有 token 共享</small></div>
            <div className="rms-vector rms-shared-weight" role="group" aria-label="共享权重 γ">
              {weight.map((value, i) => <ValueSlider key={i} value={value} limit={2} step={0.1} prefix="× "
                label={`共享权重 γ${i}`} onChange={next => setWeight(current => current.map((x, col) => col === i ? next : x))} />)}
            </div>
          </div>

          {rows.map((row, m) => (
            <div className="rms-token-row" key={m} role="group" aria-label={`token ${m}`}>
              <span className="rms-token-index">{m}</span>
              <div className="rms-vector rms-input-vector">
                {row.map((value, n) => <ValueSlider key={n} value={value} limit={20} step={0.5}
                  label={`token ${m} 输入 x${n}`} onChange={next => setRows(current => current.map((values, index) =>
                    index === m ? values.map((x, col) => col === n ? next : x) : values))} />)}
              </div>
              <div className="rms-divisor"><span>÷</span> <output>{format(results[m].divisor)}</output> <span>→</span></div>
              <div className="rms-vector rms-output-vector" data-testid={`output-${m}`}>
                {results[m].output.map((value, n) => <div className="rms-value" key={n}>
                  <output aria-label={`token ${m} 输出 y${n}`}>{format(value)}</output>
                  <ValueBar value={value} limit={4} />
                </div>)}
              </div>
              <output className="rms-output-rms" aria-label={`token ${m} 输出 RMS`}>{format(results[m].rms)}</output>
            </div>
          ))}
          <div className="rms-matrix-foot" aria-hidden="true">
            <span>输入范围 −20 ～ 20</span><span>输出条刻度 −4 ～ 4</span>
          </div>
        </div>
      </div>
      <p className="demo-note">示例数据，显示保留四位小数；输出 RMS = √mean(y²)。<span className="rms-scroll-hint">左右滑动查看完整一行。</span></p>
    </section>
  )
}
