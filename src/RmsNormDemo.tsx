import { useState } from 'react';
import { computeRmsNorm, formatValue as fmt, rms } from './rmsnorm-math';
import './rmsnorm-demo.css';
import { exampleInputs, exampleWeights } from './rmsnorm-example';

const inputLimit = 20;
const outputLimit = 3;

function MatrixCell({ value, limit, stage, row, column }: {
  value: number; limit: number; stage: 'input' | 'output'; row: number; column: number;
}) {
  const width = Math.min(50, Math.abs(value) / limit * 50);
  return <div className={`matrix-cell matrix-${stage}`}>
    <output className="matrix-value" data-stage-value={stage} data-row={row} data-column={column} data-value={value}>{fmt(value)}</output>
    <div className="cell-meter" aria-hidden="true"><span className="cell-zero" /><span className="cell-fill" style={{ width: `${width}%`, left: value < 0 ? `${50 - width}%` : '50%' }} /></div>
  </div>;
}

export function RmsNormDemo() {
  const [scale, setScale] = useState(1);
  const [activeRow, setActiveRow] = useState<number | null>(null);
  const results = exampleInputs.map((row) => computeRmsNorm(row, exampleWeights, scale, 1e-6));

  return (
    <section className="rms-demo" aria-labelledby="demo-title">
      <header className="demo-header"><h2 id="demo-title">先归一化，再乘 γ</h2><span>ε = 10⁻⁶</span></header>
      <div className="demo-controls">
        <label htmlFor="demo-scale">输入倍数</label>
        <input id="demo-scale" type="range" min="-2" max="2" step="0.01" value={scale} onChange={(event) => setScale(Number(event.target.value))} />
        <output className="demo-multiplier">×{scale.toFixed(2)}</output>
      </div>
      <p className="demo-data-note">X[M, N]：沿 N 计算 RMS，γ[N] 沿 M 共享。</p>
      <div className="matrix-flow" role="table" aria-label="RMSNorm 逐行计算：输入除以每行尺度，再逐特征乘权重，得到输出">
        <div className="matrix-head flow-columns" role="row">
          <span role="columnheader">m</span>
          <span role="columnheader" className="input-heading">输入 X<span className="matrix-features">n = 0, 1, 2, 3</span></span>
          <span role="columnheader">除数<span className="matrix-features">√(RMS² + ε)</span></span>
          <span role="columnheader" className="output-heading">输出 Y<span className="matrix-features">沿 N 归一化后 × γ[n]</span></span>
          <span role="columnheader">输出<br />RMS</span>
        </div>
        <div className="matrix-weights flow-columns" role="row">
          <span role="rowheader">γ</span><span role="cell" /><span role="cell">沿 M 共享</span>
          <span className="weight-pair" role="cell">{exampleWeights.map((value, i) => <span key={i} title={`γ[${i}] = ${value}`}>× {value.toFixed(1)}</span>)}</span><span role="cell" />
        </div>
        <div className="matrix-rows" role="rowgroup" onMouseLeave={() => setActiveRow(null)}>
          {results.map((result, index) => <div className={`matrix-row flow-columns${activeRow === index ? ' is-active' : ''}`} key={index} role="row" onMouseEnter={() => setActiveRow(index)}>
            <span className="matrix-row-label" role="rowheader">{index}</span>
            <div className="matrix-pair" role="cell">
              {result.input.map((value, column) => <MatrixCell key={column} value={value} limit={inputLimit} stage="input" row={index} column={column} />)}

            </div>
            <div className="row-division" role="cell"><span aria-hidden="true">÷</span><output data-divisor-row={index} data-value={result.denominator}>{fmt(result.denominator)}</output><span aria-hidden="true">→</span></div>
            <div className="matrix-pair" role="cell">
              {result.output.map((value, column) => <MatrixCell key={column} value={value} limit={outputLimit} stage="output" row={index} column={column} />)}

            </div>
            <output className="row-rms" role="cell" data-stage="output" data-row={index} data-value={rms(result.output)}>{fmt(rms(result.output))}</output>
          </div>)}
        </div>
      </div>
      <div className="matrix-scales flow-columns" aria-label="两侧条形分别使用固定刻度">
        <span /><span>条长刻度：−20 ～ 20</span><span /><span>条长刻度：−3 ～ 3</span><span />
      </div>
      <p className="demo-source">输入与 γ 均为示意参数。</p>
    </section>
  );
}
