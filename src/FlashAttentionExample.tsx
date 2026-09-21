import { renderToString } from 'katex'
import './FlashAttentionExample.css'

const formulas = {
  max1: String.raw`m=\ln2`,
  max2: String.raw`m=\ln8`,
  weights1: String.raw`\widetilde p_1=\begin{bmatrix}e^{0-\ln2}&e^{\ln2-\ln2}\end{bmatrix}=\begin{bmatrix}\frac12&1\end{bmatrix}`,
  weights2: String.raw`\widetilde p_2=\begin{bmatrix}e^{\ln4-\ln8}&e^{\ln8-\ln8}\end{bmatrix}=\begin{bmatrix}\frac12&1\end{bmatrix}`,
  sum: String.raw`\frac12+1=\frac32`,
  value1: String.raw`\begin{bmatrix}\frac12&1\end{bmatrix}\begin{bmatrix}1&0\\0&1\end{bmatrix}=\begin{bmatrix}\frac12&1\end{bmatrix}`,
  value2: String.raw`\begin{bmatrix}\frac12&1\end{bmatrix}\begin{bmatrix}2&0\\0&2\end{bmatrix}=\begin{bmatrix}1&2\end{bmatrix}`,
  scale: String.raw`\alpha=e^{\ln2-\ln8}=\frac14`,
  ell: String.raw`\ell`,
  u: String.raw`U`,
  oldSum: String.raw`\frac14\times\frac32=\frac38`,
  oldValue: String.raw`\frac14\begin{bmatrix}\frac12&1\end{bmatrix}=\begin{bmatrix}\frac18&\frac14\end{bmatrix}`,
  newSum: String.raw`\frac32`,
  newValue: String.raw`\begin{bmatrix}1&2\end{bmatrix}`,
  totalSum: String.raw`\frac{15}{8}`,
  totalValue: String.raw`\begin{bmatrix}\frac98&\frac94\end{bmatrix}`,
  output: String.raw`O=\frac{U}{\ell}=\begin{bmatrix}\frac{9/8}{15/8}&\frac{9/4}{15/8}\end{bmatrix}=\begin{bmatrix}\frac35&\frac65\end{bmatrix}`,
  check: String.raw`\operatorname{softmax}(aS)V=\frac1{15}\begin{bmatrix}1&2&4&8\end{bmatrix}\begin{bmatrix}1&0\\0&1\\2&0\\0&2\end{bmatrix}=\begin{bmatrix}\frac35&\frac65\end{bmatrix}`,
}
const rendered = Object.fromEntries(Object.entries(formulas).map(([key, tex]) =>
  [key, { __html: renderToString(tex, { throwOnError: true }) }],
))
function Math({ name }: { name: keyof typeof formulas }) {
  return <span className="fa-example-math" dangerouslySetInnerHTML={rendered[name]} />
}

export function FlashAttentionExample() {
  return <figure className="fa-example" aria-label="两个 tile 的 Attention 累加示例">
    <div className="fa-example-stages">
      <section className="fa-example-stage fa-example-old">
        <header><strong>Tile 1 · 建立状态</strong><Math name="max1" /></header>
        <p className="fa-example-label">分数减去最大值，再取指数</p>
        <div className="fa-example-formula"><Math name="weights1" /></div>
        <div className="fa-example-contribution">
          <p className="fa-example-label">权重求和</p><Math name="sum" />
          <p className="fa-example-label">权重 × V 子矩阵</p><Math name="value1" />
        </div>
      </section>
      <section className="fa-example-stage fa-example-new">
        <header><strong>Tile 2 · 计算新贡献</strong><Math name="max2" /></header>
        <p className="fa-example-label">使用更新后的最大值</p>
        <div className="fa-example-formula"><Math name="weights2" /></div>
        <div className="fa-example-contribution">
          <p className="fa-example-label">权重求和</p><Math name="sum" />
          <p className="fa-example-label">权重 × V 子矩阵</p><Math name="value2" />
        </div>
      </section>
    </div>
    <div className="fa-example-merge-heading">
      <strong>旧状态缩放 + 新贡献 = 两个 tile 的累计值</strong>
      <span>最大值改变，Tile 1 的两项累计值都乘以 <Math name="scale" /></span>
    </div>
    <div className="table-scroll fa-example-merge" tabIndex={0} role="region" aria-label="旧状态缩放与新贡献的累加">
      <table>
        <thead><tr><th>状态</th><th>Tile 1 × 换算系数</th><th>+ Tile 2 贡献</th><th>= 累计结果</th></tr></thead>
        <tbody>
          <tr><th scope="row"><Math name="ell" /></th><td><Math name="oldSum" /></td><td><Math name="newSum" /></td><td><Math name="totalSum" /></td></tr>
          <tr><th scope="row"><Math name="u" /></th><td><Math name="oldValue" /></td><td><Math name="newValue" /></td><td><Math name="totalValue" /></td></tr>
        </tbody>
      </table>
    </div>
    <div className="fa-example-output">
      <div><strong>最后统一归一化</strong><p>向量的每个分量除以同一个分母</p></div>
      <Math name="output" />
    </div>
    <div className="fa-example-check">
      <p>核对：与一次性 Softmax 计算结果相同</p>
      <div className="fa-example-formula"><Math name="check" /></div>
    </div>
  </figure>
}
