import Markdown from 'react-markdown'
import { ArticleMarkdown, highlightPython } from './ArticleMarkdown'
import article from '../content/rmsnorm.md?raw'
import eagerKernels from '../content/rmsnorm-eager-kernels.md?raw'
import compiledForward from '../public/measurements/rmsnorm-compile/4096x8192/forward.py?raw'
import compiledBackward from '../public/measurements/rmsnorm-compile/4096x8192/backward.py?raw'
import { RmsNormDemo } from './RmsNormDemo'
import { RmsNormGraph } from './RmsNormGraph'
import { RmsNormBenchmark } from './RmsNormBenchmark'
import { RmsNormTimeline } from './RmsNormTimeline'
import { RmsNormCompileGraph } from './RmsNormCompileGraph'
import { RmsNormCompileBenchmark } from './RmsNormCompileBenchmark'
import { RmsNormTritonBenchmark } from './RmsNormTritonBenchmark'
import { RmsNormTritonGraph } from './RmsNormTritonGraph'

const compiledSources = [
  { label: 'Forward 原始 Triton kernel', source: compiledForward,
    mapping: 'in_ptr0 → x，in_ptr1 → gamma，in_out_ptr0 → r，out_ptr0 → y。' },
  { label: 'Backward 原始 Triton kernels', source: compiledBackward,
    mapping: 'dgamma kernel：in_ptr0 → g，in_ptr1 → x，in_ptr2 → r。dx kernel：in_ptr0 → g，in_ptr1 → gamma，in_ptr2 → x，in_ptr3 → r。各自的 out_ptr1 指向对应梯度输出。' },
].map(item => ({ ...item, code: highlightPython((item.source.match(/@triton\.jit\n[\s\S]*?(?=\n''', device_str=)/g) ?? []).join('\n\n')) }))

export function App() {
  return (
    <main>
      <article className="article" id="article-content">
        <ArticleMarkdown
          source={article}
          components={{
            p({ children }) {
              if (children === '::rmsnorm-demo::') return <RmsNormDemo />
              if (children === '::rmsnorm-graph::') return <RmsNormGraph />
              if (children === '::rmsnorm-benchmark::') return <RmsNormBenchmark />
              if (children === '::rmsnorm-timeline::') return <RmsNormTimeline />
              if (children === '::rmsnorm-eager-ops::') return <div className="compile-source">
                <details>
                  <summary>完整算子清单 · Forward 9 个 / Backward 17 个 kernel</summary>
                  <Markdown>{eagerKernels}</Markdown>
                </details>
              </div>
              if (children === '::rmsnorm-compile-graph::') return <RmsNormCompileGraph />
              if (children === '::rmsnorm-compile-benchmark::') return <RmsNormCompileBenchmark />
              if (children === '::rmsnorm-triton-benchmark::') return <RmsNormTritonBenchmark />
              if (children === '::rmsnorm-triton-graph::') return <RmsNormTritonGraph />
              if (children === '::rmsnorm-triton-artifacts::') return <p className="compile-note">
                <a href={`${import.meta.env.BASE_URL}measurements/rmsnorm-fused/kernel.py`} download>完整手写 kernel</a>
                {' · '}<a href={`${import.meta.env.BASE_URL}measurements/rmsnorm-fused/results.json`} download>原始测量数据</a>
                {' · '}<a href={`${import.meta.env.BASE_URL}measurements/rmsnorm-fused.zip`} download>完整实验记录</a>
              </p>
              if (children === '::rmsnorm-compile-source::') return <div className="compile-source">
                {compiledSources.map(item => <details key={item.label}>
                  <summary>{item.label}</summary>
                  <p>原样节选 kernel 函数，保留变量名、索引与循环。完整模块可通过图下链接下载。</p>
                  <p className="compile-source-mapping">{item.mapping}</p>
                  <pre><code className="language-python">{item.code}</code></pre>
                </details>)}
              </div>
              return <p>{children}</p>
            },
          }}
        />
      </article>
    </main>
  )
}
