import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { createHighlighterCoreSync } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'
import python from 'shiki/langs/python.mjs'
import githubLight from 'shiki/themes/github-light.mjs'
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

const highlighter = createHighlighterCoreSync({
  themes: [githubLight],
  langs: [python],
  engine: createJavaScriptRegexEngine(),
})

function highlightPython(source: string) {
  const { tokens } = highlighter.codeToTokens(source.replace(/\n$/, ''), { lang: 'python', theme: 'github-light' })
  return tokens.map((line, lineIndex) => <span className="code-line" key={lineIndex}>
    {line.map((token, tokenIndex) => <span key={tokenIndex} style={{ color: token.color }}>{token.content}</span>)}
    {'\n'}
  </span>)
}

const compiledSources = [
  { label: 'Forward 原始 Triton kernel', source: compiledForward,
    mapping: 'in_ptr0 → x，in_ptr1 → gamma，in_out_ptr0 → r，out_ptr0 → y。' },
  { label: 'Backward 原始 Triton kernels', source: compiledBackward,
    mapping: 'dgamma kernel：in_ptr0 → g，in_ptr1 → x，in_ptr2 → r。dx kernel：in_ptr0 → g，in_ptr1 → gamma，in_ptr2 → x，in_ptr3 → r。各自的 out_ptr1 指向对应梯度输出。' },
].map(item => ({ ...item, code: highlightPython((item.source.match(/@triton\.jit\n[\s\S]*?(?=\n''', device_str=)/g) ?? []).join('\n\n')) }))

// The article uses LaTeX delimiters; remark-math expects dollar delimiters.
// Leave fenced code and inline code unchanged.
const markdown = article
  .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
  .map((part, index) => index % 2 ? part : part
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => `\n$$\n${math.trim()}\n$$\n`)
    .replace(/\\\((.*?)\\\)/g, (_, math: string) => `$${math}$`))
  .join('')

export function App() {
  return (
    <main>
      <article className="article">
        <Markdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
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
            table({ children }) {
              return <div className="table-scroll" tabIndex={0} role="region" aria-label="数据表"><table>{children}</table></div>
            },
            code({ children, className }) {
              if (className !== 'language-python') return <code className={className}>{children}</code>
              return <code className={className}>{highlightPython(String(children))}</code>
            },
          }}
        >{markdown}</Markdown>
      </article>
    </main>
  )
}
