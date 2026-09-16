import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { createHighlighterCoreSync } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'
import python from 'shiki/langs/python.mjs'
import githubLight from 'shiki/themes/github-light.mjs'

const highlighter = createHighlighterCoreSync({
  themes: [githubLight],
  langs: [python],
  engine: createJavaScriptRegexEngine(),
})

export function highlightPython(source: string) {
  const { tokens } = highlighter.codeToTokens(source.replace(/\n$/, ''), { lang: 'python', theme: 'github-light' })
  return tokens.map((line, lineIndex) => <span className="code-line" key={lineIndex}>
    {line.map((token, tokenIndex) => <span key={tokenIndex} style={{ color: token.color }}>{token.content}</span>)}
    {'\n'}
  </span>)
}

export function ArticleMarkdown({ source, components }: { source: string; components?: Components }) {
  // Normalize LaTeX delimiters outside fenced and inline code.
  const markdown = source.split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, index) => index % 2 ? part : part
      .replace(/\\\[([\s\S]*?)\\\]/g, (_, math: string) => `\n$$\n${math.trim()}\n$$\n`)
      .replace(/\\\((.*?)\\\)/g, (_, math: string) => `$${math}$`))
    .join('')

  return <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{
    table({ children }) {
      return <div className="table-scroll" tabIndex={0} role="region" aria-label="数据表"><table>{children}</table></div>
    },
    code({ children, className }) {
      if (className !== 'language-python') return <code className={className}>{children}</code>
      return <code className={className}>{highlightPython(String(children))}</code>
    },
    ...components,
  }}>{markdown}</Markdown>
}
