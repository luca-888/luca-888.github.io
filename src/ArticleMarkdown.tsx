import type { CSSProperties } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { createHighlighterCoreSync } from 'shiki/core'
import { createJavaScriptRegexEngine } from 'shiki/engine/javascript'
import python from 'shiki/langs/python.mjs'
import cpp from 'shiki/langs/cpp.mjs'
import githubLight from 'shiki/themes/github-light.mjs'
import githubDark from 'shiki/themes/github-dark.mjs'

const highlighter = createHighlighterCoreSync({
  themes: [githubLight, githubDark],
  langs: [python, cpp],
  engine: createJavaScriptRegexEngine(),
})

function highlightCode(source: string, lang: 'python' | 'cpp') {
  const { tokens } = highlighter.codeToTokens(source.replace(/\n$/, ''), { lang, themes: { light: 'github-light', dark: 'github-dark' } })
  return tokens.map((line, lineIndex) => <span className="code-line" key={lineIndex}>
    {line.map((token, tokenIndex) => <span key={tokenIndex} style={{ color: token.color, ...token.htmlStyle } as CSSProperties}>{token.content}</span>)}
    {'\n'}
  </span>)
}

export function highlightPython(source: string) {
  return highlightCode(source, 'python')
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
      if (className === 'language-cpp') return <code className={className}>{highlightCode(String(children), 'cpp')}</code>
      if (className !== 'language-python') return <code className={className}>{children}</code>
      return <code className={className}>{highlightPython(String(children))}</code>
    },
    ...components,
  }}>{markdown}</Markdown>
}
