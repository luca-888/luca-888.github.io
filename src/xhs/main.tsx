import { useEffect, useRef } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from '../ArticleMarkdown'
import { CompactContext } from '../figure-kit'
import { articles } from './articles'
import { paginate, type XhsResult } from './paginate'
import 'katex/dist/katex.min.css'
import '../styles.css'
import './xhs.css'

declare global { interface Window { __xhs?: XhsResult & { ready: boolean } } }

const slug = new URLSearchParams(location.search).get('slug') ?? ''
const article = articles[slug]

function App() {
  const sourceRef = useRef<HTMLDivElement>(null)
  const pagesRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    (async () => {
      await document.fonts.ready
      await Promise.all([...document.images].map(img => img.decode().catch(() => undefined)))
      await new Promise(resolve => setTimeout(resolve, 200))
      // 图片里链接无法点击，默认去掉“参考”一节。
      const source = sourceRef.current!
      const refs = [...source.querySelectorAll('h2')].find(h => h.textContent?.trim() === '参考')
      while (refs?.nextElementSibling && refs.nextElementSibling.tagName !== 'H2') refs.nextElementSibling.remove()
      refs?.remove()
      const headerTitle = article.subtitle ? `${article.title} · ${article.subtitle}` : article.title
      window.__xhs = { ...paginate(sourceRef.current!, pagesRef.current!, headerTitle), ready: true }
    })()
  }, [])
  if (!article) return <p className="xhs-missing">未知文章：{slug || '（缺少 ?slug=）'}。可用：{Object.keys(articles).join('、')}</p>
  return <>
    <div className="xhs-pages" ref={pagesRef} />
    <div className="xhs-source xhs-body" ref={sourceRef}>
      <h1>{article.title}</h1>
      {article.subtitle && <p className="xhs-subtitle">{article.subtitle}</p>}
      <CompactContext.Provider value={true}>
        <ArticleMarkdown source={article.source} components={article.components} />
      </CompactContext.Provider>
    </div>
  </>
}

createRoot(document.getElementById('root')!).render(<App />)
