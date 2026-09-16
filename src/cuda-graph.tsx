import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from './ArticleMarkdown'
import { CudaGraphTrace } from './CudaGraphFigures'
import { CudaGraphLifecycle, CudaGraphStructure } from './CudaGraphMechanics'
import { SiteHeader, SiteFooter } from './SiteLayout'
import article from '../content/cuda-graph.md?raw'
import 'katex/dist/katex.min.css'
import './styles.css'
import './site.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main>
        <article className="article" id="article-content">
          <ArticleMarkdown source={article} components={{
            p({ children }) {
              if (children === '::cuda-graph-trace::') return <CudaGraphTrace />
              if (children === '::cuda-graph-lifecycle::') return <CudaGraphLifecycle />
              if (children === '::cuda-graph-structure::') return <CudaGraphStructure />
              return <p>{children}</p>
            },
          }} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
