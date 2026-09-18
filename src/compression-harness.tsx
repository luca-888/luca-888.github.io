import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import article from '../content/compression-harness.md?raw'
import { SiteHeader, SiteFooter } from './SiteLayout'
import { CompressionOverview } from './CompressionOverview'
import latentDiffusion from './assets/compression-harness/latent-diffusion.png'
import trainingStages from './assets/compression-harness/training-stages.png'
import './styles.css'
import './site.css'
import './compression-harness.css'

const illustrations: Record<string, string> = {
  '../src/assets/compression-harness/latent-diffusion.png': latentDiffusion,
  '../src/assets/compression-harness/training-stages.png': trainingStages,
}

function CompressionArticle() {
  return (
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main>
        <article className="article article-essay" id="article-content">
          <Markdown remarkPlugins={[remarkGfm]} components={{
            p({ children, node }) {
              const child = node?.children[0]
              if (node?.children.length === 1 && child?.type === 'element' && child.tagName === 'img') {
                return <figure className="compression-overview">{children}<figcaption>{String(child.properties.alt ?? '')}</figcaption></figure>
              }
              return <p>{children}</p>
            },
            img({ src, alt }) {
              return <img src={illustrations[src ?? ''] ?? src} alt={alt} width={1536} height={1024} loading="lazy" />
            },
            h2({ children }) {
              return <>{children === '一、压缩与泛化' && <CompressionOverview />}<h2>{children}</h2></>
            },
            table({ children }) {
              return <div className="table-scroll" tabIndex={0} role="region" aria-label="任务瓶颈与改进方式"><table>{children}</table></div>
            },
          }}>{article}</Markdown>
        </article>
      </main>
      <SiteFooter />
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><CompressionArticle /></StrictMode>)
