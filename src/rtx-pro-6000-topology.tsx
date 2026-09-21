import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from './ArticleMarkdown'
import { RtxPairCommunication } from './RtxCommunicationCharts'
import { RtxMeasuredTopology } from './RtxMeasuredTopology'
import directData from '../docs/data/rtx-pro-6000-20260921-full.json'
import { RtxReferenceTopology } from './RtxReferenceTopology'
import { SiteHeader, SiteFooter } from './SiteLayout'
import article from '../content/rtx-pro-6000-topology.md?raw'
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
              if (children === '::rtx-bandwidth::') return <RtxPairCommunication {...directData} bandwidthOnly />
              if (children === '::rtx-topology::') return <RtxMeasuredTopology devices={directData.devices} />
              if (children === '::rtx-reference::') return <RtxReferenceTopology />
              if (typeof children === 'string' && children.startsWith('【编辑占位：')) return null
              return <p>{children}</p>
            },
          }} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
