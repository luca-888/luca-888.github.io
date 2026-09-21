import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from './ArticleMarkdown'
import { FlashAttentionTiling, FlashAttentionBlocks, FlashAttentionWarps } from './FlashAttentionFigures'
import { FlashAttentionExecution } from './FlashAttentionExecution'
import { FlashAttentionCausalTiles } from './FlashAttentionCausalFigure'
import { FlashAttentionExample } from './FlashAttentionExample'
import { SiteHeader, SiteFooter } from './SiteLayout'
import article from '../content/flash-attention.md?raw'
import 'katex/dist/katex.min.css'
import './styles.css'
import './site.css'
import './flash-attention.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main>
        <article className="article fa-article" id="article-content">
          <ArticleMarkdown source={article} components={{
            p({ children }) {
              if (children === '::flash-attention-tiling::') return <FlashAttentionTiling />
              if (children === '::flash-attention-blocks::') return <FlashAttentionBlocks />
              if (children === '::flash-attention-warps::') return <FlashAttentionWarps />
              if (children === '::flash-attention-execution::') return <FlashAttentionExecution />
              if (children === '::flash-attention-causal-tiles::') return <FlashAttentionCausalTiles />
              if (children === '::flash-attention-example::') return <FlashAttentionExample />
              return <p>{children}</p>
            },
          }} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
