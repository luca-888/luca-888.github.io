import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from '../../ArticleMarkdown'
import { SiteHeader, SiteFooter, ArticleVideo } from '../../SiteLayout'
import { attentionKvCompression as post } from './attention-kv-compression.blocks'
import 'katex/dist/katex.min.css'
import '../../styles.css'
import '../../site.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main>
        <article className="article" id="article-content">
          <h1>{post.title}</h1>
          <p className="article-subtitle">{post.subtitle}</p>
          <ArticleVideo slug="attention-kv-compression" />
          <ArticleMarkdown source={post.source} components={post.components} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
