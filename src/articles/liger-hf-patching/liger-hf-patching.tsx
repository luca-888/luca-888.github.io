import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from '../../ArticleMarkdown'
import { SiteHeader, SiteFooter, ArticleVideo } from '../../SiteLayout'
import { ligerHfPatching as post } from './liger-hf-patching.blocks'
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
          <ArticleVideo slug="liger-hf-patching" />
          <ArticleMarkdown source={post.source} components={post.components} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
