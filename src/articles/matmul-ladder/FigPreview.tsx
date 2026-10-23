// 仅本地预览重绘的插图（不在 vite.config.ts 的 build 入口里，不会发布）。删除本文件与 posts/_figs-preview/ 即撤掉。
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from '../../ArticleMarkdown'
import { SiteHeader, SiteFooter } from '../../SiteLayout'
import { matmulComponents } from './blocks'
import 'katex/dist/katex.min.css'
import '../../styles.css'
import '../../site.css'
import './matmul-ladder.css'

const source = `### ml-outer

::ml-outer::

### ml-hierarchy2

::ml-hierarchy2::

### ml-coalesce2

::ml-coalesce2::

### ml-smem

::ml-smem::

### ml-bank

::ml-bank::

### ml-eighth

::ml-eighth::

### ml-regtile2

::ml-regtile2::

### ml-tc-rate

::ml-tc-rate::

### ml-path:wmma

::ml-path:wmma::

### ml-swizzle2

::ml-swizzle2::

### ml-path:wgmma

::ml-path:wgmma::

### ml-wgmma-size

::ml-wgmma-size::

### ml-ws

::ml-ws::

### ml-tile-order2

::ml-tile-order2::

### ml-cluster2

::ml-cluster2::

### ml-path:tstore

::ml-path:tstore::
`

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <SiteHeader />
      <main>
        <article className="article" id="article-content">
          <h1>插图预览</h1>
          <ArticleMarkdown source={source} components={matmulComponents} />
        </article>
      </main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
