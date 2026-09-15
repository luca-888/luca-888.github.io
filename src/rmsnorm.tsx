import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { SiteHeader, SiteFooter } from './SiteLayout'
import 'katex/dist/katex.min.css'
import './styles.css'
import './site.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <App />
      <SiteFooter />
    </div>
  </StrictMode>,
)
