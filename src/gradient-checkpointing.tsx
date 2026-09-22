import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from './ArticleMarkdown'
import { GcOfficialFigure, GcOfficialBenchmark } from './GradientCheckpointingFigures'
import lifecycle from './assets/gradient-checkpointing/lifecycle-imagegen-v2.png'
import { SiteHeader, SiteFooter } from './SiteLayout'
import article from '../content/gradient-checkpointing.md?raw'
import 'katex/dist/katex.min.css'
import './styles.css'
import './site.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main><article className="article" id="article-content">
        <ArticleMarkdown source={article} components={{ p({ children }) {
          if (children === '::gc-concept::') return <figure className="gc-concept"><img src={lifecycle} alt="Eager 持续保存内部 activation 到 backward；GC 在 forward 后释放内部结果，backward 时重算，同时保留边界输入。" /></figure>
          if (children === '::gc-official-sac::') return <GcOfficialFigure selective />
          if (children === '::gc-official-benchmark::') return <GcOfficialBenchmark />
          return <p>{children}</p>
        } }} />
      </article></main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
