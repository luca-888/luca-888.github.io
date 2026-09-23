import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ArticleMarkdown } from './ArticleMarkdown'
import { SiteHeader, SiteFooter } from './SiteLayout'
import article from '../content/kl-nll-ce.md?raw'
import crossEntropy from './assets/kl-nll-ce/colah-cross-entropy.png'
import forwardKl from './assets/kl-nll-ce/forward-KL.png'
import reverseKl from './assets/kl-nll-ce/reverse-KL.png'
import 'katex/dist/katex.min.css'
import './styles.css'
import './site.css'
import './kl-nll-ce.css'

const figures = {
  '::colah-cross-entropy::': {
    src: crossEntropy, width: 873, height: 441, author: 'Christopher Olah',
    url: 'https://colah.github.io/posts/2015-09-Visual-Information/',
    alt: '同一消息分布使用不同编码时的平均编码长度：p 使用自身编码为 1.75 bits，使用 q 的编码为 2.375 bits；q 使用自身编码为 1.75 bits，使用 p 的编码为 2.25 bits。',
    caption: String.raw`原图使用 bits。图中的 $H_q(p)$ 对应本文的 $H(p,q)$：按 $p$ 的频率，使用 $q$ 的编码。`,
  },
  '::eric-forward-kl::': {
    src: forwardKl, width: 789, height: 365, author: 'Eric Jang',
    url: 'https://blog.evjang.com/2016/08/variational-bayes.html',
    alt: 'Forward KL：绿色单峰分布若漏掉蓝色目标的另一个峰，会受到较大惩罚；变宽后覆盖两个峰。',
    caption: String.raw`固定蓝色 $P$，优化绿色单峰 $Q$；比较 $D_{\mathrm{KL}}(P\|Q)$。`,
  },
  '::eric-reverse-kl::': {
    src: reverseKl, width: 740, height: 393, author: 'Eric Jang',
    url: 'https://blog.evjang.com/2016/08/variational-bayes.html',
    alt: 'Reverse KL：绿色近似分布在蓝色目标的低密度区域分配概率会受惩罚；集中到一个峰附近可能减小 loss。',
    caption: String.raw`保持同样的目标与近似族，改为比较 $D_{\mathrm{KL}}(Q\|P)$。`,
  },
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="site-shell site-article-shell">
      <a className="skip-link" href="#article-content">跳到正文</a>
      <SiteHeader />
      <main><article className="article" id="article-content">
        <ArticleMarkdown source={article} components={{ p({ children }) {
          const figure = typeof children === 'string' ? figures[children as keyof typeof figures] : undefined
          if (!figure) return <p>{children}</p>
          return <figure className="information-source-figure" style={{ maxWidth: figure.width }}>
            <a href={figure.src} target="_blank" rel="noreferrer" aria-label={`查看 ${figure.author} 原图`}>
              <img src={figure.src} alt={figure.alt} width={figure.width} height={figure.height} loading="lazy" />
            </a>
            <figcaption><span>原图：<a href={figure.url}>{figure.author}</a>。</span><ArticleMarkdown source={figure.caption} /></figcaption>
          </figure>
        } }} />
      </article></main>
      <SiteFooter />
    </div>
  </StrictMode>,
)
