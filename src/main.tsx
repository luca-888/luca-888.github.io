import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { orderedPosts } from './posts';
import { SiteHeader } from './SiteLayout';
import './styles.css';
import './site.css';

function ArticleMark({ slug }: { slug: string }) {
  return <svg className="article-mark" viewBox="0 0 180 120" fill="none" aria-hidden="true">
    {slug === 'kl-nll-ce' ? <>
      <rect className="mark-core" x="22" y="24" width="54" height="22" rx="3" />
      <rect className="mark-paper" x="22" y="51" width="34" height="22" rx="3" />
      <rect className="mark-result" x="22" y="78" width="22" height="22" rx="3" />
      <rect className="mark-core" x="108" y="24" width="22" height="22" rx="3" />
      <rect className="mark-paper" x="108" y="51" width="54" height="22" rx="3" />
      <rect className="mark-result" x="108" y="78" width="34" height="22" rx="3" />
      <path className="mark-line" d="M88 22v80" />
    </> : slug === 'gradient-checkpointing' ? <>
      <path className="mark-line" d="M24 42h132M146 70v24H34V70m0 0-7 8m7-8 7 8" />
      <rect className="mark-core" x="16" y="28" width="28" height="28" rx="5" />
      <rect className="mark-paper" x="76" y="28" width="28" height="28" rx="5" />
      <rect className="mark-result" x="136" y="28" width="28" height="28" rx="5" />
      <path className="mark-detail" d="M84 42h12m-6-6v12" />
    </> : slug === 'compression-harness' ? <>
      <path className="mark-line" d="M24 28h30l28 32-28 32H24M82 60h27m25 0h23M146 80v22H95V79" />
      <rect className="mark-paper" x="15" y="16" width="25" height="24" rx="4" />
      <rect className="mark-paper" x="15" y="48" width="25" height="24" rx="4" />
      <rect className="mark-paper" x="15" y="80" width="25" height="24" rx="4" />
      <rect className="mark-core" x="73" y="39" width="43" height="42" rx="10" />
      <path className="mark-detail" d="m83 59 10-10 11 10-10 12-11-12Z" />
      <circle className="mark-result" cx="148" cy="60" r="17" />
      <path className="mark-detail" d="m141 60 5 5 9-11" />
    </> : slug === 'flash-attention' ? <>
      <rect className="mark-paper" x="20" y="18" width="72" height="84" rx="4" />
      <path className="mark-line" d="M44 18v84m24-84v84M20 46h72M20 74h72" />
      <rect className="mark-core" x="21" y="47" width="70" height="26" rx="2" />
      <path className="mark-line" d="M101 60h24m-7-6 7 6-7 6" />
      <rect className="mark-result" x="136" y="18" width="24" height="84" rx="4" />
      <path className="mark-detail" d="M140 46h16m-16 28h16" />
    </> : slug === 'cuda-graph' ? <>
      <path className="mark-line" d="m90 22-49 38 49 38 49-38-49-38Z" />
      <rect className="mark-core" x="76" y="8" width="28" height="28" rx="7" />
      <rect className="mark-paper" x="27" y="46" width="28" height="28" rx="7" />
      <rect className="mark-result" x="125" y="46" width="28" height="28" rx="7" />
      <rect className="mark-core" x="76" y="84" width="28" height="28" rx="7" />
      <path className="mark-detail" d="M41 58h0m98 0h0" />
    </> : <>
      <path className="mark-line" d="M23 91V27m27 64V44m27 47V18m27 73V51m27 40V34m27 57V58" />
      <path className="mark-curve" d="M17 70c17-16 28 8 45-1s26-1 43 1 28-6 53-2" />
      <circle className="mark-core" cx="23" cy="27" r="5" />
      <circle className="mark-core" cx="50" cy="44" r="5" />
      <circle className="mark-core" cx="77" cy="18" r="5" />
      <circle className="mark-core" cx="104" cy="51" r="5" />
      <circle className="mark-core" cx="131" cy="34" r="5" />
      <circle className="mark-core" cx="158" cy="58" r="5" />
    </>}
  </svg>;
}

function App() {
  const [query, setQuery] = useState('');
  const search = query.trim().toLocaleLowerCase();
  const visiblePosts = orderedPosts.filter((post) =>
    [post.title, post.description, post.category, ...post.tags].join(' ').toLocaleLowerCase().includes(search),
  );

  return (
    <div className="site-shell site-home">
      <a className="skip-link" href="#articles">跳到文章目录</a>
      <SiteHeader home />

      <main>
        <section className="article-section" id="articles" aria-label="文章目录">
          <div className="toolbar">
            <div className="search-field" role="search">
              <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" strokeWidth="1.5" /><path d="m16 16 5 5" stroke="currentColor" strokeWidth="1.5" /></svg>
              <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文章…" aria-label="搜索文章" />
            </div>
          </div>

          <p className="sr-only" role="status">找到 {visiblePosts.length} 篇文章</p>
          <ul className="article-list">
            {visiblePosts.map((post) => (
              <li key={post.slug}>
                <a className="article-row" href={`${import.meta.env.BASE_URL}posts/${post.slug}/`}>
                  <ArticleMark slug={post.slug} />
                  <div className="article-copy">
                    <h2>{post.title}</h2>
                    <p>{post.description}</p>
                  </div>
                </a>
              </li>
            ))}
          </ul>
          {visiblePosts.length === 0 && <div className="empty-state"><p>没有找到相关文章</p><button type="button" onClick={() => setQuery('')}>查看全部文章</button></div>}
        </section>
      </main>

    </div>
  );
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
