import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { orderedPosts } from './posts';
import { SiteHeader } from './SiteLayout';
import './styles.css';
import './site.css';

// 冒号前是主题，冒号后的细节用浅色。
function TitleText({ title }: { title: string }) {
  const index = title.indexOf('：');
  if (index < 0) return <>{title}</>;
  return <>{title.slice(0, index)}<span className="title-detail">{title.slice(index)}</span></>;
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
                  <time dateTime={post.publishedAt}>{post.publishedAt ?? '草稿'}</time>
                  <h2><TitleText title={post.title} /></h2>
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
