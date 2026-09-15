import { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { orderedPosts } from './posts';
import { ThemeToggle } from './ThemeToggle';
import './styles.css';

const topics = [...new Set(orderedPosts.map((post) => post.category))];

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d={diagonal ? 'M6 18 18 6M6 6h12v12' : 'M4 12h15m-6-6 6 6-6 6'} stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function App() {
  const [topic, setTopic] = useState('全部');
  const [query, setQuery] = useState('');
  const search = query.trim().toLocaleLowerCase();
  const visiblePosts = orderedPosts.filter((post) =>
    (topic === '全部' || post.category === topic) &&
    [post.title, post.description, post.category, ...post.tags].join(' ').toLocaleLowerCase().includes(search),
  );

  return (
    <div className="site-shell">
      <a className="skip-link" href="#articles">跳到文章目录</a>
      <header className="site-header">
        <a className="wordmark" href={import.meta.env.BASE_URL} aria-label="luca’s blog 首页">
          luca’s blog
        </a>
        <nav aria-label="主导航">
          <a className="nav-current" href="#articles" aria-current="page">文章</a>
          <a href="https://github.com/luca-888/luca-888.github.io">GitHub <Arrow diagonal /></a>
          <ThemeToggle />
        </nav>
      </header>

      <main>
        <section className="article-section" id="articles" aria-labelledby="articles-title">
          <div className="section-heading">
            <h1 id="articles-title">文章目录 <span>INDEX</span></h1>
            <p className="total-count">{orderedPosts.length} 篇文章</p>
          </div>

          <div className="toolbar">
            <div className="topic-filters" role="group" aria-label="按主题筛选">
              {['全部', ...topics].map((item) => (
                <button key={item} type="button" aria-pressed={topic === item} onClick={() => setTopic(item)}>
                  {item}<span>{item === '全部' ? orderedPosts.length : orderedPosts.filter((post) => post.category === item).length}</span>
                </button>
              ))}
            </div>
            <div className="search-field" role="search">
              <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none"><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" strokeWidth="1.5" /><path d="m16 16 5 5" stroke="currentColor" strokeWidth="1.5" /></svg>
              <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文章…" aria-label="搜索文章" />
            </div>
          </div>

          <div className="list-heading" aria-hidden="true"><span>日期 / 状态</span><span>主题</span><span>标题</span></div>
          <p className="sr-only" role="status">找到 {visiblePosts.length} 篇文章</p>
          <ul className="article-list">
            {visiblePosts.map((post) => (
              <li key={post.slug}>
                <a className="article-row" href={`${import.meta.env.BASE_URL}posts/${post.slug}/`}>
                  <div className="article-date">{post.status === 'draft' ? <span className="draft-label">编写中</span> : <time dateTime={post.publishedAt}>{post.publishedAt.replaceAll('-', '.')}</time>}</div>
                  <span className="article-topic">{post.category}</span>
                  <div className="article-copy"><h2>{post.title}</h2><p>{post.description}</p></div>
                  <span className="article-arrow"><Arrow /></span>
                </a>
              </li>
            ))}
          </ul>
          {visiblePosts.length === 0 && <div className="empty-state"><p>没有找到相关文章</p><button type="button" onClick={() => { setQuery(''); setTopic('全部'); }}>查看全部文章 <Arrow /></button></div>}
          <div className="list-end"><span>{search || topic !== '全部' ? `显示 ${visiblePosts.length} / ${orderedPosts.length} 篇` : '以上是全部文章'}</span><span aria-hidden="true">—</span></div>
        </section>
      </main>

      <footer className="site-footer"><span>luca’s blog</span><a href="https://github.com/luca-888/luca-888.github.io">在 GitHub 查看源码 <Arrow diagonal /></a></footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
