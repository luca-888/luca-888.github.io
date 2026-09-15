import { StrictMode, useEffect, useState, type MouseEvent } from 'react';
import { createRoot } from 'react-dom/client';
import { createPortal } from 'react-dom';
import document from '../content/rmsnorm.md';
import { ThemeToggle } from './ThemeToggle';
import { RmsNormDemo } from './RmsNormDemo';
import 'katex/dist/katex.min.css';
import './styles.css';
import './article.css';

function Article() {
  const [copyStatus, setCopyStatus] = useState('');
  const [demoTarget, setDemoTarget] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setDemoTarget(window.document.getElementById('rmsnorm-demo'));
  }, []);

  useEffect(() => {
    if (!demoTarget) return;
    // React inserts the headings after initial HTML parsing, so restore deep links.
    if (window.location.hash) {
      const target = window.document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
      target?.scrollIntoView();
    }
  }, [demoTarget]);

  async function copyCode(event: MouseEvent<HTMLDivElement>) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest<HTMLButtonElement>('[data-copy-code]');
    const code = button?.closest('.code-block')?.querySelector('code');
    if (!button || !code) return;
    try {
      await navigator.clipboard.writeText(code.textContent ?? '');
      button.textContent = '已复制';
      setCopyStatus('代码已复制');
    } catch {
      const selection = window.getSelection();
      const range = window.document.createRange();
      range.selectNodeContents(code);
      selection?.removeAllRanges();
      selection?.addRange(range);
      button.textContent = '已选中';
      setCopyStatus('无法自动复制，已选中代码，请手动复制');
    }
  }

  return (
    <div className="site-shell post-shell">
      <a className="skip-link" href="#post-body">跳到正文</a>
      <header className="site-header">
        <a className="wordmark" href={import.meta.env.BASE_URL}>luca’s blog</a>
        <nav aria-label="主导航"><a href={import.meta.env.BASE_URL}>文章</a><a href="https://github.com/luca-888/luca-888.github.io">GitHub ↗</a><ThemeToggle /></nav>
      </header>
      <main className="post-layout">
        <article className="post-content">
          <header className="post-header">
            <div className="post-meta"><span>算子优化</span><span className="draft-label">草稿</span></div>
            <h1 id="post-title">{document.title}</h1>
            <div className="post-author">luca <span aria-hidden="true">/</span> 数学推导 · GPU 实现 · 性能优化</div>
          </header>
          <div id="post-body" className="prose" onClick={copyCode} dangerouslySetInnerHTML={{ __html: document.html }} />
          {demoTarget && createPortal(<RmsNormDemo />, demoTarget)}
          <p className="sr-only" role="status">{copyStatus}</p>
          <div className="post-end"><a href={import.meta.env.BASE_URL}>← 返回文章目录</a><a href="#post-title">回到顶部 ↑</a></div>
        </article>
      </main>
      <footer className="site-footer"><span>luca’s blog</span><a href="https://github.com/luca-888/luca-888.github.io">在 GitHub 查看源码 ↗</a></footer>
    </div>
  );
}

createRoot(window.document.getElementById('root')!).render(<StrictMode><Article /></StrictMode>);
