export function SiteHeader({ home = false }: { home?: boolean }) {
  return <header className="site-header">
    <a className="wordmark" href={import.meta.env.BASE_URL}>luca’s blog</a>
    <nav aria-label="主导航">
      <a href={`${import.meta.env.BASE_URL}#articles`} aria-current={home ? 'page' : undefined}>文章</a>
      <a href="https://github.com/luca-888/luca-888.github.io">GitHub</a>
    </nav>
  </header>
}

export function SiteFooter() {
  return <footer className="site-footer">
    <span>luca’s blog</span>
    <a href="https://github.com/luca-888/luca-888.github.io">在 GitHub 查看源码</a>
  </footer>
}
