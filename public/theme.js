// Runs in the document head so a saved dark theme is applied before first paint.
(() => {
  let saved;
  try { saved = localStorage.getItem('luca-blog-theme'); } catch {}
  const theme = saved === 'light' || saved === 'dark'
    ? saved
    : matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#1b1d1a' : '#f7f7f2');
})();
