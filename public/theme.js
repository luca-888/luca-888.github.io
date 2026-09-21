// Run before rendering so every entry page starts in the selected theme.
(() => {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  let preference;
  try { preference = localStorage.getItem('luca-blog-theme'); } catch {}
  const apply = () => {
    const theme = preference === 'light' || preference === 'dark' ? preference : media.matches ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#181b1f' : '#ffffff');
    window.dispatchEvent(new Event('themechange'));
  };
  window.addEventListener('theme-preference', event => {
    preference = event.detail;
    try { localStorage.setItem('luca-blog-theme', preference); } catch {}
    apply();
  });
  window.addEventListener('storage', event => {
    if (event.key === 'luca-blog-theme' || event.key === null) {
      preference = event.newValue;
      apply();
    }
  });
  media.addEventListener('change', apply);
  apply();
})();
