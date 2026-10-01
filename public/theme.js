// 站点只提供浅色模式；清掉旧版本存下的主题偏好。
(() => {
  document.documentElement.dataset.theme = 'light';
  document.documentElement.style.colorScheme = 'light';
  try { localStorage.removeItem('luca-blog-theme'); } catch {}
})();
