import { useDarkTheme } from './theme';

export function ThemeToggle() {
  const dark = useDarkTheme();
  const label = dark ? '切换到浅色模式' : '切换到深色模式';

  function toggle() {
    const next = dark ? 'light' : 'dark';
    window.dispatchEvent(new CustomEvent('theme-preference', { detail: next }));
  }

  return (
    <button className="theme-toggle" type="button" onClick={toggle} aria-label={label} title={label}>
      <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        {dark ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></> : <path d="M20.8 13.1A9 9 0 0 1 10.9 3.2a9 9 0 1 0 9.9 9.9Z" />}
      </svg>
    </button>
  );
}
