import { useSyncExternalStore } from 'react'

function subscribe(onChange: () => void) {
  window.addEventListener('themechange', onChange)
  return () => window.removeEventListener('themechange', onChange)
}

export function useDarkTheme() {
  return useSyncExternalStore(subscribe, () => document.documentElement.dataset.theme === 'dark')
}
