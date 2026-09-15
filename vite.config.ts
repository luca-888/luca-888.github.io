import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { markdown } from './build/markdown';

export default defineConfig({
  plugins: [markdown(), react()],
  base: '/',
  build: {
    rolldownOptions: {
      input: {
        main: 'index.html',
        rmsnorm: 'posts/rmsnorm/index.html',
      },
    },
  },
});
