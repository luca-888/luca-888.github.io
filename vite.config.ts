import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        rmsnorm: 'posts/rmsnorm/index.html',
        cudaGraph: 'posts/cuda-graph/index.html',
        compressionHarness: 'posts/compression-harness/index.html',
      },
    },
  },
})
