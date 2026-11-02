import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { publishedPosts } from './src/posts'

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: { port: Number(process.env.PORT) || 5173 },
  build: {
    rollupOptions: {
      // 草稿只在 dev 中可见，不进入构建产物。
      input: {
        main: 'index.html',
        ...Object.fromEntries(
          publishedPosts.map((post) => [post.slug, `posts/${post.slug}/index.html`]),
        ),
      },
    },
  },
})
