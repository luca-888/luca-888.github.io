# luca’s blog

个人技术博客，分享技术文章、学习笔记与实践经验。站点：https://luca-888.github.io/

## 本地开发

使用 Node.js 22.12+（推荐 Node.js 22 LTS）。

```bash
npm ci
npm run dev
```

打开 http://localhost:5173/，修改后页面自动更新。

```bash
npm run build     # 类型检查并构建到 dist/
npm run preview   # 预览构建结果
npm test          # 依赖版本一致性检查
```

推送 `main` 后由 GitHub Actions（`.github/workflows/pages.yml`）构建并发布到 GitHub Pages。

## 目录

每篇文章以 slug 为键，相关文件分布在下列位置：

| 内容 | 位置 |
| --- | --- |
| 正文 | `content/<slug>.md` |
| 页面入口 | `posts/<slug>/index.html`，在 `vite.config.ts` 登记 |
| 页面代码（入口、图表、样式） | `src/articles/<slug>/` |
| 图表数据、图片 | `src/data/`、`src/assets/<slug>/` |
| 测量记录 | `public/measurements/<slug>/` |
| 实验脚本 | `scripts/<slug>/` |
| 调研与确认记录 | `docs/<slug>-notes.md` |
| 视频源文件、封面 | `videos/<slug>/`、`covers/<slug>/` |

多篇共用的代码在 `src/` 顶层：

- `posts.ts`：文章目录与发布信息；`main.tsx`：首页与搜索。
- `ArticleMarkdown.tsx`：Markdown、公式、表格与代码高亮渲染。
- `SiteLayout.tsx`：页头、页脚与文章视频播放器。
- `figure-kit.tsx`：自绘图的公共图元。
- `styles.css`、`site.css`：公共颜色变量、文章与站点样式。
- `xhs/`：小红书正文页的排版与分页，导出脚本在 `scripts/xhs/`。

写作、图表、封面与视频的约定见 `AGENTS.md` 与 `docs/`。
