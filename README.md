# luca’s blog

个人技术博客，记录机器学习、系统实现与性能优化。

## 本地开发

使用 Node.js 22.12 或更高版本。

```sh
npm ci
npm run dev
```

开发地址：`http://localhost:5173/`。

```sh
npm run build
npm run preview
```

构建会先执行 TypeScript 类型检查，再生成 `dist/`。

## 添加文章

1. 在 `src/posts.ts` 中添加文章的 `slug`、标题、简介、主题和标签。目录搜索和主题筛选自动使用这些数据。
2. 草稿设置 `status: 'draft'`。正式发布时改为 `status: 'published'`，填写真实的 `publishedAt: 'YYYY-MM-DD'`。
3. 为文章提供 `posts/<slug>/index.html`，并在 `vite.config.ts` 的 `build.rolldownOptions.input` 中登记入口。Vite 多页面构建会保留该静态路径。

RMSNorm 草稿正文位于 `content/rmsnorm.md`。编辑 Markdown 即可实时预览；支持标准 Markdown、表格、代码块，以及 `\(…\)` / `\[…\]` 或 `$…$` / `$$…$$` 数学公式。保留章节锚点，正文采用居中单栏布局。KaTeX 和 Shiki 在开发或构建时生成 HTML，浏览器只加载样式、数学字体和文章展示代码。

## 部署

使用 GitHub Pages。在仓库 Settings → Pages 中将 Source 设为 GitHub Actions。
推送到 `main` 后，`.github/workflows/pages.yml` 自动安装依赖、检查类型、构建并部署。

站点地址为 https://luca-888.github.io/，仓库名为 `luca-888.github.io`。Vite 的 `base` 为 `/`，文章地址为 `/posts/<slug>/`。

## 技术栈

React、TypeScript、Vite、原生 CSS、KaTeX、Shiki、Apache ECharts。

## RMSNorm 数值实验

正文中的 `::rmsnorm-demo::` 挂载矩阵交互组件，展示“输入矩阵 ÷ 本行尺度 × γ → 输出矩阵”。`src/rmsnorm-example.ts` 提供完整的 6×4 小整数示意输入，以及 4 个一位小数的示意 γ，各行共享同一组 γ。固定 ε = 10⁻⁶；倍数滑块范围为 −2 到 +2，步长 0.01。逐行计算除数和输出 RMS，两侧条形分别使用标注的固定刻度。演示不模拟 GPU 的 BF16/FP32 舍入。

运行 `npm test` 验证数值计算，运行 `npm run build` 检查类型与静态构建。
