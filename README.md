# ML Portal

个人技术博客，分享技术文章、学习笔记与实践经验。首篇文章为 RMSNorm 的计算原理与 GPU 算子优化。

## 本地开发

使用 Node.js 22.12+（推荐 Node.js 22 LTS）。

```bash
npm ci
npm run dev
```

打开 http://localhost:5173/ml-portal/。修改 `content/rmsnorm.md`、`src/App.tsx` 或 `src/styles.css` 后，页面会自动更新。

- `content/rmsnorm.md`：文章正文，支持 Markdown、LaTeX 公式和 Python 代码高亮。
- `src/App.tsx`：文章渲染入口；`::rmsnorm-demo::` 插入 RMSNorm 交互示意。
- `src/RmsNormDemo.tsx`：可调整输入、共享权重 γ 和 ε 的计算过程演示。
- `src/styles.css`：桌面与移动端样式。

## 构建与预览

```bash
npm run build
npm run preview
```

构建包含 TypeScript 类型检查，产物位于 `dist/`。构建预览地址为 http://localhost:4173/ml-portal/。

GitHub Actions 在推送 `main` 或手动触发时构建并部署至 GitHub Pages，资源基路径为 `/ml-portal/`。
