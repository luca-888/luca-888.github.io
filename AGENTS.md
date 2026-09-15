# 项目约定

## 项目方向

- 本项目用于展示机器学习算子的实现与优化，第一个页面主题是 RMSNorm。
- 页面内容默认使用简体中文，技术名称与代码保留原文。
- 按用户要求逐步实现；当前页面正文保持空白，等待后续内容要求。

## 已确定的技术栈

后续开发采用以下技术栈：

| 用途 | 技术 |
| --- | --- |
| 页面与交互 | React + TypeScript |
| 开发与静态构建 | Vite |
| 样式 | 原生 CSS |
| 数学公式渲染 | KaTeX |
| 代码语法高亮 | Shiki |
| 性能对比与数据图表 | Apache ECharts |
| 托管与部署 | GitHub Pages + GitHub Actions |

- 当前只有 HTML 空页，尚未初始化上述前端依赖；后续实现页面时按此约定搭建。
- KaTeX、Shiki 和 ECharts 在对应内容或功能需要时引入。

## GitHub Pages 部署

- 使用 GitHub Pages 托管静态构建产物。
- 部署工作流位于 `.github/workflows/pages.yml`，由 `main` 分支推送或手动运行触发。
- 使用 Vite 后，将 `base` 配置为 `/ml-portal/`，适配当前仓库的 GitHub Pages 子路径。
- 静态资源引用须兼容该子路径；若更换仓库名或改用自定义域名，同步调整 `base`。
- 初始化 Vite 时，同步将工作流中的 HTML 复制步骤替换为依赖安装和构建，部署 `dist/` 目录。
- 页面功能应适用于静态托管；算子性能图表使用真实测量数据，并注明测试环境，演示数据须明确标注。

## 验证

- 初始化前端工程后，提供 `npm run dev`、`npm run build` 和 `npm run preview` 命令，并提交依赖锁文件。
- 构建命令应包含 TypeScript 类型检查；发布前确保构建成功。
- 页面实现后，检查桌面与移动端布局，以及 `/ml-portal/` 子路径下的资源加载。
