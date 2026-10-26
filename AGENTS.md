# 项目约定

个人技术博客 `luca’s blog`：分享技术文章、学习笔记与实践经验，主题不限。正文使用简体中文，技术名称与代码保留原文。

## 按任务读

本文件只放每次都适用的约定。具体任务开始前读对应的文档，与任务无关的不读：

| 任务 | 读 |
| --- | --- |
| 写新文章、改稿、成稿自查 | [docs/writing.md](docs/writing.md)、[docs/terminology.md](docs/terminology.md)；去 AI 腔用 `de-ai-edit` skill |
| 画文章插图 | [docs/figures.md](docs/figures.md)、[docs/visual-references.md](docs/visual-references.md) |
| 封面、头图 | [docs/covers.md](docs/covers.md) |
| 小红书图片、竖版视频 | [docs/xiaohongshu.md](docs/xiaohongshu.md) |
| 做视频 | [docs/video.md](docs/video.md)，再按 HyperFrames 插件的 `/hyperframes` skill |
| 配音（MiniMax，付费） | [docs/tts.md](docs/tts.md) |
| GPU 实验 | [docs/gpu-experiments.md](docs/gpu-experiments.md) |
| 某篇文章的背景、参考与实测记录 | `docs/<slug>-notes.md`（只读当前这一篇的） |

不需要读的：`videos/*/build/`、`renders/`、`node_modules/`、`dist/`、`public/measurements/` 下的原始输出（要数字时只读对应的汇总文件）、其它文章的 notes、`docs/legacy/`、旧视频（2026-10 之前的 canvas 视频）的源码。

## 始终适用

- 仓库是公开的：API key 只放仓库根目录的 `.env.local`（已忽略），不写进任何会提交的文件、日志或对话。
- 一个主题的文件都按 slug 存放，各只保留一个版本；临时结果用完即删（详见下面“站点”一节的表）。
- 站点只有浅色模式；只面向桌面端。
- 需要 GPU 的实验先说明配置并与用户确认（见 docs/gpu-experiments.md）。

## 站点

| 用途 | 技术 |
| --- | --- |
| 页面与交互 | React + TypeScript |
| 构建 | Vite（`npm run dev` / `build` / `preview`，build 含类型检查） |
| 样式 | 原生 CSS，公共变量在 `src/styles.css` |
| 公式 / 代码高亮 / 图表 | KaTeX / Shiki / ECharts（按需引入） |
| 部署 | GitHub Pages + GitHub Actions（`.github/workflows/pages.yml`，推送 `main` 触发） |

- 仓库 `luca-888/luca-888.github.io`，站点 `https://luca-888.github.io/`，Vite `base` 为 `/`；静态资源通过 Vite 处理或 `import.meta.env.BASE_URL` 引用。
- 新文章：正文 `content/<slug>.md`，入口 `posts/<slug>/index.html`（head 中加载 `/theme.js`），页面代码放在 `src/articles/<slug>/`（入口、`.blocks.tsx`、图表与样式），在 `vite.config.ts` 登记，信息写入 `src/posts.ts`；复用 `src/SiteLayout.tsx` 的页头页脚。正式发布时填写真实日期与 `published` 状态。
- 一个主题的文件都按文章 slug 存放，各只保留一个版本（2026-10-07 用户要求）：

  | 内容 | 位置 |
  | --- | --- |
  | 正文 / 笔记 | `content/<slug>.md` / `docs/<slug>-notes.md` |
  | 页面代码 | `src/articles/<slug>/` |
  | 图表数据 / 图片 | `src/data/<slug>*.json` / `src/assets/<slug>/` |
  | 测量记录 | `public/measurements/<slug>/` |
  | 实验脚本 | `scripts/<slug>/` |
  | 视频源文件 / 封面定稿 | `videos/<slug>/` / `covers/<slug>/` |

  `src/` 顶层只放多篇共用的文件。临时结果不存储：渲染中间产物、草稿、样稿、检查截图、导出包、旧版本用完即删；成片与配音留在已忽略的 `videos/<slug>/build/`。
- 站点只有浅色模式，不做夜间配色与切换（2026-10-01 用户确认）。
- 首页为单列、等尺寸阅读卡片（只有标题，不放摘要与配图；2026-10-07 用户要求）加搜索；白底、细边框、无阴影。
- 站点 UI 用 `src/styles.css` 的颜色变量：白与中性浅灰为底，蓝灰 `#567891` 为主强调，灰绿 `#587767` 为辅助。

## 开发与验证

- 只面向桌面端，不做移动端适配与验证。
- 日常修改用 `npm run dev` 热更新查看，不例行跑 build、完整类型检查或测试；只在出错、改动关键逻辑或用户要求时做最小验证。
- 读论文全文：优先 arXiv 源文件包（`https://arxiv.org/src/<id>`）中的 `.tex`，其次 arXiv HTML，只有两者都没有（如 USENIX 等只提供 PDF）时才用 `scripts/pdf_tools.py text` 从 PDF 提取。导出论文原图用同一脚本的 `svg` / `png`，环境为 `~/.venvs/pdf-tools`。
- 发布前运行完整构建，检查桌面页面与资源加载；旧 `/ml-portal/` 入口重定向至首页。
