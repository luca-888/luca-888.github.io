# 线性 Attention 文章原图

两张图均为原始矢量图，用 PyMuPDF 直接导出为 SVG（文字转为路径），没有重绘或改色。下载于 2026-10-10。

| 文件 | 来源与路径 | 原文中的图 | 处理 |
| --- | --- | --- | --- |
| `qwen-gdn-module.svg` | Qwen Team, *On the Design of Qwen3.8-Next Architecture*，arXiv:2608.30320，源文件包 `figs/Qwen4Exp_GDN.drawio.pdf` | Gated DeltaNet token mixer | `scripts/pdf_tools.py svg` 原样导出 |
| `k3-prefix-cache.svg` | Kimi Team, *Kimi K3: Open Frontier Intelligence*，[k3_tech_report.pdf](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) 第 23 页 | Figure 12：Fine-grained prefix caching within a physical cache block | 报告只有整页 PDF：删去图区域以外的正文后，按 Rect(124, 331, 490, 456) 裁出图本身，不含图注 |

- 许可：arXiv 源文件包未附单独的许可声明，按 arXiv.org perpetual non-exclusive license 处理；Kimi K3 报告随仓库发布。均保留作者署名与出处，不标为本站自绘。

SHA-256：

```text
161a2d65f39b5829c92101cb3496922a9ba88cec070dbc81d2a1d2e98edeb466  qwen-gdn-module.svg
119e9d49610397b0ce0eaf265459ad0e735b6cea26bcb66de133d1bac738e345  k3-prefix-cache.svg
```
