# DSpark 文章论文原图

三张图均为 DSpark 论文的矢量原图，从 arXiv TeX 源文件包中的 PDF 用 `scripts/pdf_tools.py svg`（PyMuPDF）直接导出为 SVG，没有截图、重绘、改色或裁剪。下载于 2026-10-03。

| 文件 | 源文件包中的路径 | 论文中的图 |
| --- | --- | --- |
| `position-cond-accept.svg` | `figs/position_cond_accept.pdf` | Position-wise conditional acceptance（Qwen3-4B，Math / Code / Chat） |
| `online-service.svg` | `figs/online_service.pdf` | Throughput vs. TPS（V4-Flash 与 V4-Pro 线上流量） |
| `online-service-tradeoff.svg` | `figs/online_service_tradeoff.pdf` | Load-adaptive throughput and verification budgets |

- 论文：Xin Cheng, et al. *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*，arXiv:2607.05147。
- 论文页面：https://arxiv.org/abs/2607.05147
- 源文件包：https://arxiv.org/src/2607.05147
- 许可：arXiv 源文件包未附单独的许可声明，按 arXiv.org perpetual non-exclusive license 处理，保留作者署名与出处，不标为本站自绘。

SHA-256：

```text
bf20ecee28bce6257fa6dcef4050384ffcc431ee7b645bb5120669f5a844529f  position-cond-accept.svg
5aed06fcbe9098f5c95510cac5cdb0f9b6e631d72606d39782e1c2a4cf060671  online-service.svg
bf1cbb702091e981af673aca8c8dada0e11136368fefdac97fd374283c791d0d  online-service-tradeoff.svg
```
