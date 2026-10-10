# 稀疏 Attention 文章论文原图

三张图均为论文的矢量原图，从 arXiv TeX 源文件包中的 PDF 用 `scripts/pdf_tools.py svg`（PyMuPDF）直接导出为 SVG，没有截图、重绘、改色或裁剪。下载于 2026-10-09。

| 文件 | 论文与源文件包中的路径 | 论文中的图 |
| --- | --- | --- |
| `mistral-swa.svg` | Jiang et al., *Mistral 7B*，arXiv:2310.06825，`images/swa.pdf` | Figure 1：Sliding Window Attention |
| `streamingllm-attention.svg` | Xiao et al., *Efficient Streaming Language Models with Attention Sinks*，arXiv:2309.17453，`figures/attention_weights.pdf` | Figure 2：Llama-2-7B 的平均 attention logits |
| `streamingllm-scheme.svg` | 同上，`figures/scheme.pdf` | Figure 1：StreamingLLM 与现有方法的对比 |

- 源文件包：`https://arxiv.org/src/<id>`。
- 许可：各源文件包未附单独的许可声明，按 arXiv.org perpetual non-exclusive license 处理，保留作者署名与出处，不标为本站自绘。

SHA-256：

```text
df91ffd525c5982693e89d7a495f3af83fbadf1e174040464576566555a2a081  mistral-swa.svg
b95eeb0ea6b0b3621f9541a6e02f3e6572874c65627a66e711305c38179903c8  streamingllm-attention.svg
aca368c73cb5ca57d18d671afb1f43d04c3d9213fbc762905aa7ced91dfc3deb  streamingllm-scheme.svg
```
