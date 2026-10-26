# LLM Roofline 素材出处

| 文件 | 内容 | 来源 | 取法 |
| --- | --- | --- | --- |
| `williams2009-fig1a.png` | Roofline 原论文 Figure 1(a)：AMD Opteron X2 的 roofline（peak memory bandwidth 斜线、peak floating-point performance 平顶，两条 operational intensity 虚线） | Samuel Williams, Andrew Waterman, David Patterson, “Roofline: An Insightful Visual Performance Model for Multicore Architectures”, Communications of the ACM 52(4), 2009；预印本 https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf | 用 pymupdf 从预印本第 2 页取出内嵌位图（1200 × 1560），未改动 |

视频里其余的实测数字与日志来自本站测量记录 `public/measurements/vllm-cpu-overhead/20260930-094235/default/qwen3-8b/`（Qwen3-8B，H100 80GB HBM3，vLLM 0.30.0）。H100 与 B200 的规格取自 NVIDIA 官方规格页（dense 数字），见 `docs/llm-roofline-notes.md`。片中的 H100 外形为自绘示意，未使用官方图。
