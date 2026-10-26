# PD 分离调度文章论文原图

两张图均从 arXiv TeX 源文件包中的 PDF 用 PyMuPDF（`scripts/pdf_tools.py svg`）直接导出为 SVG，没有截图、重绘、改色或裁剪。下载于 2026-10-09。许可：arXiv.org perpetual non-exclusive license，保留作者署名与出处，不标为本站自绘。

## mooncake-load_example.svg

Mooncake 论文 §Overload-oriented Scheduling 的图 `fig:load_example`：采用 early rejection、尚未按预测接收时，20 台机器的集群 20 分钟内 prefill 与 decode 实例的负载。

- 论文：Ruoyu Qin, et al. *Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot*，FAST 2025，arXiv:2407.00079。
- 源文件包：https://arxiv.org/src/2407.00079（`figures/load_example.pdf`，由 `data/overload_scheduling.tex` 引用）

## mitra-fixed-ratios.svg

Beyond the Buzz 论文 §Dynamic rate matching considerations 的图 `fig:fixed_ratios`：DeepSeek-R1（ISL 16384、OSL 2048）在固定 Ctx:Gen 比例 0.5、3.5，动态 rate matching 与合并部署下的归一化吞吐与交互性曲线，数据来自 NVIDIA 的仿真器。

- 论文：Tiyasa Mitra, et al. *Beyond the Buzz: A Pragmatic Take on Inference Disaggregation*，arXiv:2506.05508。
- 源文件包：https://arxiv.org/src/2506.05508（`fixed_ratios.pdf`，由 `disaggregation_in_practice.tex` 引用）

SHA-256：
- 2f991920078ac6627367872992f38a54050f9d1651f789a4578f93fe2f62f1c7  mitra-fixed-ratios.svg
- 0e7dbf7a32891e7ea5d0f66a4bdd1982d7fcb0199f8289665eab0c64d539f7cd  mooncake-load_example.svg
