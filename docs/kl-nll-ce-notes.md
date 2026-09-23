# KL / NLL / CE 文章资料

## 范围与硬件

- 原理文章，CPU 数值核对即可；无需专用硬件，不报告实测训练性能。
- 三词词表及数值为构造示例；使用自然对数，单位 nats。
- 本地草稿，已接入文章入口；未部署。

## 参考博客

- Christopher Olah：https://colah.github.io/posts/2015-09-Visual-Information/ ，采用编码面积解释与四格原图。
- Eric Jang：https://blog.evjang.com/2016/08/variational-bayes.html ，采用正反 KL 原图，注明单峰近似族前提。
- Lilian Weng：https://lilianweng.github.io/posts/2018-08-12-vae/ ，作为 VAE 延伸阅读；追溯其中 KL 图到 Eric Jang 原作，不重复归属。
- John Schulman 的 KL approximation 原站本次未成功读取，不作为本篇新增引用。

## 原图来源

全部图片原样保存，没有裁切、改色或生成式处理；夜间模式仍保留白底。按原始像素限制最大显示宽度，点击可打开本地原图。未找到适用的更高分辨率矢量版本，不做虚假超分辨率放大。

- `colah-cross-entropy.png`：873 × 441，原作者仓库版本。
  - https://raw.githubusercontent.com/colah/colah.github.io/master/posts/2015-09-Visual-Information/img/CrossEntropyCompare.png
  - 原文记法 H_q(p) 对应本文 H(p,q)，原图单位为 bits。

- `forward-KL.png`：从作者页面链接取得 s1600 原文件，非 s640 缩略图。
  - https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhaSnD0TmegUI12OBQG7dborW0TsahuCz7koKV2tRzpe3DfQcHRTRtud-0xuzmdS9Oy8jHoXAw5nsow-sYduLXgM38TzPFSxiIMx7_aGGx3PH462MTV1KvaouMCFnonj4WFFXV9axdSRhQ/s1600/forward-KL.png

- `reverse-KL.png`：从作者页面链接取得 s1600 原文件，非 s640 缩略图。
  - https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgP80aN0IDF7ywcvN7-1XvndChYx7cVVaUPy4maD60TtBgtYEvh9TMX4qGqwXMEGoIZ68HeId60zx8HFWjzzrl9cG4aESJpW5wiYlfCqit1s_cKfp0qToMr2I76DSnTNA4h3mvn6mqZM7M/s1600/reverse-KL.png

## 视觉方向

- 用户本次明确认可 colah / Distill，已将偏好写入 AGENTS.md 与 docs/visual-references.md。
- 本篇原图已经覆盖编码代价和 KL 方向两个核心解释任务，因此没有额外 AI 生成替代图。
- 首页继续复用公共卡片；正文使用公共 Markdown、KaTeX、Shiki 与表格样式。

## 验证

- 正文 92 处公式与全部图注通过 KaTeX 严格解析；新 TSX 文件通过独立转译。
- 从正文提取两个 Python 片段，在 CPU 上核对硬标签 CE/NLL 的数值和梯度、CE = entropy + KL、固定软标签下 CE/KL 梯度、经验频率与平均 NLL，以及两个 KL 方向的数值。
- 三张原图均已逐一打开确认内容与原始像素尺寸，没有例行执行全量构建或重复浏览器截图。
