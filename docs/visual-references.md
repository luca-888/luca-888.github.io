# 技术图表参考

2026-09-18：用户要求参考 OpenAI 官方文章的表格与技术图，纠正 FlashAttention 初稿中大面积深蓝、深绿填色和深蓝白字表头过于刺眼的问题。以下是实际页面观察，不是 OpenAI 发布的统一设计规范，也不将某一稿自动视为用户认可的固定模板。

## 表格

- [GPT-4.1 · Appendix](https://openai.com/index/gpt-4-1/#appendix)：表头和 benchmark 名称加粗，数字保持普通字重；新模型家族的连续三列使用极浅蓝底和细蓝轮廓，其余部分以白底、深色字、浅色横线呈现。颜色强调被讨论的对象，而不是给每一类数据都铺实色。
- [GPT-4.5 · 附录](https://openai.com/zh-Hans-CN/index/introducing-gpt-4-5/#fu-lu)：主要依靠表头字重、行列对齐和浅色横线组织信息；正文白底，脚注较小且弱化，没有彩色表头。

## 技术图

- [Harness engineering](https://openai.com/index/harness-engineering/) 的分层架构图使用深色轮廓、对齐的节点、分组容器和清楚的依赖方向；用空间安排区分内外关系。
- 同文的 observability 数据流图以黑白结构为主，绿色集中标出 Codex 的反馈路径。可以直接看到主数据流与反馈路径的区别，无需把所有节点涂成不同的颜色。
- [分层架构官方原图](https://images.ctfassets.net/kftzwdyauwt9/4Rlip1H3T9apPlSmWs7Wr8/7708c176bfbe11951e06ad8e2b83bf01/OAI_Harness_engineering_Layered_domain_architecture_with_explicit_cross-cutting_boundries_desktop-light.png)
- [可观测性数据流官方原图](https://images.ctfassets.net/kftzwdyauwt9/4Xr18TZ5G4Bh8zIgsTFIVK/f7ae689ddd8c31664e39d809b0973425/OAI_Harness_engineering_Giving_Codex_a_full_observability_stack_desktop-light__1_.svg)

## 在本站的应用

- 先通过构图、对齐、留白、字号和字重建立主次，保证主文字清晰；背景淡不等于文字也要淡。
- 颜色与具体关系绑定：当前对象、跨轮状态、边界、反馈或比较结果。先找出需要强调什么，再选择强调方式。
- 浅色底、深色文字与少量深色标记可以同时提供舒适度与重点；放宽风格不等于默认采用饱和实色大块或白字彩色表头。
- 参考构图与信息组织方法，不复制官方品牌标识；继续保留每篇图表根据内容调整的自由。

当前 FlashAttention 稿保留三步主流程与较大的运算符号，取消实色蓝绿面板；淡绿只标跨轮状态，淡蓝只标 causal 边界块。表格以浅灰表头、深色字重和行分隔组织内容。此稿仍可根据用户反馈调整。
