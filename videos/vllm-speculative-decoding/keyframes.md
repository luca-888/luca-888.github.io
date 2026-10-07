# Speculative Decoding 视频关键帧（待校对）

> **2026-10-01 更新**：整片已按新方向重做为 `film2.html`（`node film.mjs film2.html build/spec-decoding-v2-1080p60.mp4 --fps 60`，静帧用 `node stills.mjs film2.html <目录> <秒>…`）。
> 场景顺序：封面→赛跑（前三轮慢放，含 bonus token）、验证几乎免费、一个猜测的三条路、猜几个（公式曲线）、三类 drafter 原理各一幕、四条赛道对比、并发、收尾。
> 下文是旧版 `keyframes.html` 的关键帧记录，保留作对照。

静态关键帧，不含动效；每帧一个点，右下角编号。图片在 `keyframes/kNN.png`，总览在 `keyframes/sheet-1..4.png`。数字均来自本文实测（`public/measurements/vllm-speculative-decoding/`）或文中已标注来源的论文 / 文档。

| 帧 | 章 | 这一帧要说的点 | 数据 / 来源 |
| --- | --- | --- | --- |
| K01 | 开场 | 标题：猜多个，验一次 | — |
| K02 | 01 验证几乎免费 | 每生成一个 token 要把 16 GB 权重读一遍：17 ms、59 tok/s | baseline 并发 1，RTX 4090 实测 |
| K03 | 01 | 多验证几个 token 几乎不多花时间：32× token，1.4× 耗时 | baseline 并发 1 / 32：17.0 / 24.3 ms |
| K04 | 02 拒绝采样 | 规则：猜对收下，猜错从残差补采（4 词表示例） | 构造示例 |
| K05 | 02 | 为什么分布不变：重叠直接收下，多出的由补采补上，合起来是 p | 同上，Leviathan et al. 的证明 |
| K06 | 02 | vLLM kernel 800 万次采样，偏差 1.7×10⁻⁴，与噪声同量级 | `kernel_check.json` |
| K07 | 03 加速比 | 取决于 α、γ、c；α=0.67、γ=3 预测 2.42，实测 2.39 | Leviathan et al. 公式；draft-k3 实测 |
| K08 | 03 | γ 越大边际收益越小，成本线性增长（α=0.8） | 公式计算 |
| K09 | 04 三类 drafter | 三类 drafter 的第 1 位接受率与每步耗时倍数 | γ=3，并发 1，T=0 |
| K10 | 05 实测 | 并发 1：draft model 最快，n-gram 只在摘要有用 | 配对 baseline 加速比 |
| K11 | 05 | 用第 1 位 α 预测平均接受长度，误差 4% 以内 | draft-k3、eagle3-k3 聊天 |
| K12 | 05 | 并发上升，收益下降 | γ=3 聊天，并发 1/4/16/32 |
| K13 | 06 边界 | 只在算力有富余时有效：本文实测、vLLM 2024 blog、MagicDec | blog 与 MagicDec 引自其原文 / 摘要 |
| K14 | 06 | 拒绝采样无损，放宽接受条件的方案不是；T=0 不逐位一致是数值问题 | EAGLE 论文对 Medusa 的说明；逐位一致率实测 |
| K15 | 06 | draft model 占显存，KV 池 26,848 → 15,024 token | 启动日志 |
| K16 | 收尾 | 猜得准、猜得便宜、算力有富余 | — |

---

## 逐章审查

按章节逐章细化，每章审完再做下一章。第 04 章“三类 drafter”将展开为每类一帧（draft model、n-gram、EAGLE-3 各一帧，再加一帧对比），做到该章时给出。

### 第 01 章 验证几乎免费（`keyframes/ch1/c1-N.png`，总览 `sheet-a.png`、`sheet-b.png`）

| 帧 | 要说的点 | 数据 |
| --- | --- | --- |
| C1-0 | 标题：猜多个，验一次 | — |
| C1-1 | 每生成一个 token，要把 16 GB 权重读一遍：17 ms、59 tok/s | baseline 并发 1 实测 |
| C1-2 | 读一遍权重只算 1 个 token，算力在空转 | 17.0 ms；算力网格为示意 |
| C1-3 | 同一次读取多带几个 token，每步只慢一点：1 / 4 / 32 个 → 17.0 / 22.4 / 24.3 ms | baseline 并发 1 / 4 / 32 实测 |
| C1-4 | 结论：32× token，1.4× 耗时 | 24.3 / 17.0 |
| C1-5 | 猜的 token 也只是多带几个，一次验完 | 4 个 token 取并发 4 的实测（22.4 ms）作示意，帧内已注明 |
| C1-6 | 钩子：验出来的 token，凭什么可信？ | — |
