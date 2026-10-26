# 视频工具包（HyperFrames）

新视频统一用 [HyperFrames](https://hyperframes.heygen.com) 做：HTML + GSAP 写画面，Studio 里预览与微调，渲染成 MP4。本目录是全系列共用的部分，取代过去每部视频各复制一份、又各自改动的 `core.js` / `render.mjs` / `vo.py`。2026-10 之前的视频保持原样，不迁移。

## 目录

| 文件 | 作用 |
| --- | --- |
| `frame.md` | 系列风格规范：颜色、字体、版式骨架、字号下限（锁死的部分） |
| `brand.css` | 同一套规范的 CSS 变量与类：页眉 `kit-*`、字幕 `kit-sub`、代码框 `k-code`、瓶颈标签 `k-bottleneck` |
| `blocks/` | 可复用的画面积木，目前有 `tower.js`（硬件塔：五层、SM 框、沿层画路径） |
| `template/` | 新视频的骨架：`index.html`、`script.json`、`CLAUDE.md`、`package.json` |
| `tools/new.mjs` | 新建一部视频 |
| `tools/sync.mjs` | 按 `script.json` 重排节拍，生成字幕、配音片段与各幕页眉 |
| `tools/vo.py` | MiniMax 配音，按句缓存（付费；`--dry-run` 先看计费） |

新建时 `brand.css`、`blocks/`、`frame.md` 会**复制**进视频目录，此后改工具包不影响已有视频，旧片随时能原样重渲。

## 一部视频的流程

```bash
node videos/_kit/tools/new.mjs <slug>          # 建 videos/<slug>/
cd videos/<slug>
# 1. 写 script.json：每幕的版本名、标题、出处、进度条，每拍的画面动作（act）与配音（vo），给要定时的拍写 id
npm run sync                                    # 2. 按估计时长排节拍、生成字幕与页眉
# 3. 写 compositions/<幕>.html 的机制画面（kit 区域之外），动画用 B.<拍 id> 定时
npm run check && npm run dev                    # 4. 检查；在 Studio 里看、拖、改
python3 ../_kit/tools/vo.py . --dry-run         # 5. 方向确认后看配音要合成几句、多少钱
npm run vo && npm run sync                      # 6. 录音，节拍按实测时长重排，画面自动跟着走
npm run render                                  # 7. 出片（renders/，不进仓库）
```

`script.json` 是文案与节拍的唯一来源：改一句配音、插一拍、换进度条数字，都只改它再 `npm run sync`。机制画面的动画引用拍的名字，不写死秒数，所以重排后不用改。

## 规则

- **锁死**：配色、字体、页眉（系列名 / 版本名 / 标题 / 出处 / 进度条）、字幕样式、瓶颈标签、语义色（A 蓝、W 黄、C 紫、显存橙、片上绿）。这些只在 `brand.css` / `frame.md` 里定义，幕文件里只用变量和类。
- **同一时刻只亮一个主体**：部件在配音讲到时才出场，讲完缩小或淡出；不要从第 0 秒摆满。
- **字号**：要读的字 ≥ 22 px，关键数字与标签 ≥ 26 px；放不下就删内容。
- **每幕一个文件**，机制画面的每个部件是带 id 的 `.clip`，Studio 里能单独选中。
- 讲法（问题链、先介绍再使用、画面与配音分工）见 `docs/video.md`。

## 防同质化：变化轴

固定的部分负责“认得出是这个系列”，下面几条每部视频重新选，开工前对照 `videos/EPISODES.md` 最近 3 部，避开相同的组合：

| 轴 | 可选 |
| --- | --- |
| 主视觉 | 硬件塔 / 矩阵与 tile 图 / 时间线与 trace / 实测代码与终端 / 论文原图 / 数据曲线 |
| 空间 | 平面分区 / 等轴测立体 / 逐层推镜头 / 一镜到底的长画布 |
| 节奏 | 多台阶快推 / 少数转折点慢讲 / 先给结论再倒推 |
| 强调色 | teal 之外本片的第二强调（蓝 / 紫 / 橙 中选一） |
| 本片特别设计 | 一个只属于这一部的画面或动效（写一句话） |

完成一部后在 `videos/EPISODES.md` 记一行。
