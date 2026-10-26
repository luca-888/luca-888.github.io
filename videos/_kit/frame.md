---
name: luca888 GPU 讲解系列
canvas: { width: 1920, height: 1080, fps: 30 }
css: brand.css          # 下面的颜色与字体都在 brand.css 里定义为 CSS 变量，scene 里只用变量
colors:
  paper: "#f1ede4"      # 全片唯一背景
  paper2: "#e6e0d3"     # 次级面板、进度条底
  panel: "#e7e1d4"      # 代码框底
  ink: "#141414"        # 正文、标题
  ink2: "#3a3a3a"       # 次要文字、代码
  note: "#6b6556"       # 注释、单位、出处（纸色底上 4.96:1）
  gray: "#8a8474"       # 只用于线条、虚线框、刻度，不用于要读的字（3.19:1）
  line: "#cfc8b8"
  cell: "#e2ddd0"
  teal: "#0fa391"       # 系列主强调：Tensor Core、当前版本、实测进度（实心块）
  tealD: "#0b7d70"      # 标题里的强调词（大字）
  blue: "#2350d8"       # 矩阵 A、选中的线程
  warp: "#a9c2ee"       # 同一 warp
  yel: "#f4c21b"        # 矩阵 W / B
  purple: "#7c3ff5"     # 结果 C
  hot: "#c4561d"        # 从显存读取、瓶颈标签
  use: "#2f8a57"        # 片上数据（共享内存、寄存器里已就绪）
  red: "#e5322d"        # 划掉、错误
  small-text: "纸色底上 26 px 以下的彩色字用 brand.css 的 *-t 变体（--teal-t、--hot-t 等），对比度 ≥ 4.5:1"
fonts:
  zh: '"PingFang SC"（brand.css 里已按 PostScript 名声明各字重；只写 local("PingFang SC") 在无头 Chrome 里加载不到）'
  mono: '"JetBrains Mono"（HyperFrames 自带）'
type:
  series-tag: { size: 26, weight: 600, color: ink }
  version-chip: { font: mono, size: 30, weight: 700, color: "#fff", bg: ink }
  title: { size: 56, weight: 800, color: ink, accent: tealD }
  label: { size: 26, weight: 600, color: ink2 }
  note: { size: 22, weight: 500, color: note }
  code: { font: mono, size: 24, weight: 500, color: ink2, bg: panel }
  subtitle: { size: 36, weight: 600, color: "#fff", bg: "#1d1c19", position: "底部居中，距底 64 px" }
---

# 系列风格

颜色、字体、版式骨架、字幕样式全系列锁死；**每幕画面的构成、镜头与动画按内容自己设计**，不沿用上一部的版式（防同质化的做法见 `README.md` 的“变化轴”）。讲法原则见 `docs/video.md`。

## 版式骨架（`blocks/header.html` 已实现）

- 左上：系列名小字 → 黑底白字版本名色块（等宽）→ 大标题“第 N 步：改动”，改动名用 tealD 强调。
- 右上：出处一行小字；其下随台阶更新的小进度条（实心 teal 条 + 白色数字，右端虚线刻度标参照值，如 cuBLAS）。结果数字只在这里更新，不单独占一屏。
- 底部居中：黑底白字字幕，一句一条（`blocks/captions.html`，由 `tools/sync.mjs` 按配音生成）。
- 中部是机制画面。**同一时刻只亮一个主体**，其它部件缩小、淡出或还没出现；部件在配音讲到时才出场，不要从第 0 秒摆满。

## 视觉语言

- 语义色全片一致：A 蓝、W 黄、C 紫；从显存读取 = 橙，片上就绪 = 绿；Tensor Core 与当前版本 = teal；瓶颈 = 橙底白字“瓶颈”标签（`.k-bottleneck`），贴在**上一版自己的画面**上。
- 系列母题“硬件塔”（`blocks/tower.html`）：显存 HBM → L2 → 共享内存 → 寄存器 → Tensor Core 五层等轴测薄板自下而上，SM 内三层虚线框住。数据用沿真实路径移动的小方块表示；每一步改动只改塔上的路径。
- 矩阵按经典位置排开（A 左、Wᵀ 上、C 右下），按比例画，正在用的块高亮。
- 实心色块加细线，不用阴影、渐变、光效、emoji；圆角 2–4 px。
- 运动用缓入缓出，动的是机制本身（数据移动、块滑动、计数变化）；需要读的数字停留到能读完。

## 字号下限

成片会在手机上看（视频号、小红书）：要读的字不小于 22 px，关键数字与标签不小于 26 px，代码 24 px。放不下就删内容，不缩字。
