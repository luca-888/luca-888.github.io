# matmul-ladder-tc-hf：Tensor Core 矩阵乘法视频（第六版，待制作画面）

H100 上 BF16 矩阵乘法从 wmma 优化到接近 cuBLAS 的讲解视频，HyperFrames 制作。对应文章 `content/matmul-ladder-tc.md`。

## 目的与用户定下的规则（2026-10-27）

- **教原理，不是流水账和技术报告。** 每一步只教一个机制：为什么会卡、新方法为什么能解决，用动画演示；数字最多一句佐证。证据与数字留在文章和 EVIDENCE.md。
- **念出来的数字只有几个关键的**（32 个 bank、8 路冲突 → 1 轮、三分之二、16 个 tile）；TFLOPS、毫秒、百分比、功耗等不念。进度只靠右上角速度条（cuBLAS = 100%），不念出来。
- **标题写优化方式**（如“swizzle：把数据错开存，避免 bank 冲突”），不写“瓶颈 N”。
- **padding 不提。** 不放 Nsight 截图；改为重绘一张存储层级图（地图）贯穿全片。
- **画面跟字幕走、要有动作。** 每句配音对应画面上的一个动作（出现、点亮、流动、数字变化），动作在这句开始时发生；不是整屏静帧淡入淡出（第五版因此被否）。页眉、速度条、地图常驻，不随每句重绘。
- 系列风格不变：`frame.md`、`assets/brand.css`（纸色底、语义色、苹方 + JetBrains Mono、底部黑底字幕）。正文书面语；英文术语首次出现带一句中文解释。
- 流程：先做 s01、s02 **无声**低帧率预览给用户看，确认风格后再铺开全片、配音、出成片。审稿时用户逐条提意见，先记下，用户说改再一起改。

## 文件

| 文件 | 用途 |
| --- | --- |
| `OUTLINE.md` | 已定的大纲：骨架（存储层级图）、每步的机制与画面动作、删掉的内容 |
| `narration.md` | 已定的文字稿：9 幕（s01 开场、s02–s08 七步、s09 收尾）、52 句；每句下面的 `# 动作：` 行是该句画面动作（配音脚本跳过）；`## 读法` 是送 TTS 的改写 |
| `EVIDENCE.md` | 实测数据与证据（只作依据，不上画面） |
| `CODE.md` | 各版本 kernel 主循环带注释 |
| `frame.md`、`assets/brand.css` | 系列风格与语义色表 |
| `timing.py` | 排节拍：重写 beats.json、index.html 的幕宿主、compositions/captions.html，并把各幕 `compositions/<幕>.html` 里的 `const B = [...]`（每句起点，秒）与 `const DUR = …` 改成新值 |
| `assets/vendor/gsap.min.js`、`assets/kit/tower.js` | GSAP；可选的硬件塔组件 |

实测原始数据：`public/measurements/matmul-ladder-tc/`（`summary.json`、CSV）；脚本：`scripts/matmul-ladder/`。

## 制作流程

1. 每幕写 `compositions/sNN.html`（`<template>` + `#root`，`data-composition-id="sNN"`，一条 paused GSAP 时间线注册到 `window.__timelines["sNN"]`，脚本包在 IIFE 里）。时间线里写 `const B = [0.6, …]; const DUR = …;`，所有动作写成 `B[i] + 偏移`，这样换配音后只需重跑 timing.py。字体用 `var(--zh)` / `var(--mono)`（写字面 “PingFang SC” 会被 lint 拦）。
2. `python3 timing.py --only s01,s02`：无声，按字数估计每句时长；index.html 只挂列出的幕。
3. `npm run check`（必须 0 error）；`npx hyperframes@0.8.200 snapshot --at <秒> --describe false` 抽帧自查。check 和 snapshot 不要并行跑（抢浏览器）；macOS 没有 `timeout` 命令。
4. 预览用低帧率渲染：`npx hyperframes@0.8.200 render --fps 24 --quality draft --output build/preview.mp4`（build/ 已忽略）。
5. 配音：`python3 ../_kit/vo.py . --dry-run` 看费用（全片约 1–2 元，MiniMax，扣开放平台余额；运行前 unset 掉 *_proxy 环境变量），再去掉 `--dry-run`。录音缓存按句复用，改过的句子才重录。然后 `python3 timing.py --vo`，再 check、渲染。
6. 读音：`narration.md` 的 `## 读法` 里已有 `行 → 航`（防止读成 xíng）；稿子里不要用“进行、执行、并行”等读 xíng 的词。新发现的误读加到读法表。

## 现状

- 文字稿已定，画面还没做：`compositions/` 里只有 captions.html，index.html 已按 `timing.py --only s01,s02` 只挂 s01、s02（两个文件待写），字幕按无声估计排好。
- `assets/vo/` 里是第五版的配音缓存（已忽略），第六版要重新配音。
