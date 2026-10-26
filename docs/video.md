# 视频

按 HyperFrames 的标准流程做，本仓库只加一份系列风格和几条约定。怎么讲、怎么动、怎么检查，以 HyperFrames 插件的 `/hyperframes` skill 为准。旧的做法在 `docs/legacy/`，不用读。

## 流程

```bash
node videos/_kit/new.mjs <slug>     # = hyperframes init + 放入系列风格（frame.md、assets/brand.css）
cd videos/<slug>
npm run dev                         # Studio 预览，边改边看
npm run check                       # 检查
npm run render                      # 出片到 renders/
```

中间的事交给 `/hyperframes` skill：先给方案（几幕、每幕讲什么）等用户确认，再写 `compositions/`，check、截帧自查，打开 Studio 交用户看，用户认可后再渲染。

## 约定

- **风格**：颜色、字体、页眉、字幕样式照项目里的 `frame.md` 与 `assets/brand.css`，不换；每幕的画面构成按内容自己设计。
- **画幅**：默认 16:9（1920 × 1080）；用户要竖版时 9:16，内容放在中间 80% 宽度。
- **配音**：MiniMax，写好 `narration.md` 后用 `python3 ../_kit/vo.py .` 合成（付费，先加 `--dry-run` 看费用）；读法见 [docs/tts.md](tts.md)。配音逐句烧成底部字幕。
- **文案**：书面、准确的短句；配音讲“为什么、看哪里”，不念画面上的字。渲染前用 `de-ai-edit` skill 过一遍。
- **素材**：讲代码、性能时放真实的代码、输出与实测数字；示意内容标明。
- **文件**：源文件进仓库；`renders/`、`assets/vo/`、`.hyperframes/` 不进。上传 B 站后把 BV 号填进 `src/posts.ts`。
