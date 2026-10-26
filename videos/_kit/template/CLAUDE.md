# __SLUG__（HyperFrames 视频）

这部视频由 `videos/_kit/` 新建。先读 `videos/_kit/README.md`（流程与规则）、本目录的 `frame.md`（系列风格，锁死的部分）、仓库的 `docs/video.md`（讲法）。改画面前读 HyperFrames 插件的 `/hyperframes` skill。

## 哪里改什么

| 要改 | 改哪里 | 之后 |
| --- | --- | --- |
| 配音稿、字幕、页眉标题、版本名、出处、进度条数字、拍的顺序 | `script.json` | `npm run sync` |
| 配音录音（MiniMax，付费） | `script.json` 改好后 `npm run vo`（先加 `--dry-run` 看要合成几句、多少字符） | `npm run sync` |
| 某一幕的机制画面 | `compositions/<幕>.html` 里 kit 区域**以外**的部分，动画用 `B.<拍的 id>` 定时 | `npm run check` |
| 颜色、字体、页眉与字幕样式 | `assets/brand.css`（只影响本片） | — |

- `compositions/captions.html` 与各幕的 `kit:header` / `kit:beats` 区域、`index.html` 的 `kit:hosts` 区域由 sync 生成，手改会在下次 sync 时被覆盖。
- 在 Studio 里拖动的幕长、配音位置也会被下次 sync 按 `script.json` 改回；要固定某一拍的时间，在 `script.json` 里给它写 `"t"`。
- 每次改完：`npm run check`，再 `npx hyperframes snapshot --at <秒>` 看改到的时刻。
- 完成一部后，在 `videos/EPISODES.md` 记一行本片在各“变化轴”上的选择。
