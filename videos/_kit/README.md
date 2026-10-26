# 视频工具包

HyperFrames 标准项目之外，本系列只多这几样（流程见 `docs/video.md`）：

| 文件 | 作用 |
| --- | --- |
| `new.mjs` | `node videos/_kit/new.mjs <slug>`：`hyperframes init` 后放入下面三样 |
| `frame.md` | 系列风格：颜色、字体、版式骨架、字号下限 |
| `brand.css` | 同一套风格的 CSS 变量与类（页眉 `.kit-*`、字幕 `.kit-sub`、代码框 `.k-code`、瓶颈标签 `.k-bottleneck`） |
| `blocks/tower.js` | 硬件塔 `kitTower()` |
| `vo.py` | MiniMax 配音：读 `narration.md`，每句一个 wav 加 `vo.json`（付费，先 `--dry-run`） |

这些文件在新建时**复制**进视频目录，之后改工具包不影响已有视频。
