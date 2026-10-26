// node videos/_kit/new.mjs <slug>
// 新建一部视频：运行 hyperframes init（blank），再放入系列风格：frame.md、assets/brand.css、assets/kit/（积木）。
// 目标目录 videos/<slug>/ 已存在就停下，不覆盖。
import { cpSync, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const HF = 'hyperframes@0.8.200';
const KIT = dirname(fileURLToPath(import.meta.url));
const slug = process.argv[2];
if (!slug || !/^[a-z0-9][a-z0-9-]*$/.test(slug)) { console.error('用法：node videos/_kit/new.mjs <slug>（小写字母、数字与连字符）'); process.exit(1); }
const videos = resolve(KIT, '..'), dir = join(videos, slug);
if (existsSync(dir)) { console.error(`${dir} 已存在，不覆盖`); process.exit(1); }

execFileSync('npx', ['--yes', HF, 'init', slug, '--example', 'blank', '--non-interactive'], { cwd: videos, stdio: 'inherit', env: { ...process.env, HYPERFRAMES_SKIP_SKILLS: '1' } });

mkdirSync(join(dir, 'assets'), { recursive: true });
copyFileSync(join(KIT, 'frame.md'), join(dir, 'frame.md'));
copyFileSync(join(KIT, 'brand.css'), join(dir, 'assets/brand.css'));
cpSync(join(KIT, 'blocks'), join(dir, 'assets/kit'), { recursive: true });

// index.html：去掉 blank 模板的占位样式，改用系列样式
const ip = join(dir, 'index.html');
let html = readFileSync(ip, 'utf8').replace(/\s*<style>[\s\S]*?<\/style>/, '');
html = html.replace(/\n\s*<\/head>/, '\n    <link rel="stylesheet" href="assets/brand.css" />\n    <script src="assets/kit/tower.js"></script>\n  </head>');
writeFileSync(ip, html);

// init 生成的通用说明有 7 KB，换成几行
const note = `# ${slug}

HyperFrames 视频。流程与约定见仓库的 \`docs/video.md\`；系列风格见本目录 \`frame.md\` 与 \`assets/brand.css\`（不换）。
改画面前先读 HyperFrames 插件的 \`/hyperframes\` skill。改完 \`npm run check\`，截帧用 \`npx ${HF} snapshot --at <秒> --describe false\`。
配音稿写在 \`narration.md\`，合成：\`python3 ../_kit/vo.py . --dry-run\` 看费用，再去掉 --dry-run。
`;
writeFileSync(join(dir, 'CLAUDE.md'), note);
writeFileSync(join(dir, 'AGENTS.md'), note);
writeFileSync(join(dir, 'narration.md'), `# ${slug} 配音稿\n\n每幕一个 \`## 幕 id\`，每行一句（即一条字幕）。\n\n## s01\n第一句配音。\n`);
console.log(`→ videos/${slug}\n下一步：cd videos/${slug} && npm run dev；让 Claude 用 /hyperframes 按 frame.md 做。`);
