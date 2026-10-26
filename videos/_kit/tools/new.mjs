// node videos/_kit/tools/new.mjs <slug> [目标目录]
// 新建一部 HyperFrames 视频：复制 template/、brand.css、blocks/、frame.md，再跑一次 sync 生成首版时间线。
// 目标目录默认 videos/<slug>/；已存在就停下，不覆盖。
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const KIT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const slug = process.argv[2];
if (!slug || !/^[a-z0-9][a-z0-9-]*$/.test(slug)) { console.error('用法：node videos/_kit/tools/new.mjs <slug>（小写字母、数字与连字符）'); process.exit(1); }
const dir = resolve(process.argv[3] || join(KIT, '..', slug));
if (existsSync(dir)) { console.error(`${dir} 已存在，不覆盖`); process.exit(1); }

cpSync(join(KIT, 'template'), dir, { recursive: true, filter: (f) => !f.endsWith('compositions/scene.html') });   // scene.html 是 sync 新建幕用的模板，不进项目
mkdirSync(join(dir, 'assets'), { recursive: true });
copyFileSync(join(KIT, 'brand.css'), join(dir, 'assets/brand.css'));
cpSync(join(KIT, 'blocks'), join(dir, 'assets/kit'), { recursive: true });
copyFileSync(join(KIT, 'frame.md'), join(dir, 'frame.md'));
for (const f of ['package.json', 'CLAUDE.md']) { const p = join(dir, f); writeFileSync(p, readFileSync(p, 'utf8').replaceAll('__SLUG__', slug)); }
writeFileSync(join(dir, 'meta.json'), JSON.stringify({ id: slug, name: slug }, null, 2) + '\n');
execFileSync('node', [join(KIT, 'tools/sync.mjs'), dir], { stdio: 'inherit' });
console.log(`→ ${dir}\n下一步：写 script.json 的文案与节拍 → npm run sync → 写各幕机制画面 → npm run check → npm run dev 在 Studio 里看`);
