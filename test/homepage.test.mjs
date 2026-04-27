import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");
const exists = (path) => existsSync(new URL(`../${path}`, import.meta.url));
const articlePath = "src/content/blog/qwen3-vl.md";

test("project is an Astro blog instead of a Jekyll site", () => {
  const packageJson = JSON.parse(read("package.json"));
  const workflow = read(".github/workflows/deploy.yml");
  const astroConfig = read("astro.config.mjs");

  assert.equal(packageJson.scripts.dev, "astro dev");
  assert.equal(packageJson.scripts.build, "astro build");
  assert.ok(packageJson.dependencies.astro);
  assert.match(workflow, /withastro\/action@v/);
  assert.match(astroConfig, /site:\s*["']https:\/\/luca-888\.github\.io["']/);
  assert.equal(exists(".github/workflows/jekyll-gh-pages.yml"), false);
  assert.equal(exists("_config.yml"), false);
  assert.equal(exists("_layouts/default.html"), false);
  assert.equal(exists("_posts/2026-04-20-welcome.md"), false);
});

test("homepage is a minimal writing-first notes index", () => {
  const index = read("src/pages/index.astro");

  assert.match(index, /getCollection\(["']blog["']/);
  assert.match(index, />Latest</);
  assert.doesNotMatch(index, /Topics|topic-index|topics/);
  assert.doesNotMatch(index, /home-hero|site-title|hero-copy|AI, Systems, and Research Notes/);
  assert.doesNotMatch(index, /About me|自我介绍|bio/i);
});

test("homepage does not keep unused hero styling", () => {
  const css = read("src/styles/global.css");

  assert.doesNotMatch(css, /\.home-hero|\.hero-copy|\.kicker/);
  assert.match(css, /\.post-index\s*\{\s*padding:\s*36px 0 44px;/);
});

test("site theme uses a soft light palette instead of harsh black", () => {
  const css = read("src/styles/global.css");
  const astroConfig = read("astro.config.mjs");

  assert.match(css, /--paper:\s*#eef3ee;/);
  assert.match(css, /--surface:\s*#fffdf8;/);
  assert.match(css, /--ink:\s*#26312b;/);
  assert.doesNotMatch(css, /--ink:\s*#(?:000|111111|171817)/i);
  assert.match(css, /background:\s*linear-gradient\(180deg, #f7fbf8 0%, var\(--paper\) 46%, #f5efe6 100%\);/);
  assert.match(css, /\.post-content :where\(pre, \.astro-code\)/);
  assert.match(css, /background:\s*#fff9ee !important;/);
  assert.match(astroConfig, /shikiConfig:\s*\{\s*theme:\s*["']github-light["']/);
});

test("navigation stays minimal and does not promote GitHub", () => {
  const layout = read("src/layouts/BaseLayout.astro");

  assert.match(layout, />Notes</);
  assert.doesNotMatch(layout, />Topics</);
  assert.doesNotMatch(layout, />Research Notes</);
  assert.doesNotMatch(layout, />GitHub</);
});

test("Astro content collection defines blog post metadata", () => {
  const config = read("src/content.config.ts");
  const post = read(articlePath);

  assert.match(config, /defineCollection/);
  assert.match(config, /glob\(\{\s*pattern:\s*["']\*\*\/\*\.md["']/);
  assert.match(config, /pubDate:\s*z\.coerce\.date/);
  assert.match(post, /title:/);
  assert.match(post, /description:/);
  assert.match(post, /pubDate:/);
  assert.match(post, /tags:/);
  assert.match(post, /image:\s*"\/assets\/blog\/qwen3-vl\/qwen3vl_arc\.jpg"/);
});

test("article tables are styled and model overview remains tabular", () => {
  const post = read(articlePath);
  const css = read("src/styles/global.css");

  assert.match(post, /\| 项目 \| 内容 \|/);
  assert.match(post, /\| 模型名称 \| Qwen3-VL \|/);
  assert.match(css, /\.post-content table/);
  assert.match(css, /\.post-content th/);
  assert.match(css, /\.post-content td/);
});

test("article references are styled as visible external links", () => {
  const post = read(articlePath);
  const css = read("src/styles/global.css");

  assert.match(post, /<div class="reference-list">/);
  assert.match(post, /class="reference-card"/);
  assert.match(post, /class="reference-icon"/);
  assert.match(post, /Qwen3-VL-8B-Instruct Model Card/);
  assert.match(post, /Qwen3-VL Technical Report/);
  assert.match(post, /QwenLM\/Qwen3-VL/);
  assert.doesNotMatch(post, /Qwen2\.5-VL Technical Report|Qwen2-VL Technical Report|Qwen3 Technical Report/);
  assert.doesNotMatch(post, /quickstart code|architecture updates|release notes/);
  assert.match(css, /\.post-content a\[href\^="http"\]/);
  assert.match(css, /\.reference-card::after/);
  assert.match(css, /grid-template-columns:\s*40px 1fr;/);
  assert.match(css, /\.reference-icon img/);
});

test("reference icons are loaded automatically from link domains", () => {
  const layout = read("src/layouts/BaseLayout.astro");

  assert.match(layout, /document\.querySelectorAll\("\.reference-card\[href\]"\)/);
  assert.match(layout, /new URL\("\/favicon\.ico", url\.origin\)/);
  assert.match(layout, /icon\.dataset\.fallback/);
});

test("mermaid support is not bundled", () => {
  const packageJson = JSON.parse(read("package.json"));
  const layout = read("src/layouts/BaseLayout.astro");
  const css = read("src/styles/global.css");
  const post = read(articlePath);

  assert.equal(packageJson.dependencies.mermaid, undefined);
  assert.doesNotMatch(layout, /mermaid|data-language="mermaid"/);
  assert.doesNotMatch(css, /\.mermaid-block|mermaid-error/);
  assert.doesNotMatch(post, /```mermaid|Qwen3-VL 总体结构：/);
});

test("qwen article has cleaned export formatting", () => {
  const post = read(articlePath);

  assert.doesNotMatch(post, /docoder-only|关键技术改进:|架构关键词:|注:|注意:|qianwen-res/);
  assert.match(post, /decoder-only/);
  assert.match(post, /关键技术改进：/);
  assert.match(post, /架构关键词：/);
  assert.match(post, /"Qwen3VLForConditionalGeneration"/);
  assert.match(post, /Interleaved-MRoPE/);
  assert.match(post, /deepstack_visual_indexes/);
});

test("post routes render collection entries", () => {
  const route = read("src/pages/writing/[...slug].astro");

  assert.match(route, /getStaticPaths/);
  assert.match(route, /render/);
  assert.match(route, /entry\.data\.title/);
});
