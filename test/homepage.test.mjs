import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");
const exists = (path) => existsSync(new URL(`../${path}`, import.meta.url));

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

test("homepage remains a writing-first bilingual research blog", () => {
  const index = read("src/pages/index.astro");

  assert.match(index, /getCollection\(["']blog["']/);
  assert.match(index, /AI, Systems, and Research Notes/);
  assert.match(index, /写一些关于大模型、机器学习、工程实践和论文阅读的笔记/);
  assert.match(index, /Latest Notes/);
  assert.doesNotMatch(index, /About me|自我介绍|bio/i);
});

test("Astro content collection defines blog post metadata", () => {
  const config = read("src/content.config.ts");
  const post = read("src/content/blog/welcome.md");

  assert.match(config, /defineCollection/);
  assert.match(config, /glob\(\{\s*pattern:\s*["']\*\*\/\*\.md["']/);
  assert.match(config, /pubDate:\s*z\.coerce\.date/);
  assert.match(post, /title:/);
  assert.match(post, /description:/);
  assert.match(post, /pubDate:/);
  assert.match(post, /tags:/);
});

test("post routes render collection entries", () => {
  const route = read("src/pages/writing/[...slug].astro");

  assert.match(route, /getStaticPaths/);
  assert.match(route, /render/);
  assert.match(route, /entry\.data\.title/);
});
