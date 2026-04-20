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

test("homepage is a minimal writing-first notes index", () => {
  const index = read("src/pages/index.astro");

  assert.match(index, /getCollection\(["']blog["']/);
  assert.match(index, />Latest</);
  assert.match(index, />Topics</);
  assert.doesNotMatch(index, /home-hero|site-title|hero-copy|AI, Systems, and Research Notes/);
  assert.doesNotMatch(index, /About me|自我介绍|bio/i);
});

test("homepage does not keep unused hero styling", () => {
  const css = read("src/styles/global.css");

  assert.doesNotMatch(css, /\.home-hero|\.hero-copy|\.kicker/);
  assert.match(css, /\.post-index\s*\{\s*padding:\s*36px 0 44px;/);
});

test("navigation stays minimal and does not promote GitHub", () => {
  const layout = read("src/layouts/BaseLayout.astro");

  assert.match(layout, />Notes</);
  assert.match(layout, />Topics</);
  assert.doesNotMatch(layout, />Research Notes</);
  assert.doesNotMatch(layout, />GitHub</);
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
