import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const read = (path) => readFileSync(new URL(`../${path}`, import.meta.url), "utf8");

test("homepage is a writing-first bilingual Jekyll blog", () => {
  const index = read("index.html");

  assert.match(index, /AI, Systems, and Research Notes/);
  assert.match(index, /写一些关于大模型、机器学习、工程实践和论文阅读的笔记/);
  assert.match(index, /site\.posts/);
  assert.match(index, /Latest Notes/);
  assert.doesNotMatch(index, /About me|自我介绍|bio/i);
});

test("Jekyll layouts and styling are present", () => {
  const defaultLayout = read("_layouts/default.html");
  const postLayout = read("_layouts/post.html");
  const stylesheet = read("assets/css/main.css");

  assert.match(defaultLayout, /stylesheet/);
  assert.match(defaultLayout, /Writing/);
  assert.match(postLayout, /article/);
  assert.match(stylesheet, /--paper/);
  assert.match(stylesheet, /@media/);
});

test("sample post gives the empty blog a real latest-note entry", () => {
  const post = read("_posts/2026-04-20-welcome.md");

  assert.match(post, /layout: post/);
  assert.match(post, /subtitle:/);
  assert.match(post, /tags:/);
  assert.match(post, /LLM/);
});

test("brainstorm artifacts are ignored", () => {
  const gitignore = read(".gitignore");

  assert.match(gitignore, /^\.superpowers\/$/m);
});
