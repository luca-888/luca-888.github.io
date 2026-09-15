import test from 'node:test';
import assert from 'node:assert/strict';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';
import { remarkBoldGuard } from '../build/remark-bold-guard.ts';

function render(value) {
  return unified().use(remarkParse).use(remarkMath).use(remarkBoldGuard)
    .use(remarkRehype).use(rehypeStringify)
    .processSync({ value, path: 'example.md' }).toString();
}

test('rejects broken Chinese bold boundaries and reports the source line', () => {
  for (const prose of ['**算尺度：**每个 token', '**一句话。**下一句', '**变换：**BatchNorm']) {
    assert.throws(() => render(`开头\n\n${prose}`), error => {
      assert.equal(error.ruleId, 'unparsed-bold');
      assert.equal(error.line, 3);
      assert.match(error.message, /未解析的加粗标记/);
      return true;
    });
  }
});

test('safe punctuation and spacing produce actual strong elements', () => {
  const html = render('**算尺度**：每个 token\n\n**一句话。** 下一句\n\n**$r$ 是整行共享的系数。**');
  assert.equal((html.match(/<strong>/g) ?? []).length, 3);
  assert.ok(!html.includes('**'));
});

test('allows literal stars in code, math, escapes, and entities', () => {
  for (const prose of ['`x ** 2`', '```python\nx ** 2\n```', '$x^{**}$', String.raw`\*\*`, '&#42;&#42;']) {
    assert.doesNotThrow(() => render(prose));
  }
});
