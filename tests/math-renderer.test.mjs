import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const rendererRequire = createRequire(require.resolve('rehype-katex'));

test('KaTeX renderer matches the package providing browser styles and fonts', () => {
  assert.equal(rendererRequire('katex').version, require('katex').version,
    'Mismatched KaTeX versions can break fraction, radical and box positioning.');
});
