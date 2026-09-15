import type { Plugin } from 'vite';
import type { Element, Root } from 'hast';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeShiki from '@shikijs/rehype';
import rehypeStringify from 'rehype-stringify';
import { visit, SKIP } from 'unist-util-visit';
import { toString } from 'hast-util-to-string';
import { remarkBoldGuard } from './remark-bold-guard';

// Accept the LaTeX delimiters used in the source draft, outside fenced code.
function normalizeMath(source: string) {
  let fenced = false;
  return source.split('\n').map((line) => {
    if (line.startsWith('```')) fenced = !fenced;
    return fenced ? line : line.replaceAll('\\[', () => '$$').replaceAll('\\]', () => '$$').replaceAll('\\(', '$').replaceAll('\\)', '$');
  }).join('\n');
}

function element(tagName: string, properties: Element['properties'], children: Element['children']): Element {
  return { type: 'element', tagName, properties, children };
}

export function markdown(): Plugin {
  return {
    name: 'blog-markdown',
    enforce: 'pre',
    async transform(source, id) {
      if (!id.endsWith('.md')) return;
      let title = '';
      const codeLanguages: string[] = [];
      let headingCount = 0;
      const output = await unified()
        .use(remarkParse)
        .use(remarkGfm)
        .use(remarkMath)
        .use(remarkBoldGuard)
        .use(remarkRehype)
        .use(() => (tree: Root) => {
          tree.children = tree.children.filter((node) => {
            if (node.type === 'element' && node.tagName === 'h1') {
              title = toString(node);
              return false;
            }
            return true;
          });
          visit(tree, 'element', (node) => {
            if (node.tagName === 'p' && toString(node) === '::rmsnorm-demo::') {
              node.tagName = 'div';
              node.properties.id = 'rmsnorm-demo';
              node.children = [];
            }
            if (node.tagName === 'h2' || node.tagName === 'h3') {
              const headingId = `section-${++headingCount}`;
              node.properties.id = headingId;
              node.properties.tabIndex = -1;
            }
            if (node.tagName === 'pre') {
              const code = node.children.find((child): child is Element => child.type === 'element' && child.tagName === 'code');
              const languageClass = (code?.properties.className as string[] | undefined)?.find((name) => name.startsWith('language-'));
              const language = languageClass?.slice(9) ?? 'text';
              if (language !== 'math') codeLanguages.push(language);
            }
          });
        })
        .use(rehypeKatex, { strict: 'error' })
        .use(rehypeShiki, { themes: { light: 'github-light', dark: 'github-dark' }, langs: ['python'] })
        .use(() => (tree: Root) => {
          visit(tree, 'element', (node, index, parent) => {
            if (node.tagName === 'span' && (node.properties.className as string[] | undefined)?.includes('katex-display')) {
              node.properties.tabIndex = 0;
              node.properties.role = 'group';
              node.properties.ariaLabel = '公式，可横向滚动';
            }
            if (node.tagName !== 'pre' || index === undefined || !parent) return;
            node.properties.tabIndex = 0;
            const language = codeLanguages.shift() === 'python' ? 'Python' : 'Text';
            parent.children[index] = element('div', { className: ['code-block'] }, [
              element('div', { className: ['code-toolbar'] }, [
                element('span', {}, [{ type: 'text', value: language }]),
                element('button', { type: 'button', 'data-copy-code': '', ariaLabel: '复制代码' }, [{ type: 'text', value: '复制' }]),
              ]),
              node,
            ]);
            return SKIP;
          });
        })
        .use(rehypeStringify)
        .process({ value: normalizeMath(source), path: id });
      if (output.messages.length) this.error(output.messages.map(String).join('\n'));
      return { code: `export default ${JSON.stringify({ title, html: String(output) })}`, map: null };
    },
  };
}
