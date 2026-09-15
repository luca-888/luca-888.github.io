import type { Root } from 'mdast';
import type { VFile } from 'vfile';
import { visit } from 'unist-util-visit';

// Check parsed prose, so code and math may still contain literal **.
export function remarkBoldGuard() {
  return (tree: Root, file: VFile) => {
    const source = String(file);
    visit(tree, 'text', (node) => {
      if (!node.value.includes('**')) return;
      const raw = source.slice(node.position?.start.offset, node.position?.end.offset);
      // Explicitly escaped stars and character entities are intentional text.
      if (!/(?<!\\)(?:\\\\)*\*\*/u.test(raw)) return;
      file.fail(
        '未解析的加粗标记 **：请将末尾标点放在加粗范围外（**标题**：正文），整句加粗后留空格或换行；字面星号请使用行内代码或转义。',
        node,
        'blog-markdown:unparsed-bold',
      );
    });
  };
}
