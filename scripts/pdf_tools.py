"""论文调研用的 PDF 工具：把矢量图原样导出为 SVG / PNG；没有 .tex 源文件或 HTML 版时兜底提取全文。

运行环境：~/.venvs/pdf-tools（pypdf + pymupdf）
    ~/.venvs/pdf-tools/bin/python scripts/pdf_tools.py text paper.pdf paper.txt
    ~/.venvs/pdf-tools/bin/python scripts/pdf_tools.py svg figure.pdf figure.svg [--page 0]
    ~/.venvs/pdf-tools/bin/python scripts/pdf_tools.py png figure.pdf figure.png [--page 0] [--dpi 200]

论文原图优先从 arXiv 源文件包取（https://arxiv.org/src/<id>，解压后 figures/ 下多为矢量 PDF），
导出 SVG 时文字转为路径，不截图、不重绘。
"""
import argparse
import pymupdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['text', 'svg', 'png'])
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--page', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    doc = pymupdf.open(args.src)
    if args.mode == 'text':
        with open(args.dst, 'w') as f:
            f.write('\n'.join(page.get_text() for page in doc))
    elif args.mode == 'svg':
        with open(args.dst, 'w') as f:
            f.write(doc[args.page].get_svg_image(text_as_path=True))
    else:
        doc[args.page].get_pixmap(dpi=args.dpi).save(args.dst)
    print(f'{args.mode}: {args.src} -> {args.dst}')


if __name__ == '__main__':
    main()
