#!/usr/bin/env python3
"""Create a marker-based Zhihu import draft and a local PNG upload manifest."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from urllib.parse import unquote, urlsplit

IMAGE = re.compile(r'!\[(?P<alt>(?:\\.|[^\]\\])*)\]\((?P<path><[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prose_lines(text):
    """Yield offsets and lines outside code fences, including neither fence."""
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line.rstrip('\r\n'))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
        else:
            yield offset, line
        offset += len(line)
    if fence:
        raise ValueError('Unclosed code fence; fix the Markdown before importing.')


def code_blocks(text):
    result, fence, language, lines = [], None, '', []
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line.rstrip('\r\n'))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                result.append({'language': language, 'content': ''.join(lines)})
                fence, lines = None, []
            else:
                lines.append(line)
        elif marker:
            fence, language = marker[1], marker[2].strip()
    if fence:
        raise ValueError('Unclosed code fence.')
    return result


def prepare(md_path, source=None, title=None):
    md_path = md_path.resolve()
    text = md_path.read_text(encoding='utf-8')
    lines = list(prose_lines(text))
    if any(re.fullmatch(r'\s*::[a-z][a-z0-9-]*::\s*', line) for _, line in lines):
        raise ValueError('Unexpanded component marker; export static content first.')
    if 'ZHIMG-' in text:
        raise ValueError('Use the static original Markdown, not an already prepared web-import draft.')
    first = next(((offset, line) for offset, line in lines if line.strip()), None)
    heading = re.match(r'^#\s+(.+?)\s*$', first[1].rstrip()) if first else None
    article_title = title or (heading[1] if heading else None)
    if not article_title:
        raise ValueError('Add a first-line # title or pass --title.')
    blocks = code_blocks(text)
    if source is not None and code_blocks(source.read_text(encoding='utf-8')) != blocks:
        raise ValueError('Fenced code differs from the source article; review before importing.')

    edits, images = [], []
    if heading:
        edits.append((first[0], first[0] + len(first[1]), ''))
    for offset, line in lines:
        for match in IMAGE.finditer(line):
            if line[:match.start()].count('`') % 2:
                continue
            raw_path = match['path'].strip('<>')
            parsed = urlsplit(raw_path)
            if parsed.scheme or parsed.netloc:
                raise ValueError(f'Expected an exported local PNG, got {raw_path}')
            asset = (md_path.parent / unquote(parsed.path)).resolve()
            data = asset.read_bytes()
            if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
                raise ValueError(f'Expected a PNG image: {asset}')
            width, height = struct.unpack('>II', data[16:24])
            if not width or not height:
                raise ValueError(f'Invalid image dimensions: {asset}')
            ordinal = len(images) + 1
            token = f'ZHIMG-{ordinal:03d}'
            alt = match['alt'] or asset.stem
            placeholder = f'【配图 {token}｜{alt}】'
            start, end = offset + match.start(), offset + match.end()
            images.append({
                'ordinal': ordinal, 'token': token, 'placeholder': placeholder,
                'alt': alt, 'markdown_path': raw_path, 'absolute_path': str(asset),
                'sha256': sha(data), 'width_px': width, 'height_px': height,
                'context_before': IMAGE.sub('', text[max(0, start - 260):start]).strip()[-180:],
                'context_after': IMAGE.sub('', text[end:end + 260]).strip()[:180],
            })
            edits.append((start, end, '\n' + placeholder + '\n'))
    web = text
    for start, end, replacement in sorted(edits, reverse=True):
        web = web[:start] + replacement + web[end:]
    web = web.lstrip('\n')
    assert code_blocks(web) == blocks
    web_path = md_path.with_name(md_path.stem + '-web-import.md')
    manifest_path = md_path.parent / 'upload-map.json'
    manifest = {
        'title': article_title, 'markdown_path': str(md_path),
        'markdown_sha256': sha(text.encode()), 'web_import_path': str(web_path),
        'web_import_sha256': sha(web.encode()), 'image_count': len(images),
        'code_blocks': [{'language': b['language'], 'sha256': sha(b['content'].encode())} for b in blocks],
        'source_code_checked': source is not None,
        'headings': [line.strip() for _, line in lines if re.match(r'^#{2,6}\s', line)],
        'images': images,
    }
    web_path.write_text(web, encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('markdown', type=Path)
    parser.add_argument('--source', type=Path, help='Optional original Markdown: verify fenced code is unchanged.')
    parser.add_argument('--title', help='Title when the draft has no first-line # heading.')
    args = parser.parse_args()
    try:
        result = prepare(args.markdown, args.source, args.title)
    except (OSError, ValueError, AssertionError) as error:
        parser.exit(1, f'Export check failed: {error}\n')
    print(f"Prepared {result['image_count']} images: {result['web_import_path']}")
    print(f"Upload map: {args.markdown.resolve().parent / 'upload-map.json'}")


if __name__ == '__main__':
    main()
