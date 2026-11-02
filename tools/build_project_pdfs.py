"""Render maintained Markdown sources into Chinese PDFs; no camera imports.

Optional deps: requirements-docs.txt. Usage: python tools/build_project_pdfs.py
Use --only knowledge on a public checkout without private history/screenshots.
"""
import argparse
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                               Image, Table, TableStyle, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = A4
MARGIN = 43
CONTENT_W = WIDTH - 2 * MARGIN
INK = colors.HexColor('#142D3B')
TEAL = colors.HexColor('#087F8C')
MUTED = colors.HexColor('#566974')


def register_fonts(font_dir=None):
    fonts = dict(Latin='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 LatinBold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 Mono='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
                 CJK='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font_dir:
        fonts = {key: str(font_dir / Path(path).name) for key, path in fonts.items()}
    for name, path in fonts.items():
        if not Path(path).exists():
            raise FileNotFoundError(f'Need {Path(path).name}; provide --font-dir with the four TTF fonts')
        pdfmetrics.registerFont(TTFont(name, path))


def runs(text, latin='Latin'):
    """Droid fallback contains Chinese but not ASCII; select fonts explicitly."""
    text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
    chunks, current, buffer = [], None, []
    for char in text:
        font = latin if ord(char) in pdfmetrics.getFont(latin).face.charWidths else 'CJK'
        if ord(char) not in pdfmetrics.getFont(font).face.charWidths and not char.isspace():
            raise ValueError(f'Unsupported glyph {char!r} U+{ord(char):04X}')
        if font != current and buffer:
            chunks.append(f'<font name="{current}">{escape("".join(buffer))}</font>')
            buffer = []
        current = font
        buffer.append(char)
    if buffer:
        chunks.append(f'<font name="{current}">{escape("".join(buffer))}</font>')
    return ''.join(chunks)


TOKEN = re.compile(r'(\[[^\]]+\]\([^\s)]+\)|\*\*.+?\*\*|`[^`]+`)')


def inline(text, bold=False):
    parts = []
    for token in TOKEN.split(text):
        if token.startswith('[') and '](' in token:
            label, url = token[1:-1].split('](', 1)
            parts.append(f'<link href="{escape(url, quote=True)}" color="#087F8C">{runs(label)}</link>')
        elif token.startswith('**') and token.endswith('**'):
            parts.append(runs(token[2:-2], 'LatinBold'))
        elif token.startswith('`') and token.endswith('`'):
            parts.append(runs(token[1:-1], 'Mono'))
        else:
            parts.append(runs(token, 'LatinBold' if bold else 'Latin'))
    return ''.join(parts)


def styles():
    base = dict(fontName='Latin', textColor=INK, wordWrap='CJK', splitLongWords=True,
                allowWidows=0, allowOrphans=0)
    return {
        'body': ParagraphStyle('body', fontSize=9.6, leading=15.5, spaceAfter=7, **base),
        'h1': ParagraphStyle('h1', fontSize=19, leading=28, spaceAfter=15, keepWithNext=True, **base),
        'h2': ParagraphStyle('h2', fontSize=12, leading=18, spaceBefore=10, spaceAfter=7, keepWithNext=True, **base),
        'cell': ParagraphStyle('cell', fontSize=8.7, leading=13.6, **base),
        'head': ParagraphStyle('head', fontSize=8.8, leading=14, **dict(base, textColor=colors.white)),
        'caption': ParagraphStyle('caption', fontSize=8.4, leading=13, spaceBefore=5, spaceAfter=9,
                                  **dict(base, textColor=MUTED)),
        'code': ParagraphStyle('code', fontSize=8.4, leading=13, leftIndent=9, rightIndent=9,
                               spaceBefore=3, spaceAfter=3, backColor=colors.HexColor('#EDF4F6'), **base),
        'bullet': ParagraphStyle('bullet', fontSize=9.6, leading=15.5, leftIndent=10,
                                 firstLineIndent=-8, spaceAfter=6, **base),
    }


class BookDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'h1':
            key = f'chapter-{self.page}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, 0, False)


def page_chrome(kind):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(2)
        canvas.line(MARGIN, HEIGHT - 34, WIDTH - MARGIN, HEIGHT - 34)
        canvas.setFont('LatinBold', 8)
        canvas.setFillColor(INK)
        canvas.drawString(MARGIN, HEIGHT - 24, 'DESK CLUTTER MVP')
        canvas.setFont('Latin', 7)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(WIDTH - MARGIN, HEIGHT - 24, kind)
        canvas.setStrokeColor(colors.HexColor('#D5E2E7'))
        canvas.setLineWidth(.5)
        canvas.line(MARGIN, 34, WIDTH - MARGIN, 34)
        canvas.drawString(MARGIN, 22, 'Evidence through 2026-11-01 | Documentation edition 2026-11-02')
        canvas.drawRightString(WIDTH - MARGIN, 22, f'{doc.page:02d}')
        canvas.restoreState()
    return draw


def markdown_story(path, style):
    lines = path.read_text(encoding='utf-8').splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('# '):
            story.append(Paragraph(inline(line[2:], bold=True), style['h1']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:], bold=True), style['h2']))
        elif line.startswith('```'):
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            i += 1
            story.append(KeepTogether([Paragraph(runs(x or ' ', 'Mono'), style['code']) for x in code]))
            story.append(Spacer(1, 5))
        elif line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip())
                i += 1
            rows = [r.strip('|').split('|') for r in rows if not re.fullmatch(r'[\s|:\-]+', r)]
            count = len(rows[0])
            weights = {2: [.35, .65], 3: [.44, .28, .28], 4: [.25] * 4}.get(count, [1/count] * count)
            if count == 2 and rows[0][0].strip() == '编号':
                weights = [.12, .88]
            cells = [[Paragraph(inline(c.strip(), bold=n == 0), style['head' if n == 0 else 'cell'])
                      for c in row] for n, row in enumerate(rows)]
            table = Table(cells, colWidths=[CONTENT_W * w for w in weights], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), INK),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EDF4F6'), colors.white]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#C8D9E0')),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
            if not match:
                raise ValueError(f'Invalid image syntax: {line}')
            picture = Image(str((path.parent / match[2]).resolve()))
            factor = min(CONTENT_W / picture.imageWidth, 255 / picture.imageHeight)
            picture.drawWidth, picture.drawHeight = picture.imageWidth * factor, picture.imageHeight * factor
            picture.hAlign = 'CENTER'
            story.append(KeepTogether([Spacer(1, 4), picture, Paragraph(inline(match[1]), style['caption'])]))
        elif line.startswith('- ') or re.match(r'^\d+\. ', line):
            text = '• ' + line[2:] if line.startswith('- ') else line
            story.append(Paragraph(inline(text), style['bullet']))
        else:
            para = [line]
            while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|```|<!--|- |\d+\. )', lines[i]):
                para.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(' '.join(para)), style['body']))
    return story


def build(source, destination, title, kind):
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = BookDoc(str(destination), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                  topMargin=53, bottomMargin=48, title=title,
                  author='Desk Clutter MVP project notes', subject=kind)
    doc.build(markdown_story(source, styles()), onFirstPage=page_chrome(kind), onLaterPages=page_chrome(kind))
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['all', 'history', 'knowledge'], default='all')
    parser.add_argument('--font-dir', type=Path, help='Directory containing the four TTF fonts in register_fonts')
    args = parser.parse_args()
    register_fonts(args.font_dir)
    documents = [
        ('history', ROOT / 'data/documentation/PROJECT_HISTORY.md', 'desk-clutter-change-review.pdf',
         'Desk Clutter MVP - 改动复盘', 'CHANGE REVIEW / LOCAL EVIDENCE'),
        ('knowledge', ROOT / 'docs/reference/KNOWLEDGE_HANDBOOK.md', 'desk-clutter-knowledge-handbook.pdf',
         'Desk Clutter MVP - 知识点手册', 'KNOWLEDGE HANDBOOK / REFERENCE'),
    ]
    for key, source, name, title, kind in documents:
        if args.only in ('all', key):
            build(source, ROOT / 'output/pdf' / name, title, kind)


if __name__ == '__main__':
    main()
