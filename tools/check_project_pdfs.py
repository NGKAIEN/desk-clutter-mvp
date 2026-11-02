"""PDF text/bounds checks and optional contact sheets of existing Poppler PNGs.

Run after build_project_pdfs.py and pdftoppm; contact sheets supplement (not
replace) visual review. Optional deps: requirements-docs.txt (Pillow via ReportLab).
"""
import argparse
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact-sheets', action='store_true')
    args = parser.parse_args()
    audit = []
    for name, prefix, expected in [('desk-clutter-change-review.pdf', 'history', 13),
                                  ('desk-clutter-knowledge-handbook.pdf', 'knowledge', 16)]:
        path = ROOT / 'output/pdf' / name
        reader = PdfReader(path)
        assert len(reader.pages) == expected, (name, 'Unexpected page count', len(reader.pages))
        assert len(reader.outline) == expected, (name, 'Missing chapter bookmarks')
        details = []
        with pdfplumber.open(path) as pdf:
            for number, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ''
                assert len(text) > 250, (name, number, 'Possible orphan/blank page')
                assert '\ufffd' not in text and '\x00' not in text, (name, number, 'Broken text')
                bad = [c for c in page.chars if c['x0'] < 35 or c['x1'] > page.width - 35
                       or c['top'] < 10 or c['bottom'] > page.height - 12]
                assert not bad, (name, number, 'Out-of-page text')
                details.append(dict(page=number, characters=len(text), images=len(page.images)))
        audit.append(dict(file=name, pages=expected, bookmarks=len(reader.outline), details=details))
        print(f'{name}: {expected} pages; text, bounds and chapter bookmarks OK.')
        if args.contact_sheets:
            for start in range(1, expected + 1, 4):
                sheet = Image.new('RGB', (1100, 1600), '#d6e0e4')
                draw = ImageDraw.Draw(sheet)
                for offset, number in enumerate(range(start, min(start + 4, expected + 1))):
                    page = Image.open(ROOT / 'tmp/pdfs' / f'{prefix}-{number:02d}.png').convert('RGB')
                    page.thumbnail((530, 752))
                    x, y = 10 + (offset % 2) * 550, 30 + (offset // 2) * 800
                    sheet.paste(page, (x, y))
                    draw.text((x, y - 20), f'{prefix.upper()} / PAGE {number:02d}', fill='black')
                target = ROOT / 'tmp/pdfs' / f'contact-{prefix}-{start:02d}.png'
                sheet.save(target)
                print(target)
    target = ROOT / 'data/documentation/pdf_qa_20261102.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
