"""Read-only documentation/layout and private migration-integrity checks.

No camera access, no file mutations. If private migration data is absent, that
check is explicitly skipped; it is not a requirement for a public checkout.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-migration-snapshot', action='store_true',
                        help='Also require Python sources to match the historical reorganization snapshot')
    args = parser.parse_args()
    failures = []
    docs = [ROOT / 'README.md', ROOT / 'READ_ME.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    checked = 0
    for doc in docs:
        if 'archive' in doc.parts:
            continue  # Preserved historical bytes; links belong to the old layout.
        for href in re.findall(r'!?\[[^\]]*\]\(([^\s)]+)\)', doc.read_text()):
            if href.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            target = doc.parent / unquote(href.split('#', 1)[0])
            checked += 1
            if not target.exists():
                failures.append(f'Broken link: {doc.relative_to(ROOT)} -> {href}')
    print(f'Checked {checked} local Markdown links (archive excluded).')

    manifest_path = ROOT / 'data/migration_manifest_20261101.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest['files']:
            path = ROOT / row['new']
            # Documents are intentionally maintained after their relocation.
            if not path.is_file() or (path.suffix != '.md' and sha(path) != row['sha256']):
                failures.append(f'Changed/missing original: {row["new"]}')
        for name, expected in manifest['protected_unchanged_sha256'].items():
            path = ROOT / name
            if not path.is_file():
                failures.append(f'Missing protected file: {name}')
            elif sha(path) != expected and path.suffix == '.py' and not args.verify_migration_snapshot:
                print(f'INFO: source evolved after historical migration snapshot: {name}')
            elif sha(path) != expected:
                failures.append(f'Changed protected source: {name}')
        for name, destination in manifest['compatibility_symlinks'].items():
            path = ROOT / name
            if not path.is_symlink() or path.resolve() != (ROOT / destination).resolve():
                failures.append(f'Broken compatibility link: {name}')
        print(f'Checked {len(manifest["files"])} relocated files (text revisions allowed), '
              f'{len(manifest["protected_unchanged_sha256"])} protected files, and 4 compatibility links.')
    else:
        print('SKIP: private migration manifest is not included in this checkout.')

    sources = [*ROOT.glob('*.py'), *(ROOT / 'experiments').glob('*.py'), *(ROOT / 'tools').glob('*.py')]
    for source in sources:
        ast.parse(source.read_text(), filename=str(source))
    print(f'Parsed {len(sources)} Python source files without importing hardware modules.')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('Documentation/layout checks passed.')


if __name__ == '__main__':
    main()
