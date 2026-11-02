"""Export an explicit public-file allowlist; never initialize Git or upload.

Creates a new directory with source/, a ZIP and a SHA-256 inventory. No symlinks,
private captures or environment files are accepted. Text scanning is a guard,
not a substitute for the author's final review of the archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / 'tools/release_files.txt'
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                    r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{20,})')
PERSONAL_PATH = re.compile('/' + '(?:home|Users)/' + r'[A-Za-z0-9_.-]+/')


def checked_source(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or not rel.parts:
        raise ValueError(f'Unsafe relative path: {name}')
    if any(p.startswith(('.git', '.venv', '.env')) and p != '.github' and p != '.gitignore'
           for p in rel.parts):
        raise ValueError(f'Private/tooling path: {name}')
    if rel.parts[0] in {'height_backgrounds', 'height_diagnostics', 'plane_calibrations',
                        'outputs', 'output', 'tmp', 'releases'}:
        raise ValueError(f'Private/output path: {name}')
    if rel.parts[0] == 'data' and name != 'data/README.md':
        raise ValueError(f'Private data path: {name}')
    if name.startswith('docs/archive/') or rel.suffix in {'.npz', '.bag', '.pcd', '.ply', '.pdf'}:
        raise ValueError(f'Archive/capture format: {name}')
    if rel.name.startswith('calibration') and rel.suffix == '.json':
        raise ValueError(f'Private calibration: {name}')
    if rel.suffix.lower() in {'.png', '.jpg', '.jpeg'} and name != 'docs/images/v05-demo.png':
        raise ValueError(f'Image needs explicit privacy review: {name}')
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Symlink not allowed: {name}')
    if not current.is_file():
        raise ValueError(f'Missing release file: {name}')
    if current.suffix.lower() != '.png':
        text = current.read_text(encoding='utf-8')
        if SECRET.search(text) or PERSONAL_PATH.search(text):
            raise ValueError(f'Potential credential/personal path; review file: {name}')
    return current


def prepare(output_dir):
    names = [line.strip() for line in ALLOWLIST.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if (ROOT / 'LICENSE').exists():
        names.append('LICENSE')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate allowlist entry')
    sources = [(name, checked_source(ROOT, name)) for name in names]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = output_dir / f'desk-clutter-mvp-v0.5-{stamp}'
    destination.mkdir(parents=True, exist_ok=False)
    source_dir = destination / 'source'
    inventory = []
    archive_path = destination / 'desk-clutter-mvp-v0.5-source.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sources:
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            inventory.append(dict(path=name, bytes=target.stat().st_size, sha256=digest))
            archive.write(target, 'desk-clutter-mvp/' + name)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), files=inventory,
                  file_count=len(inventory), license_present=(source_dir / 'LICENSE').is_file(),
                  notice_present=(source_dir / 'NOTICE').is_file(),
                  zip_name=archive_path.name, zip_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  review_required=True, uploaded=False)
    (destination / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared: {destination}')
    print(f'Files: {len(inventory)} | LICENSE present: {report["license_present"]} | Nothing uploaded')
    if not report['license_present']:
        print('DRAFT: choose a project license before presenting this as an open-source release.')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'releases')
    args = parser.parse_args()
    try:
        prepare(args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Release preparation failed: {error}\n')


if __name__ == '__main__':
    main()
