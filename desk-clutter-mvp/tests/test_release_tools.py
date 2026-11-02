"""Public-export guards, using synthetic temporary files only."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import zipfile

from tools import prepare_release as release


class ReleaseTests(unittest.TestCase):
    def test_private_and_traversal_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['../secret', '/secret', 'x\\secret', '.env', '.git/config',
                         '.venv/bin/python', 'data/background.json', 'calibration.json',
                         'outputs/report.json', 'docs/archive/old.md', 'photo.png', 'sample.npz']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    release.checked_source(Path(directory), name)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'original.md').write_text('synthetic')
            (root / 'link.md').symlink_to('original.md')
            with self.assertRaises(ValueError):
                release.checked_source(root, 'link.md')

    def test_suspicious_text_rejected_without_echoing_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for content in ['ghp_' + 'x' * 30, '/' + 'home/' + 'example/private/file']:
                (root / 'README.md').write_text(content)
                with self.assertRaises(ValueError) as error:
                    release.checked_source(root, 'README.md')
                self.assertNotIn(content, str(error.exception))

    def test_public_file_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('Only synthetic content')
            self.assertEqual(release.checked_source(root, 'README.md'), root / 'README.md')

    def test_export_inventory_and_no_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'README.md').write_text('Synthetic public file')
            (root / 'calibration.json').write_text('Do not publish')
            allowlist = root / 'files.txt'
            allowlist.write_text('README.md\n')
            with patch.object(release, 'ROOT', root), patch.object(release, 'ALLOWLIST', allowlist):
                result = release.prepare(root / 'releases')
            manifest = json.loads((result / 'manifest.json').read_text())
            self.assertEqual(manifest['file_count'], 1)
            self.assertFalse(manifest['license_present'])
            self.assertFalse((result / 'source/calibration.json').exists())
            with zipfile.ZipFile(result / manifest['zip_name']) as archive:
                self.assertEqual(archive.namelist(), ['desk-clutter-mvp/README.md'])

    def test_license_and_notice_preserved_in_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            license_text = 'Synthetic license fixture\n'
            notice_text = 'Synthetic author notice\n'
            (root / 'LICENSE').write_text(license_text)
            (root / 'NOTICE').write_text(notice_text)
            allowlist = root / 'files.txt'
            allowlist.write_text('NOTICE\n')
            with patch.object(release, 'ROOT', root), patch.object(release, 'ALLOWLIST', allowlist):
                result = release.prepare(root / 'releases')
            manifest = json.loads((result / 'manifest.json').read_text())
            self.assertTrue(manifest['license_present'])
            self.assertTrue(manifest['notice_present'])
            self.assertEqual(manifest['file_count'], 2)
            with zipfile.ZipFile(result / manifest['zip_name']) as archive:
                self.assertEqual(archive.read('desk-clutter-mvp/LICENSE').decode(), license_text)
                self.assertEqual(archive.read('desk-clutter-mvp/NOTICE').decode(), notice_text)


if __name__ == '__main__':
    unittest.main()
