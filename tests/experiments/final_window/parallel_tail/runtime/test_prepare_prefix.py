"""Source preparation must work in a previously built checkout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefix_prepare_test', HERE / 'prepare_prefix.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparePrefixTest(unittest.TestCase):
    def test_build_artifacts_and_dangling_links_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            for name in ('lightning_indexer_with_fetch', 'common', 'lidu'):
                base = root / 'csrc/operators' / name
                (base / 'op_kernel').mkdir(parents=True)
                (base / 'op_kernel/source.h').write_text('// source\n')
                for build in ('build', 'build_old', 'op_kernel/build'):
                    folder = base / build
                    folder.mkdir(parents=True)
                    (folder / 'liboptiling.so').symlink_to('missing.so')
                (base / 'libunused.so').symlink_to('missing.so')
            host = root / 'csrc/operators/lightning_indexer_with_fetch/op_host'
            host.mkdir()
            (host / 'launch.cpp').write_text('// launch\n')
            dest = Path(tmp) / 'prepared'
            with patch.object(module, 'ROOT', root):
                module.prepare_prefix(dest)
            for name in ('lightning_indexer_with_fetch', 'common', 'lidu'):
                base = dest / 'csrc/operators' / name
                self.assertEqual((base / 'op_kernel/source.h').read_text(), '// source\n')
                self.assertFalse(list(base.rglob('*.so')))
                self.assertFalse(list(base.rglob('build*')))
            self.assertTrue((dest / 'producer_sources.json').is_file())
