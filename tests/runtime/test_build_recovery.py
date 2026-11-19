# SPDX-License-Identifier: Apache-2.0
"""Failed CANN preprocess builds must never reuse modified object files."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kvip.runtime import build


class BuildRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'csrc/operators/lightning_indexer_with_fetch'
        source.mkdir(parents=True)
        (source / 'indexer.cpp').write_text('// kernel\n')
        selector = source.parent / 'lidu/op_kernel/hot_slot_selection.h'
        selector.parent.mkdir(parents=True)
        selector.write_text('// selector\n')
        headers = self.root / 'headers'
        headers.mkdir()
        (headers / 'kvaf_batch_get_device.h').write_text('// interface\n')
        self.options = dict(cache_dir=self.root / 'cache', cann=self.root / 'cann',
                            headers=headers, soc='Ascend950PR_9599')
        self.addCleanup(patch.stopall)
        patch.object(build, '__file__', str(self.root / 'python/kvip/runtime/build.py')).start()

    def test_failed_attempt_preserved_and_retry_is_fresh_then_cached(self):
        attempts = []

        def compile_once(command, **kwargs):
            if command[1] != '--build':
                return SimpleNamespace(returncode=0)
            attempt = Path(command[2])
            attempts.append(attempt)
            # A failed attempt may already contain a binary and rewritten .o.
            (attempt / 'libv9_bridge.so').write_bytes(b'partial' if len(attempts) == 1 else b'complete')
            if len(attempts) == 1:
                obj = attempt / 'preprocess/CMakeFiles/aiv_obj.dir/indexer.cpp.o'
                obj.parent.mkdir(parents=True)
                obj.write_bytes(b'\x7fELF\x02\x01' + bytes(10) + b'\x02\x00\x29\x10' + bytes(12))
                kwargs['stdout'].write('ld.lld: error: indexer.cpp.o: unknown file type\n')
                return SimpleNamespace(returncode=1)
            self.assertFalse((attempt / 'preprocess').exists())
            return SimpleNamespace(returncode=0)

        with patch.object(build.subprocess, 'run', side_effect=compile_once) as run:
            with self.assertRaisesRegex(RuntimeError, 'EXEC/already linked'):
                build.build_libraries(**self.options)
            cache = attempts[0].parent
            failed_log = (attempts[0] / 'build.log').read_bytes()
            self.assertFalse((cache / 'complete.json').exists())
            self.assertFalse((cache / 'libv9_bridge.so').exists())
            libraries = build.build_libraries(**self.options)
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertEqual(attempts[0].parent, attempts[1].parent)
            self.assertEqual((attempts[0] / 'build.log').read_bytes(), failed_log)
            self.assertEqual((cache / 'build.log').resolve(), (attempts[1] / 'build.log').resolve())
            self.assertEqual(libraries[0].read_bytes(), b'complete')
            self.assertEqual(json.loads((cache / 'complete.json').read_text())['build_directory'], str(attempts[1]))
            self.assertEqual(build.build_libraries(**self.options), libraries)
            self.assertEqual(run.call_count, 4)

    def test_configure_failure_retained_without_completion(self):
        with patch.object(build.subprocess, 'run', return_value=SimpleNamespace(returncode=2)):
            with self.assertRaisesRegex(RuntimeError, 'configure failed'):
                build.build_libraries(**self.options)
        cache = next((self.root / 'cache').iterdir())
        self.assertTrue((cache / 'build.log').is_file())
        self.assertFalse((cache / 'complete.json').exists())
        self.assertFalse((cache / 'libv9_bridge.so').exists())

    def test_missing_library_cannot_publish_completion(self):
        with patch.object(build.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(RuntimeError, 'required libraries'):
                build.build_libraries(**self.options)
        cache = next((self.root / 'cache').iterdir())
        self.assertFalse((cache / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
