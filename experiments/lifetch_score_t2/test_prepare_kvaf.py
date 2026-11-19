# SPDX-License-Identifier: Apache-2.0
"""Offline source-isolation checks; no NPU build or remote Git access."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_kvaf


class DependencyPreparation(unittest.TestCase):
    def fixture(self, root):
        source=root/'current';source.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(source),*args],text=True,stderr=subprocess.DEVNULL).strip()
        git('init','-b','main')
        git('config','user.name','test');git('config','user.email','test@example.invalid')
        (source/'api.h').write_text('pinned API\n')
        git('add','api.h');git('commit','-m','pinned');pinned=git('rev-parse','HEAD')
        (source/'api.h').write_text('new API\n')
        git('commit','-am','new version');head=git('rev-parse','HEAD')
        (source/'api.h').write_text('uncommitted user edit\n')
        (source/'build-npu').mkdir();(source/'build-npu/binary.so').write_bytes(b'new ABI binary')
        return source,pinned,head,git

    def test_pinned_clone_preserves_source_and_excludes_binaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,pinned,head,git=self.fixture(root)
            before=git('status','--porcelain')
            umdk=source/'third_party/umdk/include';umdk.mkdir(parents=True)
            (umdk/'urma_api.h').write_text('header')
            before=git('status','--porcelain')
            target=root/'pinned'
            with patch.object(prepare_kvaf,'REVISION',pinned), redirect_stdout(io.StringIO()):
                result,headers=prepare_kvaf.prepare(source,target)
            self.assertEqual(git('rev-parse','HEAD'),head)
            self.assertEqual(git('status','--porcelain'),before)
            self.assertEqual((source/'api.h').read_text(),'uncommitted user edit\n')
            self.assertEqual((result/'api.h').read_text(),'pinned API\n')
            self.assertFalse((result/'build-npu').exists())
            self.assertEqual(headers,umdk.parent.resolve())
            self.assertTrue((result/'third_party/umdk').is_symlink())
            # Git history has its own objects and does not need the source.
            source.rename(root/'source-moved')
            self.assertEqual(prepare_kvaf.git(result,'rev-parse','HEAD'),pinned)

    def test_unmanaged_destination_and_missing_history_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,pinned,head,git=self.fixture(root)
            target=root/'existing';target.mkdir();(target/'keep').write_text('keep')
            with patch.object(prepare_kvaf,'REVISION',pinned), self.assertRaisesRegex(RuntimeError,'not a managed'):
                prepare_kvaf.prepare(source,target)
            self.assertEqual((target/'keep').read_text(),'keep')
            missing=root/'missing'
            with patch.object(prepare_kvaf,'REVISION','f'*40), self.assertRaisesRegex(RuntimeError,'No network fetch was attempted'):
                prepare_kvaf.prepare(source,missing)
            self.assertFalse(missing.exists())
            self.assertEqual(git('rev-parse','HEAD'),head)


if __name__=='__main__':unittest.main()
