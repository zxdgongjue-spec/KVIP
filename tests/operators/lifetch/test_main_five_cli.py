"""Cover the reference selection used by the parent and its timed child."""
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
import main_five_run as replay


class ReplayReferenceTest(unittest.TestCase):
    def test_npu4_reference_selection_in_parent_and_worker(self):
        base = ['main_five_run.py', '--physical-npu', '4', '--main', 'main',
                '--trace', 'input', '--kvaf-root', 'kvaf', '--output', 'output',
                '--kvaf-server', '127.0.0.1:12345']
        for group in (10, 4, 17, 7, 2):
            for child in (False, True):
                for archive in (False, True):
                    with self.subTest(group=group, child=child, archive=archive):
                        reference = ['--owner-archive', 'archive'] if archive else ['--native-topk-reference']
                        argv = base + ['--group', str(group)] + reference + (['--worker'] if child else [])
                        with patch.object(sys, 'argv', argv), patch.object(replay, 'main') as parent, patch.object(replay, 'worker') as worker:
                            replay.run_cli()
                        called, other = (worker, parent) if child else (parent, worker)
                        called.assert_called_once()
                        other.assert_not_called()
                        args, = called.call_args.args
                        self.assertEqual(args.owner_archive, Path('archive') if archive else None)
                        self.assertEqual(args.native_topk_reference, not archive)
        for reference in ([], ['--owner-archive', 'archive', '--native-topk-reference']):
            with self.subTest(reference=reference):
                with patch.object(sys, 'argv', base + ['--group', '10'] + reference), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        replay.run_cli()
                self.assertEqual(error.exception.code, 2)
