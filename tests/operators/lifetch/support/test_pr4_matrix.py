"""Host-only checks for matrix accounting and failed/incomparable runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('matrix', Path(__file__).resolve().parents[1] / 'compare_pr4_matrix.py')
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class MatrixSummaryTest(unittest.TestCase):
    def test_preserve_cann_acl_path(self):
        inherited = '/usr/local/Ascend/cann/python/site-packages:/custom/python'
        result = matrix.worker_pythonpath(Path('/selected'), Path('/kvaf'), inherited)
        self.assertEqual(result, '/selected/python:/kvaf/build-npu/python:' + inherited)
        self.assertEqual(matrix.worker_pythonpath(Path('/selected'), Path('/kvaf'), None),
                         '/selected/python:/kvaf/build-npu/python')

    def populate(self, root):
        for batch in matrix.BATCHES:
            for miss in matrix.MISSES:
                for variant in ('pr4', 'main'):
                    path = root / f'{variant}-b{batch}-m{miss}' / 'case0'
                    path.mkdir(parents=True)
                    report = dict(status='pass', fixture={'miss': miss}, native_misses_per_request=[miss]*batch,
                                  timing={'native_chain': {'p50_us': 100, 'p99_us': 110},
                                          'prototype_with_update': {'p50_us': 60 if variant == 'pr4' else 80,
                                                                    'p99_us': 90}})
                    (path / 'report.json').write_text(json.dumps(report))

    def test_twenty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate(root)
            result = matrix.summarize(root)
            self.assertIn('Completed comparison rows: 20/20', result)
            self.assertEqual(result.count('+40.00%'), 20)
            self.assertEqual(len((root / 'results.csv').read_text().splitlines()), 21)

    def test_missing_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('0/20', matrix.summarize(Path(tmp)))



if __name__ == '__main__':
    unittest.main()
