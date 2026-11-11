"""Host-only checks for matrix accounting and failed/incomparable runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('matrix', Path(__file__).resolve().parents[1] / 'compare_matrix.py')
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


class MatrixSummaryTest(unittest.TestCase):
    def populate(self, root):
        for batch in matrix.BATCHES:
            for miss in matrix.MISSES:
                for variant in ('current', 'main'):
                    path = root / f'{variant}-b{batch}-m{miss}' / 'case0'
                    path.mkdir(parents=True)
                    report = dict(status='pass', fixture={'miss': miss}, native_misses_per_request=[miss]*batch,
                                  timing={'native_chain': {'p50_us': 100, 'p99_us': 110},
                                          'prototype_with_update': {'p50_us': 60 if variant == 'current' else 80,
                                                                    'p99_us': 90}})
                    (path / 'report.json').write_text(json.dumps(report))

    def test_sixty_rows_and_twenty_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate(root)
            result = matrix.summarize(root)
            self.assertIn('Completed comparison rows: 60/60', result)
            self.assertEqual(result.count('+40.00% | +25.00%'), 20)
            self.assertEqual(len((root / 'results.csv').read_text().splitlines()), 61)

    def test_missing_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn('0/60', matrix.summarize(Path(tmp)))

    def test_different_fixtures_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.populate(root)
            path = root / 'main-b1-m3/case0/report.json'
            report = json.loads(path.read_text())
            report['fixture'] = {'miss': 99}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'Fixture metadata differs'):
                matrix.summarize(root)


if __name__ == '__main__':
    unittest.main()
