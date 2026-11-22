import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('failure_report_test', HERE / 'failure_report.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FailureReportTest(unittest.TestCase):
    def test_index_evidence_and_nonfailed_rows(self):
        record = [16, 0, 0, 1, 0, 0, 230, 57, 58, 0, 0, 1, 60, 8, 17000, 9, 59]
        result = module.decode([[record]*4, [[0]*32]*4], [16, 0])
        self.assertEqual(len(result), 1)
        q = result[0]['queries'][0]
        self.assertEqual(q['index'], 1)
        self.assertEqual(q['bad_ordinal'], 60)
        self.assertEqual(q['source'], 17000)
        self.assertEqual(q['previous_ordinal'], 59)

    def test_success(self):
        self.assertEqual(module.decode([[[0]*32]*4], [0]), [])
