from pathlib import Path
import unittest

from report import parse_run


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.log = Path(__file__).with_name('capacity_v10-2048.log').read_text()

    def test_complete_run(self):
        self.assertEqual(parse_run(self.log)['duration_us'], 57.79)

    def test_rejects_incomplete_or_failed_evidence(self):
        variants = [
            self.log.replace('verified_launch=2', ''),
            self.log.replace('audit_index=0', 'audit_index=1', 1),
            self.log.replace('audit_output=0', '', 1),
            self.log.replace('mode=9 posted=146 bad=0', 'mode=9 posted=146 bad=1', 1),
            self.log.replace('Op profiling finish.', ''),
            self.log.replace('core1.veccore1', 'core1.veccore0'),
        ]
        for variant in variants:
            with self.subTest(variant=variants.index(variant)):
                with self.assertRaises(ValueError):
                    parse_run(variant)


if __name__ == '__main__':
    unittest.main()
