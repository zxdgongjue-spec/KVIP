"""Reject masked or partial per-launch correctness evidence."""
import unittest

from report import validate_audits, cq_timing_available, resolve_simt_post_clocks


class AuditValidationTest(unittest.TestCase):
    def test_single_launch(self):
        self.assertEqual(validate_audits('mode=4 posted=146 bad=0\n'), (4, 1))

    def test_both_launches_verified(self):
        log = ('mode=9 posted=146 bad=0\nverified_launch=1\n'
               'mode=9 posted=146 bad=0\nverified_launch=2\n')
        self.assertEqual(validate_audits(log), (9, 2))

    def test_retained_queue_requires_both_launches(self):
        first = 'mode=15 posted=146 bad=0\nverified_launch=1\n'
        with self.assertRaises(ValueError):
            validate_audits(first)
        self.assertEqual(validate_audits(
            first + 'mode=15 posted=146 bad=0\nverified_launch=2\n'), (15, 2))
        with self.assertRaises(ValueError):
            validate_audits(first + 'mode=15 posted=146 bad=1\nverified_launch=2\n')

    def test_incomplete_or_failed_evidence(self):
        logs = [
            'mode=9 posted=146 bad=0\n',
            'mode=9 posted=146 bad=0\nmode=9 posted=146 bad=0\n',
            'mode=9 posted=146 bad=1\nmode=9 posted=146 bad=0\n',
            'mode=9 posted=146 bad=0\nmode=9 posted=146 bad=1\n',
            'mode=4 posted=146 bad=0\nFAIL retirement\n',
            'mode=4 posted=145 bad=0\n',
            'mode=9 posted=146 bad=0\nmode=4 posted=146 bad=0\n',
        ]
        for log in logs:
            with self.subTest(log=log), self.assertRaises(ValueError):
                validate_audits(log)


class CQTimingValidationTest(unittest.TestCase):
    def test_complete_or_overlapped_wait(self):
        for end in (15.0, 19.0):
            self.assertTrue(cq_timing_available([
                {'stamps_us': {4: 10.0, 16: 14.0, 17: end}} for _ in range(4)]))

    def test_missing_or_invalid_post(self):
        for stamps in ({16: 14.0, 17: 15.0}, {4: 16.0, 16: 14.0, 17: 15.0},
                       {4: 10.0, 16: 11.0, 17: 11.01}):
            self.assertFalse(cq_timing_available([{'stamps_us': stamps} for _ in range(4)]))


class SimtClockValidationTest(unittest.TestCase):
    def test_exact_native_clock_match(self):
        events=[{'name':'SIMT_S2RL','ts':t,'args':{'code':'__clang_cce_simt_spr.h:33'}}
                for t in (20.0,21.0)]
        times={18000:10.0,54000:30.0}
        resolve_simt_post_clocks(events,times,{4:36000,5:37800},0)
        self.assertEqual(times[36000],20.0)
        self.assertEqual(times[37800],21.0)

    def test_wrong_clock_or_ambiguous_event_rejected(self):
        events=[{'name':'SIMT_S2RL','ts':t,'args':{'code':'__clang_cce_simt_spr.h:33'}}
                for t in (20.0,21.0)]
        for found,stamps in ((events,{4:36000,5:38000}),(events[:1],{4:36000,5:37800})):
            with self.assertRaises(ValueError):
                resolve_simt_post_clocks(found,{18000:10.0,54000:30.0},stamps,0)


if __name__ == '__main__':
    unittest.main()
