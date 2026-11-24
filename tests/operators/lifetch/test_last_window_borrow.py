"""Normal vs terminal-only LI schedule evidence, with no NPU dependency."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'support'))
from last_window_borrow import check_flags, decode_mailbox, summarize


class BorrowAuditTest(unittest.TestCase):
    def setUp(self):
        self.words = np.zeros((16, 4, 32), dtype=np.uint64)
        self.words[:, 2, 16] = 2
        self.words[:, 2, 20] = 1

    def decode(self):
        return decode_mailbox(self.words, 16384, 131328)

    def test_zero_help_and_two_arrival_points(self):
        no_help = self.decode()
        self.assertEqual(summarize(dict.fromkeys(range(20), no_help))['retained'], 320)
        ready = (131328 << 32) | 91000
        for row in range(2):
            self.words[row, 2, 16:21] = [1, ready, ready, 0, row]
            self.words[row, 3, 16:18] = [1, 1]
        counts = summarize(dict.fromkeys(range(20), self.decode()))
        self.assertEqual((counts['delegated'], counts['before_first_q'], counts['before_second_q']), (40, 20, 20))

    def test_bad_mailbox_fails(self):
        for q, word, value in ((2, 16, 0), (2, 16, 1), (2, 16, 3), (2, 20, 0), (3, 17, 1), (3, 16, 2)):
            with self.subTest(q=q, word=word, value=value):
                saved = self.words[0, q, word]
                self.words[0, q, word] = value
                with self.assertRaises(ValueError):
                    self.decode()
                self.words[0, q, word] = saved
        self.words[0, 2, 16] = self.words[0, 3, 16] = self.words[0, 3, 17] = 1
        with self.assertRaisesRegex(ValueError, 'ready range'):
            self.decode()

    def test_mismatched_combined_and_stale_options(self):
        for detail in (True, False):
            check_flags(0x18020400, int(detail), detail)
            for li, tail in ((0x98020400, int(detail)), (0x18020400, 1-int(detail)), (0x18020400, 3)):
                with self.assertRaisesRegex(ValueError, 'diagnostics flags'):
                    check_flags(li, tail, detail)
        with self.assertRaises(ValueError):
            summarize(dict.fromkeys(range(19), self.decode()))


if __name__ == '__main__':
    unittest.main()
