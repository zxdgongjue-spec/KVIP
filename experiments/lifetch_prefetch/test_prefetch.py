# SPDX-License-Identifier: Apache-2.0
"""Host checks for the A5 report and production CQ join state machine."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_a5_prefetch_report import ProtocolReportTest  # Included in this entry point.


ROOT = Path(__file__).resolve().parents[2]


class CqJoinTest(unittest.TestCase):
    def test_one_cqe_per_nonempty_window_and_missing_tail(self):
        source = (ROOT / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/staged_fetch_worker.h').read_text()
        implementation = source.split('namespace rotating {', 1)[1].split('#undef V9_CQ_INLINE', 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            code = work / 'check.cpp'
            code.write_text(
                '#include <cassert>\n#include <cstdint>\n#define V9_CQ_INLINE inline\nnamespace rotating {'
                + implementation + '''
struct Cq {
    uint32_t words[4]{};
    uint64_t committed = 0;
    uint32_t Read(uint32_t slot) { return words[slot]; }
    void Commit(uint64_t consumer) { committed = consumer; }
};

int main() {
    Cq cq;
    // Two nonempty logical windows. The first has two doorbells and
    // contributes only its signaled tail. The next CQE wraps the CQ.
    cq.words[3] = 4; cq.words[0] = 0;
    auto ok = rotating::DrainCq(cq, 3, 4, 2, 3, 11, 12, 13);
    assert(!ok.error && ok.completed == 2 && cq.committed == 5);

    Cq missing;
    auto timedOut = rotating::DrainCq(missing, 3, 4, 1, 3, 11, 12, 13);
    assert(timedOut.error == 13 && !timedOut.completed && !missing.committed);

    Cq failed;
    failed.words[3] = 4; failed.words[0] = 1U << 16;
    auto error = rotating::DrainCq(failed, 3, 4, 2, 3, 11, 12, 13);
    assert(error.error == 12 && error.completed == 1 && failed.committed == 5);

    Cq empty;
    auto noPost = rotating::DrainCq(empty, 3, 4, 0, 3, 11, 12, 13);
    assert(!noPost.error && !noPost.polls && !empty.committed);
}
''')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            str(code), '-o', str(work / 'check')], check=True)
            subprocess.run([str(work / 'check')], check=True)


if __name__ == '__main__':
    unittest.main()
