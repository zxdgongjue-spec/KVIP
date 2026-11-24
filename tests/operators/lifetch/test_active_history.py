# SPDX-License-Identifier: Apache-2.0
"""Exercise the shared scorer/service/terminal schedule, including unequal row lengths."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_all_live_tiles_and_both_shards_drain(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for the native schedule")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "schedule.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include "lightning_indexer_with_fetch/include/runtime_shape.h"
namespace rotating {
#include "common/lifetch/sequence_schedule.h"
}
using namespace rotating;
int main() {
    RuntimeShape s{};
    s.batch = 16; s.serviceLanes = 4; s.history = 524544; s.resident = 16384;
    s.splitQ = 1; s.indexHeads = 32; s.rounds = 4;
    uint32_t ends[64]{};
    // A high causal token can be in any Q of any row assigned to the same lane.
    ends[4*12+1] = 524291; ends[4*13+2] = 130290;
    assert(ActiveScanHistory(s, ends, 0) == s.history);
    assert(ActiveScanHistory(s, ends, 1) == 130304);
    assert(ActiveScanHistory(s, ends, 13) == 130304);
    assert(SequenceHalfWindow(s, 130304) == SequenceHalfWindow(s, 131328));
    assert(ActiveScanHistory(s, static_cast<const uint32_t*>(nullptr), 2) == s.history);
    ends[4*14] = UINT32_MAX;
    assert(ActiveScanHistory(s, ends, 2) == s.history);
    // Each active physical tile must be scored exactly once, and the final
    // window must have a real last tile on both workers for cross-core flags.
    for (uint32_t windows = 2; windows <= 32; ++windows) {
        s.rounds = windows;
        if (!UseSequenceSplit(s)) continue;
        for (uint32_t valid = 0; valid < s.history; valid += 127) {
            ends[3 * 4 + 1] = valid;
            const auto active = ActiveScanHistory(s, ends, 3);
            const auto half = SequenceHalfWindow(s, active);
            assert(valid <= active && active <= s.history && active % 256 == 0);
            uint32_t cursor = s.resident;
            for (uint32_t epoch = 1; epoch < windows; ++epoch) {
                for (uint32_t shard = 0; shard < 2; ++shard) {
                    auto begin = s.resident + (2 * (epoch - 1) + shard) * half;
                    auto end = std::min(begin + half, active);
                    assert(begin == cursor && begin < end);
                    auto slice = SequenceSliceAt(begin, s.resident, half);
                    assert(slice.shard == shard && slice.end >= end);
                    cursor = end;
                }
            }
            assert(cursor == active);
        }
    }
    s.history = 131328; s.rounds = 4;
    assert(ActiveScanHistory(s, ends, 3) == s.history);
}
""")
    executable = tmp_path / "schedule"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(root / "csrc/operators"),
                    str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
