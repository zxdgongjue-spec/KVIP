# SPDX-License-Identifier: Apache-2.0
"""The same graph input buffers can switch dedup storage between replays."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_batch_window_storage_tracks_all_query_ends(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for the native window planner")
    operators = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = tmp_path / "window_plan.cpp"
    source.write_text(r"""
#include <array>
#include <cassert>
#include "lightning_indexer_with_fetch/include/window_dedup.h"

int main() {
    constexpr uint32_t history = 524544;
    constexpr uint32_t batch = 16;
    constexpr uint32_t cutoff = 157696;
    static_assert(rotating::kDirectWindowTokens == cutoff);
    std::array<uint32_t, 4 * batch> ends{};
    auto words = [&] { return rotating::BatchWindowWords(history, batch, ends.data()); };

    assert(words() == 8);  // All padded rows.
    ends.fill(131072);
    assert(words() == 131072 / 32);
    for (auto& end : ends) {
        end = cutoff - 1;
        assert(words() == rotating::kDirectWindowWords);
        end = cutoff;
        assert(words() == rotating::kDirectWindowWords);
        end = cutoff + 1;
        assert(words() == rotating::kDirectWindowWords + 8);
        end = 131072;
        assert(words() == 131072 / 32);
    }
    ends.back() = UINT32_MAX;
    assert(words() == history / 32);
    assert(rotating::BatchWindowWords(history, batch,
                                     static_cast<const uint32_t*>(nullptr)) == history / 32);
    assert(rotating::BatchWindowWords(131328, batch, ends.data()) == 131328 / 32);
}
""")
    binary = tmp_path / "window_plan"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(operators),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
