# SPDX-License-Identifier: Apache-2.0
"""Compact LRU reconstruction and native init ordering; no performance claims."""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from kvip.adapters.vllm.inputs import reserved_hot_reference


def test_reference_preserves_packed_row_ordinals_and_ignores_inactive_rows():
    candidates = np.array([[7, 0, 3, 5], [-1, -1, -1, -1], [4, 2, 6, 1]], dtype=np.int32)
    ends = np.array([[100, 101, 0, 0], [0, 0, 0, 0], [200, 201, 202, 203]], dtype=np.int32)
    expected = np.full((3, 8), -1, dtype=np.int32)
    expected[0, [7, 0, 3, 5]] = range(4)
    expected[2, [4, 2, 6, 1]] = range(4)
    np.testing.assert_array_equal(reserved_hot_reference(candidates, ends, 8), expected)
    # Stable owners are intentionally absent: this table follows packed rows.
    order = [2, 1, 0]
    np.testing.assert_array_equal(reserved_hot_reference(candidates[order], ends[order], 8), expected[order])


@pytest.mark.parametrize("slots", ([0, 0, 2, 3], [-1, 1, 2, 3], [0, 1, 2, 8]))
def test_reference_rejects_invalid_active_candidate_lists(slots):
    with pytest.raises(ValueError, match="unique in-range"):
        reserved_hot_reference(np.array([slots], dtype=np.int32), np.ones((1, 4), dtype=np.int32), 8)


def test_shared_native_helper_rebuilds_active_rows_after_dma_clear(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the shared native init model")
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "reserved-hot-model"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-I{root / 'csrc/operators/common/lifetch'}",
            str(root / "tests/operators/lifetch/reserved_hot_model.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
