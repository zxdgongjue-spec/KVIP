# SPDX-License-Identifier: Apache-2.0
"""Execute exact bitmap planners, including full-domain fallback and failures."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_actual_bitmap_index_plans_match_hash_and_ordered_oracle(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the native bitmap planner model")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "bitmap-index"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-pthread",
            "-fsanitize=address,undefined",
            "-I" + str(root / "csrc/operators/common/lifetch"),
            str(Path(__file__).with_name("bitmap_index_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30)
