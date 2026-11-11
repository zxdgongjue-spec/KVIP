# SPDX-License-Identifier: Apache-2.0
"""Execute the native logger's actual indexing/stores under sanitizers."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_threshold_trace_writes_exclusive_cache_lines(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "threshold-trace"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-I" + str(root / "csrc/operators/common/lightning_indexer"),
            str(Path(__file__).with_name("threshold_trace_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
