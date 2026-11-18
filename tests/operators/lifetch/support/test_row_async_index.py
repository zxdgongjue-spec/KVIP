# SPDX-License-Identifier: Apache-2.0
"""Run the production row-local Index path with independent, concurrent row teams."""

import subprocess
from pathlib import Path


def test_concurrent_row_teams(tmp_path):
    here = Path(__file__).resolve().parent
    kernel = here.parents[3] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    text = (kernel / "high_index_balance.h").read_text()
    (tmp_path / "transfer_segments.h").write_text(text.split("__aicore__ inline void ApplyGlobalIndex(", 1)[0])
    binary = tmp_path / "row_index"
    subprocess.run([
        "c++", "-std=c++20", "-O1", "-fno-strict-aliasing", "-pthread",
        "-fsanitize=address,undefined", "-I", str(tmp_path), "-I", str(kernel),
        str(here / "row_async_index_model.cpp"), "-o", str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True, timeout=60)
