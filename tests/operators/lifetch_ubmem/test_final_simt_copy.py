# SPDX-License-Identifier: Apache-2.0
"""Run the actual final-copy helper with concurrent lanes and delayed stores."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPERATOR = ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem"


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_final_simt_copy_bytes_completion_and_sector_ownership(tmp_path, workers):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for actual final-copy model")
    definitions = dict(
        SIM_BATCH=1,
        SIM_HISTORY=131328,
        SIM_RESIDENT=16384,
        TOKEN_BYTES=656,
        INDEX_HEADS=32,
        V9_AIC_COUNT=32,
        V9_SPLIT_Q=1,
        V9_INTERMEDIATE_FETCHES=8,
        V9_FINAL_AIVS=workers,
        V9_FINAL_REPLICATED=1,
        V9_DEDICATED_SERVICE=0,
        V9_UBPIPE_DEPTH=8,
        V9_UBPIPE_ROWS=4,
        V9_WINDOW_UBPIPE_DEPTH=8,
        V9_USER_UB_BYTES=223232,
        V9_UBMEM_SIMT_FINAL=1,
    )
    options = [
        compiler,
        "-std=c++17",
        "-O1",
        "-g",
        "-pthread",
        "-fsanitize=address,undefined",
        *[f"-D{name}={value}" for name, value in definitions.items()],
        f"-I{OPERATOR / 'include'}",
    ]
    binary = tmp_path / "simt-copy"
    subprocess.run([*options, str(OPERATOR / "tests/final_simt_copy_model.cpp"), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    assert f"PASS workers={workers}" in result.stdout
    # The option retains the same production launches and completion ordering.
    abi = tmp_path / "abi"
    subprocess.run(
        [*options, str(OPERATOR / "tests/abi_test.cpp"), str(OPERATOR / "op_host/launch.cpp"), "-o", str(abi)],
        check=True,
    )
    subprocess.run([str(abi)], check=True)
