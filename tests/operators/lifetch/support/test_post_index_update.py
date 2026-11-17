# SPDX-License-Identifier: Apache-2.0
"""Exercise production Index exchanges with concurrent CPU warp groups."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_recent_index import INDEXER


@pytest.mark.parametrize("threads", [64, 256, 512])
def test_post_index_warp_contract(tmp_path, threads):
    header = (INDEXER.parent / "post_index_update.h").read_text()
    header, changes = re.subn(
        r"constexpr uint32_t kThreads = \d+;",
        f"constexpr uint32_t kThreads = {threads};",
        header,
    )
    assert changes == 1
    (tmp_path / "post_index_update.h").write_text(header)
    (tmp_path / "post_index_prepare.h").write_text(
        (INDEXER.parent / "post_index_prepare.h").read_text()
    )
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler, "The Index CPU model requires a C++20 compiler"
    binary = tmp_path / "post_index_model"
    compiled = subprocess.run([
        compiler, "-std=c++20", "-O1", "-g", "-pthread",
        "-fno-strict-aliasing", "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer", "-I", str(tmp_path),
        str(Path(__file__).with_name("post_index_model.cpp")),
        "-o", str(binary),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"INDEX_WARP_CONTRACT_PASS threads={threads}" in result.stdout
