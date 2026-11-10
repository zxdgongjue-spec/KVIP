# SPDX-License-Identifier: Apache-2.0
"""Production service loops skip impossible speculative admission, not final Get."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("ubmem", (False, True))
def test_actual_service_preserves_epochs_cursors_and_pending_when_budget_exhausted(tmp_path, ubmem):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the service control model")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    path = (
        "lightning_indexer_with_fetch_ubmem/op_kernel/controller/window_fetch.h"
        if ubmem
        else "lightning_indexer_with_fetch/op_kernel/prefetch.h"
    )
    source = (root / path).read_text()
    start = source.index("__aicore__ inline void FastService(")
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    (tmp_path / "service_function.h").write_text(source[start:end])
    (tmp_path / "batch_get_abi.h").write_text("#pragma once\n")
    binary = tmp_path / "service"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DMODEL_UBMEM={int(ubmem)}",
            "-I" + str(tmp_path),
            "-I" + str(root / "lightning_indexer_with_fetch/include"),
            "-I" + str(root / "lightning_indexer_with_fetch_ubmem/include"),
            "-I" + str(root / "common/lifetch"),
            str(Path(__file__).with_name("prefetch_budget_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
