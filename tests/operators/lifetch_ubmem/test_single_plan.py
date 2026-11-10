# SPDX-License-Identifier: Apache-2.0
"""Actual descriptor reload and existing final DMA, without NPU claims."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("width,ordered,rows", ((656, False, 2), (2624, False, 2), (2624, True, 2), (2624, False, 4)))
def test_single_plan_reload_preserves_sources_owners_and_copy_bytes(tmp_path, width, ordered, rows):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for descriptor/DMA model")
    root = Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch_ubmem"

    def function(path, name):
        source = path.read_text()
        start = source.index(name + "(")
        start = source.rfind("\n", 0, start) + 1
        end, depth = source.index("{", start) + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    functions = "\n".join(
        function(root / "op_kernel/resolver/direct_slots.h", name) for name in ("OwnerRow", "OutputStride")
    )
    functions += "\n" + function(root / "op_kernel/resolver/resolver.h", "LoadFinalCopyPlan")
    (tmp_path / "load_plan_under_test.h").write_text(functions)
    definitions = {
        "SIM_BATCH": 2,
        "SIM_HISTORY": 32768,
        "SIM_RESIDENT": 8192,
        "SIM_Q": 4,
        "TOKEN_BYTES": width,
        "V9_AIC_COUNT": 32,
        "V9_SPLIT_Q": 1,
        "V9_INTERMEDIATE_FETCHES": 8,
        "V9_FINAL_REPLICATED": 1,
        "V9_DEDICATED_SERVICE": 0,
        "V9_UBPIPE_DEPTH": 8,
        "V9_UBPIPE_ROWS": 4,
        "V9_WINDOW_UBPIPE_DEPTH": 8,
        "V9_UBMEM_GROUPED_COPY": int(width == 2624),
        "V9_UBMEM_FINAL_COPY_STAGES": 6 if rows == 4 else 8 if width == 2624 else 4,
        "V9_UBMEM_FINAL_COPY_ROWS": rows,
        "V9_UBMEM_ORDERED_PAIR": int(ordered),
        "V9_USER_UB_BYTES": 223232,
    }
    binary = tmp_path / "load-plan"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            *[f"-D{k}={v}" for k, v in definitions.items()],
            f"-I{root / 'tests'}",
            f"-I{root / 'include'}",
            f"-I{root / 'op_kernel/copy'}",
            f"-I{root / 'op_kernel/resolver'}",
            f"-I{tmp_path}",
            str(Path(__file__).with_name("single_plan_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30, capture_output=True, text=True)
