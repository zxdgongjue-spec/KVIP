# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual prefix helper and resident-publication control flow.

The CPU model supplies histogram instruction semantics, not NPU scheduling.
The native CANN build and replay remain the checks for hardware execution.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


def _function(source, name):
    start = source.index("__aicore__ inline void " + name + "(")
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("ubmem", (False, True))
@pytest.mark.parametrize("cold_only", (False, True))
@pytest.mark.parametrize("threshold_only", (False, True))
def test_actual_prefix_threshold_and_publication(tmp_path, ubmem, cold_only, threshold_only):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the prefix model")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    backend = "lightning_indexer_with_fetch" + ("_ubmem" if ubmem else "")
    source = (root / backend / "op_kernel/indexer/indexer_vector.h").read_text()
    (tmp_path / "publication_under_test.h").write_text(_function(source, "PublishResident"))
    # Replace only the include of the established SIMD histogram primitives.
    # The helper under test retains all rank parameters, scratch offsets,
    # cutoff broadcast, and gating. Full TopK is an independent sort oracle.
    helper = (root / "common/lightning_indexer/resident_threshold.h").read_text()
    (tmp_path / "threshold_under_test.h").write_text(helper.replace('#include "topk_common.h"', ""))
    binary = tmp_path / "resident-threshold"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DV9_THRESHOLD_ONLY={int(threshold_only)}",
            f"-DV9_COLD_ONLY_PREFETCH={int(cold_only)}",
            "-I" + str(tmp_path),
            str(Path(__file__).with_name("resident_threshold_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30, capture_output=True, text=True)
