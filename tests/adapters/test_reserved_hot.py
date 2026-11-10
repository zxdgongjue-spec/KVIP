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


@pytest.mark.parametrize("variant", ("urma", "ubmem", "ubmem_parallel", "ubmem_tagged"))
@pytest.mark.parametrize("resident", (8192, 16384, 131072))
@pytest.mark.parametrize("compact", (0, 1))
def test_shared_native_helper_rebuilds_active_rows_after_dma_clear(tmp_path, variant, resident, compact):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the shared native init model")
    root = Path(__file__).resolve().parents[2]
    operator = "lightning_indexer_with_fetch" + ("_ubmem" if variant != "urma" else "")
    kernel = root / "csrc/operators" / operator / "op_kernel"
    lifecycle = kernel / "lifecycle.cpp"
    tail = kernel / ("prefetch.h" if variant == "urma" else "resolver/direct_slots.h")

    def function(path, name):
        source = path.read_text()
        begin = source.rfind("\n", 0, source.index(name + "(")) + 1
        brace = source.index("{", begin)
        end, depth = brace + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[begin:end]

    # Compile actual init call sites, not a second implementation of their
    # ordering. The model delays MTE3 until WaitFlag and counts Duplicate/VF.
    functions = [function(tail, name) for name in ("OutputStride", "InitializeTailMappings")]
    names = (
        ("FillRoundWords", "InitializeRound")
        if variant == "urma"
        else ("FillWords", "InitializeControl", "CopyInitialLruSlots", "InitializeRound")
    )
    functions.extend(function(lifecycle, name) for name in names)
    if variant == "ubmem_parallel":
        functions.extend(function(lifecycle, name) for name in ("InitializeTailShard", "InitializeRoundParallel"))
    if variant == "ubmem_tagged":
        functions.extend(function(tail, name) for name in ("CurrentMappingTag", "RecordChange"))
        functions.extend(
            function(kernel / "controller/window_fetch.h", name)
            for name in ("PrepareUbmemWindow", "GatherCandidateMap")
        )
    (tmp_path / "reserved_hot_lifecycle.h").write_text("\n".join(functions))
    binary = tmp_path / "reserved-hot-model"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DV9_COMPACT_RESERVED={compact}",
            "-DSIM_BATCH=4",
            "-DSIM_HISTORY=131328",
            f"-DSIM_RESIDENT={resident}",
            f"-DMODEL_URMA={int(variant == 'urma')}",
            f"-DMODEL_PARALLEL={int(variant == 'ubmem_parallel')}",
            f"-DV9_UBMEM_TAGGED_MAP={int(variant == 'ubmem_tagged')}",
            f"-I{root / 'csrc/operators/common/lifetch'}",
            f"-I{root / 'csrc/operators/lightning_indexer_with_fetch_ubmem/include'}",
            f"-I{tmp_path}",
            str(root / "tests/operators/lifetch/reserved_hot_model.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
