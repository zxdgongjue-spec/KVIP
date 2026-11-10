# SPDX-License-Identifier: Apache-2.0
"""Compare actual init bodies and operation counts with delayed DMA."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _function(source, name):
    start = source.rfind("\n", 0, source.index(name + "(")) + 1
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("transport,tagged", (("urma", 0), ("ubmem", 0), ("ubmem", 1)))
@pytest.mark.parametrize("batch", (1, 16))
def test_reserved_ub_matches_records_with_one_dma_and_no_gm_scatter(tmp_path, transport, tagged, batch):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the actual init model")
    operator = ROOT / "csrc/operators" / ("lightning_indexer_with_fetch" + ("_ubmem" if transport == "ubmem" else ""))
    kernel = operator / "op_kernel"
    lifecycle = (kernel / ("lifecycle.cpp" if transport == "urma" else "init.h")).read_text()
    tail = (kernel / ("resolver/direct_slots.h" if transport == "ubmem" else "prefetch.h")).read_text()
    functions = [
        _function(tail, name) for name in ("OutputStride", "InitializeTailMappingsBody", "InitializeTailMappings")
    ]
    names = (
        ("InitializeReservedLocal", "FillRoundWords", "InitializeRound")
        if transport == "urma"
        else ("InitializeReservedLocal", "FillWords", "InitializeControl", "CopyInitialLruSlots", "InitializeRound")
    )
    functions.extend(_function(lifecycle, name) for name in names)
    if tagged:
        functions.extend(_function(tail, name) for name in ("CurrentMappingTag", "RecordChange"))
        window = (kernel / "controller/window_fetch.h").read_text()
        functions.extend(_function(window, name) for name in ("PrepareUbmemWindow", "GatherCandidateMap"))
    (tmp_path / "reserved_hot_lifecycle.h").write_text("\n".join(functions))
    results = []
    for enabled in (0, 1):
        binary = tmp_path / f"reserved-ub-{enabled}"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                f"-DV9_RESERVED_UB={enabled}",
                "-DV9_COMPACT_RESERVED=1",
                f"-DSIM_BATCH={batch}",
                "-DSIM_HISTORY=131328",
                "-DSIM_RESIDENT=16384",
                f"-DV9_UBMEM_TAGGED_MAP={tagged}",
                f"-DMODEL_URMA={int(transport == 'urma')}",
                "-DMODEL_PARALLEL=0",
                f"-I{ROOT / 'csrc/operators/common/lifetch'}",
                f"-I{ROOT / 'csrc/operators/lightning_indexer_with_fetch_ubmem/include'}",
                f"-I{tmp_path}",
                str(ROOT / "tests/operators/lifetch/reserved_hot_model.cpp"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        results.append([int(value) for value in subprocess.check_output([str(binary)], text=True).split()])
    before, after = results
    assert before[0] == after[0]  # All persistent/output records, including padding and epoch wrap.
    assert before[1] == after[2] * 2048 and after[1] == 0
    assert before[2] == after[2] * 4  # Four sentinel copies become one complete inverse copy.
    assert before[3:] == after[3:]  # No additional tail VF or all-ones Duplicate.
