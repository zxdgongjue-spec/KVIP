# SPDX-License-Identifier: Apache-2.0
"""Model real entry ordering; do not substitute host timing for hardware."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel"


def _function(source, name):
    start = source.rfind("\n", 0, source.index(name + "(")) + 1
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("batch,aics", ((1, 2), (2, 4), (16, 32)))
def test_every_mixed_aiv_waits_for_complete_init_before_li(tmp_path, batch, aics):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the fused-entry model")
    (tmp_path / "before_li_actual.h").write_text(_function((OP / "init.h").read_text(), "InitializeBeforeLi"))
    (tmp_path / "run_indexer_actual.h").write_text(
        "template<bool WithInit>\n" + _function((OP / "indexer.cpp").read_text(), "RunIndexer")
    )
    binary = tmp_path / "fused-init-model"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-pthread",
            "-fsanitize=address,undefined",
            f"-DSIM_BATCH={batch}",
            f"-DMODEL_AICS={aics}",
            f"-DV9_AIC_COUNT={aics}",
            "-DV9_UBMEM_FUSED_INIT=1",
            "-DSIM_HISTORY=131328",
            "-DSIM_RESIDENT=16384",
            "-DINDEX_HEADS=32",
            "-DV9_SPLIT_Q=1",
            f"-I{tmp_path}",
            str(Path(__file__).with_name("fused_init_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=20)


def test_cube_data_access_retains_both_original_vector_credit_dependencies():
    source = (ROOT / "csrc/operators/common/lightning_indexer/indexer_cube.h").read_text()
    body = _function(source, "QLIMatmul<QLIT>::ComputeMm1")
    # Both vector credits precede the first branch that can load Q/K or write
    # cube output; initializing views/buffer addresses cannot bypass them.
    waits = body[: body.index("uint64_t s2GmBaseOffset")]
    assert waits.count("CrossCoreWaitFlag<") == 2
    assert "AIV0_AIV1_OFFSET" in waits
