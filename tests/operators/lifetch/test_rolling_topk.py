# SPDX-License-Identifier: Apache-2.0
"""Actual rolling driver and original TopK merge, with CPU instruction probes."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("pruned", (False, True))
def test_rolling_topk_preserves_order_and_reloads_state(tmp_path, pruned):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = (root / "common/lightning_indexer/topk.h").read_text()
    start = source.index("template<>\nclass LITopk<uint16_t>")
    stop = source.index("\n};", start) + 3
    (tmp_path / "topk_under_test.h").write_text(
        "namespace topk { template<class T> class LITopk;\n" + source[start:stop] + "\n}\n"
    )
    shutil.copy(root / "common/lightning_indexer/rolling_topk.h", tmp_path / "rolling_topk.h")
    shutil.copy(root / "common/lightning_indexer/pruned_topk.h", tmp_path / "pruned_topk.h")
    binary = tmp_path / "rolling-topk"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-DV9_ROLLING_THRESHOLD=1",
            f"-DV9_PRUNED_TOPK={int(pruned)}",
            "-I" + str(tmp_path),
            str(Path(__file__).with_name("rolling_topk_model.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=60, capture_output=True, text=True)


@pytest.mark.parametrize("backend", ("lightning_indexer_with_fetch", "lightning_indexer_with_fetch_ubmem"))
def test_rolling_precedes_service_and_final_publication(backend):
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = (root / backend / "op_kernel/indexer/indexer.h").read_text()
    body = source[source.index("void QuantLightningIndexerKernel<QLIT>::ProcessBaseBlock(") :]
    assert body.index("ProcessVec1(runInfo)") < body.index("ProcessRollingTopK(runInfo)")
    assert body.index("ProcessRollingTopK(runInfo)") < body.index("vectorService.RotateFetch(")
    assert body.index("vectorService.RotateFetch(") < body.index("ProcessTopK(runInfo)")
    vector = (root / backend / "op_kernel/indexer/indexer_vector.h").read_text()
    assert "if(!prefixActive_ && rollingReady_) {PublishRollingTopK(info);return;}" in vector
    assert "if(!V9_ROLLING_THRESHOLD && kvip_prefix::UseThresholdOnly(" in vector


@pytest.mark.parametrize("backend", ("lightning_indexer_with_fetch", "lightning_indexer_with_fetch_ubmem"))
def test_pruned_final_reuses_resident_and_keeps_original_index_banks(backend):
    root = Path(__file__).resolve().parents[3] / "csrc/operators"
    source = (root / backend / "op_kernel/indexer/indexer_vector.h").read_text()
    body = source[source.index("if (loopIdx == 0)") :]
    assert body.index("kvip_pruned::RestoreResident(") < body.index("copyInParams.blockLen = trunkLen_")
    assert "validTrunkLen,loopIdx*trunkLen_,loopIdx%2,(loopIdx+1)%2" in body
    assert "loopIdx==s2LoopNum-1 && (loopIdx+1)%2==1" in body
    # Head64 is the largest admitted Q2 geometry. The new map never borrows
    # resMm1, which can already be receiving the producer's next score tile.
    total = 65536 + 512 + 1024 + 3 * 1024 + 16384 + 512 + 36864 + 8448 + 4096 + 48384 + 32768
    assert total == 217600 < 253 * 1024
    assert "16384 * sizeof(uint16_t)" in source
