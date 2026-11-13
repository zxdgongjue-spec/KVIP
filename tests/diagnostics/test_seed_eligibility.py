# SPDX-License-Identifier: Apache-2.0
"""Predict the UB seed fast path without claiming device dispatch was observed."""

from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics import replay_measure
from kvip.diagnostics.fused_replay import validate_fused_tuning, verify_build_options
from kvip.diagnostics.seed_eligibility import predict_seed_eligibility


def _archive(*, history=131328, resident=16384, batch=16, aics=32, begin=130000, end=130512):
    geometry = dict(batch=batch, history=history, resident=resident, aic_count=aics, heads=32, requested_windows=8)
    arrays = {
        "pre.threshold_bounds": np.tile(np.array([begin, end], dtype=np.int32), (batch, 1)),
        "pre.query_ends": np.tile(np.arange(end - 3, end + 1, dtype=np.int32), (batch, 1)),
        "pre.query_lengths": np.full(batch, 4, dtype=np.int32),
        "pre.key_lengths": np.full(batch, history, dtype=np.int32),
    }
    return SimpleNamespace(manifest={"geometry": geometry}, array=arrays.__getitem__, arrays=arrays)


def test_five_input_geometry_has_64_eligible_queries_with_explicit_prediction_scope():
    report = predict_seed_eligibility(_archive())
    assert report["predicted_eligible_queries"] == 64 and report["total_queries"] == 64
    assert report["predicted_fallback_queries"] == 0
    assert report["split_q"] == 1 and report["sequence_split"] is False
    assert report["padded_words"] == [17152] * 16
    assert "not observed native dispatch" in report["scope"]


def test_actual_runtime_shape_must_match_the_prediction():
    archive = _archive()
    shape = SimpleNamespace(batch=16, history=131328, resident=16384, aicCount=32, indexHeads=32, splitQ=1, rounds=8)
    assert predict_seed_eligibility(archive, shape)["predicted_eligible_queries"] == 64
    shape.splitQ = 0
    with pytest.raises(ValueError, match="actual native RuntimeShape"):
        predict_seed_eligibility(archive, shape)


@pytest.mark.parametrize("history,aics", [(131584, 32), (131328, 16)])
def test_sequence_split_and_two_queries_per_aiv_fall_back(history, aics):
    report = predict_seed_eligibility(_archive(history=history, aics=aics))
    assert report["predicted_eligible_queries"] == 0
    assert report["sequence_split"] is (history == 131584)


def test_halo_capacity_uses_batch_max_query_end_not_smaller_own_query():
    archive = _archive(begin=128001, end=130049)
    # A nominal 2048-position exact range needs 2176 physical words after
    # tile alignment. A shorter first query cannot shrink the shared storage.
    archive.arrays["pre.query_ends"][0] = [128002, 128003, 130048, 130049]
    report = predict_seed_eligibility(archive)
    assert report["padded_words"][0] == 18688
    assert report["predicted_eligible_queries"] == 0
    archive.arrays["pre.threshold_bounds"][:] = [128000, 130048]
    archive.arrays["pre.query_ends"][:] = [130045, 130046, 130047, 130048]
    assert predict_seed_eligibility(archive)["predicted_eligible_queries"] == 64


def test_partial_history_and_short_query_report_partial_fallback():
    archive = _archive()
    archive.arrays["pre.key_lengths"][0] -= 128
    archive.arrays["pre.query_lengths"][1] = 2
    archive.arrays["pre.query_ends"][2] = [16382, 16383, 16384, 16385]
    report = predict_seed_eligibility(archive)
    assert report["predicted_eligible_queries"] == 54
    assert report["eligible_mask"][:3] == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]]


def test_empty_recent_is_resident_only_eligible_and_invalid_bounds_reject():
    archive = _archive(begin=0, end=16384)
    archive.arrays["pre.query_ends"][:] = [16384, 16385, 16386, 16387]
    assert predict_seed_eligibility(archive)["predicted_eligible_queries"] == 64
    archive.arrays["pre.threshold_bounds"][0] = [10, 9]
    with pytest.raises(ValueError, match="threshold bounds"):
        predict_seed_eligibility(archive)


def test_all_fallback_study_rejects_before_runtime_allocation(monkeypatch):
    def allocate(*args):
        pytest.fail("All-fallback study must not allocate or import DDR")

    monkeypatch.setattr(replay_measure, "allocate_context", allocate)
    args = SimpleNamespace(study="seed-ub")
    with pytest.raises(ValueError, match="all-fallback"):
        replay_measure.run(args, _archive(aics=16), {})


def test_seed_flag_requires_threshold_only_and_exact_binary_getter():
    with pytest.raises(ValueError, match="requires recent_threshold_only"):
        validate_fused_tuning({"recent_seed_ub": True}, "urma")
    options = {"recent_seed_ub": True, "recent_threshold_only": True, "recent_profile": False}
    assert validate_fused_tuning(options, "urma") == options
    library = SimpleNamespace(**{"kvip_" + key: (lambda value=int(value): value) for key, value in options.items()})
    verify_build_options(library, options)
    library.kvip_recent_profile = lambda: 1
    with pytest.raises(RuntimeError, match="recent_profile=False"):
        verify_build_options(library, options)


def test_cpu_prediction_matches_actual_native_seed_layout_helper(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "seed.cpp"
    source.write_text("""#include <iostream>
#include "csrc/operators/common/lifetch/recent_index.h"
int main() {
  int32_t begin, end; uint32_t r,h,available,q,maxq;
  while (std::cin >> begin >> end >> r >> h >> available >> q >> maxq) {
    auto s=kvip_recent_index::MakeSeedLayout(begin,end,r,h,available,q,maxq,18432);
    std::cout << s.enabled << "\\n";
  }
}""")
    binary = tmp_path / "seed"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(binary)], check=True, capture_output=True
    )
    cases = [_archive(begin=b, end=e) for b, e in [(130000, 130512), (128000, 130048), (128001, 130049), (0, 16384)]]
    lines, expected = [], []
    for archive in cases:
        report = predict_seed_eligibility(archive)
        for row in range(16):
            begin, end = archive.array("pre.threshold_bounds")[row]
            ends = archive.array("pre.query_ends")[row]
            for query, value in enumerate(ends):
                lines.append(f"{begin} {end} 16384 131328 131328 {value} {ends.max()}")
                expected.append(report["eligible_mask"][row][query])
    process = subprocess.run([str(binary)], input="\n".join(lines) + "\n", text=True, capture_output=True, check=True)
    assert list(map(int, process.stdout.split())) == expected
