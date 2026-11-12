# SPDX-License-Identifier: Apache-2.0
"""P0-A2 CPU admission of per-request actual key lengths."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from actual_k_admission import parse_actual_key_lengths, score_oracle_for_actual_k  # noqa: E402
from real_fetch_inputs import prepare_full_chain_fixture  # noqa: E402


def test_parse_k():
    assert parse_actual_key_lengths("32768,65536,98304,131072", 4, 131072, 8192) == (
        32768, 65536, 98304, 131072)
    assert parse_actual_key_lengths(None, 4, 131072, 8192) is None
    for val in ("32768,65536", "32769,65536,98304,131072",
                "4096,65536,98304,131072", "32768,65536,98304,131200",
                "32768,65536,NA,131072"):
        with pytest.raises(ValueError, match="actual_key_lengths_invalid"):
            parse_actual_key_lengths(val, 4, 131072, 8192)


def test_score_oracle_excludes_invalid_suffix():
    n = 4096
    scores = np.zeros((4, n), dtype=np.int32)
    scores[:, :2048] = np.arange(2048, 0, -1)
    scores[:, 2048:] = 100000  # would win if K were erroneously n
    reports, selected = score_oracle_for_actual_k(
        scores, np.arange(n), (2048, 2048), 4096)
    assert selected.shape == (2, 4, 2048)
    assert np.array_equal(selected[0, 0], np.arange(2048))
    assert all(x["actual_key_length"] == 2048 for x in reports)
    with pytest.raises(ValueError, match="boundary_tie"):
        score_oracle_for_actual_k(np.zeros((4, n)), np.arange(n), (n,), 4096)


def test_generated_mixed_k_fixture_has_strict_topk(tmp_path):
    lengths = (32768, 49152, 65536, 65536)
    folder = tmp_path / "fixture"
    result = prepare_full_chain_fixture(
        folder, 65536, 8192, 10, index_heads=32,
        native_union_fit=True, actual_key_lengths=lengths)
    report = result["actual_key_admission"]
    assert report["key_lengths"] == list(lengths)
    order = np.load(folder / "physical_to_logical.npy")
    logical = np.load(folder / "logical_scores.npy")
    derived, topk = score_oracle_for_actual_k(logical, order, lengths, 8192)
    assert topk.shape == (4, 4, 2048)
    assert all(x["unique_cold"] > 0 for x in derived)
    assert derived == report["requests"]
    assert (json.loads((folder / "fixture.json").read_text())
            ["full_chain_fixture"]["actual_key_admission"]["requests"] == derived)


def test_wiring_stays_in_test_harness_only():
    root = Path(__file__).resolve().parents[3]
    cli = (root / "tests/operators/lifetch/test_lifetch.py").read_text()
    runner = (root / "tests/operators/lifetch/support/lifetch.py").read_text()
    assert cli.count('add_argument("--actual-key-lengths"') == 2
    assert "'--actual-key-lengths', a.actual_key_lengths" in cli
    assert "actual_key_lengths=actual_lengths" in runner
    assert "arguments[6].view(torch.int32)" in runner
    assert "query_ends=query_ends" in runner
