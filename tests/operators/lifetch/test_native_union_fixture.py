# SPDX-License-Identifier: Apache-2.0
"""P0-A1: test Native LIDU's simultaneous four-query TopK capacity."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
from real_fetch_inputs import prepare_full_chain_fixture  # noqa: E402


def _union_from_oracle(folder, resident):
    order = np.load(folder / "physical_to_logical.npy")
    physical_scores = np.load(folder / "logical_scores.npy")[:, order]
    selected = [np.argpartition(row, -2048)[-2048:] for row in physical_scores]
    ids = np.unique(np.concatenate(selected))
    return len(ids), int(np.count_nonzero(ids >= resident))


def test_legacy_q4_hot_span_is_rejected_before_device_setup(tmp_path):
    with pytest.raises(ValueError, match="native_mtp_union_exceeds_resident"):
        prepare_full_chain_fixture(
            tmp_path / "legacy", 8192, 4096, 10,
            index_heads=32, native_union_fit=False,
        )


def test_compact_q4_hot_union_admitted_and_reported(tmp_path):
    folder = tmp_path / "admitted"
    info = prepare_full_chain_fixture(
        folder, 8192, 4096, 10, index_heads=32,
        native_union_fit=True,
    )
    count, cold = _union_from_oracle(folder, 4096)
    assert cold > 0
    assert count <= 4096
    admission = info["native_union_admission"]
    assert admission["admissible"] is True
    assert admission["hot_stride"] == 128
    assert admission["unique_topk"] == count
    assert admission["unique_cold"] == cold
    saved = json.loads((folder / "fixture.json").read_text())["full_chain_fixture"]
    assert saved["native_union_admission"] == admission


def test_r2048_rejected_even_with_compact_q4_hot(tmp_path):
    with pytest.raises(ValueError, match="native_mtp_union_exceeds_resident"):
        prepare_full_chain_fixture(
            tmp_path / "small", 8192, 2048, 10,
            index_heads=32, native_union_fit=True,
        )


def test_opt_in_flag_reaches_full_chain_fixture():
    root = Path(__file__).resolve().parents[3]
    cli = (root / "tests/operators/lifetch/test_lifetch.py").read_text()
    runner = (root / "tests/operators/lifetch/support/lifetch.py").read_text()
    assert cli.count('add_argument("--native-union-fit"') == 2
    assert "common.append('--native-union-fit')" in cli
    assert 'native_union_fit=bool(getattr(args, "native_union_fit", False))' in runner
