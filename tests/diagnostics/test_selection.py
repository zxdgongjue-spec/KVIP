# SPDX-License-Identifier: Apache-2.0
"""Only proven BF16 boundary exchanges may differ from captured logical sets."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import array_digest
from kvip.diagnostics.selection import SelectionOracle


def case(batch=1, permute=False):
    rows, heads, history, owners = batch * 4, 32, 2176, batch + 1
    captured = np.tile(np.arange(2048, dtype=np.int32), (rows, 1))
    query = torch.zeros((rows, heads, 128), dtype=torch.float32)
    query[:, 0, 0] = 1
    query = query.to(torch.float8_e4m3fn)
    weights = torch.zeros((rows, heads), dtype=torch.bfloat16)
    weights[:, 0] = 1
    scale = torch.ones((rows, heads), dtype=torch.float32)
    tensors = {"pre.query": query, "pre.weights": weights, "pre.query_scale": scale}
    tensor_calls = []

    def tensor(name, torch_module, device):
        tensor_calls.append(name)
        assert device == "cpu"
        return tensors[name]

    archive = SimpleNamespace(
        manifest=dict(
            geometry=dict(batch=batch, heads=heads),
            tensors={name: dict(dtype=str(value.dtype).removeprefix("torch.")) for name, value in tensors.items()},
        ),
        array=lambda name: captured if name == "post.logical_output" else None,
        tensor=tensor,
    )
    keys = np.zeros((owners, history, 128), dtype=np.uint8)
    keys[:, :, 0] = 0x38  # FP8 e4m3fn 1.0
    scales = np.ones((owners, history), dtype=np.float32)
    scales[:, 2048] = 1.003  # Same BF16 bin as 1.0, not equal FP64 scores.
    unpack = np.arange(rows)
    owner_order = np.arange(1, owners)
    if permute:
        assert batch == 2
        owner_order = owner_order[::-1].copy()
        unpack = np.r_[np.arange(4, 8), np.arange(4)]
        weights[4:, 0] = 2
        # A wrong owner lookup would reject row zero's otherwise valid tie.
        scales[1, 2048] = 1.1
    inputs = dict(
        query_ends=np.tile(np.arange(2151, 2155), (batch, 1)),
        owners=owner_order,
        unpack=unpack,
        keys=keys,
        scales=scales,
    )
    return archive, inputs, captured.copy(), tensor_calls


def test_exact_set_is_order_independent_and_never_loads_score_arrays():
    archive, inputs, selected, calls = case()
    del inputs["keys"], inputs["scales"]
    oracle = SelectionOracle(archive, inputs)
    result = oracle.verify(selected[:, ::-1])
    assert result["status"] == "exact"
    assert result["selected_set_sha256"] == array_digest(np.sort(selected, axis=1).astype("<i4"))
    assert result["changed_queries"] == result["total_replacements"] == 0
    assert result["min_overlap_fraction"] == 1
    assert calls == []


def test_accepts_only_same_bf16_minimum_bin_and_caches_score_work(monkeypatch):
    archive, inputs, selected, calls = case()
    selected[0, 2047] = 2048
    oracle = SelectionOracle(archive, inputs)
    result = oracle.verify(selected)
    assert result["status"] == "bf16_boundary_ties"
    assert result["total_replacements"] == result["changed_queries"] == 1
    assert result["compared_queries"] == 4
    assert result["min_overlap_fraction"] == 2047 / 2048
    assert result["boundary_details"] == [dict(query=0, owner=1, replacements=1, boundary_bf16=1.0)]
    assert "not full-history LI proof" in result["reference"]
    assert len(calls) == 3
    monkeypatch.setattr(oracle, "_scores", lambda *_: pytest.fail("Cached set must not recompute scores"))
    assert oracle.verify(selected[:, ::-1]) == result


def test_repeated_runs_must_keep_one_selected_set_even_for_other_valid_ties():
    archive, inputs, selected, _ = case()
    first, second = selected.copy(), selected.copy()
    first[0, 2047] = 2048
    second[0, 2046] = 2048
    assert SelectionOracle(archive, inputs).verify(second)["status"] == "bf16_boundary_ties"
    oracle = SelectionOracle(archive, inputs)
    oracle.verify(first)
    with pytest.raises(AssertionError, match="Native LI selected set changed across eager/graph replays"):
        oracle.verify(second)


@pytest.mark.parametrize("value", [0.98, 1.02])
def test_rejects_lower_or_higher_bin_replacement_with_diagnostics(value):
    archive, inputs, selected, _ = case()
    inputs["scales"][1, 2048] = value
    selected[0, 2047] = 2048
    with pytest.raises(AssertionError, match=r"query=0 owner=1 overlap=2047/2048.*native_only_ids=\[2048\]"):
        SelectionOracle(archive, inputs).verify(selected)


def test_tie_above_minimum_captured_score_is_not_accepted():
    archive, inputs, selected, _ = case()
    inputs["scales"][1, [2047, 2048]] = 2
    selected[0, 2047] = 2048
    with pytest.raises(AssertionError, match="not a proven BF16 boundary tie"):
        SelectionOracle(archive, inputs).verify(selected)


@pytest.mark.parametrize("sign", [1, -1])
def test_cast_round_midpoint_is_not_torch_ties_to_even(sign):
    archive, inputs, selected, _ = case()
    inputs["scales"][1, 2048] = 1.00390625
    archive.tensor("pre.weights", torch, "cpu")[:, 0] = sign
    selected[0, 2047] = 2048
    # Torch would round +/-1.00390625 to +/-1.0. Ascend CAST_ROUND
    # instead produces +/-1.0078125, a different bin from the capture.
    with pytest.raises(AssertionError, match="not a proven BF16 boundary tie"):
        SelectionOracle(archive, inputs).verify(selected)


@pytest.mark.parametrize("fault", ["duplicate", "future", "negative", "float"])
def test_checks_uniqueness_and_causality_before_any_score_comparison(fault):
    archive, inputs, selected, calls = case()
    if fault == "duplicate":
        selected[0, 1] = selected[0, 0]
    elif fault == "future":
        selected[0, 1] = inputs["query_ends"][0, 0]
    elif fault == "negative":
        selected[0, 1] = -1
    else:
        selected = selected.astype(np.float32)
    with pytest.raises(AssertionError):
        SelectionOracle(archive, inputs).verify(selected)
    assert calls == []


def test_query_unpack_and_owner_permutation_use_the_same_tnd_row():
    archive, inputs, selected, _ = case(batch=2, permute=True)
    selected[0, 2047] = 2048
    result = SelectionOracle(archive, inputs).verify(selected)
    assert result["boundary_details"] == [dict(query=0, owner=2, replacements=1, boundary_bf16=2.0)]


def test_nonfinite_input_and_unexpected_dtype_are_not_tolerated():
    archive, inputs, selected, _ = case()
    selected[0, 2047] = 2048
    inputs["scales"][1, 2048] = np.nan
    with pytest.raises(AssertionError, match="invalid Index"):
        SelectionOracle(archive, inputs).verify(selected)
    archive.manifest["tensors"]["pre.weights"]["dtype"] = "float32"
    with pytest.raises(AssertionError, match="unexpected captured weights dtype"):
        SelectionOracle(archive, inputs).verify(selected)
