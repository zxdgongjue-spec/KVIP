# SPDX-License-Identifier: Apache-2.0
"""Synthetic token replay validates real selection geometry, never model timing."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.demand import SnapshotDemand
from kvip.diagnostics.oracle import DecodeOracle
from kvip.diagnostics.replay import inspect_snapshot
from kvip.diagnostics.selection import SelectionOracle
from kvip.diagnostics.serial_replay import serial_inputs
from kvip.diagnostics.token_replay import materialize_token_trace
from kvip.operators.lifetch.geometry import final_aivs, li_geometry


@pytest.fixture(params=[656, 2624])
def token_trace(request):
    b, h, r, heads = 2, 4096, 2048, 32
    owners = np.asarray([2, 1], dtype=np.int32)
    ids = np.tile(np.arange(h, dtype=np.int32), (b, 1))
    ids[:, [0, 2048]] = ids[:, [2048, 0]]
    metadata = np.zeros((5, b * 4), dtype=np.int64)
    metadata[0] = np.arange(b * 4)
    metadata[1:4:2] = 1
    metadata[2] = [4, 5, 6, 7, 0, 1, 2, 3]
    metadata[4].view(np.int32)[:b] = owners
    ends = np.tile(np.arange(3969, 3973, dtype=np.int32), (b, 1))
    selected = np.empty((b * 4, 2048), dtype=np.int32)
    for tnd, source in enumerate(metadata[2]):
        selected[tnd] = [*range(2044), 3000, 3400, 3500, ends.reshape(-1)[source] - 1]
    slots = np.full((b, h // 128), -1, dtype=np.int32)
    slots[0, :31] = 300 + np.arange(31) * 2
    slots[1, :31] = 700 + np.arange(31) * 3
    slots[1, :8] = slots[0, :8]  # Actual shared-prefix aliases must survive synthesis.
    geometry = li_geometry(b, "auto", 8, 32)
    manifest = dict(
        schema_version=3,
        status="complete",
        kind="kvaf_decode_token_trace",
        content="token_ids",
        validation=dict(passed=True, kind="token_id_integrity"),
        geometry=dict(
            batch=b,
            history=h,
            resident=r,
            output_stride=r + 384,
            heads=heads,
            token_bytes=request.param,
            aic_count=32,
            requested_windows=8,
            rounds=geometry["intermediate_fetches"],
            service_lanes=geometry["service_aivs_per_window"],
            final_workers=final_aivs(b, "auto", 32),
            owners=3,
            token_count=b * 4,
            tokens_per_block=128,
            prefetch_limit=512,
            window_limit=64,
        ),
        metadata=dict(
            capture_mode="eager_diagnostic",
            content="token_ids",
            dp_rank=0,
            dp_size=8,
            group=0,
            layer_names=["model.layers.0"],
            geometry=dict(batch=b),
            context=dict(min_context=3968, max_context=4096, global_decode_confirmed=True, round=12),
            requests=[
                dict(request_id=str(owner), owner=int(owner), query_begin=row * 4, query_stop=(row + 1) * 4)
                for row, owner in enumerate(owners)
            ],
        ),
    )
    arrays = {
        "pre.logical_ids": ids,
        "pre.hot_valid": np.ones((b, r), dtype=np.uint8),
        "pre.row_owners": owners,
        "pre.address_valid": (ids < 3968).astype(np.uint8),
        "pre.ddr_page_slots": slots,
        "pre.metadata": metadata,
        "pre.query_ends": ends,
        "pre.tail_bounds": np.tile([3968, 3972], (b, 1)).astype(np.int32),
        "pre.lru_slots": np.tile(np.arange(2048, dtype=np.int32), (b, 1)),
        "post.logical_output": selected,
    }
    return SimpleNamespace(manifest=manifest, arrays=arrays, array=lambda name: arrays[name])


def test_synthetic_fp8_scores_recover_every_captured_topk_with_exact_boundary(token_trace, tmp_path):
    archive = materialize_token_trace(token_trace, tmp_path / "synthetic")
    restored = load_snapshot(archive.directory)
    result = inspect_snapshot(restored)
    assert result["content"] == "synthetic_token_replay"
    assert "not real model" in result["fidelity"]
    assert restored.manifest["synthetic"]["native_validated"] is False
    q = restored.tensor("pre.query", torch, "cpu").float().numpy()
    w = restored.tensor("pre.weights", torch, "cpu").float().numpy()
    keys = torch.from_numpy(restored.array("pre.keys").copy()).view(torch.float8_e4m3fn).float().numpy()
    assert np.all(w[:, 0] == 1) and not np.any(w[:, 1:])
    assert not np.any(q[:, :, 4:])
    assert not np.array_equal(keys[1, 10], keys[1, 11])
    ids, metadata = restored.array("pre.logical_ids"), restored.array("pre.metadata")
    owners = metadata[4].view(np.int32)[:2]
    for tnd, source in enumerate(metadata[2]):
        row = source // 4
        end = restored.array("pre.query_ends")[source]
        scores = keys[owners[row], :end] @ q[source, 0]
        assert np.all((scores == 0) | (scores == 1))
        assert np.count_nonzero(scores == 1) == 2048
        actual = ids[owners[row], :end][scores == 1]
        np.testing.assert_array_equal(np.sort(actual), np.sort(token_trace.array("post.logical_output")[tnd]))
    inputs = serial_inputs(restored, restored.manifest["ddr"]["region_base"])
    assert SelectionOracle(restored, inputs).verify(token_trace.array("post.logical_output"))["status"] == "exact"
    demand = SnapshotDemand(restored).classify(token_trace.array("post.logical_output"))
    assert demand["abc_hit"] == 8 and demand["cold_demand"] > 0 and demand["candidate_hot_refetch"] > 0
    assert demand == restored.manifest["synthetic"]["captured_selection_demand"]


def test_synthesis_preserves_shared_pages_and_unpublished_zero_addresses(token_trace, tmp_path):
    archive = materialize_token_trace(token_trace, tmp_path / "synthetic")
    ddr = archive.manifest["ddr"]
    assert ddr["record_count"] == 31 * 2 - 8
    assert ddr["region_bytes"] == ddr["record_count"] * ddr["slot_bytes"]
    ids, addresses = archive.array("pre.logical_ids"), archive.array("pre.addresses")
    logical_addresses = np.take_along_axis(addresses, np.argsort(ids), axis=1)
    np.testing.assert_array_equal(logical_addresses[1, :1024], logical_addresses[2, :1024])
    assert not np.any(logical_addresses[:, 3968:])
    oracle = DecodeOracle(archive)
    np.testing.assert_array_equal(oracle.expected(1, [0, 127, 500]), oracle.expected(2, [0, 127, 500]))
    assert not np.array_equal(oracle.expected(1, [2000]), oracle.expected(2, [2000]))
    # No inactive owner state or real payload is inferred from the token trace.
    assert not np.any(archive.array("pre.hot_valid")[0])
    assert not np.any(addresses[0])
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_token_trace(token_trace, archive.directory)


def _valid_post(archive):
    """Construct a legal different post permutation to exercise the unchanged byte/index oracle."""
    g = archive.manifest["geometry"]
    oracle = DecodeOracle(archive)
    ids = archive.array("pre.logical_ids").copy()
    before = ids.copy()
    selected = archive.array("post.logical_output").copy()
    slots = np.empty_like(selected)
    hot_valid = archive.array("pre.hot_valid").copy()
    output = archive.array("pre.output").copy().reshape(len(ids), g["output_stride"], g["token_bytes"])
    metadata = archive.array("pre.metadata")
    owners = metadata[4].view(np.int32)[: g["batch"]]
    for row, owner in enumerate(owners):
        tnd_rows = np.flatnonzero(metadata[2] // 4 == row)
        begin, end = archive.array("pre.tail_bounds")[row]
        union = np.unique(selected[tnd_rows])
        cold_or_hot = union[(union < begin) | (union >= end)]
        assert len(cold_or_hot) <= g["resident"]
        for destination, token in enumerate(cold_or_hot):
            source = np.flatnonzero(ids[owner] == token)[0]
            ids[owner, [destination, source]] = ids[owner, [source, destination]]
        hot_valid[owner] = 1
        output[owner, : g["resident"]] = oracle.expected(owner, ids[owner, : g["resident"]])
        inverse = np.argsort(ids[owner])
        for tnd in tnd_rows:
            tokens = selected[tnd]
            slots[tnd] = inverse[tokens]
            tail = (tokens >= begin) & (tokens < end)
            slots[tnd, tail] = g["resident"] + tokens[tail] % (g["output_stride"] - g["resident"])
    state = dict(logical_ids=ids, hot_valid=hot_valid, output=output, logical_output=selected, slot_output=slots)
    for name in ("keys", "key_scales", "addresses"):
        pre = archive.array("pre." + name)
        state[name] = np.empty_like(pre)
        for owner in range(len(ids)):
            state[name][owner] = pre[owner, np.argsort(before[owner])[ids[owner]]]
    return state


def test_existing_oracle_checks_synthetic_payload_and_index_alignment(token_trace, tmp_path):
    archive = materialize_token_trace(token_trace, tmp_path / "synthetic")
    state = _valid_post(archive)
    expected = archive.array("post.logical_output")
    DecodeOracle(archive).verify(state, expected_topk=expected)
    for fault in ("payload", "keys", "addresses", "selection"):
        changed = {name: array.copy() for name, array in state.items()}
        if fault == "payload":
            changed["output"][1, 0, 40] ^= 1
        elif fault == "selection":
            changed["logical_output"][0, 0] = 3900
        else:
            changed[fault][1, 0] = 0
        with pytest.raises(AssertionError):
            DecodeOracle(archive).verify(changed, expected_topk=expected)


@pytest.mark.parametrize("fault", ["duplicate", "future", "causal_permutation", "availability", "owner", "slot"])
def test_rejects_nonreplayable_token_contract(token_trace, tmp_path, fault):
    arrays = {name: values.copy() for name, values in token_trace.arrays.items()}
    if fault == "duplicate":
        arrays["post.logical_output"][0, 1] = arrays["post.logical_output"][0, 0]
    elif fault == "future":
        arrays["post.logical_output"][0, 0] = 4095
    elif fault == "causal_permutation":
        arrays["pre.logical_ids"][0, [0, 4095]] = arrays["pre.logical_ids"][0, [4095, 0]]
    elif fault == "availability":
        arrays["pre.address_valid"][0, 17] = 0
    elif fault == "slot":
        arrays["pre.ddr_page_slots"][0, 1] = -1
    else:
        arrays["pre.row_owners"][0] = 0
    invalid = SimpleNamespace(manifest=copy.deepcopy(token_trace.manifest), array=lambda name: arrays[name])
    with pytest.raises(ValueError):
        materialize_token_trace(invalid, tmp_path / "invalid")
    assert not (tmp_path / "invalid" / "manifest.json").exists()


def test_actual_token_recorder_to_standard_replay_archive(tmp_path):
    from kvip.diagnostics.token_trace import TokenIdTraceRecorder, load_token_trace
    from test_token_trace import fixture

    selector, source, complete = fixture(batch=2, high_addresses=True)
    shape = li_geometry(2, "auto", 8, 32)
    selector.rotating.rounds = shape["intermediate_fetches"]
    selector.rotating.service_lanes = shape["service_aivs_per_window"]
    recorder = TokenIdTraceRecorder(torch, tmp_path / "tokens", source)
    recorder.before(selector)
    complete()
    recorder.after(selector)
    tokens = load_token_trace(recorder.directory)
    synthetic = materialize_token_trace(tokens, tmp_path / "synthetic")
    inspected = inspect_snapshot(load_snapshot(synthetic.directory))
    assert inspected["content"] == "synthetic_token_replay"
    assert inspected["native_validated"] is False
    assert inspected["captured_fusion_demand"] == tokens.manifest["summary"]["demand"]
    np.testing.assert_array_equal(synthetic.array("post.logical_output"), tokens.array("post.logical_output"))


def test_comparison_propagates_synthetic_content_and_rejects_mixing_real_inputs():
    from kvip.diagnostics.compare import summarize
    from test_compare import results

    cases = results()
    for case in cases.values():
        case.update(
            content="synthetic_token_replay",
            fidelity="Synthetic fixture, not real-model timing",
            input_validation=dict(kind="synthetic_fixture_consistency", native_validated=False),
        )
    summary = summarize(cases)
    assert summary["content"] == "synthetic_token_replay"
    assert "not real-model timing" in summary["fidelity"]
    assert summary["input_validation"]["native_validated"] is False
    cases["urma-fused"]["content"] = "full"
    with pytest.raises(ValueError, match="content"):
        summarize(cases)
