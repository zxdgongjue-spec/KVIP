# SPDX-License-Identifier: Apache-2.0
"""Real score analysis, DP namespace merging and unchanged LI input materialization."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics.archive import load_snapshot
from kvip.diagnostics.score_replay import analyze_scores, load_round, main, merge_traces, score_value
from kvip.diagnostics.token_replay import materialize_score_trace, materialize_token_trace
from kvip.diagnostics.token_trace import trace_summary
from test_token_replay import token_trace as _source_fixture


@pytest.fixture(name="token_trace", params=[656, 2624])
def _token_fixture(request):
    return _source_fixture.__wrapped__(request)


def score_trace(base, rank=0, *, real=True):
    manifest, arrays = copy.deepcopy(base.manifest), copy.deepcopy(base.arrays)
    b, h, heads, r = (manifest["geometry"][name] for name in ("batch", "history", "heads", "resident"))
    slot = ((manifest["geometry"]["token_bytes"] * 128 + 4095) // 4096) * 4096
    manifest.update(schema_version=4, kind="kvaf_decode_score_trace", content="scores", real_inputs_saved=real)
    manifest["validation"] = dict(passed=True, kind="score_trace_integrity")
    manifest["geometry"].update(
        slot_bytes=slot, region_bytes=1024 * slot, page_mode="thp", numa_node=rank % 4, soc="Ascend950DT_9582"
    )
    manifest["metadata"].update(dp_rank=rank, capture_real_inputs=real, content="scores")
    scores = np.full((b, 4, h), 0x8000, dtype=np.uint16)  # sortable +0
    real_keys = np.zeros((b, h, 128), dtype=np.uint8)
    real_keys[:, :, 12] = np.arange(h, dtype=np.uint8)  # unused columns must remain unchanged
    for tnd, source in enumerate(arrays["pre.metadata"][2]):
        row, q = divmod(int(source), 4)
        selected = np.isin(arrays["pre.logical_ids"][row], arrays["post.logical_output"][tnd])
        scores[row, q, selected] = 0xBF80  # sortable +1
        real_keys[row, selected, q] = 0x38
        # Future scores are not candidates, regardless of how large they are.
        scores[row, q, arrays["pre.query_ends"][row, q] :] = 0xC100
    records = np.zeros((b, 4, (h + 16383) // 16384 + 2, 4), dtype=np.int32)
    for row in range(b):
        for q in range(4):
            records[row, q, 0] = [r, r, int(np.partition(scores[row, q, :r], r - 2048)[-2048]), 1]
            records[row, q, -1] = [arrays["pre.query_ends"][row, q], h, 0xBF80, 4]
    arrays.update(
        {
            "post.score_keys": scores,
            "post.threshold_records": records,
            "post.prefetch_records": np.array([[0, 3500, 3500], [0, 3600, 3600]], dtype=np.int32),
            "post.final_records": np.array([[0, 3400, 0]], dtype=np.int32),
        }
    )
    if real:
        query = np.zeros((b, 4, heads, 128), dtype=np.uint8)
        for q in range(4):
            query[:, q, 0, q] = 0x38
        weights = np.zeros((b, 4, heads), dtype=np.uint16)
        weights[:, :, 0] = 0x3F80
        arrays.update(
            {
                "real.keys": real_keys,
                "real.key_scales": np.ones((b, h), dtype=np.float32),
                "real.query": query,
                "real.weights": weights,
                "real.query_scale": np.ones((b, 4, heads), dtype=np.float32),
            }
        )
    trace = SimpleNamespace(manifest=manifest, arrays=arrays, array=lambda name: arrays[name])
    manifest["summary"] = trace_summary(trace)
    return trace


def predecessor(current):
    previous = copy.deepcopy(current)
    previous.array = lambda name: previous.arrays[name]
    previous.manifest["metadata"]["context"]["round"] -= 1
    return previous


def test_scores_validate_full_causal_topk_thresholds_and_observed_misses(token_trace):
    current = score_trace(token_trace)
    previous = predecessor(current)
    previous.arrays["post.logical_output"][:, 0] = 3600
    report = analyze_scores(current, previous)
    assert report["status"] == "pass"
    assert len(report["queries"]) == 8
    assert {q["final_value"] for q in report["queries"]} == {1.0}
    assert {q["previous_seed_value"] for q in report["queries"]} == {0.0}
    assert report["reads"]["unused_prefetch"] == 1
    assert report["transfer_classes"]["unused_prefetch_cold"] == 1
    assert report["demand"]["cold_demand"] > 0
    assert all(q["previous_seed_candidates"] == q["query_end"] for q in report["queries"])
    assert all(q["previous_seed_above"] == 2048 and q["final_above"] == 0 for q in report["queries"])
    assert all(q["final_equal"] == 2048 for q in report["queries"])
    assert "Native admission additionally" in report["candidate_count_rule"]
    np.testing.assert_array_equal(score_value(np.array([0x407F, 0x8000, 0xBF80], dtype=np.uint16)), [-1, 0, 1])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("score", "higher-scored"),
        ("nan", "NaN"),
        ("threshold", "threshold disagrees"),
        ("gap", "immediately preceding"),
        ("identity", "lifetime"),
        ("summary", "summary differs"),
    ],
)
def test_analysis_rejects_false_score_and_predecessor_contract(token_trace, mutation, match):
    current = score_trace(token_trace)
    previous = predecessor(current)
    if mutation == "score":
        current.arrays["post.score_keys"][0, 0, 3600] = 0xC000
    elif mutation == "nan":
        current.arrays["post.score_keys"][0, 0, 3600] = 0xFFC1
    elif mutation == "threshold":
        current.arrays["post.threshold_records"][0, 0, 0, 2] += 1
    elif mutation == "gap":
        previous.manifest["metadata"]["context"]["round"] -= 1
    elif mutation == "identity":
        previous.manifest["metadata"]["requests"][0]["request_id"] = "new-lifetime"
    else:
        current.manifest["summary"]["reads"]["unused_prefetch"] += 1
    with pytest.raises(ValueError, match=match):
        analyze_scores(current, previous)


def test_dp8_b2_merge_keeps_tnd_identity_causality_and_dp_local_slot_aliases(token_trace):
    traces = [score_trace(token_trace, rank) for rank in range(8)]
    # Same IDs on different DPs must remain different request lifetimes.
    merged = merge_traces(traces, numa_node=-1)
    assert merged.manifest["geometry"]["batch"] == 16
    assert merged.manifest["geometry"]["rounds"] == 8
    assert traces[0].manifest["geometry"]["rounds"] == 4
    np.testing.assert_array_equal(merged.array("pre.row_owners"), np.arange(1, 17))
    np.testing.assert_array_equal(merged.array("pre.metadata")[4].view(np.int32)[:16], np.arange(1, 17))
    for rank in range(8):
        rows = slice(2 * rank, 2 * rank + 2)
        np.testing.assert_array_equal(merged.array("pre.query_ends")[rows], traces[rank].array("pre.query_ends"))
        np.testing.assert_array_equal(
            merged.array("pre.metadata")[2, rank * 8 : (rank + 1) * 8], traces[rank].array("pre.metadata")[2] + rank * 8
        )
        np.testing.assert_array_equal(
            merged.array("pre.ddr_page_slots")[2 * rank, :8], merged.array("pre.ddr_page_slots")[2 * rank + 1, :8]
        )
    assert len({int(merged.array("pre.ddr_page_slots")[2 * rank, 0]) for rank in range(8)}) == 8
    requests = merged.manifest["metadata"]["requests"]
    assert len({item["request_id"] for item in requests}) == 16
    assert merged.manifest["geometry"]["region_bytes"] == 8 * traces[0].manifest["geometry"]["region_bytes"]
    assert merged.manifest["geometry"]["page_mode"] == "thp"
    assert merged.manifest["geometry"]["numa_node"] == -1


def test_real_materializer_preserves_li_bits_sparse_slots_and_score_keys(token_trace, tmp_path):
    traces = [score_trace(token_trace, rank) for rank in range(2)]
    merged = merge_traces(traces, numa_node=3)
    result = materialize_score_trace(merged, tmp_path / "real", numa_node=3)
    archive = load_snapshot(result.directory)
    assert archive.manifest["content"] == "real_li_synthetic_kv"
    assert archive.manifest["oracle"]["native_validated"] is False
    assert "not the original" in archive.manifest["fidelity"]
    np.testing.assert_array_equal(archive.array("pre.keys")[1:], merged.array("real.keys"))
    np.testing.assert_array_equal(archive.array("pre.key_scales")[1:], merged.array("real.key_scales"))
    np.testing.assert_array_equal(archive.array("pre.query"), merged.array("real.query").reshape(16, 32, 128))
    np.testing.assert_array_equal(archive.array("pre.weights"), merged.array("real.weights").reshape(16, 32))
    np.testing.assert_array_equal(archive.array("post.score_keys"), merged.array("post.score_keys"))
    ddr = archive.manifest["ddr"]
    assert ddr["region_bytes"] == 2 * traces[0].manifest["geometry"]["region_bytes"]
    assert ddr["numa_node"] == 3 and ddr["page_mode"] == "thp"
    keys = np.load(archive.directory / "ddr/keys.npy")
    np.testing.assert_array_equal(keys, ddr["synthetic_slot_identities"])
    assert keys[0] == 300 and 1324 in keys
    addresses = archive.array("pre.addresses")
    assert addresses[3, 10] - addresses[1, 10] == traces[0].manifest["geometry"]["region_bytes"]
    assert addresses[1, 10] == addresses[2, 10]
    assert not np.array_equal(
        archive.array("post.selected_payload"), np.zeros_like(archive.array("post.selected_payload"))
    )


def test_score_only_cannot_silently_become_a_timing_fixture(token_trace, tmp_path):
    trace = score_trace(token_trace, real=False)
    assert analyze_scores(trace, predecessor(trace))["status"] == "pass"
    with pytest.raises(ValueError, match="saved real inputs"):
        merge_traces([trace], numa_node=0)
    with pytest.raises(ValueError, match="saved real inputs"):
        materialize_score_trace(trace, tmp_path / "missing", numa_node=0)
    with pytest.raises(ValueError, match="membership rewriting"):
        materialize_token_trace(trace, tmp_path / "rewritten")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ("round", "scheduler round"),
        ("geometry", "core geometry"),
        ("slot", "outside"),
        ("duplicate", "Duplicate DP"),
        ("numa", "explicit"),
    ],
)
def test_merge_rejects_incompatible_sources(token_trace, change, match):
    traces = [score_trace(token_trace, rank) for rank in range(2)]
    numa = 0
    if change == "round":
        traces[1].manifest["metadata"]["context"]["round"] += 1
    elif change == "geometry":
        traces[1].manifest["geometry"]["soc"] = "Other"
    elif change == "slot":
        traces[1].arrays["pre.ddr_page_slots"][0, 0] = 1024
    elif change == "duplicate":
        traces[1].manifest["metadata"]["dp_rank"] = 0
    else:
        numa = None
    with pytest.raises(ValueError, match=match):
        merge_traces(traces, numa_node=numa)


def test_series_resolver_uses_recorded_paths_and_cli_has_no_npu_import(token_trace, tmp_path, monkeypatch, capsys):
    trace = score_trace(token_trace, real=False)
    old = predecessor(trace)
    directory = tmp_path / "dp0"
    directory.mkdir()
    (directory / "series.json").write_text(
        json.dumps(
            dict(
                content="scores",
                status="complete",
                dp_rank=0,
                consecutive_required=True,
                seed_rounds=1,
                request_owner_signature=[
                    dict(request_id=item["request_id"], owner=item["owner"])
                    for item in trace.manifest["metadata"]["requests"]
                ],
                rounds=[
                    dict(
                        round_index=i,
                        scheduler_round=t.manifest["metadata"]["context"]["round"],
                        groups=[dict(group=0, manifest=f"chosen{i}/manifest.json")],
                    )
                    for i, t in [(1, old), (2, trace)]
                ],
            )
        )
    )
    monkeypatch.setattr(
        "kvip.diagnostics.token_trace.load_token_trace", lambda path: old if path.name == "chosen1" else trace
    )
    current, previous = load_round(tmp_path, [0], 2, 0)
    assert current == [trace] and previous == [old]
    main(["--root", str(tmp_path), "--round", "2", "--group", "0", "--dp-ranks", "0"])
    assert "SCORES_ONLY" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--root",
                str(tmp_path),
                "--round",
                "2",
                "--group",
                "0",
                "--dp-ranks",
                "0",
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert exc.value.code == 2
    series = json.loads((directory / "series.json").read_text())
    series["status"] = "capturing"
    (directory / "series.json").write_text(json.dumps(series))
    with pytest.raises(ValueError, match="complete consecutive"):
        load_round(tmp_path, [0], 2, 0)


def test_real_recorder_schema4_load_analyze_and_merge_contract(tmp_path):
    import torch

    from kvip.diagnostics.token_trace import TokenIdTraceRecorder, load_token_trace
    from test_score_trace import score_fixture

    selector, metadata, publish = score_fixture(real=True)
    selector.rotating.rounds = 4  # Actual auto B2/AIC32 geometry, not the fixture's generic W8.
    metadata.update(dp_rank=0, dp_size=8, group=10, layer_names=["model.layers.34"])
    metadata["context"].update(round=51, global_decode_confirmed=True)
    recorder = TokenIdTraceRecorder(torch, tmp_path / "captured", metadata)
    recorder.before(selector)
    publish(recorder)
    recorder.after(selector)
    current = load_token_trace(recorder.directory)
    report = analyze_scores(current, predecessor(current))
    assert report["status"] == "pass"
    assert {q["final_value"] for q in report["queries"]} == {1.0}
    merged = merge_traces([current], numa_node=2)
    np.testing.assert_array_equal(merged.array("real.query"), current.array("real.query"))
    np.testing.assert_array_equal(merged.array("post.score_keys"), current.array("post.score_keys"))
