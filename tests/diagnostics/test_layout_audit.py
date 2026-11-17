"""Physical permutation, scheduling and semantic/physical state contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from kvip.diagnostics import layout_survey, layout_state, replay
from score_archive_fixture import score_archive, write_archive
from test_threshold_survey import recorded_series


def test_reversal_uses_physical_positions_not_sorted_token_ids():
    h, r = 16384, 4096
    ids = np.arange(h, dtype=np.int32)[None, :]
    ids[:, [6000, 12000]] = ids[:, [12000, 6000]]
    topk = np.tile(np.r_[np.arange(2047), 12000], (1, 4, 1))
    metadata = np.zeros((5, 4), np.int64)
    metadata[2] = np.arange(4)
    a = {
        "pre.logical_ids": ids, "post.logical_ids": ids.copy(),
        "pre.hot_valid": np.ones((1, r), np.uint8),
        "pre.query_ends": np.full((1, 4), 16000, np.int32),
        "pre.tail_bounds": np.array([[15872, 16000]], np.int32),
        "pre.lru_slots": np.arange(2048, dtype=np.int32)[None, :],
        "pre.address_valid": (ids < 15872).astype(np.uint8),
        "pre.metadata": metadata, "post.topk": topk,
        "post.logical_output": ids[0, topk[0]],
    }
    previous = dict(ids=ids.copy(), post_ids=ids.copy(), ends=a["pre.query_ends"])
    m = dict(geometry=dict(history=h, resident=r))
    result, blocks = layout_survey.request_metrics(m, a, 0, previous)
    assert result["cold"] == 1 and result["displaced_block"] == 2
    assert result["reverse_lead"] > result["forward_lead"]
    assert sum(b["cold_demand"] for b in blocks) == 1
    assert next(b for b in blocks if b["cold_demand"])["physical_start"] == 10240
    # Changing token age in the same physical slot must not change discovery.
    a["pre.logical_ids"] = np.arange(h, dtype=np.int32)[None, :]
    a["post.logical_ids"] = a["pre.logical_ids"].copy()
    a["post.logical_output"] = topk[0].copy()
    result2, _ = layout_survey.request_metrics(m, a, 0, previous)
    assert result2["forward_lead"] == result["forward_lead"]
    assert result2["reverse_lead"] == result["reverse_lead"]


def test_eight_cold_windows_and_partial_causal_block_are_preserved():
    h, r, ends = 16640, 4096, [16001, 16002, 16003, 16004]
    f = layout_survey.scan_windows(h, r, ends, False)
    rev = layout_survey.scan_windows(h, r, ends, True)
    assert set(f) == set(rev) == set(range(9))
    np.testing.assert_array_equal(f[:r], rev[:r])
    np.testing.assert_array_equal(f[14336:], rev[14336:])
    assert f[4096] == 1 and f[-1] == 8
    assert rev[14335] < f[14335]


def test_reader_does_not_decompress_score_or_real_index_members(tmp_path, monkeypatch):
    trace = score_archive(real=True)
    write_archive(trace, tmp_path / "sample")
    original = np.lib.npyio.NpzFile.__getitem__

    def getitem(self, name):
        assert name in layout_survey.FIELDS
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    _, arrays = layout_survey.read_ids(tmp_path / "sample/manifest.json")
    assert set(arrays) == set(layout_survey.FIELDS)


def test_complete_survey_cli_outputs_every_round_layer(recorded_series, tmp_path, capsys):
    output = tmp_path / "survey"
    layout_survey.main(["--root", str(recorded_series), "--result-dir", str(output),
                        "--workers", "2", "--expect-dp", "0", "1", "--expect-rounds", "3",
                        "--expect-layers", "4", "--cold-windows", "8"])
    d = json.loads((output / "summary.json").read_text())
    assert d["request_rows"] == 8 and d["round_layer_rows"] == 8
    assert d["cold_windows"] == 8 and len(d["groups"]) == 1
    assert not (output / "INCOMPLETE").exists()
    assert (output / "physical_blocks.csv.gz").is_file()
    assert "LAYOUT_SURVEY_PASS" in capsys.readouterr().out


def state_fixture(base=10000):
    h, r, stride, width = 8, 4, 6, 2
    ids = np.tile(np.arange(h, dtype=np.int64), (2, 1))
    state = dict(logical_ids=ids, keys=np.repeat(ids[..., None], 2, axis=-1).astype(np.uint8),
                 key_scales=ids.astype(np.float32), addresses=base+ids*width,
                 hot_valid=np.array([[0]*r, [1, 0, 1, 0]], np.int32),
                 output=np.tile(np.arange(stride, dtype=np.uint8)[None, :, None], (2, 1, width)))
    oracle = SimpleNamespace(geometry=dict(resident=r, token_bytes=width, output_stride=stride),
                             owners=[1], owner_count=2, pre={"query_ends": np.full((1, 4), 7)},
                             ddr={"region_base": 10000})
    return state, oracle


def test_semantic_hot_equality_is_separate_from_physical_slot_equality():
    state, oracle = state_fixture()
    first = layout_state.summarize(state, oracle, 0)
    # Swap two valid hot entries consistently. Set/payload stay semantically equal.
    for name in ("logical_ids", "keys", "key_scales", "addresses", "output"):
        state[name][1, [0, 2]] = state[name][1, [2, 0]]
    second = layout_state.summarize(state, oracle, 0)
    report = layout_state.compare(first, second)
    assert report["hot_tokens_equal"] and report["hot_payload_by_token_equal"]
    assert not report["persistent_state_equal"]
    assert report["changed_index_positions"] == report["changed_hot_slots"] == 2
    assert report["hot_token_symmetric_difference"] == 0


def test_address_normalization_and_ignored_invalid_bytes():
    a, oracle = state_fixture()
    b, _ = state_fixture(base=20000)
    b["output"][1, [1, 3]] = 99
    left = layout_state.summarize(a, oracle, 0)
    right = layout_state.summarize(b, oracle, 10000)
    assert layout_state.compare(left, right)["persistent_state_equal"]
    b["output"][1, 0] = 77
    assert not layout_state.compare(left, layout_state.summarize(b, oracle, 10000))["persistent_state_equal"]


def test_schedule_override_is_in_memory_and_rejects_insufficient_cores():
    g = dict(batch=16, aic_count=32, history=131328, resident=16384, requested_windows=8)
    archive = SimpleNamespace(manifest={"geometry": g})
    layout_state.configure_windows(archive, 8)
    assert g["requested_windows"] == 8
    assert archive.manifest["geometry"]["rounds"] == 9
    assert len(archive.manifest["layout_audit_schedule"]["boundaries"]) == 9
    g["batch"] = 2
    with pytest.raises(ValueError, match="Insufficient"):
        layout_state.configure_windows(SimpleNamespace(manifest={"geometry": g}), 8)


def test_full_layout_audit_arguments(tmp_path):
    args = replay.arguments([
        "--snapshot", "input", "--build-dir", "build", "--transport", "urma", "--mode", "fused",
        "--threshold-policy", "recent", "--study", "scan-order", "--scan-variants",
        "forward", "reverse", "reverse-end", "forward-end", "--cold-windows", "8", "--compare-layout",
        "--upstream-root", "native", "--upstream-revision", "1"*40, "--soc", "Ascend950DT_9582",
        "--kvaf-server", "127.0.0.1:29091", "--output", str(tmp_path / "result.json"),
    ])
    assert args.cold_windows == 8 and args.compare_layout
