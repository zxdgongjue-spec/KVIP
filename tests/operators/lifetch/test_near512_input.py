import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))

import near512_input
from native_li_reference import digest
from near512_input import _manifest_matches, _near_ends, _stitched, make_input
from workloads import CaptureReference, ReplayRound


def test_stitch_reads_distinct_donor_keys_and_requires_independent_native_scores():
    batch, history, resident = 3, 512, 64
    n = np.arange(history)
    row = np.arange(batch)[:, None]
    keys = np.zeros((batch, history, 128), np.uint8)
    keys[:, :, 0] = n % 251
    keys[:, :, 1] = row
    scales = (1 + n + 1000 * row).astype(np.float32)
    layout = np.tile(n, (batch, 1)).astype(np.int32)
    layout[:, :64], layout[:, 448:512] = layout[:, 448:512].copy(), layout[:, :64].copy()
    g = dict(batch=batch, history=history, resident=resident, output_stride=resident + 384)
    source = ReplayRound(
        1,
        2,
        tuple((0, str(r), r) for r in range(batch)),
        g,
        keys,
        scales,
        np.ones((batch, 4, 1, 128), np.uint8),
        np.ones((batch, 4, 1), np.uint16),
        np.ones((batch, 4, 1), np.float32),
        np.tile(np.arange(509, 513), (batch, 1)),
        np.tile([380, 512], (batch, 1)),
        np.ones((batch, history), np.uint8),
        (),
        layout,
        np.ones((batch, resident), np.uint8),
        layout[:, :32],
        (),
        np.ones((batch, 4, history), np.uint16),
        np.ones((batch, 4, history), bool),
    )
    source.logical_address_valid[:, 128:256] = 0  # A missing page must never contribute donor Key bytes.
    source.logical_score_valid[:, :, 256:384] = 0
    ends = np.array([[953, 954, 955, 956], [922, 923, 924, 925], [880, 881, 882, 883]], np.int32)

    result, proof = _stitched(source, ends, 1024)
    donor = result.logical_keys[:, :, 1]
    source_id = (result.logical_scales - 1 - donor.astype(int) * 1000).astype(int)
    active = np.arange(1024) < ends[:, -1:]
    ids = source_id[active]
    assert np.all((ids < 128) | (ids >= 384))
    np.testing.assert_array_equal(result.logical_keys[:, :, 0][active], ids % 251)
    assert any(len(np.unique(donor[b, active[b]])) == batch for b in range(batch))
    with pytest.raises(RuntimeError, match="independent native"):
        result.validate_topk(None)
    assert result.captured_lru is source.captured_lru
    assert result.geometry["history"] == 1024
    np.testing.assert_array_equal(result.tail_bounds[:, 1], ends[:, -1])
    np.testing.assert_array_equal(result.tail_bounds[:, 0], ends[:, -1] - 132)
    np.testing.assert_array_equal(np.sort(result.captured_layout), np.tile(np.arange(1024), (batch, 1)))
    np.testing.assert_array_equal(result.logical_score_valid, np.arange(1024) < ends[:, :, None])
    np.testing.assert_array_equal(
        result.logical_keys[row, result.captured_layout[:, :resident]],
        source.logical_keys[row, source.captured_layout[:, :resident]],
    )
    assert proof["source_pages_available_per_row"] == [2, 2, 2]
    assert proof["captured_hot_slot_ids_remapped_per_row"] == [64, 64, 64]
    assert proof["distinct_target_key_rows"] == batch
    published = [key for rows in result.page_keys for key in rows if key is not None]
    assert len(set(published)) == len(published)


def test_real_input_requires_real_operands_and_every_q_near_512k(tmp_path):
    ends = np.tile(524273 + np.arange(4, dtype=np.int32), (2, 1))
    np.savez(tmp_path / "scores.npz", **{"pre.query_ends": ends})
    g = dict(history=524544, resident=16384, heads=32, token_bytes=656)
    manifest = tmp_path / "manifest.json"
    fields = dict(geometry=g, metadata=dict(group=4), archive=dict(file="scores.npz"), real_inputs_saved=True)
    manifest.write_text(json.dumps(fields))
    assert _manifest_matches(manifest, 4)
    ends[1] -= 131072
    np.savez(tmp_path / "scores.npz", **{"pre.query_ends": ends})
    assert not _manifest_matches(manifest, 4)
    assert not _near_ends(np.array([[524288, 524288, 524288, 524288]]), 524544)
    fields["real_inputs_saved"] = False
    manifest.write_text(json.dumps(fields))
    assert not _manifest_matches(manifest, 4)


def test_explicit_bad_native_path_does_not_become_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("PEER_NEAR512_TRACE", str(tmp_path))
    with pytest.raises(ValueError, match="No complete"):
        make_input(tmp_path, 4, None, tmp_path)


@pytest.mark.parametrize("group,source_round", [(10,51), (4,2), (17,2), (7,2), (2,2)])
def test_five_cases_stitch_only_the_verified_group(tmp_path, monkeypatch, group, source_round):
    fields = ("query", "weights", "query_scale", "logical_keys", "logical_scales", "query_ends", "captured_layout")
    state = ("captured_lru", "captured_hot_valid", "logical_address_valid", "tail_bounds")
    refs = [CaptureReference(dp,1,900,group,tmp_path / f"dp{dp}",((f"a{dp}",1),(f"b{dp}",2))) for dp in range(8)]
    source = SimpleNamespace(**{f: np.array([group, i]) for i, f in enumerate((*fields, *state))},
                             identities=tuple((r.dp_rank, n, o) for r in refs for n,o in r.request_signature),
                             source_round=1, scheduler_round=900, source_paths=[str(r.manifest_path) for r in refs],
                             geometry=dict(batch=16, heads=32, history=131328, resident=16384, token_bytes=656))
    proof = tmp_path / "short.json"
    expected = {"capture_operand_hashes": {f: digest(getattr(source, f)) for f in fields},
                "history_cap": {"capture_state_sha256": {f: digest(getattr(source, f)) for f in state}}}
    proof.write_text(json.dumps(expected))
    class Sequence:
        def __init__(self, path, g, rounds):
            assert path == tmp_path and g == group and rounds == (1,1)
            self.rounds = (refs,)
        def cache_ranked_round(self):
            self.cached_ranked_round = source
        def __iter__(self):
            yield self.cached_ranked_round

    monkeypatch.setattr(near512_input, "ReplaySequence", Sequence)
    monkeypatch.setattr(near512_input, "_find_long_input", lambda *_: pytest.fail("unexpected long capture lookup"))
    monkeypatch.setattr(near512_input, "_stitched", lambda src, ends, cap: (src, {}))
    monkeypatch.setenv("PEER_NEAR512_TRACE", "an unrelated long trace")
    _, _, result = make_input(tmp_path, group, tmp_path, tmp_path, stitch_source=True, source_proof=proof)
    donor = result["donor"]
    assert (donor["group"], donor["selector_round"], donor["scheduler_round"]) == (group, source_round, 900)
    assert donor["dp_ranks"] == list(range(8)) and donor["requests"] == 16
    source.query[:] = -1
    with pytest.raises(ValueError, match="128K source differs.*query"):
        make_input(tmp_path, group, tmp_path, tmp_path, stitch_source=True, source_proof=proof)
    refs[0] = CaptureReference(0,1,900,88,tmp_path,refs[0].request_signature)
    with pytest.raises(ValueError, match="group, round or request identities"):
        make_input(tmp_path, group, tmp_path, tmp_path, stitch_source=True, source_proof=proof)
