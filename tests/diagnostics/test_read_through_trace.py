# SPDX-License-Identifier: Apache-2.0
"""CPU-only traffic counterfactuals; never pretend these are device timings."""

import json

import pytest

from kvip.diagnostics import read_through_trace as audit
from score_archive_fixture import score_archive, write_archive
from test_threshold_survey import recorded_series  # noqa: F401


def state(hot, selected, *, tail=(100, 104), identity=("request", 1)):
    return audit.RequestState(identity, frozenset(hot), frozenset(selected), *tail, 10)


def test_spans_count_repeated_cold_reads_and_reset_at_boundaries():
    frames = [
        (2, 102, [state([0], [0, 20])]),
        (3, 103, [state([0, 20], [0, 20])]),
        (4, 104, [state([0, 20], [0, 20])]),
        (5, 105, [state([0, 20], [0, 20])]),
        (6, 106, [state([0, 20], [0, 20])]),
    ]
    result = audit.audit_sequence(frames, group=2, rank=0, spans=[1, 2, 4, 8])
    assert [result[n].retained_cold_reads for n in (1, 2, 4, 8)] == [1, 2, 4, 5]
    assert all(item.baseline_cold_reads == 1 and item.samples == 5 for item in result.values())
    assert result[4].report(656)["net_extra_reads"] == 3
    assert result[4].worst["capture_round"] == 3


def test_retained_hot_can_save_evicted_tokens_and_net_is_signed():
    frames = [(2, 102, [state([0, 1], [0])]),
              (3, 103, [state([0, 20], [1])])]
    result = audit.audit_sequence(frames, group=2, rank=0, spans=[2])[2].report(656)
    assert result["baseline_cold_reads"] == 1
    assert result["retained_cold_reads"] == 0
    assert result["saved_reads"] == 1
    assert result["net_extra_reads"] == -1
    assert result["net_extra_percent"] == -100


def test_abc_advances_without_freezing_and_duplicate_q_ids_do_not_duplicate_get():
    frames = [(2, 102, [state([0], [0, 100, 100])]),
              (3, 103, [state([0, 100], [100, 104, 104], tail=(104, 108))])]
    result = audit.audit_sequence(frames, group=2, rank=0, spans=[2])[2]
    assert result.selected == 4
    assert result.baseline_cold_reads == 0
    assert result.retained_cold_reads == result.newly_missing_reads == 1
    assert result.report(656)["net_extra_percent"] is None


@pytest.mark.parametrize("change", ["request", "owner", "round"])
def test_unknown_request_lifetime_or_missing_round_rejected(change):
    identity = ("other", 1) if change == "request" else ("request", 2) if change == "owner" else ("request", 1)
    frames = [(2, 102, [state([0], [20])]),
              (3, 104 if change == "round" else 103, [state([20], [20], identity=identity)])]
    with pytest.raises(ValueError, match="consecutive|lifetime"):
        audit.audit_sequence(frames, group=2, rank=0, spans=[8])


def test_reader_decompresses_only_named_state_members(tmp_path, monkeypatch):
    path = write_archive(score_archive(real=True), tmp_path / "sample") / "manifest.json"
    original = audit.np.load
    accessed = []
    class Guard:
        def __enter__(self):
            self.source = original(path.parent / "scores.npz", allow_pickle=False)
            return self

        def __exit__(self, *args):
            self.source.close()

        def __getitem__(self, name):
            assert name in audit.MEMBERS
            accessed.append(name)
            return self.source[name]
    monkeypatch.setattr(audit.np, "load", lambda *args, **kwargs: Guard())
    manifest, requests = audit.read_requests(path)
    assert set(accessed) == set(audit.MEMBERS)
    assert manifest["real_inputs_saved"]
    assert len(requests) == 2
    assert all(request.cold(request.hot) == {5000} for request in requests)
    assert all(8064 not in request.cold(request.hot) for request in requests)


def test_reader_rechecks_captured_demand(tmp_path):
    path = write_archive(score_archive(), tmp_path / "sample") / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["summary"]["requests"][0]["cold_demand"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="demand differs"):
        audit.read_requests(path)


def test_cli_validated_schema4_series(recorded_series, tmp_path, capsys):
    output = tmp_path / "traffic.json"
    audit.main(["--root", str(recorded_series), "--groups", "2", "--expect-rounds", "3",
                "--expect-dp", "0", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert len(report["cases"]) == 4
    assert all(row["samples"] == 8 and row["net_extra_reads"] == 0 for row in report["cases"])
    assert "not elapsed time" in report["scope"]["limits"]
    terminal = capsys.readouterr().out
    assert terminal.count("TRAFFIC G2") == 4
    assert "scores_loaded=false" in terminal
