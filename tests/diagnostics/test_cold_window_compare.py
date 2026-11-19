import numpy as np
import pytest

from kvip.diagnostics.cold_window_compare import decode, edges


def test_g7_physical_coverage_and_partial_last_pair():
    original = np.diff(edges(131328, 16384, reference=False))
    reference = np.diff(edges(131328, 16384, reference=True))
    assert list(original) == [112, 112, 112, 113] * 2
    assert list(reference) == [116] * 7 + [86]
    assert sum(reference) == sum(original) == 898


def test_own_fetch_exclusion_including_the_last_window():
    t = np.zeros((64, 8, 8), np.int64)
    cuts = edges(131328, 16384, reference=False)
    for c, e in np.ndindex(64, 8):
        t[c, e, :2] = [1000 + 100*e, 1040 + 100*e]
        t[c, e, 4:6] = cuts[e:e+2]
        if e == c//8:
            t[c, e, 2:4] = [1042 + 100*e, 1050 + 100*e]
    out = decode(t, 2, 1)
    assert out["scan_us"][0] == [20] * 8
    assert out["outside_us"][0] == (740-8)/2
    assert out["outside_us"][-1] == (750-8)/2
    t[63, 7, 1] = 0
    with pytest.raises(ValueError):
        decode(t, 2, 1)


def test_swapped_or_omitted_shard_is_rejected():
    t = np.zeros((64, 8, 8), np.int64)
    with pytest.raises(ValueError, match="tile bounds"):
        decode(t, 1, 3)


def test_one_run_gate_uses_both_halves_and_both_controls(tmp_path, capsys):
    import json

    from kvip.diagnostics.cold_window_compare import graph_report

    def save(name, samples):
        count = 4 if name == "ranked" else 1
        timeline = dict(li=dict(count=count, span_us=200), streaming=dict(count=1, span_us=100))
        case = dict(
            status="pass",
            snapshot_sha256="g7",
            geometry=dict(rounds=8, service_lanes=8, token_bytes=656),
            tuning=dict(window_limit=64, prefetch_limit=512),
            score_comparison=dict(status="bit_exact"),
            threshold_comparison=dict(initial_keys=[1], final_keys=[2]),
            timing=dict(samples_us=samples),
            reads_samples=[dict(unused_prefetch=2, final=5, transfers=10)],
            system_profiles=dict(passes=[dict(timeline=timeline, metric="none")]),
        )
        (tmp_path / f"g7-{name}-1.json").write_text(json.dumps(case))

    save("forward", [120] * 10)
    save("reverse", [100] * 10)
    save("ranked", [85] * 5 + [101] * 5)
    graph_report(tmp_path)
    assert "GATE KEEP_BASELINES" in capsys.readouterr().out
    save("ranked", [85] * 5 + [95] * 5)
    graph_report(tmp_path)
    assert "GATE BOTH_BASELINES_BEATEN" in capsys.readouterr().out
