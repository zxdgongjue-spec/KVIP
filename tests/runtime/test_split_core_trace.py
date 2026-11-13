"""Trace summaries must preserve time domains and reject incomplete raw data."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("schedule_trace", ROOT / "experiments/split_core/scheduling_trace.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample(start=100, end=200):
    return f"[info] +++core: 0, subcore: 1, start: {start}, tick: {end}, kernelId: 0, blkID: 0, blkDim: 32, streamId: 0, done++++."


def test_native_footer_is_one_complete_envelope():
    assert module.footer(sample()) == dict(core=0, subcore=1, start=100, tick=200, block=0, blocks=32)
    for text in ("incomplete", sample() + sample(), sample(200, 100)):
        with pytest.raises(ValueError):
            module.footer(text)


def test_pipe_occupancy_preserves_vendor_clock_and_union():
    events = [
        dict(ph="X", pid="core0.veccore0", tid="MTE2", ts=100.5, dur=2),
        dict(ph="X", pid="core0.veccore0", tid="MTE2", ts=101.5, dur=3),
        dict(ph="X", pid="core0.veccore1", tid="MTE2", ts=102, dur=1),
        dict(ph="M", pid="core0.veccore0", tid="MTE2", name="thread_name"),
    ]
    result = module.occupancy(events)
    assert result["x_events"] == 3 and result["origin"] == 100.5 and result["end"] == 104.5
    assert result["lanes"][0]["intervals"] == [[100.5, 104.5]]
    assert result["lanes"][1]["intervals"] == [[102, 103]]
    assert events[0]["ts"] == 100.5  # No normalization or counter/frequency conversion in raw data.


def test_invalid_or_empty_trace_is_rejected():
    with pytest.raises(ValueError):
        module.occupancy([])
    with pytest.raises(ValueError):
        module.occupancy([dict(ph="X", pid=0, tid=0, ts=2, dur=-1)])
    with pytest.raises(ValueError):
        module.interval_union([(float("nan"), 1)])


def test_historical_kernel_names_are_classified_without_counting_duplicates():
    assert module.stage("v9_li_kernel_0_mix_aic") == "LI"
    assert module.stage("kvop_direct_update_outputs_early_kernel_3") == "PublishPlan"
    assert module.stage("v9_fusion3_tail_kernel_0") == "Fusion3 Tail"
    with pytest.raises(ValueError):
        module.stage("unknown_kernel")


def test_unrelated_archive_cannot_be_labeled_historical_comparison(tmp_path):
    path = tmp_path / "unrelated.tar.gz"
    path.write_bytes(b"not the audited archive")
    with pytest.raises(ValueError, match="pinned"):
        module.archive(path, 12)
