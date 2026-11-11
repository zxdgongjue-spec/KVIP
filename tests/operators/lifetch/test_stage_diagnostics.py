"""Unit tests for durable LIFetch stage diagnostics."""

import json
import sys
from pathlib import Path

import pytest

SUPPORT = Path(__file__).resolve().parent / "support"
sys.path.insert(0, str(SUPPORT))
from stage_diagnostics import run_synchronized_stage  # noqa: E402


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_submit_and_sync_completion(tmp_path):
    path = tmp_path / "stages.jsonl"
    calls = []

    result = run_synchronized_stage(
        path,
        "mixed_qk_round1",
        "v9_li",
        lambda: calls.append("submit") or "submitted-result",
        lambda: calls.append("synchronize"),
    )

    assert result == "submitted-result"
    assert calls == ["submit", "synchronize"]
    events = read_events(path)
    assert [event["event"] for event in events] == [
        "begin",
        "submitted",
        "sync_begin",
        "completed",
    ]
    assert all(event["stage"] == "v9_li" for event in events)
    assert all(event["scenario"] == "mixed_qk_round1" for event in events)
    assert all(event["timestamp_monotonic_ns"] > 0 for event in events)


def test_submit_failure_is_logged_and_reraised(tmp_path):
    path = tmp_path / "stages.jsonl"

    def fail_submit():
        raise ValueError("launch rejected")

    with pytest.raises(ValueError, match="launch rejected"):
        run_synchronized_stage(
            path, "round1", "v9_init", fail_submit, lambda: None
        )

    events = read_events(path)
    assert [event["event"] for event in events] == ["begin", "failed"]
    assert events[-1]["phase"] == "submit"
    assert events[-1]["error_type"] == "ValueError"


def test_sync_failure_is_logged_and_reraised(tmp_path):
    path = tmp_path / "stages.jsonl"

    def fail_sync():
        raise RuntimeError("aicore timeout")

    with pytest.raises(RuntimeError, match="aicore timeout"):
        run_synchronized_stage(
            path, "round2", "v9_wait", lambda: None, fail_sync
        )

    events = read_events(path)
    assert [event["event"] for event in events] == [
        "begin",
        "submitted",
        "sync_begin",
        "failed",
    ]
    assert events[-1]["phase"] == "synchronize"
    assert events[-1]["error_type"] == "RuntimeError"
    assert events[-1]["error"] == "aicore timeout"
