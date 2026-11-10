# SPDX-License-Identifier: Apache-2.0
"""KVAF transfer-vs-stable-time attribution (CPU only, no NPU or service).

These tests pin the arithmetic that separates the round time spent waiting for
KVAF BatchGet (``kvaf_fetch_us``) from the transport-independent work
(``unaffected_us``). They are the deterministic counterpart to the NPU trace and
run without the simulator, an Ascend device, or a KVAF service.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/operators/lifetch/support"))

from device_trace import (  # noqa: E402
    TOKEN_BYTES,
    _annotate_transport,
    bandwidth_line,
    bandwidth_summary,
)


def sample(fetch_us, prefetch, final, total_us=None, gap=0.07, boundary=70.0, init=12.0, li=285.0, resolve=24.0):
    stages = {"init": init, "li": li, "resolve": resolve, "fetch": fetch_us}
    total = total_us if total_us is not None else sum(stages.values()) + gap + boundary
    return {
        "round": 0,
        "us": total,
        "prefetch": prefetch,
        "final": final,
        "fetch_bytes": (prefetch + final) * TOKEN_BYTES,
        "accounting": {
            "stages_us": stages,
            "inter_kernel_gap_us": gap,
            "boundary_residual_us": boundary,
        },
    }


def test_fetch_bytes_uses_token_size():
    row = sample(fetch_us=20.0, prefetch=8000, final=200)
    assert row["fetch_bytes"] == 8200 * TOKEN_BYTES == 8200 * 656


def test_annotate_splits_transfer_from_unaffected():
    row = sample(fetch_us=100.0, prefetch=8200, final=0)
    _annotate_transport(row)
    assert row["kvaf_fetch_us"] == 100.0
    # 8200 * 656 bytes in 100 us -> 53.792 GB/s
    assert row["kvaf_fetch_GBps"] == pytest.approx(8200 * 656 / (100.0 * 1000.0), rel=1e-6)
    # unaffected = total - fetch = init + li + resolve + gap + boundary
    expected = 12.0 + 285.0 + 24.0 + 0.07 + 70.0
    assert row["unaffected_us"] == pytest.approx(expected, abs=1e-6)
    assert row["unaffected_pct"] == pytest.approx(expected * 100.0 / row["us"], abs=1e-2)


def test_bandwidth_contention_moves_fetch_not_unaffected():
    """Same bytes, different fetch time: only fetch/bandwidth move."""
    fast = sample(fetch_us=15.0, prefetch=8200, final=0)
    slow = sample(fetch_us=150.0, prefetch=8200, final=0)
    _annotate_transport(fast)
    _annotate_transport(slow)
    assert fast["fetch_bytes"] == slow["fetch_bytes"]
    assert slow["kvaf_fetch_GBps"] < fast["kvaf_fetch_GBps"]
    # The transport-independent work is identical across both samples.
    assert fast["unaffected_us"] == pytest.approx(slow["unaffected_us"])


def test_zero_or_missing_fetch_does_not_divide():
    row = sample(fetch_us=0.0, prefetch=0, final=0)
    _annotate_transport(row)
    assert row["kvaf_fetch_GBps"] is None
    row2 = {"us": 400.0, "fetch_bytes": 0, "accounting": {"stages_us": {}}}
    _annotate_transport(row2)
    assert row2["kvaf_fetch_us"] is None
    assert row2["unaffected_us"] is None
    assert row2["unaffected_pct"] is None


def test_bandwidth_summary_aggregates_swing_and_unaffected():
    rows = [
        sample(fetch_us=15.0, prefetch=8200, final=0),
        sample(fetch_us=150.0, prefetch=8200, final=0),
        sample(fetch_us=90.0, prefetch=8200, final=0),
    ]
    for row in rows:
        _annotate_transport(row)
    summary = bandwidth_summary({"samples": rows})
    assert summary["rounds"] == 3
    assert summary["fetch_bytes_mean"] == pytest.approx(8200 * 656)
    assert summary["kvaf_fetch_us_mean"] == pytest.approx((15.0 + 150.0 + 90.0) / 3)
    assert summary["kvaf_fetch_us_min"] == 15.0
    assert summary["kvaf_fetch_us_max"] == 150.0
    assert summary["kvaf_fetch_us_swing"] == pytest.approx(10.0)
    assert summary["unaffected_us_min"] == pytest.approx(summary["unaffected_us_max"], abs=1e-6)


def test_bandwidth_line_reports_unaffected_part():
    rows = [sample(fetch_us=15.0, prefetch=8200, final=0), sample(fetch_us=150.0, prefetch=8200, final=0)]
    for row in rows:
        _annotate_transport(row)
    line = bandwidth_line({"samples": rows})
    assert line.startswith("KVAF BANDWIDTH:")
    assert "swing=10.00x" in line
    assert "unaffected_us mean=" in line
    assert "% of round" in line


def test_missing_bytes_skips_summary():
    rows = [{"us": 400.0, "accounting": {"stages_us": {"fetch": 20.0}}}]  # no fetch_bytes
    assert bandwidth_summary({"samples": rows}) is None
    assert bandwidth_line({"samples": rows}) is None
