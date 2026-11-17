# SPDX-License-Identifier: Apache-2.0
import pytest

from kvip.diagnostics.counter_clock import calibration_summary


def test_calibration_is_measured_median_not_an_assumed_clock():
    values = [dict(cycles=20_000_000, event_us=us) for us in (20_000, 25_000, 40_000)]
    result = calibration_summary(values)
    assert result["cycles_per_us"] == 800
    assert result["rate_min"] == 500 and result["rate_max"] == 1000
    assert result["included_in_timing"] is False


@pytest.mark.parametrize("values", [[], [dict(cycles=20_000_000, event_us=0)]*3,
                                     [dict(cycles=19_999_999, event_us=20_000)]*3,
                                     [dict(cycles=20_000_000, event_us=float("nan"))]*3])
def test_calibration_rejects_invalid_or_short_measurement(values):
    with pytest.raises(ValueError):
        calibration_summary(values)
