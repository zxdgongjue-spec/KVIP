# SPDX-License-Identifier: Apache-2.0
from fetch_ab_report import conservative_gain


def test_gain_uses_slower_candidate_faster_control_and_larger_drift_per_metric():
    def sample(mean, p50):
        return {"timing": {"mean_us": mean, "p50_us": p50}}

    old = {"ref": sample(100, 100), "ref-end": sample(102, 95)}
    new = {"v": sample(89, 91), "v-end": sample(90, 90)}
    mean_gain, p50_gain = conservative_gain(new, "v", old, "ref")
    assert round(mean_gain, 6) == 8
    # 91 is faster than 95, but the control itself drifted by 5%.
    assert round(p50_gain, 6) == round(100*(1-91/95)-5, 6)
