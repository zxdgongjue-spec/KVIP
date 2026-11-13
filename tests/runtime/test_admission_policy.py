# SPDX-License-Identifier: Apache-2.0
import random

import pytest

from kvip.runtime.admission_policy import Demand, DemandHistory, PathSelector, allocate_controls, estimate_cost


def test_feedback_is_causal_and_lifetime_scoped():
    h = DemandHistory()
    assert h.predict((1, 1), 7, 51, fallback_reads=512).samples == 0
    h.observe((1, 1), 7, 51, selected_reads=800)
    with pytest.raises(ValueError):
        h.predict((1, 1), 7, 51, fallback_reads=512)
    assert h.predict((1, 1), 7, 52, fallback_reads=512).expected == 800
    assert h.predict((1, 2), 7, 52, fallback_reads=512).samples == 0
    assert h.predict((1, 1), 4, 52, fallback_reads=512).samples == 0
    assert h.predict((1, 1), 7, 55, fallback_reads=512).samples == 0
    with pytest.raises(ValueError):
        h.observe((1, 1), 7, 51, selected_reads=1)


def test_pool_does_not_leave_quota_in_idle_owners():
    plan = allocate_controls([Demand(0, 0, 1), Demand(1100, 0, 1)], pool_tokens=1024)
    assert [row[0] for row in plan] == [0, 1024]
    assert plan[1][1:] == (128, 0, 1)


def test_random_pool_capacity_and_owner_limits():
    rng = random.Random(1604)
    for _ in range(1000):
        batch = rng.randint(1, 16)
        predictions = [Demand(rng.random() * 8192, rng.random() * 1024, 4) for _ in range(batch)]
        pool = rng.randint(0, batch * 2048)
        controls = allocate_controls(predictions, pool_tokens=pool)
        assert sum(row[0] for row in controls) <= pool
        assert all(0 <= row[0] <= 2048 for row in controls)


def test_shape_guard_and_switch_cost_and_hysteresis():
    curves = {"serial": [(0, 280), (16000, 900)], "fused": [(0, 380), (16000, 600)]}
    p = PathSelector(fingerprint="shape-runtime", curves=curves)
    inputs = dict(fingerprint="shape-runtime", predictions=[Demand(16000, 0, 10)],
                  compatible_paths=("serial", "fused"))
    assert p.choose(**inputs)["path"] == "serial"
    assert p.choose(**inputs)["path"] == "fused"
    assert p.choose(**(inputs | {"fingerprint": "different-payload"}))["reason"] == "uncalibrated_shape_or_runtime"
    p = PathSelector(fingerprint="shape-runtime", curves=curves, confirmations=1)
    assert p.choose(**inputs, transition_us=400)["path"] == "serial"
    assert p.choose(**(inputs | {"compatible_paths": ("serial",)}))["path"] == "serial"


def test_no_latency_or_shape_extrapolation():
    assert estimate_cost([(100, 30), (200, 50)], 150) == 40
    assert estimate_cost([(100, 30), (200, 50)], 201) is None
    assert estimate_cost([(100, 30)], 99) is None
    with pytest.raises(ValueError):
        estimate_cost([(100, float("nan"))], 100)
    with pytest.raises(ValueError):
        allocate_controls([Demand(float("inf"), 0, 1)], pool_tokens=512)


@pytest.mark.parametrize("step", [-1, True, False, 1.0, float("nan"), "2", None])
def test_history_requires_integer_decode_steps(step):
    history = DemandHistory()
    with pytest.raises(ValueError, match="nonnegative integer"):
        history.observe("request-generation", 7, step, selected_reads=12)
    with pytest.raises(ValueError, match="nonnegative integer"):
        history.predict("request-generation", 7, step, fallback_reads=12)


@pytest.mark.parametrize("confirmations", [-1, 0, True, 1.0, float("nan"), "2", None])
def test_hysteresis_requires_a_positive_integer(confirmations):
    with pytest.raises(ValueError, match="hysteresis"):
        PathSelector(fingerprint="G7-runtime", curves={"serial": [(0, 200)]},
                     confirmations=confirmations)


def test_cold_owner_resets_pending_switch_and_transition_is_charged_once():
    curves = {"serial": [(0, 200), (1000, 200)], "fused": [(0, 100), (1000, 100)]}
    selector = PathSelector(fingerprint="G7-runtime", curves=curves)
    inputs = dict(fingerprint="G7-runtime", predictions=[Demand(100, 10, 8)],
                  compatible_paths=("serial", "fused"), transition_us=20)
    assert selector.choose(**inputs)["path"] == "serial"
    cold = inputs | {"predictions": [Demand(50, 5, 8), Demand(50, 5, 0)]}
    for _ in range(3):
        report = selector.choose(**cold)
        assert report["path"] == "serial" and report["reason"] == "cold_start"
        assert report["transition_us"] == 0 and report["predicted_us"] == 200
    assert selector.choose(**inputs)["path"] == "serial"
    report = selector.choose(**inputs)
    assert report["path"] == "fused"
    assert report["predicted_us"] == 120
    assert report["predicted_path_us"] == 100 and report["transition_us"] == 20
    report = selector.choose(**inputs)
    assert report["predicted_us"] == 100 and report["transition_us"] == 0


@pytest.mark.parametrize("expensive_endpoint", [0, 200])
def test_switch_requires_both_uncertainty_endpoints_to_win(expensive_endpoint):
    # The mean strongly favors fusion, but one calibrated endpoint does not.
    fused = [(x, 220 if x == expensive_endpoint else 100) for x in (0, 100, 200)]
    selector = PathSelector(fingerprint="G7-runtime", confirmations=1,
                            curves={"serial": [(0, 200), (200, 200)], "fused": fused})
    inputs = dict(fingerprint="G7-runtime", predictions=[Demand(100, 100, 8)],
                  compatible_paths=("serial", "fused"))
    report = selector.choose(**inputs)
    assert report["estimates"]["fused"] == 100
    assert report["demand_interval"] == (0, 200) and report["path"] == "serial"
    assert selector.choose(**(inputs | {"predictions": [Demand(100, 0, 8)]}))["path"] == "fused"


def test_transition_cost_is_checked_at_the_uncertainty_endpoints():
    selector = PathSelector(fingerprint="G7-runtime", confirmations=1,
                            curves={"serial": [(0, 200), (200, 200)],
                                    "fused": [(0, 180), (100, 100), (200, 180)]})
    inputs = dict(fingerprint="G7-runtime", predictions=[Demand(100, 100, 8)],
                  compatible_paths=("serial", "fused"))
    assert selector.choose(**inputs, transition_us=15)["path"] == "serial"
    report = selector.choose(**inputs, transition_us=5)
    assert report["path"] == "fused" and report["predicted_us"] == 105


def test_uncertainty_outside_calibration_and_other_groups_do_not_switch():
    selector = PathSelector(fingerprint=("G7", "shape-runtime"), confirmations=1,
                            curves={"serial": [(0, 200), (200, 200)],
                                    "fused": [(0, 100), (200, 100)]})
    inputs = dict(fingerprint=("G7", "shape-runtime"), predictions=[Demand(100, 101, 8)],
                  compatible_paths=("serial", "fused"))
    report = selector.choose(**inputs)
    assert report["path"] == "serial" and report["reason"] == "outside_calibrated_uncertainty"
    assert report["demand_interval"] == (0, 201)
    other_group = inputs | {"fingerprint": ("G4", "shape-runtime"), "predictions": [Demand(100, 0, 8)]}
    report = selector.choose(**other_group)
    assert report["path"] == "serial" and report["predicted_us"] is None


@pytest.mark.parametrize("samples", [-1, True, 1.0, float("nan")])
def test_prediction_sample_counts_are_checked(samples):
    selector = PathSelector(fingerprint="G7-runtime", curves={"serial": [(0, 200), (200, 200)]})
    with pytest.raises(ValueError, match="samples"):
        selector.choose(fingerprint="G7-runtime", predictions=[Demand(100, 0, samples)],
                        compatible_paths=("serial",))
