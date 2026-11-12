# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for P0-A2.3 ABBA timing aggregation and fail-closed gates."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).with_name("run_active_k_abba.py")
spec = importlib.util.spec_from_file_location("active_k_abba", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def make_run(label, init, core, full):
    return {
        "switch": label,
        "fixture_sha256": {"arg1.bin": "equal", "logical_scores.npy": "equal"},
        "native_unique_misses": [820, 824, 828, 832],
        "metrics_p50_us": dict(
            round_init=init, native_chain=385, prototype_core=core,
            prototype_with_update=full, aiv_cache_update=98,
        ),
    }


def test_abba_uses_two_balanced_off_on_pairs():
    runs = [
        make_run("OFF", 38, 480, 530),
        make_run("ON", 11, 470, 520),
        make_run("ON", 12, 471, 521),
        make_run("OFF", 39, 481, 531),
    ]
    r = mod.paired_analysis(runs)
    assert r["order"] == ["OFF", "ON", "ON", "OFF"]
    assert r["metrics"]["round_init"]["both_pairs_positive"]
    assert r["metrics"]["round_init"]["median_saved_us"] == 27
    assert r["metrics"]["prototype_with_update"]["median_saved_us"] == 10


def test_reject_mismatched_fixture_even_if_p50_positive():
    runs = [
        make_run("OFF", 38, 480, 530), make_run("ON", 11, 470, 520),
        make_run("ON", 12, 471, 521), make_run("OFF", 39, 481, 531),
    ]
    runs[2]["fixture_sha256"]["arg1.bin"] = "different"
    with pytest.raises(ValueError, match="fixture_sha256"):
        mod.paired_analysis(runs)


def test_wrong_order_cannot_be_mislabeled_abba():
    runs = [
        make_run("ON", 10, 470, 520), make_run("OFF", 40, 480, 530),
        make_run("ON", 11, 471, 521), make_run("OFF", 41, 481, 531),
    ]
    with pytest.raises(ValueError, match="ABBA must use"):
        mod.paired_analysis(runs)


def test_negative_result_is_not_reported_as_positive():
    runs = [
        make_run("OFF", 10, 470, 520), make_run("ON", 11, 475, 525),
        make_run("ON", 12, 476, 526), make_run("OFF", 9, 471, 521),
    ]
    result = mod.paired_analysis(runs)
    assert not result["metrics"]["round_init"]["both_pairs_positive"]
    assert result["metrics"]["prototype_core"]["median_saved_us"] < 0
