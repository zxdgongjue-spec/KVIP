# SPDX-License-Identifier: Apache-2.0
"""P0-A2.3 regression: Native baseline drift can invert apparent V9 gains."""
import importlib.util
from pathlib import Path
import sys

import pytest

script = Path(__file__).with_name("analyze_active_k_abba.py")
spec = importlib.util.spec_from_file_location("active_k_attribution", script)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def make_run(label, native, core, full, init):
    return {
        "switch": label,
        "fixture_sha256": {"data": "same"},
        "native_unique_misses": [820, 824, 828, 832],
        "metrics_p50_us": {
            "native_chain": native, "prototype_core": core,
            "prototype_with_update": full, "round_init": init,
            "aiv_cache_update": 90,
        },
    }


def test_positive_raw_can_be_negative_native_normalized():
    runs = [
        make_run("OFF", 400, 500, 560, 30),
        make_run("ON", 380, 492, 552, 33),
        make_run("ON", 381, 491, 551, 34),
        make_run("OFF", 401, 501, 561, 31),
    ]
    d = mod.analyze({"runs": runs})
    assert d["aggregate"]["prototype_core"]["raw_saved_us"] == 9
    assert d["aggregate"]["prototype_core"]["raw_saved_pct"] > 0
    assert d["aggregate"]["prototype_core"]["native_normalized_saved_pct"] < 0
    assert d["aggregate"]["prototype_with_update"]["excess_latency_saved_us"] < 0
    assert d["aggregate"]["round_init"]["raw_saved_us"] < 0


def test_wrong_order_rejected():
    runs = [
        make_run("ON", 390, 490, 550, 30),
        make_run("OFF", 400, 500, 560, 32),
        make_run("ON", 391, 491, 551, 30),
        make_run("OFF", 401, 501, 561, 32),
    ]
    with pytest.raises(ValueError, match="ABBA"):
        mod.analyze({"runs": runs})


def test_fixture_and_native_miss_mismatch_rejected():
    runs = [
        make_run("OFF", 400, 500, 560, 30),
        make_run("ON", 380, 490, 550, 32),
        make_run("ON", 381, 491, 551, 34),
        make_run("OFF", 401, 501, 561, 31),
    ]
    runs[2]["fixture_sha256"] = {"data": "different"}
    with pytest.raises(ValueError, match="fixture"):
        mod.analyze({"runs": runs})
    runs[2]["fixture_sha256"] = {"data": "same"}
    runs[2]["native_unique_misses"][0] = 999
    with pytest.raises(ValueError, match="Native miss"):
        mod.analyze({"runs": runs})
