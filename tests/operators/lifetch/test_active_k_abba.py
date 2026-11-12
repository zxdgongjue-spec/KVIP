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


def test_k_profiles_have_75_37p5_and_zero_percent_reduction():
    assert mod.mapping_clear_geometry("short")["mapping_clear_reduction_pct_theoretical"] == 75
    assert mod.mapping_clear_geometry("mixed")["mapping_clear_reduction_pct_theoretical"] == 37.5
    assert mod.mapping_clear_geometry("full")["mapping_clear_reduction_pct_theoretical"] == 0
    assert mod.mapping_clear_geometry("short")["mapping_copy_chunks_on"] == 32
    assert mod.mapping_clear_geometry("mixed")["mapping_copy_chunks_on"] == 80
    assert mod.mapping_clear_geometry("full")["mapping_copy_chunks_on"] == 128


def test_k_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown Active-K profile"):
        mod.mapping_clear_geometry("missing")


def test_k_profile_appears_in_subprocess_command():
    from types import SimpleNamespace
    args = SimpleNamespace(
        device=0, kvaf_server="127.0.0.1:28999",
        kvaf_root=Path("/opt/KVAF"), cann="/opt/cann",
        soc="Ascend950DT_9582", warmup=4, iterations=20,
        k_profile="short",
    )
    command = mod.base_command(args, Path("/tmp/run"), enabled=True)
    assert command[command.index("--actual-key-lengths") + 1] == "32768,32768,32768,32768"
    assert "--active-k-map-clear" in command
    args.k_profile = "full"
    command = mod.base_command(args, Path("/tmp/run"), enabled=False)
    assert command[command.index("--actual-key-lengths") + 1] == "131072,131072,131072,131072"
    assert "--active-k-map-clear" not in command
