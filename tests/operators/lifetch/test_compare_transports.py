# SPDX-License-Identifier: Apache-2.0
"""Result comparison is host-only and rejects misleading performance pairs."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("compare_transports", Path(__file__).with_name("compare_transports.py"))
compare_transports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_transports)


def result(transport="ubmem", samples=(10.0, 20.0, 30.0), device=6, width=656):
    contract = dict(
        batch=16,
        queries_per_request=4,
        mtp_speculative_tokens=3,
        history_capacity=131328,
        valid_sequence_length=131072,
        resident=8192,
        token_bytes=width,
        index_heads=64,
        index_head_dim=128,
        topk=2048,
        output_stride=16768,
        requested_windows=8,
        prefetch_limit=2048,
        window_limit=480,
        lru_order="existing_SlotLRU_after_oracle_hot_commit",
        fixture_sha256="f" * 64,
        requested_miss_percent=10.0,
        scope="LI+TopK+Fetch+IndexUpdate+TND_outputs",
        visibility="Q4_same_valid_prefix_no_ABC_growth",
        snapshot="restore_before_every_replay",
    )
    geometry = dict.fromkeys(compare_transports.GEOMETRY_FIELDS, 1)
    geometry.update(available_aic=32, cores_per_batch=2, intermediate_fetches=8, service_groups=[[0, 2, 4, 6]])
    reads = dict(prefetch_hot=30, prefetch_cold=20, unused_hot=5, unused_cold=3, final=10, total=60, unused=8)
    row = dict(
        schema_version=1,
        status="pass",
        transport=transport,
        device=device,
        host="server20",
        soc="Ascend950DT_9582",
        device_visibility="",
        kvip_commit="a" * 40,
        contract=contract,
        geometry=geometry,
        final_workers=2,
        ddr_slot_bytes=86016,
        warmup=4,
        final_cold_per_query=[205] * 4,
        distinct_final_cold_per_request=400,
        timing=dict(samples_us=list(samples), count=len(samples), p50_us=sorted(samples)[len(samples) // 2]),
        reads_per_replay=reads,
        reads_samples=[copy.deepcopy(reads) for _ in samples],
    )
    fingerprint(row)
    return row


def fingerprint(row):
    row["comparison_key"] = hashlib.sha256(json.dumps(row["contract"], sort_keys=True).encode()).hexdigest()


def test_cli_pairs_devices_and_widths_without_creating_outputs(tmp_path, capsys):
    paths = {"ubmem": [], "urma": []}
    for transport in paths:
        for device, width in ((6, 656), (7, 2624)):
            row = result(
                transport, samples=(10, 20, 30) if transport == "ubmem" else (20, 40, 60), device=device, width=width
            )
            path = tmp_path / f"{transport}_{device}_{width}.json"
            path.write_text(json.dumps(row))
            paths[transport].append(str(path))
    before = set(tmp_path.iterdir())
    assert compare_transports.main(["--ubmem", *paths["ubmem"], "--urma", *reversed(paths["urma"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(tmp_path.iterdir()) == before
    assert [(row["device"], row["token_bytes"]) for row in report["comparisons"]] == [(6, 656), (7, 2624)]
    for row in report["comparisons"]:
        assert row["urma_over_ubmem"] == 2
        assert row["reduction_percent"] == 50
        assert row["equal_actual_read_counts"]
        assert row["equal_actual_read_bytes"]


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("contract", "queries_per_request", 2),
        ("contract", "index_heads", 32),
        ("contract", "history_capacity", 131072),
        ("contract", "resident", 16384),
        ("contract", "requested_miss_percent", 20),
        ("contract", "requested_windows", 4),
        ("contract", "fixture_sha256", "b" * 64),
        ("geometry", "available_aic", 24),
        ("geometry", "cores_per_batch", 1),
        (None, "device_visibility", "6,7"),
        (None, "host", "server12"),
        (None, "final_cold_per_query", [206] * 4),
    ],
)
def test_rejects_changed_demand_geometry_or_device_identity(section, field, value):
    local, remote = result(), result("urma")
    target = remote[section] if section else remote
    target[field] = value
    with pytest.raises(ValueError, match=field):
        compare_transports.compare({(6, 656): local}, {(6, 656): remote})


def test_different_speculative_traffic_is_reported_without_claiming_equal_bytes():
    local, remote = result(), result("urma", samples=(5, 10, 15))
    remote["reads_per_replay"].update(prefetch_cold=30, total=70, unused_cold=13, unused=18)
    remote["reads_samples"] = [copy.deepcopy(remote["reads_per_replay"]) for _ in range(3)]
    compare_transports.validate(remote, "urma")
    pair = compare_transports.compare({(6, 656): local}, {(6, 656): remote})["comparisons"][0]
    assert not pair["equal_actual_read_counts"]
    assert not pair["equal_actual_read_bytes"]
    assert pair["ubmem_read_bytes_per_replay"] == 60 * 656
    assert pair["urma_read_bytes_per_replay"] == 70 * 656
    assert pair["reduction_percent"] == -100
    assert "different speculative Fetch traffic" in pair["interpretation"]


def test_allows_prefetch_final_redistribution_and_preserves_timed_evidence():
    local, remote = result(), result("urma", samples=(20, 40, 60))
    reference = dict(
        prefetch_hot=3840,
        prefetch_cold=6485,
        unused_hot=3840,
        unused_cold=0,
        final=75,
        total=10400,
        unused=3840,
    )
    for row in (local, remote):
        row["reads_per_replay"] = copy.deepcopy(reference)
        row["reads_samples"] = [copy.deepcopy(reference) for _ in range(3)]
    remote["reads_samples"][1].update(prefetch_cold=6503, final=57)
    remote["reads_samples"][2].update(prefetch_cold=6500, final=60)
    original_samples = [copy.deepcopy(row["reads_samples"]) for row in (local, remote)]
    for row in (local, remote):
        compare_transports.validate(row, row["transport"])
    report = compare_transports.compare({(6, 656): local}, {(6, 656): remote})
    pair = report["comparisons"][0]
    assert pair["equal_actual_read_bytes"]
    assert not pair["equal_actual_read_counts"]
    assert [row["reads_samples"] for row in (local, remote)] == original_samples
    assert "ubmem_reads_samples" not in pair and "urma_reads_samples" not in pair
    assert pair["urma_read_count_ranges"]["prefetch_cold"] == dict(min=6485, max=6503)
    assert pair["urma_read_count_ranges"]["final"] == dict(min=57, max=75)
    assert pair["urma_read_bytes_per_replay"] == pair["ubmem_read_bytes_per_replay"] == 10400 * 656
    assert "equal total Fetch bytes" in pair["interpretation"]
    assert "different speculative Fetch traffic" not in pair["interpretation"]
    assert report["metric"] == "fixed_snapshot_fusion_latency_us"
    assert report["note"] == "Not model TPOT."


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda row: row.update(status="failed"), "correctness"),
        (lambda row: row.update(schema_version=2), "schema_version"),
        (lambda row: row.update(comparison_key="old"), "comparison_key"),
        (lambda row: row["timing"].update(samples_us=[10, float("nan"), 30]), "samples_us"),
        (lambda row: row["timing"].update(samples_us=[10, 0, 30]), "samples_us"),
        (lambda row: row["timing"].update(p50_us=999), "median"),
        (lambda row: row["timing"].update(count=4), "sample"),
        (lambda row: row["reads_per_replay"].update(total=61), "inconsistent"),
        (lambda row: row["reads_samples"][1].update(final=11, total=61), "changed"),
        (lambda row: row["reads_samples"][1].update(total=61), "inconsistent"),
        (lambda row: row["reads_samples"][1].update(prefetch_hot=31, prefetch_cold=19), "changed"),
        (lambda row: row["reads_samples"][1].update(unused_hot=6, unused=9), "changed"),
        (lambda row: row["reads_samples"][1].update(prefetch_cold=2, final=28), "[Uu]nused"),
        (lambda row: row["reads_samples"][1].update(prefetch_cold=-1, final=31), "prefetch_cold"),
        (lambda row: row["reads_samples"][1].update(final=True), "final"),
    ],
)
def test_rejects_invalid_measurements(mutation, match):
    row = result()
    mutation(row)
    with pytest.raises(ValueError, match=match):
        compare_transports.validate(row, "ubmem")


def test_rejects_missing_or_duplicate_pairs_and_wrong_transport(tmp_path):
    local, remote = result(), result("urma")
    with pytest.raises(ValueError, match="pairs differ"):
        compare_transports.compare({(6, 656): local}, {(7, 656): remote})
    path = tmp_path / "result.json"
    path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match="duplicate"):
        compare_transports.load_results([path, path], "ubmem")
    with pytest.raises(ValueError, match="transport"):
        compare_transports.load_results([path], "urma")
