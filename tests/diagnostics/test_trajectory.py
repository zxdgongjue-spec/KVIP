# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from kvip.diagnostics.trajectory import inspect_series, main, materialize_snapshot


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def series(tmp_path):
    for dp in (0, 1):
        requests = [
            dict(request_id=f"dp{dp}-r{r}", owner=r + 1, query_begin=r * 4, query_stop=(r + 1) * 4) for r in range(16)
        ]
        groups = [dict(group=g, layer_names=[f"model.layers.{g}.self_attn.attn"]) for g in (0, 1)]
        rounds = []
        for number in (1, 2):
            entries = []
            for group in groups:
                relative = f"round{number:04d}/group{group['group']:03d}/manifest.json"
                snapshot = dict(
                    schema_version=2,
                    status="complete",
                    oracle={"passed": True},
                    geometry={"batch": 16, "token_count": 64},
                    metadata=dict(
                        dp_rank=dp,
                        **group,
                        context=dict(round=number + 10, global_decode_confirmed=True),
                        requests=requests,
                    ),
                )
                _write(tmp_path / f"dp{dp}" / relative, snapshot)
                entries.append(dict(group=group["group"], manifest=relative))
            rounds.append(dict(round_index=number, scheduler_round=number + 10, groups=entries))
        summary = dict(
            schema_version=1,
            kind="kvaf_decode_snapshot_series",
            status="complete",
            scope="target",
            dp_rank=dp,
            dp_ranks=[0, 1],
            dp_size=2,
            expected_rounds=2,
            completed_rounds=2,
            batch=16,
            queries=4,
            expected_target_layer_count=2,
            target_layer_names=[name for group in groups for name in group["layer_names"]],
            groups=groups,
            request_owner_signature=[dict(request_id=p["request_id"], owner=p["owner"]) for p in requests],
            rounds=rounds,
        )
        _write(tmp_path / f"dp{dp}/series.json", summary)
    return tmp_path


def test_two_dp_all_layer_complete_trace_summary(series, capsys):
    report = inspect_series(series, expected_dp=[0, 1], expected_rounds=2, expected_layers=2)
    assert [item["snapshots"] for item in report] == [4, 4]
    main(["--root", str(series), "--expect-dp", "0", "1", "--expect-rounds", "2", "--expect-layers", "2"])
    assert capsys.readouterr().out.count("TRACE_DP_PASS") == 2


def _token_series(root):
    for series_path in root.glob("dp*/series.json"):
        data = json.loads(series_path.read_text())
        data["content"] = "token_ids"
        _write(series_path, data)
        for entry in data["rounds"]:
            for group in entry["groups"]:
                path = series_path.parent / group["manifest"]
                manifest = json.loads(path.read_text())
                manifest.pop("oracle")
                manifest.update(
                    schema_version=3,
                    kind="kvaf_decode_token_trace",
                    content="token_ids",
                    validation={"passed": True, "kind": "token_id_integrity"},
                    archive={"file": "token_ids.npz", "bytes": 3},
                )
                _write(path, manifest)
                (path.parent / "token_ids.npz").write_bytes(b"ids")


def test_token_trace_coverage_has_separate_integrity_contract(series, capsys):
    _token_series(series)
    reports = inspect_series(series, expected_content="token_ids")
    assert all(report["content"] == "token_ids" for report in reports)
    main(["--root", str(series), "--expect-content", "token_ids"])
    assert "real_payload=not_collected" in capsys.readouterr().out
    with pytest.raises(ValueError, match="content"):
        inspect_series(series, expected_content="full")
    path = series / "dp1/round0002/group001/token_ids.npz"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="snapshot contract"):
        inspect_series(series)


def test_token_trace_cannot_claim_full_snapshot_extraction(series, tmp_path):
    _token_series(series)
    with pytest.raises(SystemExit, match="token_replay"):
        main(
            [
                "--root",
                str(series),
                "--dp-rank",
                "0",
                "--round",
                "1",
                "--group",
                "0",
                "--extract",
                str(tmp_path / "full"),
            ]
        )


def _score_series(root):
    _token_series(root)
    for path in root.glob("dp*/series.json"):
        data = json.loads(path.read_text())
        data.update(
            content="scores", seed_rounds=1, consecutive_required=True, real_input_rounds=[2], real_input_groups=[1]
        )
        _write(path, data)
        for entry in data["rounds"]:
            for group in entry["groups"]:
                manifest_path = path.parent / group["manifest"]
                manifest = json.loads(manifest_path.read_text())
                real = entry["round_index"] == 2 and group["group"] == 1
                manifest.update(
                    schema_version=4,
                    kind="kvaf_decode_score_trace",
                    content="scores",
                    real_inputs_saved=real,
                    validation={"passed": True, "kind": "score_trace_integrity"},
                )
                manifest["metadata"]["capture_real_inputs"] = real
                _write(manifest_path, manifest)


def test_scores_require_seed_and_sampled_real_inputs(series, capsys):
    _score_series(series)
    reports = inspect_series(series, expected_content="scores")
    assert all(r["analysis_rounds"] == 1 and r["real_input_snapshots"] == 1 for r in reports)
    main(["--root", str(series), "--expect-content", "scores"])
    assert "seed_rounds=1 analysis_rounds=1 real_input_snapshots=1" in capsys.readouterr().out
    path = series / "dp0/round0002/group001/manifest.json"
    data = json.loads(path.read_text())
    data["real_inputs_saved"] = False
    _write(path, data)
    with pytest.raises(ValueError, match="contract differs"):
        inspect_series(series)


def test_scores_reject_missing_consecutive_predecessor(series):
    _score_series(series)
    path = series / "dp0/series.json"
    data = json.loads(path.read_text())
    data["rounds"][1]["scheduler_round"] += 1
    _write(path, data)
    with pytest.raises(ValueError, match="consecutive captured predecessor"):
        inspect_series(series)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(status="capturing"),
        lambda data: data["rounds"][0]["groups"].pop(),
        lambda data: data["rounds"][0].update(round_index=2),
        lambda data: data["groups"][1].update(layer_names=data["groups"][0]["layer_names"]),
    ],
)
def test_partial_or_duplicate_coverage_is_not_success(series, change):
    path = series / "dp1/series.json"
    data = json.loads(path.read_text())
    change(data)
    _write(path, data)
    with pytest.raises(ValueError):
        inspect_series(series)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["metadata"]["context"].update(global_decode_confirmed=False),
        lambda data: data["metadata"]["requests"][0].update(query_stop=128),
        lambda data: data["metadata"]["requests"][0].update(request_id="replacement-request"),
        lambda data: data["oracle"].update(passed=False),
    ],
)
def test_each_snapshot_preserves_decode_batch_cohort_and_oracle(series, change):
    path = series / "dp0/round0002/group001/manifest.json"
    data = json.loads(path.read_text())
    change(data)
    _write(path, data)
    with pytest.raises(ValueError, match="snapshot contract"):
        inspect_series(series)


def test_missing_rank_and_unexpected_layer_count_fail(series):
    with pytest.raises(ValueError, match="expected 78"):
        inspect_series(series, expected_layers=78)
    (series / "dp1/series.json").unlink()
    with pytest.raises(ValueError, match="Missing/unexpected DP"):
        inspect_series(series)


def test_different_coordinated_rounds_fail(series):
    path = series / "dp1/series.json"
    summary = json.loads(path.read_text())
    summary["rounds"][1]["scheduler_round"] += 1
    for group in summary["rounds"][1]["groups"]:
        snapshot_path = path.parent / group["manifest"]
        snapshot = json.loads(snapshot_path.read_text())
        snapshot["metadata"]["context"]["round"] += 1
        _write(snapshot_path, snapshot)
    _write(path, summary)
    with pytest.raises(ValueError, match="same coordinated"):
        inspect_series(series)


def test_extract_series_snapshot_preserves_raw_tensors_and_ddr(tmp_path):
    import numpy as np

    from kvip.diagnostics.archive import TensorArchive, load_snapshot
    from kvip.diagnostics.ddr import export_pages, validate_pages
    from kvip.diagnostics.storage import PackedObjectStore
    from test_ddr import CpuTorch, Runtime

    runtime = Runtime()
    runtime.seed([1, 4, 9])
    source, destination = tmp_path / "dp0/round0001/group000", tmp_path / "standalone"
    source.mkdir(parents=True)
    manifest = dict(schema_version=2, status="complete", tensors={})
    raw = np.asarray([0, 0x7FC0, 0xFFFF], dtype=np.uint16)
    with PackedObjectStore(tmp_path / "dp0/objects", create=True) as store:
        archive = TensorArchive(source, manifest, object_store=store)
        archive.save("pre.query", raw, dtype="bfloat16")
        manifest["ddr"] = export_pages(
            runtime,
            [1, 4, 9],
            source / "ddr",
            page_mode="thp",
            generation=1,
            torch_module=CpuTorch(),
            object_store=store,
        )
        store.flush()
        _write(source / "manifest.json", manifest)
        materialize_snapshot(source, destination)
    restored = load_snapshot(destination)
    assert restored.manifest["schema_version"] == 1
    assert "object_store" not in restored.manifest
    np.testing.assert_array_equal(restored.array("pre.query"), raw)
    assert restored.manifest["tensors"]["pre.query"]["dtype"] == "bfloat16"
    assert validate_pages(destination / "ddr", restored.manifest["ddr"])["record_count"] == 3
    assert (destination / "ddr/payload.bin").read_bytes() == b"".join(runtime.pages[k].tobytes() for k in (1, 4, 9))
    assert not (destination / "INCOMPLETE").exists()
    with pytest.raises(FileExistsError):
        materialize_snapshot(source, destination)
