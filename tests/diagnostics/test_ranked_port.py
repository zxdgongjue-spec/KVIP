import numpy as np
import pytest

from kvip.diagnostics.ranked_port import decode_windows, prefix_oracle


def test_prefix_oracle_includes_retired_recent_and_boundary_ties():
    rng = np.random.default_rng(91)
    scores = rng.integers(14000, 14020, size=(1, 4, 20480), dtype=np.uint16)
    ends = np.array([[20001, 20002, 20003, 20004]])
    bounds = np.array([[18000, 20004]])
    # The seed is not merely the current 512-position ABC allocation.
    scores[:, :, 18000:20004] = 14021
    result = prefix_oracle(scores, ends, bounds)
    assert result["checks"] == 32
    assert result["timing"] == "NOT_SIMULATED"
    assert result["unchanged"] + result["boundary_only"] + result["full_rank"] == result["checks"]


def test_window_rank_timer_excludes_wait_but_open_timer_includes_it():
    raw = np.zeros((8, 2, 16), dtype="<u8")
    raw[0, 0, [1, 4, 11, 13, 14]] = [350, 100, 330, 200, 300]
    result = decode_windows(raw, 8, 2, 10)
    assert result["windows"]["rank_update"]["mean"] == 10
    assert result["windows"]["predecessor_wait"]["mean"] == 10
    assert result["windows"]["open_to_first_post"]["mean"] == 23
    assert result["windows"]["open_to_return"]["mean"] == 25
    raw[0, 0, 8] = 30
    split = decode_windows(raw[:1, :1], 1, 1, 10, split=True)["windows"]
    assert split["post_wait"]["mean"] == 3
    assert split["post_work"]["mean"] == 2
    assert split["combined_wait"]["mean"] == 13
    with pytest.raises(ValueError, match="Missing ranked window markers"):
        decode_windows(raw, 8, 2, 10, split=True)
    raw[0, 0, 11] = 250
    with pytest.raises(ValueError, match="First doorbell"):
        decode_windows(raw, 8, 2, 10)


def test_ranked_cli_uses_eight_cold_windows_without_legacy_flags(tmp_path):
    from kvip.diagnostics.replay import arguments

    args = arguments([
        "--snapshot", str(tmp_path / "input"), "--build-dir", str(tmp_path / "build"),
        "--transport", "urma", "--mode", "fused", "--threshold-policy", "recent",
        "--study", "threshold", "--ranked-port",
        "--upstream-root", str(tmp_path / "native"), "--upstream-revision", "a"*40,
        "--kvaf-server", "127.0.0.1:28999", "--soc", "Ascend950DT_9582",
        "--cold-windows", "8", "--output", str(tmp_path / "result.json"),
    ])
    assert args.cold_windows == 8 and args.ranked_port
    assert args.scan_variants is None and args.pipeline_variants is None


def test_prefix_oracle_exercises_independent_and_unready_queries():
    rng = np.random.default_rng(12)
    scores = rng.integers(14000, 14500, size=(2, 4, 20480), dtype=np.uint16)
    ends = np.full((2, 4), 20480)
    bounds = np.array([[19000, 19500], [19500, 20000]])
    result = prefix_oracle(scores, ends, bounds, skew=True)
    assert result["checks"] == 64
    assert result["delayed_windows"] > 0


def test_report_uses_both_repeats_and_actual_traffic(tmp_path, capsys):
    import json
    from kvip.diagnostics.ranked_port import report

    for name, time in (("forward", 500), ("reverse", 480), ("ranked", 470)):
        for run in (1, 2):
            case = dict(
                status="pass", snapshot_sha256="same",
                geometry=dict(rounds=8, service_lanes=8, token_bytes=656),
                tuning=dict(window_limit=64, prefetch_limit=512),
                threshold_comparison=dict(initial_keys=[1], final_keys=[2]),
                timing=dict(mean_us=time, p50_us=time-1),
                reads_samples=[dict(prefetch=100, unused_prefetch=20, final=50,
                                    transfers=150, cold_demand=100, selected_unique=1000)],
            )
            # A slow second candidate run prevents a false positive gate.
            if name == "ranked" and run == 2:
                case["timing"]["mean_us"] = 490
            if run == 2 and name in ("reverse", "ranked"):
                timeline = {key: dict(count=1, sum_us=time) for key in ("init", "li", "streaming")}
                case["system_profiles"] = {"passes": [dict(metric="none", timeline=timeline)]}
            payload = dict(status="pass", cases={name: case}) if name != "ranked" else case
            (tmp_path / f"g7-{name}-{run}.json").write_text(json.dumps(payload))
    report(tmp_path, 7)
    out = capsys.readouterr().out
    assert "GATE ranked KEEP_BASELINES" in out
    assert "conservative_mean/P50=-2.08/+2.09%" in out
    assert "51.2/12.8/32.0/96.1" in out
    assert "KERNEL_DELTA_US ranked-minus-reverse init/LI/resolve_update=-10.00/-10.00/-10.00" in out

    for name in ("reverse", "ranked", "ranked-46924fb", "ranked-a92b7c8"):
        path = tmp_path / f"g7-{name}-2.json"
        data = json.loads((path if path.exists() else tmp_path / "g7-ranked-2.json").read_text())
        case = data["cases"]["reverse"] if name == "reverse" else data
        case["kvip_commit"] = name.split("-")[-1]
        case["native_diagnostics"] = dict(samples=[dict(final_get=dict(
            status="available", q_workers=8, p50_us=12 if name == "reverse" else 5,
            max_us=18 if name == "reverse" else 11))])
        path.write_text(json.dumps(data))
    report(tmp_path, 7, history=True)
    out = capsys.readouterr().out
    assert "HISTORY_GRAPH_US ranked-46924fb" in out
    assert "FINAL_GET_SAVED_US ranked vs_reverse Q_worker_P50/max=+7.00/+7.00" in out
    assert "LI_added=-10.00 tail_saved=+10.00" in out


def test_final_get_joins_phases_per_worker_before_taking_median():
    from kvip.diagnostics.ranked_port import final_get_times

    batch, workers = 4, 2
    raw = np.zeros(batch * 256, dtype=np.uint64)
    rows = raw[batch*128:].reshape(batch, 128)[:, :96].reshape(batch, 4, 24)
    for b in range(batch):
        for q in range(4):
            start = 1000 + q
            rows[b, q, [3, 4, 7, 8, 20]] = [start, start+10+q, start+50, start+53+b%2, 3]
    result = final_get_times(raw, batch, workers, 2)
    assert result["q_workers"] == 8
    assert result["p50_us"] == 7.5
    assert result["max_us"] == 8.5
    rows[2, 0, 4] += 1
    with pytest.raises(ValueError, match="Replicated"):
        final_get_times(raw, batch, workers, 2)
    rows[:, :, 20] = 4
    rows[:, :, 21] = 32
    assert final_get_times(raw, batch, workers, 2)["max_us"] == 16
    rows[2, 0, 20] = 0
    assert final_get_times(raw, batch, workers, 2)["status"] == "unavailable"


def test_kernel_costs_falls_back_to_actual_profiler_export(tmp_path):
    from kvip.diagnostics.ranked_port import kernel_costs

    p = tmp_path / "kernel_details.csv"
    p.write_text("Name,Start Time(us),Duration(us)\n"
                 "v9_init_kernel_1,1000.01,12\n"
                 "v9_li_kernel_0,1012.4,355\n"
                 "v9_streaming_resolve_update_5,1369.05,215\n")
    report = dict(metric="none", directory=str(tmp_path), timeline={"li": None},
                  artifacts={"csv_files": [{"path": p.name}]})
    assert kernel_costs({"system_profiles": {"passes": [report]}}) == [12, 355, 215]
