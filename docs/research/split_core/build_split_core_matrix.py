"""Reproduce analytic scenarios; no timing or hardware performance predictions."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = "81b1472809b7a5f3469e26467feee0ca1c61d71a"
import subprocess
import types
geometry = types.ModuleType("archived_geometry")
exec(compile(subprocess.check_output(["git","-C",str(ROOT),"show",BASE+":python/kvip/operators/lifetch/geometry.py"]),"pinned geometry","exec"),geometry.__dict__)


def scenarios():
    rows = []
    for cores in (32, 28):
        for batch in (1, 2, 4, 8, 16):
            g = geometry.li_geometry(batch, "auto", 32, cores)
            a = g["cores_per_batch"]
            active = g["rotating_aic"]
            epochs = g["intermediate_fetches"]
            final = geometry.final_aivs(batch, "auto", cores)
            limit, window = geometry.prefetch_budget(512, None, 32)
            for percent in (30, 50, 100):
                # Keep exact requested effective length, use padded K lengths for costs.
                valid = (131072 * percent + 99) // 100
                history = (valid + 255) // 256 * 256
                tiles = history // 128
                resident = 16384
                for miss in (3, 5, 10, 15):
                    m = round(2048 * miss / 100)
                    lower, upper = m, min(4 * m, valid - resident)
                    pref_cap = min(limit, epochs * window)
                    row = dict(
                        source_commit=BASE, evidence="analytic_not_measured",
                        aic_budget=cores, aiv_budget=2 * cores, batch=batch,
                        history_percent=percent, valid_history=valid,
                        shape_history=history, padded_key_lengths=history,
                        query_end=valid, padding_tokens=history - valid,
                        resident=resident, resident_hot_valid="all",
                        extra_abc_hits="none", miss_percent_target=miss,
                        miss_per_query=m, miss_percent_actual=100 * m / 2048,
                        split_q=a - 1, queries_per_aic=4 // a,
                        queries_per_aiv=2 // a, li_aic=active, li_aiv=2 * active,
                        li_budget_fraction=active / cores,
                        li_resident_waves=1,
                        history_tiles_per_aic=tiles,
                        logical_aic_tiles=active * tiles,
                        s2_shards_max_simultaneous=cores // active,
                        # Per tile: (M, N, K), G sensitivity in separate columns.
                        mm_m_g32=4 // a * 32, mm_m_g64=4 // a * 64,
                        mm_n=128, mm_k=128,
                        dot_ops_g32=2 * batch * 4 * 32 * history * 128,
                        dot_ops_g64=2 * batch * 4 * 64 * history * 128,
                        key_logical_read_bytes=batch * a * history * 128,
                        key_q4_logical_read_bytes=batch * history * 128,
                        score_write_bytes=batch * 4 * history * 2,
                        final_score_logical_read_bytes=batch * 4 * valid * 2,
                        qk_fix_logical_bytes_g32=batch * 4 * 32 * history * 4,
                        cold_hits_write_bytes=batch * 4 * (history - resident) // 4,
                        resident_fraction=resident / valid,
                        final_workers_per_request=final,
                        resolve_aiv_blocks=batch * final,
                        apply_aiv_blocks=4 * batch,
                        apply_min_waves=(4 * batch + 2 * cores - 1) // (2 * cores),
                        topk_trunk_tokens=16384 if a == 2 else 8192,
                        final_topk_trunks_per_query=(valid + (16384 if a == 2 else 8192) - 1) // (16384 if a == 2 else 8192),
                        requested_windows=32, effective_windows=epochs,
                        service_aivs_per_window=g["service_aivs_per_window"],
                        post_groups=(batch + g["service_aivs_per_window"] - 1) // g["service_aivs_per_window"],
                        prefetch_total_limit=limit, prefetch_window_limit=window,
                        prefetch_nominal_cap_per_request=pref_cap,
                        unique_miss_lower_per_request=lower,
                        unique_miss_upper_per_request=upper,
                        required_miss_bytes_lower=batch * lower * 656,
                        required_miss_bytes_upper=batch * upper * 656,
                        repair_count_lower_at_unique_upper=max(0, upper - pref_cap),
                        policy=("test_S2_shards_and_effective_windows" if batch <= 4 else
                                "compare_Q4_Q2_and_two_shards" if batch == 8 else
                                "Q2_baseline_reduce_vector_tail" if cores == 32 else
                                "Q4_fallback_test_persistent_history_tasks"),
                    )
                    rows.append(row)
    return rows


def main():
    rows = scenarios()
    with (OUT / "split_core_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        schema_version=1, rows=len(rows), source_commit=BASE,
        measured_performance=False,
        defaults=dict(query_count=4, head_dim=128, topk=2048, resident=16384,
                      token_bytes=656, resident_hot_valid="all", extra_abc_hits="none",
                      index_heads_primary=32,
                      index_heads_sensitivity=64, requested_windows=32,
                      geometry="auto", key_lengths="padded shape_history",
                      query_ends="valid_history", miss_count="standard fixture round"),
        caveats=[
            "Logical read/FIX volumes are not physical HBM traffic or time.",
            "Required miss bytes exclude headers, speculative waste, retries and cache apply.",
            "Occupancy fraction is a grid resource upper bound, not measured engine utilization.",
            "Prefetch cap is a loose admission bound, not useful completed prefetch count.",
            "New S2 sharding is a design proposal; existing live protocols require redesign.",
        ],
    )
    (OUT / "split_core_model.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "measured": False}))


if __name__ == "__main__":
    main()
