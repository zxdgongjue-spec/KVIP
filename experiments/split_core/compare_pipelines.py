"""Validate archived whole-Model cohorts and regenerate their CSV/optional figure.

No device-time conversion or attribution of cross-driver differences is performed.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

DOC = Path(__file__).resolve().parents[2] / "docs/research/split_core"
BASE = "81b1472809b7a5f3469e26467feee0ca1c61d71a"
STREAMING = "13d2898892652382ea36a0652030874efc31097d"


def load(name):
    return json.loads((DOC / "results" / name).read_text())


def validate():
    proof = load("pipeline-comparison-proof.json")
    captures = load("fusion3-captures.json")["captures"]
    comparison = load("fusion3-direct-comparison.json")
    records = load("model-records.json")["records"]
    assert proof["historical_base"] == comparison["main_sha"] == BASE
    assert (
        hashlib.sha256((DOC / "results/fusion3-direct-comparison.json").read_bytes()).hexdigest()
        == proof["fusion3_comparison_sha256"]
    )
    assert len(proof["shared_fixture_files_sha256"]) == 20
    assert len(proof["shared_key_outputs_sha256"]) == 6
    assert proof["effective_batch"] == 1 and proof["source_manifest_declared_batch"] == 16
    assert not proof["full_loaded_library_set_verified"]
    assert not proof["physical_npu"] and not proof["native_transport"]
    assert comparison["single_pair"] and comparison["active_plans_equal"]
    assert all(comparison["output_bytes_equal"].values())
    assert len(captures) == 2 and {c["variant"] for c in captures} == {"baseline", "candidate"}
    assert {c["raw_capture_sha256"] for c in captures} == set(comparison["capture_sha256"])
    assert {c["name"] for c in captures} == {"direct-b1-baseline-r1", "direct-b1-candidate-r1"}
    rows = []
    for c in captures:
        assert c["clock_equivalence_unverified"] is True
        assert not c["physical_npu"] and not c["native_transport"]
        runtime = {Path(r["path"]).name: r["sha256"] for r in c["loaded_libraries"]}
        runtime.update({Path(k).name: v for k, v in c["hardware_configs_sha256"].items()})
        assert all(runtime[n] == h for n, h in proof["common_recorded_runtime_sha256"].items())
        assert c["host_memory_limit_bytes"] == int(proof["host_limits"]["memory.max"])
        assert c["host_swap_limit_bytes"] == int(proof["host_limits"]["memory.swap.max"])
        assert c["host_cpu_max"] == proof["host_limits"]["cpu.max"]
        assert c["valid_for_standard_performance"] is False
        assert c["full_owner_oracle_passed"] and c["exit_code"] == 0
        g = c["geometry"]
        assert (g["history"], g["resident"], g["batch"], g["heads"]) == (131072, 16384, 1, 32)
        assert (g["service_rounds"], g["miss_per_query"], g["unique_misses_per_request"]) == (0, 205, 410)
        assert c["fixture_manifest_sha256"] == proof["fixture_manifest_sha256"]
        assert c["raw_output_sha256"] == proof["shared_key_outputs_sha256"]
        data = (DOC / "results" / c["exported_log"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == c["model_log_sha256"]
        assert re.findall(rb"Total tick:\s*(\d+)", data) == [str(c["total_model_ticks"]).encode()]
        assert b"Model stopped successfully." in data
        fused = c["variant"] == "candidate"
        assert c["total_model_ticks"] == comparison["candidate_ticks" if fused else "baseline_ticks"]
        rows.append(
            dict(
                cohort="fusion3_direct",
                variant="Fusion3" if fused else "serial6",
                tested_base=BASE,
                whole_model_ticks=c["total_model_ticks"],
                control_ticks=comparison["baseline_ticks"],
                driver_sha256=c["driver_sha256"],
                evidence=c["exported_log"],
                status="PASSED_MODEL",
                producer_shards="",
                topk_shards="",
                aic_budget=32,
                aiv_budget=64,
            )
        )
    split = {r["name"]: r for r in records if r["id"] in proof["verified_split_record_ids"]}
    assert len(split) == 7
    control = split["original"]
    for name in [
        "original",
        "q4-score-s1",
        "q4-score-p1-t8",
        "history-s32",
        "history-s28",
        "history-p32-t8",
        "history-p28-t8",
    ]:
        r = split[name]
        assert r["tested_base"] == BASE and r["driver_sha256"] == control["driver_sha256"]
        assert r["case"] == "standard" and r["batch"] == 1 and r["index_heads"] == 32
        assert r["service_rounds"] == 0 and r["per_token_parity_with_original"]
        assert all(r["input_files_sha256"][n] == h for n, h in proof["shared_fixture_files_sha256"].items())
        assert all(r["raw_output_sha256"][n] == h for n, h in proof["shared_key_outputs_sha256"].items())
        data = (DOC / "results" / r["exported_log"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r["exported_log_sha256"]
        assert re.findall(rb"Total tick:\s*(\d+)", data) == [str(r["total_model_ticks"]).encode()]
        assert b"Model stopped successfully." in data
        rows.append(
            dict(
                cohort="history_pool_same_binary",
                variant=name,
                tested_base=BASE,
                whole_model_ticks=r["total_model_ticks"],
                control_ticks=control["total_model_ticks"],
                driver_sha256=r["driver_sha256"],
                evidence=r["exported_log"],
                status="PASSED_MODEL",
                producer_shards=r["shards"] or "",
                topk_shards=r["topk_shards"] or "",
                aic_budget=r["aic_budget"],
                aiv_budget=r["aiv_budget"],
            )
        )
    for row in rows:
        row["reduction_vs_own_control_pct"] = round(100 * (1 - row["whole_model_ticks"] / row["control_ticks"]), 6)
    rows.append(
        dict(
            cohort="current_main",
            variant="streaming",
            tested_base=STREAMING,
            whole_model_ticks="",
            control_ticks="",
            driver_sha256="",
            evidence="",
            status="NOT_RUN",
            producer_shards="",
            topk_shards="",
            aic_budget="",
            aiv_budget="",
            reduction_vs_own_control_pct="",
        )
    )
    return rows


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def plot(rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2.1]})
    for ax, cohort, title in zip(
        axes,
        ["fusion3_direct", "history_pool_same_binary"],
        ["Fusion3 direct pair (different binaries)", "History split controls (one binary)"],
    ):
        items = [r for r in rows if r["cohort"] == cohort]
        labels = [r["variant"].replace("history-", "").replace("q4-score-", "Q4-") for r in items]
        colors = ["#64748b" if i == 0 else "#197b86" for i in range(len(items))]
        bars = ax.barh(labels[::-1], [r["whole_model_ticks"] / 1e6 for r in items[::-1]], color=colors[::-1])
        for bar, row in zip(bars, items[::-1]):
            ax.text(
                bar.get_width() + 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{row['whole_model_ticks']:,}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1.62)
        ax.set_xlabel("Whole CA Model Total tick (millions)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Historical 81b1472 + mock overlay | B1 / Q4 / H128K / m205", fontsize=12)
    fig.text(
        0.5,
        0.025,
        "Service0, synchronous GM-copy mock, n=1; no physical-time conversion. Current main streaming: NOT_RUN.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(DOC / "pipeline-comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate CSV from archived evidence")
    parser.add_argument("--plot", action="store_true", help="Regenerate optional static PNG (matplotlib required)")
    args = parser.parse_args()
    rows = validate()
    output = DOC / "pipeline-comparison.csv"
    expected = csv_text(rows)
    if args.write:
        output.write_text(expected)
    else:
        assert output.read_text() == expected
    if args.plot:
        plot(rows)
    print(
        json.dumps(
            dict(
                status="passed",
                measured_rows=9,
                not_run_rows=1,
                scope="whole-Model archive consistency; not raw tensor replay",
            )
        )
    )


if __name__ == "__main__":
    main()
