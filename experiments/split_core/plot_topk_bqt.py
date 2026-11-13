"""Plot verified Model results and same-run raw worker execution envelopes."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(result, output):
    rows = result["rows"]
    assert len(rows) == 9 and result["complete_worker_csvs"] == 548
    colors = {1: "#4477aa", 4: "#ee7733", 8: "#228833"}
    batches = (1, 8, 16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(3)
    for i, t in enumerate((1, 4, 8)):
        points = [next(r for r in rows if r["batch"] == b and r["shards"] == t) for b in batches]
        xx = x + (i - 1) * 0.24
        a = axes[0].bar(
            xx, [r["raw_kernel_envelope_sum_ticks"] / 1000 for r in points], 0.23, color=colors[t], label=f"T{t}"
        )
        axes[0].bar_label(a, fmt="%.1f", fontsize=8, padding=2)
        a = axes[1].bar(
            xx, [r["exported_calls_per_query"] / 1000 for r in points], 0.23, color=colors[t], label=f"T{t}"
        )
        axes[1].bar_label(a, fmt="%.1f", fontsize=8, padding=2)
    for ax in axes:
        ax.set_xticks(x, [f"BS={b}" for b in batches])
        ax.set_xlabel("Q=4, H=131072, m=205/query, AIV budget=56")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.margins(y=0.17)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Sum of kernel envelopes (thousands of ticks)")
    axes[0].set_title("Local + merge kernel envelopes; lower is better")
    axes[1].set_ylabel("Exported instruction calls/query (thousands)")
    axes[1].set_title("Local + merge work; complete active-worker coverage")
    fig.suptitle("Final TopK component: batch/query parallelism changes the useful T")
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"topk-bqt-comparison.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), constrained_layout=True)
    for r in rows:
        ax = axes[batches.index(r["batch"]), (1, 4, 8).index(r["shards"])]
        for phase in r["phases"]:
            color = "#4477aa" if phase["phase"] == "local" else "#ee7733"
            for core in phase["cores"]:
                ax.broken_barh(
                    [(core["start"] / 1000, (core["tick"] - core["start"]) / 1000)],
                    (core["block"] - 0.4, 0.8),
                    facecolors=color,
                )
        base = next(p for p in rows if p["batch"] == r["batch"] and p["shards"] == 1)
        row_max = max(p["total_model_ticks"] for p in rows if p["batch"] == r["batch"])
        ax.axvline(base["total_model_ticks"] / 1000, linestyle="--", color="#666666", linewidth=0.7)
        ax.set(xlim=(0, row_max / 1000 * 1.03), ylim=(-1, 56), title=f"BS={r['batch']}, T={r['shards']}")
        ax.grid(axis="x", alpha=0.15)
        if r["shards"] == 1:
            ax.set_ylabel("AIV block ID")
        if r["batch"] == 16:
            ax.set_xlabel("Same-run raw Model clock (thousands of ticks)")
    fig.suptitle(
        "Raw kernel start/done envelopes: blue local, orange merge; dashed T1 total\n"
        "Includes initialization; each panel is a separate Model run, not instruction PIPE detail",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"topk-bqt-envelopes.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.output)
