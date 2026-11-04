"""Controlled eligible-miss distribution using real score inputs, not injected flags.

Both queries on each AIV select the same cold token lanes; report quantized
actual fractions (e.g. requested20% becomes50/256=19.53125%).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fixture import prepare


def build(
    root,
    fraction,
    layout,
    history=8192,
    resident=4096,
    mean_per_aiv=None,
    tile_pattern="uniform",
    independent_queries=False,
):
    if not 0 <= fraction <= 1 or layout not in ("dispersed", "clustered"):
        raise ValueError("invalid fraction/layout")
    prepare(root, history, resident, 4)
    order = np.load(root / "physical_to_logical.npy")
    scale = np.fromfile(root / "arg4.bin", np.float32)
    weights = np.fromfile(root / "arg2.bin", np.float32).reshape(4, 64)
    query_scale = np.fromfile(root / "arg3.bin", np.float32).reshape(4, 64)
    tile_count = (history - resident) // 128
    if mean_per_aiv is None:
        selected_by_tile = np.full(tile_count, int(np.floor(128 * fraction)), dtype=int)
    else:
        if not 0 <= mean_per_aiv <= 256 or tile_pattern not in ("uniform", "bursty"):
            raise ValueError("invalid mean-per-AIV/tile pattern")
        # Both query masks are controlled together. Fractional means use
        # temporally spaced pairs, never silently claim one hit per tile.
        cumulative = np.floor(np.arange(tile_count + 1) * mean_per_aiv / 2).astype(int)
        selected_by_tile = np.diff(cumulative)
        if tile_pattern == "bursty":
            total = int(selected_by_tile.sum())
            selected_by_tile = np.array([min(128, max(0, total - t * 128)) for t in range(tile_count)])
    for tile, start in enumerate(range(resident, history, 128)):
        selected = selected_by_tile[tile]
        lanes = (
            np.arange(selected)
            if layout == "clustered"
            else np.floor(np.arange(selected) * 128 / max(selected, 1)).astype(int)
        )
        scale[start : start + 128] = 0
        scale[start + lanes] = 8
    scale.tofile(root / "arg4.bin")
    encoded = np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128)
    values = np.zeros_like(encoded, dtype=np.float32)
    for byte, value in ((0x38, 1), (0x40, 2), (0x44, 3)):
        values[encoded == byte] = value
    columns_by_query = [(np.arange(64) + q * 16) % 128 for q in range(4)]
    expected_counts = [selected_by_tile] * 4
    if independent_queries:
        if mean_per_aiv is None:
            raise ValueError("independent queries require mean_per_aiv")
        # Disjoint K feature bands allow one query to win without forcing its
        # partner to win. Keep all 64 heads active; only this fixture's inputs
        # change. Compare stage4/stage5 with the SAME fixture, never paired vs
        # independent inputs as a causal performance comparison.
        columns_by_query = [np.arange(64) % 32 + q * 32 for q in range(4)]
        cumulative = np.floor(np.arange(tile_count + 1) * mean_per_aiv).astype(int)
        total_by_tile = np.diff(cumulative)
        if tile_pattern == "bursty":
            total = int(total_by_tile.sum())
            total_by_tile = np.array([min(256, max(0, total - t * 256)) for t in range(tile_count)])
        first = total_by_tile // 2 + (total_by_tile % 2) * (np.arange(tile_count) % 2 == 0)
        expected_counts = [first, total_by_tile - first, first, total_by_tile - first]
        query = np.zeros((4, 64, 128), np.uint8)
        values[resident:] = 0
        scale[resident:] = 8
        for q, columns in enumerate(columns_by_query):
            query[q, np.arange(64), columns] = 0x38
            for tile, selected in enumerate(expected_counts[q]):
                lanes = (
                    np.arange(selected)
                    if layout == "clustered"
                    else np.floor(np.arange(selected) * 128 / max(selected, 1)).astype(int)
                )
                values[np.ix_(resident + tile * 128 + lanes, np.unique(columns))] = 3
        query.tofile(root / "arg0.bin")
        np.array([0, 0x38, 0x40, 0x44], np.uint8)[values.astype(np.uint8)].tofile(root / "arg1.bin")
        scale.tofile(root / "arg4.bin")
    logical = []
    actual = []
    for q in range(4):
        columns = columns_by_query[q]
        score = (values[:, columns] * (weights[q] * query_scale[q])).sum(-1) * scale
        bits = ((score.view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
        row = np.empty_like(bits)
        row[order] = bits
        logical.append(row)
        threshold = np.partition(bits[:resident], -2048)[-2048]
        counts = (bits[resident:] >= threshold).reshape(-1, 128).sum(-1)
        assert np.array_equal(counts, expected_counts[q]), "constructed scores missed requested lane count"
        actual.append(counts)
    np.save(root / "logical_scores.npy", np.stack(logical))
    config = json.loads((root / "fixture.json").read_text())
    config["stage5_gate"] = {
        "requested_fraction": fraction,
        "layout": layout,
        "actual_fraction": float(np.mean(actual) / 128),
        "scores_per_aiv_tile": 256,
        "requested_mean_per_aiv_tile": mean_per_aiv,
        "actual_mean_per_aiv_tile": float(np.mean(actual) * 2),
        "temporal_pattern": tile_pattern,
        "query_mask_correlation": (
            "independent_feature_bands; odd hits rotate query ownership"
            if independent_queries
            else "paired_query_masks; fractional means alternate zero and two hits"
        ),
        "eligible_per_aiv_tile": [(actual[0] + actual[1]).tolist(), (actual[2] + actual[3]).tolist()],
        "eligible_per_query_tile": [counts.tolist() for counts in actual],
        "slot_capacity_per_query": 2048,
        "measurement_must_exclude_capacity_bypass": True,
    }
    (root / "fixture.json").write_text(json.dumps(config, indent=2))
    return config["stage5_gate"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--layout", choices=("dispersed", "clustered"), required=True)
    parser.add_argument("--mean-per-aiv", type=float)
    parser.add_argument("--tile-pattern", choices=("uniform", "bursty"), default="uniform")
    parser.add_argument("--independent-queries", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output,
                args.fraction,
                args.layout,
                mean_per_aiv=args.mean_per_aiv,
                tile_pattern=args.tile_pattern,
                independent_queries=args.independent_queries,
            )
        )
    )
