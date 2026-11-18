# SPDX-License-Identifier: Apache-2.0
"""Summarize one verified Level0 graph replay; never combine independent clocks."""
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path


def summarize(root, label):
    manifest, = root.glob("*/none/profile_manifest.json")
    info = json.loads(manifest.read_text())
    if info["correctness"] != "passed" or info["graph_replays"] != 1 or info["included_in_headline_timing"]:
        raise ValueError(f"Not a verified independent graph capture: {manifest}")
    if label in ("before", "after") and info["experiment"]["auto_route"]["mode"] != "low":
        raise ValueError("Fused trace was not LOW")
    table, = manifest.parent.rglob("kernel_details.csv")
    with table.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted((Decimal(r["Start Time(us)"].strip()), Decimal(r["Duration(us)"].strip()), r["Name"])
                      for r in csv.DictReader(f))
    if not rows or any(length <= 0 for _, length, _ in rows):
        raise ValueError(f"No valid timed kernel rows: {table}")
    last = rows[0][0]
    gaps = Decimal(0)
    for start, duration, name in rows:
        gaps += max(Decimal(0), start - last)
        last = max(last, start + duration)
        print(f"KERNEL {label} {name} us={duration:.3f}")
    print(f"DEVICE_SPAN {label} us={last-rows[0][0]:.3f} uncovered_us={gaps:.3f} single_profile=true")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]), sys.argv[2])
