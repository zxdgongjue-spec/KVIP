#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two bench_lifetch_batch.sh result directories case by case.

Usage: python3 scripts/compare_batch.py <baseline-dir> <candidate-dir>

Reads only the saved logs, so it works after the fact and needs no device.
"""

import json
import re
import sys
from pathlib import Path

BS = re.compile(r"^BS=(\d+) .*history=(\d+) hot=(\d+)", re.M)
P50 = re.compile(r"graph_p50_us=([^\n]+)")
SPEED = re.compile(r"full_speedup=([0-9.]+)x")
ROUND = re.compile(r"^round=(\d+) total_us=([0-9.]+) (.*)$", re.M)
STAGE = re.compile(r"([a-z_]+)_us=([0-9.]+)")


def parse(path):
    """One record per (miss, batch) with timings and the steady-state stage split."""
    text = path.read_text()
    cases = {}
    blocks = re.split(r"(?=^BS=)", text, flags=re.M)
    for block in blocks:
        bs = BS.search(block)
        if not bs:
            continue
        batch = int(bs.group(1))
        times = P50.search(block)
        if not times:
            continue
        per = {}
        for item in times.group(1).split(","):
            name, _, value = item.strip().partition(":")
            try:
                per[name] = float(value)
            except ValueError:
                continue
        rounds = [m for m in ROUND.finditer(block) if int(m.group(1)) > 0]
        stages = {}
        if rounds:
            for key, value in STAGE.findall(rounds[-1].group(3)):
                stages[key] = float(value)
        speed = SPEED.search(block)
        cases[batch] = dict(times=per, stages=stages, speed=float(speed.group(1)) if speed else None)
    return cases


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    base_dir, cand_dir = Path(argv[1]), Path(argv[2])
    rows = []
    for miss in (10, 30, 50):
        base_log, cand_log = base_dir / f"miss{miss}.log", cand_dir / f"miss{miss}.log"
        if not (base_log.exists() and cand_log.exists()):
            continue
        base, cand = parse(base_log), parse(cand_log)
        for batch in sorted(set(base) & set(cand)):
            b, c = base[batch], cand[batch]
            bp50 = b["times"].get("prototype_with_update")
            cp50 = c["times"].get("prototype_with_update")
            if bp50 is None or cp50 is None:
                continue
            row = dict(
                miss=miss,
                batch=batch,
                base=bp50,
                cand=cp50,
                delta=100.0 * (cp50 - bp50) / bp50,
                base_speed=b["speed"],
                cand_speed=c["speed"],
            )
            for stage in ("init", "li", "resolve", "fetch"):
                row[f"base_{stage}"] = b["stages"].get(stage)
                row[f"cand_{stage}"] = c["stages"].get(stage)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No comparable cases under {base_dir} and {cand_dir}")
    header = ["miss", "B", "base_us", "cand_us", "delta%", "init", "li", "resolve", "fetch", "speed"]
    print("\t".join(header))
    for row in rows:
        stages = []
        for stage in ("init", "li", "resolve", "fetch"):
            base, cand = row[f"base_{stage}"], row[f"cand_{stage}"]
            stages.append(f"{base:.1f}->{cand:.1f}" if base is not None and cand is not None else "-")
        speed = f"{row['base_speed']:.3f}->{row['cand_speed']:.3f}" if row["base_speed"] and row["cand_speed"] else "-"
        print(
            f"{row['miss']}%\t{row['batch']}\t{row['base']:.3f}\t{row['cand']:.3f}\t{row['delta']:+.1f}\t"
            + "\t".join(stages)
            + f"\t{speed}"
        )
    print()
    print("delta% is candidate vs baseline on prototype_with_update p50; negative is faster.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
