# SPDX-License-Identifier: Apache-2.0
"""Summarize completed four-batch tail diagnostics. Runs on CPU and writes no files."""

import argparse
import json
import math
from collections import deque
from pathlib import Path
from statistics import median
import tarfile


def peak(posts, width):
    live = deque()
    total = highest = 0
    for timestamp, tokens in sorted(posts):
        while live and live[0][0] <= timestamp - width:
            total -= live.popleft()[1]
        total += tokens
        live.append((timestamp, tokens))
        highest = max(highest, total)
    return highest


def report(doc):
    ident = doc["identity"]
    if doc["correctness"] != "PASS" or ident["group"] != 7:
        raise ValueError("Expected a checked G7 diagnostic")
    tail = doc["tail"]
    mhz = tail["clock_mhz"]
    qs = tail["decoded"]["qs"]
    if not math.isfinite(mhz) or mhz <= 0 or len(qs) != ident["batch"] * 4:
        raise ValueError("Missing calibrated clock or query rows")
    posts = []
    drains = [[], []]
    observations = []
    for q in qs:
        end = 0
        for i, item in enumerate(q["batches"]):
            t, start, done = (item[k] for k in ("post_end", "drain_begin", "drain_end"))
            n = item["prefix_end"] - end
            if n <= 0 or not t <= start <= done:
                raise ValueError("Invalid private prefix or clock order")
            end += n
            posts.append((t, n, (q["row"] + q["query"] - 1) % 4, len(q["batches"])))
            drains[int(i != 0)].append((done - start) / mhz)
            observations.extend(((t, 1), (done, -1)))
        if end != q["own"]:
            raise ValueError("Incomplete private GET list")
    if not all(drains):
        raise ValueError("Both first and later private batches are required")
    inflight = highest = 0
    for _, delta in sorted(observations):
        inflight += delta
        highest = max(highest, inflight)
    original = [(t, n) for t, n, _, _ in posts]
    width = 4096
    gated = []
    for offset in range(0, 4 * width, width // 4):
        shifted = []
        for t, n, group, count in posts:
            phase = (t + offset) % (4 * width)
            start = group * width
            delay = 0 if count != 4 or start <= phase < start + width - 1024 else (start - phase) % (4 * width)
            shifted.append((t + delay, n))
        gated.append(peak(shifted, 4 * mhz))
    print(f"G7 BS{ident['batch']} diagnostic first/later_drain_P50_us={median(drains[0]):.2f}/{median(drains[1]):.2f} "
          f"posted_unobserved_Q_peak={highest} peak_tokens_in_4us={peak(original, 4 * mhz)} "
          f"fixed_ready_gate_sensitivity_tokens={min(gated)}..{max(gated)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    docs = []
    if args.archive.is_dir():
        paths = sorted(args.archive.glob("g7/bs*/single_diagnostic.json"))
        for p in paths:
            docs.append(json.loads(p.read_bytes()))
    elif tarfile.is_tarfile(args.archive):
        with tarfile.open(args.archive) as archive:
            for member in archive:
                p = Path(member.name)
                if p.name == "single_diagnostic.json" and p.parent.parent.name == "g7" and member.isfile():
                    docs.append(json.load(archive.extractfile(member)))
    else:
        raise SystemExit(f"Cannot open diagnostic directory or tar: {args.archive}")
    if not docs:
        raise SystemExit(f"No completed G7 private-prefix diagnostic in {args.archive}")
    for doc in sorted(docs, key=lambda d: d["identity"]["batch"]):
        report(doc)


if __name__ == "__main__":
    main()
