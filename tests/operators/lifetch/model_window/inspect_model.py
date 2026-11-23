# SPDX-License-Identifier: Apache-2.0
"""Validate complete production LI tasks, resident seeds and in-memory SQ descriptors."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

BATCH = 2
QUERIES = 4
K = 2048
HISTORY = 5120
ROW_BYTES = QUERIES * K * 6 + QUERIES * 256
CORE_DONE = re.compile(r"core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+)")
ROW_REPORT = re.compile(r"ROW=(\d+) QUEUE=(\d+) POSTS=(\d+) EPOCHS=([\d,]+) LAST_Q=([\d,]+)")


def check(condition, description):
    if not condition:
        raise ValueError(description)


def task_spans(folder, replays):
    stages = [{} for _ in range(3 * replays)]
    for path in sorted((folder / "model").glob("core*.instr_log.dump")):
        with path.open() as stream:
            found = [CORE_DONE.search(line) for line in stream if "done++" in line]
        check(len(found) == len(stages), f"Incomplete tasks in {path}")
        for stage, match in zip(stages, found):
            check(match is not None, f"Invalid completion marker in {path}")
            core, lane, start, end = map(int, match.groups())
            check((core, lane) not in stage, f"Duplicate core in {path}")
            stage[core, lane] = start, end

    spans = []
    expected = {(core, lane) for core in range(4) for lane in range(3)}
    for stage in stages:
        check(stage.keys() == expected, "Missing Cube or Vector core")
        spans.append((min(s for s, _ in stage.values()), max(e for _, e in stage.values())))
    return spans


def validate_seeds(raw, scores):
    check(len(raw) == BATCH * ROW_BYTES, "Invalid seed buffer length")
    for b in range(BATCH):
        row = memoryview(raw)[b * ROW_BYTES:(b + 1) * ROW_BYTES]
        keys = np.frombuffer(row, "<u2", QUERIES * K).reshape(QUERIES, K)
        ids = np.frombuffer(row, "<u4", QUERIES * K, QUERIES * K * 2).reshape(QUERIES, K)
        ids = ids & 0x3fffffff
        cuts = np.frombuffer(row, "<u2", offset=QUERIES * K * 6).reshape(QUERIES, 128)
        for q in range(QUERIES):
            check(np.array_equal(np.sort(ids[q]), np.arange(K)), "Invalid resident seed token set")
            check(np.array_equal(keys[q], scores[b, q, ids[q]]), "Resident seed scores disagree")
            check(cuts[q, 0] == keys[q].min(), "Incorrect resident cutoff")


def validate_window_topk(raw, scores, meta):
    """Compare every saved winner with the independent score oracle and scanned frontiers."""
    check(len(raw) == BATCH * ROW_BYTES, "Invalid incremental TopK buffer length")
    # The two shards each own half of every aligned cold window.
    half = ((HISTORY - K + 511) // 512 + 4 - 2) // (4 - 1) * 256
    tokens = np.arange(HISTORY, dtype=np.int64)
    shards = ((tokens - K) // half) % 2
    count = 0
    for batch in range(BATCH):
        row = memoryview(raw)[batch * ROW_BYTES:(batch + 1) * ROW_BYTES]
        saved_scores = np.frombuffer(row, "<u2", QUERIES * K).reshape(QUERIES, K)
        saved_ids = np.frombuffer(row, "<u4", QUERIES * K, QUERIES * K * 2).reshape(QUERIES, K)
        for q in range(QUERIES):
            ids = saved_ids[q] & 0x3fffffff
            cursor = int(meta[batch, q, 1])
            low, high = cursor & 0xffffffff, cursor >> 32
            check(K <= low <= HISTORY and K <= high <= HISTORY, "Invalid published shard frontiers")
            eligible = (tokens < K) | (tokens < np.where(shards == 0, low, high))
            check(len(np.unique(ids)) == K and np.all(ids < HISTORY), "Invalid incremental winner IDs")
            check(np.all(eligible[ids]), "Winner outside the processed sequence shards")
            oracle = scores[batch, q]
            check(np.array_equal(oracle[ids], saved_scores[q]), "Incremental ID points to a different score")
            exact = np.sort(oracle[eligible])[-K:]
            check(np.array_equal(exact, np.sort(saved_scores[q])), "Incremental TopK drops a strict winner")
            count += K
    return count


def validate_queue(queue_file, row, layout):
    words = np.fromfile(queue_file, "<u4").reshape(-1, 16).astype(np.uint64)
    source = words[:, 10] + (words[:, 11] << 32)
    target = words[:, 14] + (words[:, 15] << 32)
    remote_base = 0x1800000000 + row * HISTORY * 656
    local_base = 0x1200000000 + row * K * 656
    check(len(words) > 0 and np.all(source >= remote_base), "Empty SQ or invalid remote address")
    tokens = (source - remote_base) // 656
    check(np.all(source == remote_base + tokens * 656) and np.all(tokens < HISTORY),
          "WQE source is not a token in this row")
    slots = np.arange(len(words))
    check(np.array_equal(target, local_base + slots * 656), "WQE destination is not the configured LRU slot")
    check(len(np.unique(tokens)) == len(tokens), "Repeated prefetch WQE")
    check(np.array_equal(layout[tokens], K + slots), "WQEs do not match Index commitments")
    return tokens.tolist()


def inspect(folder):
    lines = (folder / "output.log").read_text()
    check("Model stopped successfully" in lines, f"Model did not finish in {folder}")
    scores_checked = re.findall(r"SCORE_MATCH=(\d+)/(\d+)", lines)
    expected_count = BATCH * QUERIES * HISTORY
    check(scores_checked and all(int(left) == int(right) == expected_count for left, right in scores_checked),
          f"Missing independent score oracle in {folder}")
    replays = len(scores_checked)
    rows = ROW_REPORT.findall(lines)
    check(len(rows) == replays * BATCH, f"Incomplete per-window report in {folder}")
    marker = folder / "mode.txt"
    mode = marker.read_text().strip() if marker.exists() else "timing"
    check(mode in ("fast", "timing"), f"Unknown chip-model mode in {folder}")
    spans = task_spans(folder, replays) if mode == "timing" else None

    results = []
    for replay in range(replays):
        name = "scores.bin" if replay == 0 else f"scores.bin.r{replay}"
        path = folder / name
        scores = np.fromfile(path, "<u2").reshape(BATCH, QUERIES, HISTORY)
        seeds = (folder / (name + ".seeds")).read_bytes()
        validate_seeds(seeds, scores)
        layout = np.fromfile(folder / (name + ".map"), "<u4").reshape(BATCH, HISTORY)
        meta = np.fromfile(folder / (name + ".meta"), "<u8").reshape(BATCH, QUERIES, 16)
        window = folder / (name + ".window")
        check(window.is_file() or not marker.exists(), f"Missing persistent window TopK in {folder}")
        checked_winners = validate_window_topk(window.read_bytes(), scores, meta) if window.exists() else None
        tokens_by_window = []
        for b in range(BATCH):
            row, head, total, buckets, last_q = rows[replay * BATCH + b]
            counts = list(map(int, buckets.split(",")))
            check(len(counts) == 4 and counts[-1] > 0 and sum(counts) == int(total) == int(head),
                  "Last window was empty, or queue head disagrees with window post counts")
            check(int(row) == b and list(map(int, last_q.split(","))) == meta[b, :, 3].tolist(),
                  "Incorrect query counts or row order")
            tokens = validate_queue(folder / (name + f".sq{b}"), b, layout[b])
            check(len(tokens) == sum(counts), "SQ length disagrees with window post counts")
            position = 0
            windows = []
            for count in counts:
                windows.append(tokens[position:position + count])
                position += count
            tokens_by_window.append(windows)
        timing = {}
        if spans is not None:
            hot, seed, cold = spans[3 * replay:3 * replay + 3]
            timing = dict(hot_cycles=hot[1] - hot[0], seed_cycles=seed[1] - seed[0],
                          cold_cycles=cold[1] - cold[0], three_li_cycles=cold[1] - hot[0])
        results.append(dict(**timing, mode=mode, incremental_winners_checked=checked_winners,
                            scores_sha=hashlib.sha256(scores.tobytes()).hexdigest(),
                            seed_sha=hashlib.sha256(seeds).hexdigest(),
                            frontier=meta[:, :, 4].tolist(), tokens=tokens_by_window))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="*")
    args = parser.parse_args()
    baseline = inspect(args.reference)
    reference = baseline[0]
    output = {}
    for folder in [args.reference, *args.candidates]:
        runs = baseline if folder == args.reference else inspect(folder)
        report = []
        for result in runs:
            same_scores = result["scores_sha"] == reference["scores_sha"]
            same_seeds = result["seed_sha"] == reference["seed_sha"]
            check(same_scores and same_seeds, f"Changed token scores or resident seeds in {folder}")
            order, membership = [], []
            for arow, brow in zip(reference["tokens"], result["tokens"]):
                order.append([a == b for a, b in zip(arow, brow)])
                membership.append([set(a) == set(b) for a, b in zip(arow, brow)])
            timings = {name: value for name, value in result.items() if name.endswith("_cycles")}
            if reference["mode"] == result["mode"] == "timing":
                timings["three_li_saved_pct"] = 100 * (1 - result["three_li_cycles"] / reference["three_li_cycles"])
            else:
                timings["timing_comparison"] = "unavailable; fast mode has no per-core timing"
            report.append(dict(**timings, mode=result["mode"], incremental_winners_checked=result["incremental_winners_checked"],
                               posts_by_window=[[len(w) for w in row] for row in result["tokens"]],
                               same_frontiers=result["frontier"] == reference["frontier"],
                               same_prefetch_token_sets=membership, same_prefetch_order=order))
        output[str(folder)] = report
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
