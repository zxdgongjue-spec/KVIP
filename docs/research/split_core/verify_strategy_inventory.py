"""CPU reference arithmetic/document verification, NOT an Ascend kernel test."""

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs/research/split_core/strategy_inventory.md"
DATA = ROOT / "docs/research/split_core/results/strategy-inventory.json"


def byte_histogram_select(values, k):
    # Independent illustrative threshold reference; tie order is not a kernel oracle.
    hist = [0] * 256
    for value in values:
        hist[value >> 8] += 1
    rank = len(values) - k + 1
    high = 0
    while rank > hist[high]:
        rank -= hist[high]
        high += 1
    low_hist = [0] * 256
    for value in values:
        if value >> 8 == high:
            low_hist[value & 255] += 1
    low = 0
    while rank > low_hist[low]:
        rank -= low_hist[low]
        low += 1
    threshold = (high << 8) | low
    ids = [i for i, v in enumerate(values) if v > threshold]
    ids += [i for i, v in enumerate(values) if v == threshold][: k - len(ids)]
    return ids


def main():
    record = json.loads(DATA.read_text())
    for relative, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    links = re.findall(r"\]\((\.\./\.\./\.\./[^)]+)\)", DOC.read_text())
    for link in links:
        name, _, anchor = link.partition("#L")
        path = (DOC.parent / name).resolve()
        assert path.is_file(), path
        if anchor:
            assert 1 <= int(anchor) <= len(path.read_text().splitlines()), link
    rng = random.Random(1010)
    checks = 0
    for length, k in [(31, 8), (2048, 2048), (16384, 2048), (18432, 2048)]:
        for mode in ["random", "tie", "same"]:
            values = [
                rng.randrange(65536) if mode == "random" else rng.randrange(8) if mode == "tie" else 32768
                for _ in range(length)
            ]
            ids = byte_histogram_select(values, k)
            assert len(ids) == len(set(ids)) == k
            assert sorted(values[i] for i in ids) == sorted(values)[-k:]
            checks += 1
    # Independent local/merge property: verify selected key multiset, not tied ID order.
    for length, t, k in [(128, 8, 8), (257, 4, 16), (131072, 8, 2048)]:
        values = [rng.randrange(65536) for _ in range(length)]
        candidates = []
        for shard in range(t):
            begin = length * shard // t
            end = length * (shard + 1) // t
            ids = byte_histogram_select(values[begin:end], k)
            candidates.extend(values[begin + i] for i in ids)
        merged = byte_histogram_select(candidates, k)
        assert sorted(candidates[i] for i in merged) == sorted(values)[-k:]
        checks += 1
    # Current 256-aligned component mapping covers each query without overlap.
    geometry_checks = 0
    for b in [1, 2, 4, 8, 16, 24, 32]:
        for t in [1, 4, 8]:
            h = 131072
            parts = [((h // 256 * s // t) * 256, (h // 256 * (s + 1) // t) * 256) for s in range(t)]
            assert parts[0][0] == 0 and parts[-1][1] == h
            assert all(end - begin >= 2048 for begin, end in parts)
            assert all(parts[i][1] == parts[i + 1][0] for i in range(t - 1))
            tasks = [(task // t, task % t) for task in range(b * 4 * t)]
            assert len(set(tasks)) == b * 4 * t
            geometry_checks += 1
    for row in record["conditional_no_prefetch_examples"]:
        m = round(2048 * row["nominal_miss_percent"] / 100)
        assert (row["miss_per_query"], row["missing_positions_per_request"], row["unique_repairs_per_request"]) == (
            m,
            4 * m,
            2 * m,
        )
        assert row["hash_buckets"] == (2048 if 4 * m <= 1024 else 4096 if 4 * m <= 2048 else 8192)
    groups = [list(range(205)), list(range(205, 410)), list(range(205)), list(range(205, 410))]
    seen = {}
    added = []
    for group in groups:
        count = 0
        for token in group:
            if token not in seen:
                seen[token] = len(seen)
                count += 1
        added.append(count)
    assert added == [205, 205, 0, 0] and len(seen) == 410
    assert 16 * 4 * 205 == 13120 and 16 * 410 == 6560 and 16 * 4 * 2048 == 131072
    print(
        json.dumps(
            dict(
                scope="CPU mathematical reference and source/document consistency only",
                source_hashes=len(record["source_sha256"]),
                source_links=len(links),
                histogram_and_local_merge_reference_checks=checks,
                mapping_shapes=geometry_checks,
                conditional_miss_examples=len(record["conditional_no_prefetch_examples"]),
                q4_first_owner_example="passed",
                model_runs=0,
                npu_runs=0,
            )
        )
    )


if __name__ == "__main__":
    main()
