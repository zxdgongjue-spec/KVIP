"""Score-only task/row adapters and BS boundary experiments; timings are evidence, never estimates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

VERSIONS = ("current", "historical")


def plan(batch, history=131072, heads=32, cube=32, vector=64, mode="auto", producers=None, query_group=4):
    integers = (batch, history, heads, cube, vector, query_group)
    if any(type(x) is not int for x in integers):
        raise ValueError("integer geometry required")
    if not 1 <= batch <= 32 or not 0 < history <= 210944 or history % 256:
        raise ValueError("batch 1..32 and 256-aligned history required")
    if heads not in (32, 64) or not 1 <= cube <= 32 or not 2 <= vector <= 64 or query_group not in (2, 4):
        raise ValueError("unsupported heads or physical budget/query group")
    if mode not in ("auto", "history", "batch_query"):
        raise ValueError("unknown mode")
    capacity = min(cube, vector // 2)
    if mode == "auto":
        if producers is not None or query_group != 4:
            raise ValueError("auto owns its structural choice; use explicit mode for A/B")
        producers = max(1, min(32, capacity // batch, history // 4096))
        mode = "history" if producers > 1 else "batch_query"
    elif mode == "batch_query":
        if producers not in (None, 1):
            raise ValueError("batch/query must scan whole H")
        producers = 1
    else:
        producers = 2 if producers is None else producers
        if query_group != 4:
            raise ValueError("history route preserves Q4")
    if type(producers) is not int or not 1 <= producers <= min(32, history // 256):
        raise ValueError("each producer needs a nonempty 256-token range")
    tasks = []
    for b in range(batch):
        for q in range(0, 4, query_group):
            for s in range(producers):
                tasks.append(
                    dict(
                        owner=b,
                        query_begin=q,
                        query_count=query_group,
                        begin=(history // 256 * s // producers) * 256,
                        end=(history // 256 * (s + 1) // producers) * 256,
                    )
                )
    stride = max(t["end"] - t["begin"] for t in tasks)
    waves = [tasks[i : i + capacity] for i in range(0, len(tasks), capacity)]
    return dict(
        batch=batch,
        history=history,
        heads=heads,
        cube=cube,
        vector=vector,
        mode=mode,
        producers=producers,
        query_group=query_group,
        capacity=capacity,
        row_stride=stride,
        tasks=tasks,
        waves=waves,
        score_bytes=batch * 4 * history * 2,
        logical_key_bytes=batch * (4 // query_group) * history * 128,
        max_wave_rows=max(map(len, waves)),
        measured_latency_ms=None,
        recommendation_basis="structural occupancy; no measured crossover",
    )


def gather_rows(canonical, p):
    import numpy as np

    rows = np.full((len(p["tasks"]), p["query_group"], p["row_stride"]), 0x5A5A, np.uint16)
    for i, t in enumerate(p["tasks"]):
        rows[i, :, : t["end"] - t["begin"]] = canonical[
            t["owner"], t["query_begin"] : t["query_begin"] + t["query_count"], t["begin"] : t["end"]
        ]
    return rows


def scatter_rows(rows, p):
    import numpy as np

    expected = (len(p["tasks"]), p["query_group"], p["row_stride"])
    if rows.shape != expected or rows.dtype != np.uint16:
        raise ValueError("score producer row shape/dtype mismatch")
    out = np.empty((p["batch"], 4, p["history"]), np.uint16)
    for i, t in enumerate(p["tasks"]):
        length = t["end"] - t["begin"]
        if not np.all(rows[i, :, length:] == 0x5A5A):
            raise ValueError("producer padding was overwritten")
        out[t["owner"], t["query_begin"] : t["query_begin"] + t["query_count"], t["begin"] : t["end"]] = rows[
            i, :, :length
        ]
    return out


def decode_fp8(raw):
    import numpy as np

    exponent = ((raw >> 3) & 15).astype(np.int32)
    mantissa = (raw & 7).astype(np.float32)
    if np.any((exponent == 15) & (mantissa == 7)):
        raise ValueError("finite FP8 fixture required")
    value = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7))
    return (np.where(raw & 128, -value, value)).astype(np.float32)


def sortable_bf16(value):
    import numpy as np

    bits = np.asarray(value, np.float32).view(np.uint32)
    # The original kernel uses CAST_ROUND (midpoint away from zero), as do
    # the repository's controlled fixtures. This is not round-to-even.
    bf16 = ((bits + 0x8000) >> 16).astype(np.uint16)
    return np.where(bf16 & 0x8000, ~bf16, bf16 ^ 0x8000).astype(np.uint16)


def score_oracle(query, key, weight, qscale, kscale):
    """Even/odd head accumulators mirror ReduceSumWeightedHeads, CAST_ROUND keys.

    CPU dot/FMA rounding is not device validation. Fixtures use sparse dyadic
    operands so the independent dot is exact and both routes have equal keys.
    """
    import numpy as np

    q, k = decode_fp8(query), decode_fp8(key)
    w = (weight.astype(np.uint32) << 16).view(np.float32) * qscale
    result = np.empty((q.shape[0], 4, k.shape[1]), np.uint16)
    for b in range(q.shape[0]):
        for r in range(4):
            dot = q[b, r] @ k[b].T
            accum = [np.zeros(k.shape[1], np.float32), np.zeros(k.shape[1], np.float32)]
            for g in range(q.shape[2]):
                accum[g % 2] += np.maximum(dot[g], 0) * w[b, r, g]
            value = ((accum[0] + accum[1]) * kscale[b]).astype(np.float32)
            result[b, r] = sortable_bf16(value)
    return result


def make_fixture(output, batch, history, heads):
    import numpy as np

    plan(batch, history, heads)  # fail before creating anything
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261115)
    query = np.zeros((batch, 4, heads, 128), np.uint8)
    for b in range(batch):
        for q in range(4):
            for g in range(heads):
                query[b, q, g, (g * 7 + q * 13 + b) % 128] = 0x38 if (b + q + g) % 3 else 0xB8
    codes = np.array([0x00, 0x20, 0x30, 0x38, 0x40, 0xB0, 0xB8], np.uint8)
    key = rng.choice(codes, (batch, history, 128))
    weight_f = rng.choice(np.array([0.25, 0.5, 1, -0.5], np.float32), (batch, 4, heads))
    weight = (weight_f.view(np.uint32) >> 16).astype(np.uint16)
    qscale = rng.choice(np.array([0.5, 1, 2], np.float32), weight.shape)
    kscale = rng.choice(np.array([0.5, 1, 2], np.float32), (batch, history))
    oracle = score_oracle(query, key, weight, qscale, kscale)
    files = dict(query=query, key=key, weight=weight, qscale=qscale, kscale=kscale, oracle=oracle)
    for name, value in files.items():
        value.tofile(output / (name + ".bin"))
    meta = dict(
        batch=batch,
        history=history,
        heads=heads,
        seed=20261115,
        numerical_domain="finite sparse dyadic FP8, BF16 weights, signed per-owner data",
        hashes={name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in files.items()},
    )
    (output / "fixture.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def boundary_matrix(history=131072):
    rows = []
    for version in VERSIONS:
        for cube, vector in ((32, 64), (28, 56)):
            for heads in (32, 64):
                for batch in range(1, 33):
                    choices = [("batch_query", 1, q) for q in (4, 2)]
                    choices += [("history", p, 4) for p in (2, 4, 8, 16, cube) if p <= history // 4096]
                    for mode, producers, group in choices:
                        p = plan(batch, history, heads, cube, vector, mode, producers, group)
                        rows.append(
                            {
                                k: p[k]
                                for k in (
                                    "batch",
                                    "history",
                                    "heads",
                                    "cube",
                                    "vector",
                                    "mode",
                                    "producers",
                                    "query_group",
                                    "row_stride",
                                )
                            }
                            | dict(
                                version=version,
                                waves=len(p["waves"]),
                                tasks=len(p["tasks"]),
                                status="NOT_RUN",
                                score_ms=None,
                                input_sha=None,
                                binary_sha=None,
                                oracle_passed=False,
                                device="",
                                run_mode="",
                                config_sha=None,
                                iterations=None,
                            )
                        )
    return rows


def measured_choices(rows):
    """Require a matched cohort and both routes; return per-B winners, no monotonic assumption."""
    groups = {}
    for r in rows:
        if r["status"] != "PASSED" or r["oracle_passed"] not in (True, "True", "true"):
            continue
        if r.get("run_mode") != "npu":
            continue  # Model ticks/wall time cannot establish a physical crossover.
        if r.get("score_ms") in (None, "", "null"):
            continue  # Valid Model evidence contains no physical score latency.
        cohort = tuple(
            r[k]
            for k in (
                "version",
                "batch",
                "history",
                "heads",
                "cube",
                "vector",
                "input_sha",
                "binary_sha",
                "device",
                "run_mode",
                "config_sha",
                "iterations",
            )
        )
        if any(x is None or str(x) == "" for x in cohort):
            raise ValueError("incomplete measurement provenance")
        value = float(r["score_ms"])
        if not 0 < value < float("inf"):
            raise ValueError("positive finite measured latency required")
        groups.setdefault(cohort, []).append(r)
    result = []
    for cohort, candidates in groups.items():
        modes = {r["mode"] for r in candidates}
        if not {"history", "batch_query"} <= modes:
            continue
        winner = min(candidates, key=lambda r: float(r["score_ms"]))
        result.append(
            dict(
                cohort=cohort,
                mode=winner["mode"],
                producers=int(winner["producers"]),
                query_group=int(winner["query_group"]),
                score_ms=float(winner["score_ms"]),
                basis="matched measured score-only cohort",
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--history", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=32)
    args = parser.parse_args()
    if args.fixture:
        print(json.dumps(make_fixture(args.fixture, args.batch, args.history, args.heads), indent=2))
    elif args.measurements:
        with args.measurements.open() as f:
            result = measured_choices(list(csv.DictReader(f)))
        print(json.dumps(result, indent=2))
    else:
        if not args.output:
            parser.error("--output is required for the BS matrix")
        args.output.mkdir(parents=True, exist_ok=True)
        rows = boundary_matrix(args.history)
        with (args.output / "score-boundary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (args.output / "score-occupancy.json").write_text(
            json.dumps(
                [
                    {k: v for k, v in plan(b, args.history, cube=c, vector=v).items() if k not in ("tasks", "waves")}
                    for c, v in ((32, 64), (28, 56))
                    for b in range(1, 33)
                ],
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(dict(cases=len(rows), status="NOT_RUN", measured_crossover=None)))


if __name__ == "__main__":
    main()
