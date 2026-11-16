"""Independent sparse-FP8 score and owner-scoped reference, never a device result."""

from pathlib import Path
import numpy as np
from staged_routes import stage_tasks, owner_waves


def decode(codes):
    exp = ((codes & 0x7F) >> 3).astype(np.int32)
    mant = (codes & 7).astype(np.float32)
    return np.where(exp == 0, mant * 2.0**-9, (1 + mant / 8) * np.exp2(exp - 7)) * np.where(codes & 128, -1, 1)


def sparse_qk(folder, h, heads=32):
    folder = Path(folder)
    q = np.fromfile(folder / "arg0.bin", np.uint8).reshape(4, heads, 128)
    k = np.fromfile(folder / "arg1.bin", np.uint8).reshape(h, 128)
    weights = np.fromfile(folder / "weights-bf16.bin", np.uint16).astype(np.uint32)
    weights = (weights << 16).view(np.float32).reshape(4, heads)
    qs = np.fromfile(folder / "arg3.bin", np.float32).reshape(4, heads)
    ks = np.fromfile(folder / "arg4.bin", np.float32)
    result = []
    for query in range(4):
        accum = np.zeros(h, np.float32)
        for head in range(heads):
            columns = np.flatnonzero(q[query, head])
            if len(columns) != 1:
                raise ValueError("controlled oracle requires one-hot Q")
            col = columns[0]
            dot = (decode(k[:, col]) * decode(q[query, head, col])).astype(np.float32)
            accum += (np.maximum(dot, 0) * weights[query, head] * qs[query, head]).astype(np.float32)
        score = (accum * ks).astype(np.float32)
        result.append(((score.view(np.uint32) + 0x8000) >> 16).astype(np.uint16) ^ np.uint16(0x8000))
    return np.stack(result)


def check_case(folder, case):
    h, r, b = case["history"], case["resident"], case["batch"]
    score = sparse_qk(folder, h)
    reference = np.fromfile(Path(folder) / "oracle.bin", np.uint16).reshape(4, h)
    np.testing.assert_array_equal(score, reference)
    theta = np.partition(score[:, :r], -2048, axis=1)[:, -2048]
    selected = [np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in score]
    if any(len(x) != 2048 or np.count_nonzero(x >= r) != case["miss_per_query"] for x in selected):
        raise AssertionError("Final exact set/miss count")
    union = set(int(x) for row in selected for x in row if x >= r)
    assert len(union) == 2 * case["miss_per_query"]
    protected = set(int(x) for row in selected for x in row if x < r)
    victims = [x for x in range(r - 2048, r) if x not in protected]
    assert len(victims) >= len(union)
    # Compare actual task scatter to full row; owner metadata is checked separately.
    for lo, hi in ((0, r), (r, h)):
        plan = stage_tasks(b, lo, hi)
        written = {}
        for task in plan["tasks"]:
            for q in range(task.query_begin, task.query_begin + plan["query_group"]):
                row = written.setdefault((task.owner, q), np.zeros(hi - lo, np.uint16))
                row[task.begin - lo : task.end - lo] = score[q, task.begin : task.end]
        assert len(written) == b * 4
        for (_, q), row in written.items():
            np.testing.assert_array_equal(row, score[q, lo:hi])
    # Cross-Q dedup/one slot owner and permutation invariants across owner waves.
    for first, n in owner_waves(b):
        for owner in range(first, first + n):
            logical = np.arange(h, dtype=np.uint64) + owner * h
            mapping = {}
            for token, slot in zip(sorted(union), victims):
                logical[token], logical[slot] = logical[slot], logical[token]
                mapping[token] = slot
            assert len(set(mapping.values())) == len(union)
            for row in selected:
                for token in row:
                    slot = int(token) if token < r else mapping[int(token)]
                    assert logical[slot] == owner * h + int(token)
            assert np.array_equal(np.sort(logical), np.arange(h, dtype=np.uint64) + owner * h)
    return dict(
        status="PASSED",
        scope="CPU sparse-FP8 and owner reference only",
        batch=b,
        miss_per_query=case["miss_per_query"],
        unique_miss=len(union),
        resident_theta=[int(x) for x in theta],
        physical_npu=False,
    )
