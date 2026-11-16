"""Full staged route plans and the existing 128K/Q4 default fixture adapter."""

import ctypes
import importlib
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests/operators/lifetch/support")]
cases = importlib.import_module("run_split_core_cases").cases


@dataclass(frozen=True)
class Task:
    owner: int
    query_begin: int
    begin: int
    end: int


def stage_tasks(batch, begin, end, c=28, v=56, *, force=None):
    """Structure-based initial boundary; no measured latency claim."""
    if not (1 <= batch <= 32 and begin % 128 == end % 128 == 0 and end > begin and 1 <= c <= 32 and 1 <= v <= 64):
        raise ValueError("invalid staged geometry")
    capacity = min(c, v // 2)
    if not capacity:
        raise ValueError("MIX1:2 needs at least two AIVs")
    query_group = 4 if 2 * batch <= capacity else 2
    producers = max(1, min(capacity // batch, (end - begin) // 128)) if query_group == 4 else 1
    if force == "history":
        query_group = 4
        producers = max(1, min(capacity // batch, (end - begin) // 128))
    elif force == "bsq":
        query_group, producers = 2, 1
    elif force is not None:
        raise ValueError("unknown route")
    tiles = (end - begin) // 128
    tasks = [
        Task(b, q, begin + tiles * p // producers * 128, begin + tiles * (p + 1) // producers * 128)
        for b in range(batch)
        for q in range(0, 4, query_group)
        for p in range(producers)
    ]
    return dict(
        route="history" if query_group == 4 else "bsq",
        producers=producers,
        query_group=query_group,
        tasks=tasks,
        capacity=capacity,
        waves=[tasks[i : i + capacity] for i in range(0, len(tasks), capacity)],
    )


def owner_waves(batch, v=56, workers=2):
    if not (1 <= batch <= 32 and 1 <= v <= 64 and workers in (1, 2, 4)):
        raise ValueError("invalid owner geometry")
    capacity = min(16, (v - 4 * workers) // 3)
    if capacity < 1:
        raise ValueError("no resident owner capacity")
    return [(b, min(capacity, batch - b)) for b in range(0, batch, capacity)]


def fixture(output, miss, h=131072, r=16384, heads=32):
    import numpy as np
    from real_fetch_inputs import prepare_full_chain_fixture

    output = Path(output)
    info = prepare_full_chain_fixture(output, h, r, miss, index_heads=heads, exact_miss=True)
    weights = np.fromfile(output / "arg2.bin", np.float32)
    ((weights.view(np.uint32) + 0x8000) >> 16).astype(np.uint16).tofile(output / "weights-bf16.bin")
    order = np.load(output / "physical_to_logical.npy")
    bits = np.load(output / "logical_scores.npy")[:, order]
    (bits ^ np.uint16(0x8000)).tofile(output / "oracle.bin")
    selected = np.stack([np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in bits]).astype(np.uint32)
    selected.tofile(output / "selected.bin")
    np.arange(r - 2048, r, dtype=np.uint32).tofile(output / "lru.bin")
    (output / "expected.json").write_text(
        json.dumps(dict(**info, history=h, resident=r, heads=heads, miss_per_query=round(2048 * miss / 100)), indent=2)
        + "\n"
    )
    return info


class ScoreWave(ctypes.Structure):
    _fields_ = [(x, ctypes.c_uint64) for x in ("bindings", "scores", "tasks")] + [
        (x, ctypes.c_uint32) for x in ("rows", "stride", "queryGroup", "readyEnd")
    ]


class Plan(ctypes.Structure):
    _fields_ = [
        ("canonical", ctypes.c_uint64),
        ("hot", ctypes.POINTER(ScoreWave)),
        ("cold", ctypes.POINTER(ScoreWave)),
    ] + [(x, ctypes.c_uint32) for x in ("hotCount", "coldCount", "aivBudget", "fetchWorkers")]
