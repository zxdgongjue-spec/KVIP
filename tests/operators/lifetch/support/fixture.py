"""Actual FP8 bytes, dense distinct queries, independent resident-prefix oracle."""

import json

import numpy as np


def prepare(root, history, resident, queries, cold_scale=1.0, cold_active_tiles=None, causal=False):
    if history % 256 or not 2048 <= resident <= history or resident % 256:
        raise ValueError("history/resident must be aligned and resident >= Top2048")
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(73121)
    hot = rng.choice(history, resident, replace=False)
    cold = np.setdiff1d(np.arange(history), hot)
    actual = history + (queries - 1 if causal else 0)
    storage = (actual + 127) // 128 * 128
    visible = [history + q if causal else history for q in range(queries)]
    order = np.concatenate((hot, cold, np.arange(history, storage))).astype(np.int32)
    values = rng.integers(0, 4, (storage, 128), dtype=np.uint8)
    if causal:
        values[history:actual] = 3
    key = np.array([0, 0x38, 0x40, 0x44], np.uint8)[values][order]
    query = np.zeros((queries, 64, 128), np.uint8)
    # Exact binary fractions expose stale/repeated query scaling without
    # adding CPU-vs-device accumulation-rounding ambiguity to this fixture.
    head = np.arange(64)[None, :]
    query_index = np.arange(queries)[:, None]
    query_scale = np.array([0.5, 1.0, 2.0], np.float32)[(head + query_index) % 3]
    weights = np.array([1.0, 2.0, 0.5], np.float32)[(head * 2 + query_index) % 3]
    logical_scores = []
    scale = rng.uniform(0.25, 2, storage).astype(np.float32)
    if causal:
        scale[history:actual] = 4  # Visible tails must win; invisible ones must never be fetched.
        scale[actual:] = 0
    if not np.isfinite(cold_scale) or cold_scale < 0:
        raise ValueError("cold_scale must be finite and nonnegative")
    scale[cold] *= np.float32(cold_scale)
    if cold_active_tiles is not None:
        if not 0 <= cold_active_tiles <= (history - resident) // 128:
            raise ValueError("cold_active_tiles outside cold region")
        scale[cold[cold_active_tiles * 128 :]] = 0
    for q in range(queries):
        columns = (np.arange(64) + q * 16) % 128
        query[q, np.arange(64), columns] = 0x38
        score = (values[:, columns].astype(np.float32) * (weights[q] * query_scale[q])).sum(-1) * scale
        # Ascend CAST_ROUND is ties-away, unlike NumPy/PyTorch BF16 defaults.
        logical_scores.append(((score.view(np.uint32) + 0x8000) >> 16).astype(np.uint16))
    metadata = np.zeros(1024, np.uint32)
    metadata[:8] = [1, 0, 0, 0, 0, 0, storage // 128, 0]
    arrays = [
        query,
        key,
        weights,
        query_scale,
        scale[order],
        np.array([queries], np.uint32),
        np.array([actual], np.uint32),
        np.arange(storage // 128, dtype=np.int32),
        metadata,
        np.full((queries, 2048), -1, np.int32),
        np.zeros(128, np.uint8),
        np.zeros(128, np.uint8),
        np.zeros(16 * 1024 * 1024, np.uint8),
        np.zeros(queries * 2048 * 6 + 1024 + queries * history * 2 + 128, np.uint8),
    ]
    address_offset = queries * 2048 * 6 + 1024 + queries * storage * 2
    sq_offset = address_offset + storage * 8
    arrays[13] = np.zeros(sq_offset + queries * 2048 * 64 + queries * 128, np.uint8)
    addresses = np.uint64(0x200000000) + order.astype(np.uint64) * np.uint64(656)
    arrays[13][address_offset:sq_offset] = addresses.view(np.uint8)
    for i, array in enumerate(arrays):
        array.tofile(root / f"arg{i}.bin")
    np.save(root / "logical_scores.npy", np.stack(logical_scores))
    np.save(root / "physical_to_logical.npy", order)
    (root / "fixture.json").write_text(
        json.dumps(
            {
                "history": history,
                "resident": resident,
                "queries": queries,
                "batch": 1,
                "aic": 1,
                "aiv": 2,
                "seed": 73121,
                "dense_heads": True,
                "query_inputs": "distinct_nonunit_exact_binary_scales",
                "cold_scale": cold_scale,
                "cold_active_tiles": cold_active_tiles,
                "storage": storage,
                "visible_lengths": visible,
                "visibility": "causal_history_plus_query" if causal else "same_history_controlled_baseline",
                "network": False,
            }
        )
    )
