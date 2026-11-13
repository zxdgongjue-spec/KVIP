# SPDX-License-Identifier: Apache-2.0
"""CPU prediction of the current production URMA resident/recent UB seed path.

Mirrors PrepareShape/UseSequenceSplit and recent_index::MakeSeedLayout. This is
an eligibility prediction from restored inputs, not a device execution counter.
"""

import numpy as np


def predict_seed_eligibility(archive, runtime_shape=None):
    g = archive.manifest["geometry"]
    batch, history, resident, aics, heads = (
        int(g[name]) for name in ("batch", "history", "resident", "aic_count", "heads")
    )
    split = 2 * batch <= aics
    rounds = min(int(g["requested_windows"]), batch * (1 + int(split)))
    if runtime_shape is not None:
        expected = dict(
            batch=batch,
            history=history,
            resident=resident,
            aicCount=aics,
            indexHeads=heads,
            splitQ=int(split),
            rounds=rounds,
        )
        if any(int(getattr(runtime_shape, name)) != value for name, value in expected.items()):
            raise ValueError("Seed eligibility geometry differs from the actual native RuntimeShape")
    bounds = archive.array("pre.threshold_bounds")
    ends = archive.array("pre.query_ends").reshape(batch, 4)
    q_lengths = archive.array("pre.query_lengths").reshape(batch)
    k_lengths = archive.array("pre.key_lengths").reshape(batch)
    if (
        bounds.dtype != np.int32
        or bounds.shape != (batch, 2)
        or np.any(bounds < 0)
        or np.any(bounds[:, 1] < bounds[:, 0])
        or np.any(bounds > history)
    ):
        raise ValueError("Invalid resident/recent threshold bounds")
    if np.any(ends < 0) or np.any(ends > history):
        raise ValueError("Seed query ends exceed history")
    blocks = max(0, history - resident) // 512
    half_blocks = (blocks + rounds - 2) // (rounds - 1) if rounds > 1 else 0
    sequence = bool(
        rounds > 1
        and history > resident
        and split
        and heads == 32
        and resident % 512 == 0
        and (history - resident) % 512 == 0
        and history - resident >= 512 * (rounds - 1)
        and half_blocks * (2 * rounds - 3) < 2 * blocks
        and np.all(q_lengths == 4)
        and np.all(k_lengths == history)
    )
    eligible = np.zeros((batch, 4), dtype=bool)
    padded_words = []
    tile_ranges = []
    for row in range(batch):
        first = max(int(bounds[row, 0]), resident)
        last = min(int(bounds[row, 1]), int(ends[row].max()))
        begin_tile, end_tile = (first // 128, (last + 127) // 128) if first < last else (0, 0)
        words = ((resident + (end_tile - begin_tile) * 128 + 255) // 256) * 256
        padded_words.append(words)
        tile_ranges.append([begin_tile, end_tile])
        layout = (
            not sequence
            and split
            and resident >= 2048
            and resident % 128 == 0
            and q_lengths[row] == 4
            and k_lengths[row] == history
            and words <= 18432
        )
        if layout:
            eligible[row] = (ends[row] >= resident) & (ends[row] <= history)
    count = int(eligible.sum())
    return dict(
        predicted_eligible_queries=count,
        total_queries=batch * 4,
        predicted_fallback_queries=batch * 4 - count,
        eligible_mask=eligible.astype(np.uint8).tolist(),
        padded_words=padded_words,
        recent_tile_ranges=tile_ranges,
        split_q=int(split),
        sequence_split=sequence,
        scope="CPU eligibility prediction, not observed native dispatch; fixed production BSND/Q4/no sparse mask/LD-disabled caller, current RuntimeShape and restored query lengths/bounds",
    )
