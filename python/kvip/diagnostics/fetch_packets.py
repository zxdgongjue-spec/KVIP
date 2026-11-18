# SPDX-License-Identifier: Apache-2.0
"""Check compact producer packets and budget-driven rank catch-up, without an NPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.fetch_rank_check import oracle_row
from kvip.diagnostics.fetch_window_topk import publication
from kvip.diagnostics.incremental_topk import eligibility, inputs, scan_windows, select
from kvip.diagnostics.recent_threshold import recent_positions
from kvip.diagnostics.window_audit import CachedArchive


def packetize(values, threshold):
    values = np.asarray(values, dtype=np.uint16).reshape(-1, 128)
    packets = np.zeros((len(values), 48), dtype=np.uint16)
    for tile, scores in enumerate(values):
        ids = np.flatnonzero(scores > threshold)
        n = min(16, len(ids))
        packets[tile, 0] = len(ids)
        packets[tile, 16:16+n] = scores[ids[:n]]
        packets[tile, 32:32+n] = ids[:n]
    return packets


def unpack(packets, values, threshold):
    """Overflow falls back for the entire 2K block to preserve tie ordering."""
    values = np.asarray(values).reshape(-1, 128)
    scores, ids = [], []
    for begin in range(0, len(values), 16):
        chunk = packets[begin:begin+16]
        if np.any(chunk[:, 0] > 16):
            full = values[begin:begin+16].reshape(-1)
            offsets = np.flatnonzero(full > threshold)
            scores.extend(full[offsets]); ids.extend(begin*128+offsets)
        else:
            for tile, packet in enumerate(chunk, begin):
                n = int(packet[0])
                keys, offsets = packet[16:16+n], packet[32:32+n]
                keep = keys > threshold
                scores.extend(keys[keep]); ids.extend(tile*128+offsets[keep])
    return np.asarray(scores, dtype=np.uint16), np.asarray(ids, dtype=np.int32)


def check(scores, ends, bounds, masks, resident=16384):
    prefixes = posts = updates = deferred = overflow = candidates = 0
    for row, row_scores in enumerate(scores):
        oracle = oracle_row(row_scores, ends[row], bounds[row], resident)
        windows = scan_windows(row_scores.shape[-1], resident, ends[row], True)
        for q, end in enumerate(ends[row]):
            seed = recent_positions(resident, bounds[row], int(end))
            state = select(row_scores[q, seed], seed)
            t0 = state[2]
            seeded = np.zeros(row_scores.shape[-1], dtype=bool); seeded[seed] = True
            for epoch, physical in enumerate(windows, 1):
                keys = row_scores[q, physical]
                packets = packetize(keys, t0)
                current, logical = unpack(packets, keys, state[2])
                ids = physical[logical]
                valid = (ids < end) & ~seeded[ids]
                state = select(np.r_[state[0], current[valid]], np.r_[state[1], ids[valid]])
                expected = oracle[q][epoch]
                if state[2] != expected[2] or not all(np.array_equal(x, y) for x, y in zip(state[:2], expected[:2])):
                    raise AssertionError(f"Packet rank differs row={row} q={q} epoch={epoch}")
                prefixes += 1
                candidates += int(valid.sum())
                # Count only overflow tiles containing valid non-seed positions.
                real = ((physical < end) & ~seeded[physical]).reshape(-1, 128).any(axis=1)
                overflow += int(np.count_nonzero((packets[:, 0] > 16) & real))
        # Fixed and skewed frozen frontiers; only preceding published versions
        # are used. A blocked Q keeps its old state and catches up when needed.
        for frontiers in (list(range(1, 10)), [0, 1, 1, 3, 3, 6, 7, 7, 9]):
            posted, expected_posted = set(), set()
            cursors = [0]*4
            for target in frontiers:
                wanted_lists = [oracle[q][target-1][1][masks[row, oracle[q][target-1][1]]].tolist()
                                for q in range(4)] if target else []
                wanted = publication(wanted_lists, expected_posted, 64, 512)
                observed = []
                for q in range(4):
                    if not target or len(observed) >= 64 or len(posted) >= 512:
                        deferred += target > cursors[q]
                        continue
                    updates += target-cursors[q]
                    cursors[q] = target
                    for token in wanted_lists[q]:
                        if token not in posted and len(observed) < 64 and len(posted) < 512:
                            observed.append(token); posted.add(token)
                if observed != wanted:
                    raise AssertionError(f"Lazy Q4 posting differs row={row} target={target}")
                posts += 1
    return dict(prefixes=prefixes, submissions=posts, rank_updates=updates,
                deferred_queries=deferred, valid_overflow_tiles=overflow, candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--inputs', type=Path)
    source.add_argument('--audit-root', type=Path)
    parser.add_argument('--groups', type=int, nargs='+', default=[7, 2])
    args = parser.parse_args()
    for group in args.groups:
        if args.audit_root:
            with np.load(args.audit_root/f'g{group}'/'probe_inputs.npz', allow_pickle=False) as data:
                result = check(data['scores'], data['query_ends'], data['threshold_bounds'], data['eligible'])
        else:
            directory = args.inputs/f'g{group}-input'
            archive = CachedArchive(directory, json.loads((directory/'manifest.json').read_text()))
            try:
                g, scores, ends, bounds = inputs(archive)
                masks = np.stack([eligibility(archive, row) for row in range(g['batch'])])
                result = check(scores, ends, bounds, masks, g['resident'])
            finally:
                archive.close()
        print(f"CPU_PACKETS G{group} PASS "+' '.join(f'{k}={v}' for k,v in result.items())+
              ' timing=NOT_SIMULATED', flush=True)


if __name__ == '__main__':
    main()
