"""Prepare production-helper inputs from an extracted eight-window audit.

Use every second archived boundary. All four Qs see the same pre-window bitmap.
Completed windows mark the union of their admitted IDs. Live service timing
(including skipped windows) is not recorded by this audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HISTORY = 131328
RESIDENT = 16384
TOP = 2048
QUOTA = 64
MINIMUM = 32
SAMPLES = (10, 4, 17, 7, 2)


def threshold(scores, k):
    index = len(scores) - k
    return int(np.partition(scores, index)[index])


def reference(ids, scores, covered, valid, rank, empty):
    accepted = ids < valid
    if not empty:
        accepted[accepted] &= ~covered[ids[accepted]]
    positions = np.flatnonzero(accepted)
    n = len(positions)
    selected_scores = scores[positions]
    cut = 0 if rank == TOP or n <= MINIMUM else threshold(scores, rank)
    cap = threshold(selected_scores, QUOTA) if n > QUOTA else 0
    if n > MINIMUM and cut > cap:
        cut = min(cut, threshold(selected_scores, MINIMUM))
    cut = max(cut, cap)

    # Work from the original order to check the independent packed algorithm.
    winners = np.r_[ids[accepted & (scores > cut)], ids[accepted & (scores == cut)]][:QUOTA]
    packed = (positions.astype(np.uint32) << 16) | selected_scores
    restored = np.r_[packed[(packed & 65535) > cut], packed[(packed & 65535) == cut]][:QUOTA]
    np.testing.assert_array_equal(winners, ids[restored >> 16])
    return winners, n


def record(ids, scores, covered, valid, rank, meta):
    empty = not np.any(covered)
    expected, n = reference(ids, scores, covered, valid, rank, empty)
    packed = np.zeros(7184, dtype='<u4')
    packed[:3] = [valid, rank, empty]
    packed[8:2056] = ids
    packed[2056:3080] = scores.astype('<u2').view('<u4')
    packed[3080:] = np.packbits(covered, bitorder='little').view('<u4')
    answer = np.zeros(72, dtype='<u4')
    answer[0] = len(expected)
    answer[8:8+len(expected)] = expected
    meta.update(rank=rank, empty=empty, uncovered=n, posted=len(expected))
    return packed, answer, meta


def write_case(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.bin').open('wb') as inputs, path.with_suffix('.want').open('wb') as answers:
        for packed, answer, _ in rows:
            packed.tofile(inputs)
            answer.tofile(answers)
    path.with_suffix('.json').write_text(json.dumps([meta for _, _, meta in rows], indent=2) + '\n')


def window_rank(valid, recent, window):
    low = max(RESIDENT, min(valid, int(recent[0])))
    high = max(low, min(valid, int(recent[1])))
    remaining = valid - RESIDENT - (high - low)
    if remaining == 0:
        return 1024
    # Model both shard cursors at this quarter boundary.
    end = min(valid, RESIDENT + (HISTORY - RESIDENT) // 128 * (window + 1) // 4 * 128)
    scanned = end - RESIDENT - (min(end, high) - min(end, low))
    return 1024 + (min(scanned, remaining) * 1024 + remaining // 2) // remaining


def prepare(group, data, out, all_model):
    geometry = json.loads((data/'manifest.json').read_text())['geometry']
    with np.load(data/'probe_inputs.npz', allow_pickle=False) as a:
        ids, scores = a['expected_ids'], a['expected_scores']
        ends, eligible, recent = a['query_ends'], a['eligible'], a['tail_bounds']
    if (geometry['history'], ids.shape, geometry['resident']) != (HISTORY, (16, 4, 9, TOP), RESIDENT):
        raise ValueError('Expected the full real BS16 Q4 window audit')
    all_inputs = []
    for b in range(16):
        covered = np.zeros(HISTORY, dtype=bool)
        for w, archive_w in enumerate((2, 4, 6, 8)):
            submitted = []
            for q in range(4):
                raw = ids[b, q, archive_w].astype('<u4')
                valid = int(ends[b, q])
                tagged = raw.copy()
                unavailable = ~eligible[b, raw].astype(bool)
                tail = (raw >= recent[b, 0]) & (raw < recent[b, 1])
                tagged[unavailable & ~tail] |= 0x80000000
                tagged[unavailable & tail] |= 0x40000000
                meta = dict(group=group, row=b, q=q, window=w)
                data_row = record(tagged, scores[b, q, archive_w], covered, valid,
                                  window_rank(valid, recent[b], w), meta)
                all_inputs.append(data_row)
                answer = data_row[1]
                submitted.extend(answer[8:8+answer[0]])
            covered[submitted] = True
    selected = []
    for w in range(4):
        rows = sorted((r for r in all_inputs if r[2]['window'] == w), key=lambda r: r[2]['uncovered'])
        selected.append(rows[len(rows)//2])
    measured = all_inputs if all_model else selected
    write_case(measured, out/f'g{group}')
    return dict(group=group, full_cpu_verified=len(all_inputs), model_cases=len(measured),
                eligible_p50=float(np.median([r[2]['uncovered'] for r in all_inputs])))


def edge_cases(out):
    rng = np.random.default_rng(936)
    ids = rng.choice(HISTORY, TOP, replace=False).astype('<u4')
    rows = []
    for index, count in enumerate((0, 1, 31, 32, 33, 63, 64, 65, 2047, 2048)):
        tagged = ids.copy()
        tagged[count:] = 0xffffffff
        covered = np.zeros(HISTORY, dtype=bool)
        if index % 2:
            covered[(int(ids[0]) + 1) % HISTORY] = True
        keys = np.full(TOP, 16800, dtype='<u2')
        if index % 3:
            keys[1::3] += 1
            keys[::5] -= 1
        rows.append(record(tagged, keys, covered, HISTORY, 1400 if index % 3 else TOP,
                           dict(group='edges', row=index)))
    write_case(rows, out/'edges')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--all-model', action='store_true',
                        help='Export all 1280 lists in place of the 20 stratified lists')
    cfg = parser.parse_args()
    for group in SAMPLES:
        print(json.dumps(prepare(group, cfg.audit/f'g{group}', cfg.out, cfg.all_model)), flush=True)
    edge_cases(cfg.out)
