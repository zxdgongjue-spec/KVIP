"""Offline trace statistics for fitting executable synthetic LI workloads.

Consumes existing schema-4 scores, including captures without Q/Index. Output
contains empirical fitting targets, not a claim that a generator is calibrated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reporting import atomic_json
from trace_inputs import TOPK, load_score_capture, score_values, select_series
from workloads import round_range

QUANTILES = (0, .01, .1, .25, .5, .75, .9, .99, 1)
POSITION_BINS = 16


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return dict(count=len(values), finite=len(finite),
                negative_inf=int(np.isneginf(values).sum()), positive_inf=int(np.isposinf(values).sum()),
                mean=float(finite.mean()) if len(finite) else None,
                quantiles=dict(zip(map(str, QUANTILES), np.quantile(finite, QUANTILES).tolist())) if len(finite) else {})


def position_histogram(positions, length):
    return np.histogram(np.asarray(positions) / length, bins=np.linspace(0, 1, POSITION_BINS + 1))[0].tolist()


def capture_rows(capture):
    """Return statistics plus private token sets used only for temporal matching."""
    a, g = capture.arrays, capture.manifest['geometry']
    batch, history, resident = (g[key] for key in ('batch', 'history', 'resident'))
    posted, processed = a['post.window_posted'], a['post.window_processed_ends']
    records, ordinals = a['post.prefetch_records'], a['post.prefetch_ordinals']
    final, thresholds = a['post.final_records'], a['post.threshold_records']
    if (posted.ndim != 2 or posted.shape[0] != batch or posted.dtype.kind != 'i'
            or processed.shape != (*posted.shape, 4) or processed.dtype.kind != 'i'
            or records.ndim != 2 or records.shape[1] != 3 or records.dtype.kind != 'i'
            or ordinals.shape != (len(records),) or ordinals.dtype.kind != 'i'
            or final.ndim != 2 or final.shape[1] != 4 or final.dtype.kind != 'i'
            or thresholds.ndim != 4 or thresholds.shape[:2] != (batch, 4) or thresholds.shape[-1] != 4
            or thresholds.dtype.kind != 'i'):
        raise ValueError('Calibration requires valid window, threshold and transfer record shapes')
    if (np.any(posted < 0) or np.any(processed < 0) or np.any(processed > history)
            or np.any(np.diff(processed, axis=1) < 0)
            or np.any(records[:, 0] < 0) or np.any(records[:, 0] >= batch)
            or np.any(final[:, 0] < 0) or np.any(final[:, 0] >= batch)):
        raise ValueError('Invalid calibration window/transfer bounds')
    result = []
    for row in range(batch):
        layout = a['pre.logical_ids'][row]
        topk = a['post.topk'][row]
        sets = [set(map(int, layout[query])) for query in topk]
        union = set.union(*sets)
        hot = set(map(int, layout[:resident][a['pre.hot_valid'][row] != 0]))
        begin_tail, end_tail = map(int, a['pre.tail_bounds'][row])
        if not 0 <= begin_tail <= end_tail <= history:
            raise ValueError('Invalid captured ABC bounds')
        local = hot | set(range(begin_tail, end_tail))
        fetched = records[records[:, 0] == row]
        order = ordinals[records[:, 0] == row]
        if (len(fetched) != int(posted[row].sum())
                or not np.array_equal(np.sort(order), np.arange(len(fetched)))):
            raise ValueError('Prefetch ordinals/window counts disagree')
        fetched = fetched[np.argsort(order)]
        if (np.any(fetched[:, 1] < 0) or np.any(fetched[:, 1] >= history)
                or not np.array_equal(layout[fetched[:, 1]], fetched[:, 2])):
            raise ValueError('Prefetch logical identity differs from pre-Index position')
        unused = np.array([int(token) not in union for token in fetched[:, 2]], dtype=bool)
        window_rows, offset = [], 0
        for window, count in enumerate(posted[row]):
            count = int(count)
            window_rows.append(dict(window=window + 1, posted=count,
                unused=int(unused[offset:offset + count].sum()),
                processed_ends=processed[row, window].astype(int).tolist()))
            offset += count
        queries, seen = [], set()
        for q in range(4):
            end = int(a['pre.query_ends'][row, q])
            keys = a['post.score_keys'][row, q, :end]
            scores = score_values(keys)
            cutoff = np.partition(keys, end - TOPK)[end - TOPK]
            hot_end = min(resident, end)
            hot_cutoff = np.partition(keys[:hot_end], hot_end - TOPK)[hot_end - TOPK]
            valid_thresholds = []
            for valid, applies, key, kind in thresholds[row, q]:
                if not kind:
                    continue
                if (not 0 <= valid <= end or not valid <= applies <= history
                        or not 0 <= key <= 65535 or kind not in (1, 2, 3, 4)):
                    raise ValueError('Invalid threshold record')
                expected = np.partition(keys[:valid], valid - TOPK)[valid - TOPK] if valid >= TOPK else 0
                if key != expected:
                    raise ValueError('Threshold record differs from captured scores')
                valid_thresholds.append(dict(valid=int(valid), applies=int(applies), key=int(key), kind=int(kind)))
            queries.append(dict(query=q + 1, selected=TOPK, unique_new=len(sets[q] - seen),
                cross_query_duplicates=len(sets[q] & seen), initial_cold=len(sets[q] - local),
                cutoff_key=int(cutoff), resident_cutoff_key=int(hot_cutoff),
                cutoff_tie_population=int(np.count_nonzero(keys == cutoff)),
                scores=distribution(scores), selected_scores=distribution(scores[topk[q]]),
                selected_cutoff_margin=(distribution(scores[topk[q]] - float(score_values([cutoff])[0]))
                    if np.isfinite(score_values([cutoff])[0]) else None),
                beyond_resident_above_threshold=int(np.count_nonzero(keys[resident:] > hot_cutoff)),
                physical_position_hist=position_histogram(topk[q], end),
                logical_position_hist=position_histogram(list(sets[q]), history),
                thresholds=valid_thresholds))
            seen.update(sets[q])
        overlap = [[len(left & right) / len(left | right) for right in sets] for left in sets]
        repairs = final[final[:, 0] == row]
        if np.any(repairs[:, 1] < 0) or np.any(repairs[:, 1] >= history):
            raise ValueError('Final transfer logical token outside history')
        result.append((dict(history=history, resident=resident, heads=g['heads'],
            initial_hot_valid=len(hot), selected_unique=len(union), initial_cold=len(union-local),
            prefetch=len(fetched), unused_prefetch=int(unused.sum()), final=len(repairs),
            q_jaccard=overlap, queries=queries, windows=window_rows), union))
    return result


def analyze(root, group, rounds, output):
    references = select_series(root, group, rounds)
    if not references:
        raise ValueError('No captures selected for calibration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous, rows, source_checksums = {}, [], {}
    for ref in references:
        capture = load_score_capture(ref.manifest_path, require_real=False)
        source_checksums[str(ref.manifest_path)] = capture.manifest['archive']['sha256']
        for (request_id, owner), (row, selected) in zip(ref.request_signature, capture_rows(capture), strict=True):
            identity = (ref.dp_rank, request_id, owner)
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            row.update(dp=ref.dp_rank, request_key=key, source_round=ref.source_round,
                       scheduler_round=ref.scheduler_round, temporal=None)
            prior = previous.get(identity)
            if prior is not None and prior[0] + 1 == ref.scheduler_round:
                old = prior[1]
                row['temporal'] = dict(retained=len(old & selected), previous=len(old), current=len(selected),
                                      jaccard=len(old & selected) / len(old | selected))
            previous[identity] = (ref.scheduler_round, selected)
            rows.append(row)
        atomic_json(output / 'request_rounds.json', rows)
    totals = {key: sum(row[key] for row in rows) for key in
              ('selected_unique', 'initial_cold', 'prefetch', 'unused_prefetch', 'final')}
    pairwise = np.asarray([row['q_jaccard'] for row in rows])
    targets = dict(schema_version=1, kind='lifetch_empirical_fit_targets', fitted_generator=False,
        captures=len(references), request_rounds=len(rows), totals=totals,
        cold_miss_rate=totals['initial_cold'] / totals['selected_unique'],
        unused_rate=totals['unused_prefetch'] / totals['prefetch'] if totals['prefetch'] else None,
        q_jaccard_mean=pairwise.mean(axis=0).tolist(),
        temporal_jaccard=distribution([row['temporal']['jaccard'] for row in rows if row['temporal']]),
        metrics={key: distribution([row[key] for row in rows]) for key in totals},
        physical_position_hist=np.sum([q['physical_position_hist'] for row in rows for q in row['queries']], axis=0).tolist(),
        logical_position_hist=np.sum([q['logical_position_hist'] for row in rows for q in row['queries']], axis=0).tolist(),
        scope='Captured cache trajectory; selected token identity at cutoff ties is capture-specific. Not TPOT or kernel timing.',
        model_plan='Fit Q-overlap, score margins and position locality jointly; validate generated Q/Index through actual LI.')
    atomic_json(output / 'fit_targets.json', targets)
    atomic_json(output / 'sources.json', dict(group=group, round_range=rounds, archives=source_checksums))
    lines = [f"CALIBRATION captures={len(references)} request_rounds={len(rows)}",
             f"cold/selected={totals['initial_cold']}/{totals['selected_unique']} rate={targets['cold_miss_rate']:.6f}",
             f"prefetch={totals['prefetch']} unused={totals['unused_prefetch']} final={totals['final']}",
             f"adjacent_round_pairs={targets['temporal_jaccard']['count']}",
             'Q Jaccard mean:']
    lines.extend(' '.join(f'{value:.4f}' for value in row) for row in targets['q_jaccard_mean'])
    lines.append('Empirical targets only; no claim of fitted generator or device performance.')
    (output / 'summary.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--round-range', type=round_range, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / 'analysis_status.json'
    atomic_json(status, dict(status='RUNNING'))
    try:
        analyze(args.trace_root, args.group, args.round_range, args.output)
    except Exception as error:
        atomic_json(status, dict(status='FAILED', error=str(error)))
        raise
    atomic_json(status, dict(status='COMPLETE'))


if __name__ == '__main__':
    main()
