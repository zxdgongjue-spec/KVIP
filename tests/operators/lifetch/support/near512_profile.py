# SPDX-License-Identifier: Apache-2.0
"""Check a length-only replay against the original trace before interpreting its timing."""
import json
from pathlib import Path
from statistics import mean

import numpy as np

from correctness import topk_fingerprint
from main_five_report import archived_oracle

PAGE = 128
TOPK = 2048


def _categories(inputs):
    """Index by logical ID. The terminal uses physical hot slots and ABC positions."""
    g = inputs.geometry
    categories = np.zeros((g['batch'], g['history']), dtype=np.uint8)
    capacity = g['output_stride'] - g['resident']
    for row, (lo, hi) in enumerate(inputs.tail_bounds):
        layout = inputs.captured_layout[row]
        if not (0 <= lo <= hi <= len(layout)) or hi - lo > capacity:
            raise ValueError(f'Row {row}: invalid initial ABC interval')
        categories[row, layout[lo:hi]] = 2
        hot_ids = layout[:g['resident']][inputs.captured_hot_valid[row].astype(bool)]
        categories[row, hot_ids] = 1
    return categories


def _locality(categories, selected):
    # Every request has a separate set of fetched KV. An ID selected by two
    # queries of the same request consumes only one initial remote token.
    totals = np.zeros(3, dtype=np.int64)
    for row, queries in enumerate(selected):
        totals += np.bincount(categories[row, np.unique(queries)], minlength=3)
    cold, hot, abc = map(int, totals)
    return dict(unique=cold + hot + abc, hot=hot, abc=abc, cold=cold)


def archive_profile(source, group, proof_path):
    """Reuse the stock-serial IDs and 20 completed request counters on B24."""
    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text())
    check = json.loads((proof_path.parent / 'score_topk_check.json').read_text())
    ref = check['native_reference']
    if ((check['group'], check['calls'], check['queries'], ref['origin']) != (group, 20, 1280, 'completed-owner-archive')
            or ref['identity']['inputs'] != proof['capture_operand_hashes']):
        raise ValueError(f'G{group}: serial reference and timing refer to different operands')
    old, old_check = archived_oracle(Path(ref['archive_path']), group)
    if (check['serial_set_digest'] != old_check['serial_set_digest']
            or check['serial_set_digest'] != topk_fingerprint(old)):
        raise ValueError(f'G{group}: archived serial ID digest mismatch')
    source.validate_topk(old)
    requests = proof['history_cap']['measured_requests']
    measured = {}
    for key in ('prefetch', 'final'):
        values = requests[key]
        if len(values) != 20 or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(f'G{group}: incomplete 128K traffic for {key}')
        measured[key] = mean(values)
    return dict(native_unique=_locality(_categories(source), old), measured_calls=20,
                measured_tokens_per_call_mean=measured, strong_ids_sha256=check['serial_set_digest'],
                strong_ids_reference=ref['archive_path'], archive=str(proof_path))


def bind_source_check(source, long_input, source_page_to_long_page):
    """Retain compact lookup fields until all measured TopK selections are checked."""
    pages = np.asarray(source_page_to_long_page)
    batch, history = source.captured_layout.shape
    long_history = long_input.geometry['history']
    if (pages.shape != (batch, history // PAGE) or pages.dtype.kind not in 'iu'
            or np.any(pages < 0) or np.any(pages >= long_history // PAGE) or np.any(np.diff(pages.astype(np.int64), axis=1) <= 0)):
        raise ValueError('Invalid monotonic mapping of original pages into the extended layout')
    back = np.full((batch, long_history // PAGE), -1, np.int32)
    rows = np.arange(batch)[:, None, None]
    back[rows[:, :, 0], pages] = np.arange(history // PAGE)
    old_class = _categories(source)
    new_class = _categories(long_input)
    old_scores = source.logical_score_keys
    old_valid = source.logical_score_valid
    # Captured score keys are monotonic encodings of BF16 values. On each Q
    # store the exact cutoff and the number of strictly higher valid scores.
    cutoffs = np.empty(old_scores.shape[:2], dtype=old_scores.dtype)
    above = np.empty(cutoffs.shape, dtype=np.int32)
    for i in np.ndindex(cutoffs.shape):
        population = old_scores[i][old_valid[i]]
        if len(population) < TOPK:
            raise ValueError(f'Source query {i} has fewer than {TOPK} valid scores')
        cutoffs[i] = np.partition(population, len(population) - TOPK)[-TOPK]
        above[i] = np.count_nonzero(population > cutoffs[i])
    qs = np.arange(4)[None, :, None]

    def check(selected):
        ids = np.asarray(selected)
        if (ids.shape != (batch, 4, TOPK) or ids.dtype.kind not in 'iu'
                or np.any(ids < 0) or np.any(ids >= long_history)):
            raise ValueError('Incorrect 512K TopK ID domain')
        src_page = back[rows, ids // PAGE]
        if np.any(src_page < 0):
            raise ValueError('Added history contributed a final TopK winner')
        short_ids = src_page * PAGE + (ids % PAGE).astype(np.int32)
        if (np.any(np.diff(np.sort(short_ids, axis=-1), axis=-1) == 0)
                or not old_valid[rows, qs, short_ids].all()):
            raise ValueError('Mapped TopK contains repeated or invisible original tokens')
        scores = old_scores[rows, qs, short_ids]
        if (np.any(scores < cutoffs[..., None]) or
                np.any(np.count_nonzero(scores > cutoffs[..., None], axis=-1) != above)):
            raise ValueError('Mapped winners differ from original TopK scores beyond cutoff ties')
        if not np.array_equal(old_class[rows, short_ids], new_class[rows, ids]):
            raise ValueError('A selected token changed its initial hot, ABC or cold classification')
        return short_ids

    long_input.check_donor_topk = check


def compare_profiles(short, inputs, selected):
    inputs.check_donor_topk(selected)
    old = short['native_unique']
    new = _locality(_categories(inputs), selected)
    return dict(short=old, long=new, delta={k: new[k] - old[k] for k in old}, checked_queries=selected.shape[0]*4)


def traffic_comparison(short, measured):
    if len(measured) != 10:
        raise ValueError('Expected all ten long-input traffic measurements')
    old = short['measured_tokens_per_call_mean']
    new = {k: mean(item[k] for item in measured) for k in ('prefetch', 'final')}
    return dict(short=old, long=new, delta={k: new[k] - old[k] for k in old})
