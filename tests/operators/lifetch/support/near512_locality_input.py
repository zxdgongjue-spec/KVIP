# SPDX-License-Identifier: Apache-2.0
"""Expand the scanned cold span while preserving all four captured winner populations."""
from dataclasses import replace

import numpy as np

from workloads import TOPK

PAGE = 128


def _pages_for_row(history, resident, capacity, end, abc):
    # Preserve the hot slots and keep the entire ABC and four causal ends contiguous.
    first = resident // PAGE
    last = min(end, int(abc[0]) if abc[1] > abc[0] else end) // PAGE
    if not first < last <= history // PAGE:
        raise ValueError("Expected a cold interval before the causal/ABC suffix")
    pages = np.arange(history // PAGE, dtype=np.int32)
    inserted = (capacity - history) // PAGE
    # Spread the additional pages over all cold windows, not just the final one.
    return pages + np.clip(pages - first, 0, last - first) * inserted // (last - first)


def _background(source, row, count):
    """Use this same Q/Key/scale pair's measured BF16 scores in all four queries."""
    scores = source.logical_score_keys[row]
    valid = source.logical_score_valid[row]
    hot_ids = source.captured_layout[row, :source.geometry['resident']][source.captured_hot_valid[row] != 0]
    low = source.logical_address_valid[row].astype(bool)
    final_cutoffs, prefix_cutoffs = [], []
    for q in range(4):
        population = scores[q, valid[q]]
        if population.size < TOPK:
            raise ValueError(f"Row {row} Q{q + 1}: insufficient captured scores")
        cutoff = np.partition(population, -TOPK)[-TOPK]
        hot = scores[q, hot_ids[valid[q, hot_ids]]]
        # Once the hot scores are visible, every later prefix contains these
        # entries. Exclude even near-cutoff filler from its TopK.
        prefix_cutoff = np.partition(hot, -TOPK)[-TOPK] if hot.size >= TOPK else cutoff
        low &= valid[q] & (scores[q] < min(cutoff, prefix_cutoff))
        final_cutoffs.append(int(cutoff))
        prefix_cutoffs.append(int(prefix_cutoff) if hot.size >= TOPK else None)
    pages = np.flatnonzero(low.reshape(-1, PAGE).all(axis=1))
    tokens = ((pages[:, None] * PAGE + np.arange(PAGE)).ravel() if pages.size else np.flatnonzero(low))
    if not len(tokens):
        raise ValueError(f"Row {row}: no same-request published keys below all four hot cutoffs")
    return np.resize(tokens, count), dict(eligible_full_pages=len(pages), eligible_tokens=int(low.sum()),
                                          final_cutoffs_bf16_key=final_cutoffs, hot_cutoffs_bf16_key=prefix_cutoffs)


def expand_locality(source, capacity):
    """Inject all original physical/logical positions into a larger per-row permutation.

    Every extra position is a real scored, published token. Its exact captured scores
    are known because it retains the Key, scale and Q from the same request.
    """
    g = source.geometry
    batch, history, resident = g['batch'], g['history'], g['resident']
    if history % PAGE or resident % PAGE or capacity <= history or capacity % 256:
        raise ValueError("Expected a wider page-aligned LIFetch workspace")
    delta = capacity - history
    ends = source.query_ends + delta
    bounds = source.tail_bounds.copy()
    occupied = bounds[:, 1] > bounds[:, 0]
    bounds[occupied] += delta
    offset = np.arange(PAGE, dtype=np.int32)
    keys = np.empty((batch, capacity, 128), source.logical_keys.dtype)
    scales = np.empty((batch, capacity), source.logical_scales.dtype)
    scores = np.empty((batch, 4, capacity), np.uint16)
    valid = np.empty((batch, 4, capacity), bool)
    layout = np.tile(np.arange(capacity, dtype=np.int32), (batch, 1))
    address = np.ones((batch, capacity), np.uint8)
    page_keys, maps, limits = [], [], []
    for row in range(batch):
        page_map = _pages_for_row(history, resident, capacity, source.query_ends[row, 0], source.tail_bounds[row])
        positions = (page_map[:, None] * PAGE + offset).ravel()
        filler_page = np.ones(capacity // PAGE, bool)
        filler_page[page_map] = False
        filler = np.repeat(filler_page, PAGE)
        original_ids = np.empty(capacity, np.int32)
        original_ids[positions] = np.arange(history, dtype=np.int32)
        background, stats = _background(source, row, delta)
        original_ids[filler] = background
        keys[row] = source.logical_keys[row, original_ids]
        scales[row] = source.logical_scales[row, original_ids]
        scores[row] = source.logical_score_keys[row][:, original_ids]
        valid[row] = source.logical_score_valid[row][:, original_ids]
        layout[row, positions] = positions[source.captured_layout[row]]
        address[row, positions] = source.logical_address_valid[row]
        # Original pages retain publication holes and shared identities. Inserted
        # pages have unique payload identities, even when the scored Key repeats.
        new_pages = [(-1-row, p) for p in range(capacity // PAGE)]
        for old, p in enumerate(page_map):
            new_pages[p] = source.page_keys[row][old]
        page_keys.append(tuple(new_pages))
        maps.append(page_map.tolist())
        limits.append(stats)
    inputs = replace(source, geometry=dict(g, history=capacity), query_ends=ends, tail_bounds=bounds,
                     logical_keys=keys, logical_scales=scales, captured_layout=layout,
                     logical_address_valid=address, logical_score_keys=scores, logical_score_valid=valid,
                     page_keys=tuple(page_keys))
    return inputs, dict(source_page_to_long_page=maps, extra_scored_tokens_per_row=delta, background=limits,
                        score_source='same-request unchanged Q/Index/BF16; mapped source scores plus independent native LI')
