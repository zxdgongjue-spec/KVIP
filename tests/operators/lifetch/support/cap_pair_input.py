"""Bind a frozen short trace at a larger capacity while preserving all visible inputs."""
from copy import copy
from dataclasses import replace

import numpy as np


def extend_axis(array, axis, size, fill):
    shape = list(array.shape)
    length = shape[axis]
    shape[axis] = size
    result = np.empty(shape, dtype=array.dtype)
    prefix = [slice(None)] * array.ndim
    prefix[axis] = slice(0, length)
    result[tuple(prefix)] = array
    prefix[axis] = slice(length, None)
    result[tuple(prefix)] = fill
    return result


def bind_capacity(sequence, source, pages, history, digest):
    old = source.geometry['history']
    ends = np.asarray(source.query_ends)
    if (history < old or history % 256 or old % 256 or len(sequence.rounds) != 1
            or np.any(ends <= 0) or np.any(ends > old)
            or not np.array_equal(ends, ends[:, -1:] + np.arange(-3, 1))):
        raise ValueError('Expected one aligned Q4 capture with consecutive exclusive query ends')
    if pages.geometry != source.geometry or pages.page_indices.shape != (len(source.identities), old // 128):
        raise ValueError('Captured page table and Index capacity disagree')

    fields = ('captured_hot_valid', 'captured_lru', 'tail_bounds', 'logical_address_valid',
              'logical_score_keys', 'logical_score_valid')
    proof = dict(capture_history=old, bound_history=history, effective_ends=ends.tolist(),
                 query_ends_sha256=digest(ends), captured_page_count=pages.page_count,
                 bound_page_count=pages.page_count, capture_page_table_sha256=digest(pages.page_indices),
                 capture_state_sha256={name: digest(getattr(source, name)) for name in fields})
    if history == old:
        return sequence, source, pages, proof

    # The serving IndexCacheStorage initializes unused keys to 0, scales to 1,
    # addresses to 0 and logical IDs to identity. Q inputs, validity on the old
    # slots, absolute positions, ABC, hot validity and LRU are preserved.
    g = dict(source.geometry, history=history)
    missing_pages = (history - old) // 128
    padded = replace(
        source, geometry=g, logical_keys=extend_axis(source.logical_keys, 1, history, 0),
        logical_scales=extend_axis(source.logical_scales, 1, history, 1),
        captured_layout=extend_axis(source.captured_layout, 1, history,
                                    np.arange(old, history, dtype=source.captured_layout.dtype)),
        logical_address_valid=extend_axis(source.logical_address_valid, 1, history, 0),
        page_keys=tuple(row + (None,) * missing_pages for row in source.page_keys),
        logical_score_keys=extend_axis(source.logical_score_keys, 2, history, 0x8000),
        logical_score_valid=extend_axis(source.logical_score_valid, 2, history, False),
    )
    # Never ask ReplayPagePlan to mint DDR pages for the suffix. The production
    # runner indexes this array before masking invalid addresses to zero. An
    # existing page index behind that mask preserves registration and all KV bytes.
    bound_pages = copy(pages)
    bound_pages.geometry = g
    bound_pages.page_indices = extend_axis(pages.page_indices, 1, history // 128, 0)
    bound = copy(sequence)
    bound.cached_ranked_round = padded
    return bound, padded, bound_pages, proof
