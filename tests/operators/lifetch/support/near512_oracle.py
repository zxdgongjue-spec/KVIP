# SPDX-License-Identifier: Apache-2.0
"""Independent checks for the full-length Q4 replay, outside all graph timers."""
from types import SimpleNamespace

import numpy as np


def _native_selector(inputs, torch, torch_npu, device):
    from execution import native_li
    from native_li_reference import NativeReplayLI

    def select(keys, scales, ends, sparse_mode):
        geometry = {**inputs.geometry, 'history': keys.shape[1], 'sparse_mode': sparse_mode}
        operands = NativeReplayLI(torch, device, geometry)
        operands.prepare(SimpleNamespace(logical_keys=keys, logical_scales=scales,
                                        query=inputs.query, query_scale=inputs.query_scale,
                                        weights=inputs.weights, query_ends=ends))
        selected = native_li(torch_npu, operands.fixture).cpu().numpy().reshape(*ends.shape, 2048)
        if (selected.dtype.kind not in 'iu' or np.any(selected < 0) or
                np.any(selected >= ends[..., None]) or
                np.any(np.diff(np.sort(selected, axis=-1), axis=-1) == 0)):
            raise ValueError('Stock LI produced invalid or repeated candidates')
        return selected.astype(np.int32, copy=False)
    return select


def _select_ids(inputs, select, ids, lengths, q):
    """Gather a different shortlist per request. Padded IDs lie outside actual_key."""
    batch = len(ids)
    physical = (int(lengths.max()) + 127) // 128 * 128
    padded = np.zeros((batch, physical), dtype=np.int32)
    for row, length in enumerate(lengths):
        padded[row, :length] = ids[row, :length]
    rows = np.arange(batch)[:, None]
    ends = np.broadcast_to(lengths[:, None], (batch, 4))
    local = select(inputs.logical_keys[rows, padded], inputs.logical_scales[rows, padded], ends, 0)[:, q]
    return np.take_along_axis(padded, local, axis=1)


def _sharded_reference(inputs, select, k=2048, chunk=65536):
    # The installed strong-serial operator was previously used for 128K histories.
    # A token omitted from a shard's TopK cannot strictly beat the global Kth score.
    # Run the same independent operator on the shortlist to resolve the global cutoff.
    ends = inputs.query_ends
    batch = len(ends)
    history = inputs.logical_keys.shape[1]
    if (ends.dtype.kind not in 'iu' or ends.shape != (batch, 4) or np.any(ends < k)
            or np.any(ends > history) or history % 128 or chunk % 128 or chunk < k):
        raise ValueError('Invalid full-length causal domains or reference shard size')
    # Signed subtraction also handles inactive fragments in short rows.
    ends = ends.astype(np.int64, copy=False)
    contributions = []
    for begin in range(0, int(ends.max()), chunk):
        stop = min(history, begin + chunk)
        visible = np.clip(ends - begin, 0, stop-begin).astype(np.int32)
        shard_ids = np.full((batch, 4, k), history, np.int32)
        if visible.max() > k:
            keys = inputs.logical_keys[:, begin:stop]
            scales = inputs.logical_scales[:, begin:stop]
            if np.all(visible == visible[:, :1]) and visible.min() >= k:
                shard_ids[:] = begin + select(keys, scales, visible, 0)
            elif np.all(np.diff(visible, axis=1) == 1) and visible.min() >= k:
                shard_ids[:] = begin + select(keys, scales, visible, 3)
            else:
                for q in range(4):
                    if visible[:, q].max() > k:
                        safe = np.maximum(visible[:, q], k)
                        all_q = np.broadcast_to(safe[:, None], visible.shape)
                        shard_ids[:, q] = begin + select(keys, scales, all_q, 0)[:, q]
        for row, q in np.argwhere(visible <= k):
            count = visible[row, q]
            shard_ids[row, q].fill(history)
            shard_ids[row, q, :count] = np.arange(begin, begin + count)
        contributions.append(shard_ids)
    ids = np.sort(np.concatenate(contributions, axis=-1), axis=-1)
    ref = np.empty((batch, 4, k), np.int32)
    for q in range(4):
        lengths = np.count_nonzero(ids[:, q] < history, axis=1).astype(np.int32)
        ref[:, q] = _select_ids(inputs, select, ids[:, q], lengths, q)
    return ref


def serial_reference(inputs, torch, torch_npu, device):
    mode = inputs.geometry.get('sparse_mode', 3)
    if mode not in (0, 3) or not np.all(np.diff(inputs.query_ends, axis=1) == int(mode == 3)):
        raise ValueError('Strong serial requires causal consecutive Q4 ends or equal noncausal ends')
    return _sharded_reference(inputs, _native_selector(inputs, torch, torch_npu, device))


def _certify_sets(inputs, selected, reference, select, verified):
    ends = inputs.query_ends
    if (selected.shape != reference.shape or selected.dtype.kind not in 'iu'
            or np.any(selected < 0) or np.any(selected >= ends[..., None])):
        raise AssertionError('Invalid candidate TopK domains')
    ranked = np.sort(selected, axis=-1)
    if np.any(np.diff(ranked, axis=-1) == 0):
        raise AssertionError('Candidate TopK contains duplicates')
    expected = np.sort(reference, axis=-1)
    batch, _, k = ranked.shape
    for q in range(4):
        unchecked = []
        union = np.zeros((batch, 2*k), np.int32)
        lengths = np.full(batch, k, np.int32)
        for row in range(batch):
            chosen = ranked[row, q]
            union[row, :k] = chosen
            if np.array_equal(chosen, expected[row, q]):
                continue
            key = (row, q, chosen.tobytes())
            if key in verified:
                continue
            missing = np.setdiff1d(expected[row, q], chosen, assume_unique=True)
            union[row, k:k + len(missing)] = missing
            lengths[row] += len(missing)
            unchecked.append((row, key))
        if unchecked:
            # The stock operator prefers earlier positions on equal scores. Put the
            # candidate IDs first. A strict-lower-score substitution can never pass
            # this rerank even if the tie order changes in a future vendor release.
            accepted = np.sort(_select_ids(inputs, select, union, lengths, q), axis=-1)
            for row, key in unchecked:
                if not np.array_equal(accepted[row], ranked[row, q]):
                    raise AssertionError(f'Native cutoff equivalence not proved for row {row}, Q{q+1}')
                verified[key] = True


def validate_native_sets(inputs, selected, reference, torch, torch_npu, device, verified):
    _certify_sets(inputs, selected, reference, _native_selector(inputs, torch, torch_npu, device), verified)
