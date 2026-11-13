# SPDX-License-Identifier: Apache-2.0
"""CPU input assembly for continuous replay of existing service score captures.

This layer supplies inputs, not a cache policy. Only the first captured hot/LRU
state initializes a run. Later Index layouts come from the executing path.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from trace_inputs import load_score_capture, select_series


@dataclass
class ReplayRound:
    source_round: int
    scheduler_round: int
    identities: tuple
    geometry: dict
    logical_keys: np.ndarray
    logical_scales: np.ndarray
    query: np.ndarray
    weights: np.ndarray
    query_scale: np.ndarray
    query_ends: np.ndarray
    tail_bounds: np.ndarray
    logical_address_valid: np.ndarray
    page_keys: tuple
    captured_layout: np.ndarray
    captured_hot_valid: np.ndarray
    captured_lru: np.ndarray
    source_paths: tuple
    logical_score_keys: np.ndarray
    logical_score_valid: np.ndarray

    def validate_topk(self, selected):
        selected = np.asarray(selected)
        if selected.shape != (self.geometry['batch'], 4, 2048) or selected.dtype.kind not in 'iu':
            raise AssertionError('Replay TopK shape/type differs from Q4 contract')
        for row in range(self.geometry['batch']):
            for query in range(4):
                ids = selected[row, query]
                if np.any(ids < 0) or np.any(ids >= self.geometry['history']):
                    raise AssertionError('Replay TopK logical token outside history')
                valid = self.logical_score_valid[row, query]
                scores = self.logical_score_keys[row, query]
                if len(np.unique(ids)) != 2048 or not np.all(valid[ids]):
                    raise AssertionError('Replay TopK duplicates or exceeds causal prefix')
                population = scores[valid]
                cutoff = np.partition(population, len(population) - 2048)[-2048]
                if np.min(scores[ids]) != cutoff or np.sum(scores[ids] > cutoff) != np.sum(population > cutoff):
                    raise AssertionError(f'Replay TopK ranking differs: request={row} Q={query + 1}')

    def index_for_layout(self, layout):
        """Map real LI input to a path's current layout without resetting residency."""
        layout = np.asarray(layout)
        batch, history = self.logical_scales.shape
        if (layout.shape != (batch, history) or layout.dtype.kind not in 'iu'
                or not np.all(np.sort(layout, axis=1) == np.arange(history))):
            raise ValueError('Replay Index layout must be a per-request token permutation')
        rows = np.arange(batch)[:, None]
        return (self.logical_keys[rows, layout], self.logical_scales[rows, layout],
                self.logical_address_valid[rows, layout])


def assemble_round(references):
    """Merge selected DPs; retain page aliasing within DP, separate DP arenas."""
    if not references:
        raise ValueError('Cannot assemble an empty replay round')
    source_round = references[0].source_round
    scheduler_round = references[0].scheduler_round
    if any(ref.source_round != source_round or ref.scheduler_round != scheduler_round for ref in references):
        raise ValueError('DP capture rounds are not aligned')
    identities, keys, scales, queries, weights, query_scales = [], [], [], [], [], []
    ends, tails, valid, pages, layouts, hot, lru, sources = [], [], [], [], [], [], [], []
    score_keys, score_valid = [], []
    geometry = None
    seen_ranks = set()
    for ref in references:
        if ref.dp_rank in seen_ranks:
            raise ValueError('Duplicate DP in a replay round')
        seen_ranks.add(ref.dp_rank)
        capture = load_score_capture(ref.manifest_path)
        arrays, source_geometry = capture.arrays, capture.manifest['geometry']
        current = {name: source_geometry[name] for name in
                   ('history', 'resident', 'heads', 'token_bytes', 'output_stride')}
        if geometry is None:
            geometry = current
        if current != geometry:
            raise ValueError('Selected DPs have incompatible replay geometry')
        if len(ref.request_signature) != source_geometry['batch']:
            raise ValueError('Request signature size differs from captured batch')
        for row, (request, owner) in enumerate(ref.request_signature):
            if int(arrays['pre.row_owners'][row]) != owner:
                raise ValueError('Captured row owner differs from series request identity')
            identities.append((ref.dp_rank, request, owner))
            logical_keys, logical_scales = capture.logical_keys(row)
            keys.append(logical_keys)
            scales.append(logical_scales)
            logical_valid = np.empty(source_geometry['history'], dtype=np.uint8)
            logical_valid[arrays['pre.logical_ids'][row]] = arrays['pre.address_valid'][row]
            valid.append(logical_valid)
            ordered_scores = np.empty_like(arrays['post.score_keys'][row])
            ordered_valid = np.zeros(ordered_scores.shape, dtype=bool)
            order = arrays['pre.logical_ids'][row]
            for query in range(4):
                ordered_scores[query, order] = arrays['post.score_keys'][row, query]
                ordered_valid[query, order[:arrays['pre.query_ends'][row, query]]] = True
            score_keys.append(ordered_scores)
            score_valid.append(ordered_valid)
            # Capture page slots are logical pages. Do not renumber per request:
            # repeated slots are a shared prefix, but same slot in another DP is not.
            slots = arrays['pre.ddr_page_slots'][row]
            page_valid = logical_valid.reshape(-1, 128)
            if (np.any(page_valid != page_valid[:, :1])
                    or not np.array_equal(slots >= 0, page_valid[:, 0] != 0)
                    or np.any(slots < -1)):
                raise ValueError('Published page slots differ from logical address validity')
            pages.append(tuple(None if slot < 0 else (ref.dp_rank, int(slot)) for slot in slots))
        queries.append(arrays['real.query'])
        weights.append(arrays['real.weights'])
        query_scales.append(arrays['real.query_scale'])
        ends.append(arrays['pre.query_ends'])
        tails.append(arrays['pre.tail_bounds'])
        layouts.append(arrays['pre.logical_ids'])
        hot.append(arrays['pre.hot_valid'])
        lru.append(arrays['pre.lru_slots'])
        sources.append(str(ref.manifest_path))
    if len(identities) > 16 or len(set(identities)) != len(identities):
        raise ValueError('Replay requires distinct requests and batch <= 16')
    return ReplayRound(
        source_round, scheduler_round, tuple(identities), dict(geometry, batch=len(identities)),
        np.stack(keys), np.stack(scales), np.concatenate(queries), np.concatenate(weights),
        np.concatenate(query_scales), np.concatenate(ends), np.concatenate(tails),
        np.stack(valid), tuple(pages), np.concatenate(layouts), np.concatenate(hot),
        np.concatenate(lru), tuple(sources), np.stack(score_keys), np.stack(score_valid),
    )


class ReplaySequence:
    """Lazy complete-sequence loader, repeatable without retaining every Q/Index."""
    def __init__(self, root, group, round_range):
        snapshot = Path(root) / 'manifest.json'
        if snapshot.is_file() and json.loads(snapshot.read_text()).get('kind') == 'lifetch_legacy_replay_snapshot':
            if tuple(round_range) != (1, 1):
                raise ValueError('Legacy snapshots are independent: use --round-range 1:1')
            capture = load_score_capture(snapshot)
            if capture.manifest['metadata']['group'] != group:
                raise ValueError('Requested group differs from legacy snapshot')
            self.rounds = self.from_snapshot(snapshot).rounds
            return
        references = select_series(root, group, round_range)
        self.rounds = tuple(tuple(ref for ref in references if ref.source_round == number)
                            for number in range(round_range[0], round_range[1] + 1))

    @classmethod
    def from_snapshot(cls, path):
        from trace_inputs import CaptureReference
        capture = load_score_capture(path)
        metadata = capture.manifest['metadata']
        signature = tuple((item['request_id'], item['owner']) for item in metadata['requests'])
        reference = CaptureReference(metadata['dp_rank'], 1, metadata['context']['round'],
                                     metadata['group'], capture.path, signature)
        result = cls.__new__(cls)
        result.rounds = ((reference,),)
        return result

    def __iter__(self):
        previous = None
        for references in self.rounds:
            current = assemble_round(references)
            if previous is not None:
                if current.identities != previous.identities or current.geometry != previous.geometry:
                    raise ValueError('Request identity/geometry changed within replay sequence')
                if np.any(current.query_ends < previous.query_ends):
                    raise ValueError('Query causal bounds regressed within replay sequence')
                for before, after in zip(previous.page_keys, current.page_keys):
                    if any(old is not None and old != new for old, new in zip(before, after)):
                        raise ValueError('Committed DDR page identity changed within replay sequence')
            yield current
            previous = current


class ReplayPagePlan:
    """Stable synthetic KV identities for real captured page sharing.

    Real captures do not export DDR KV bytes. Use deterministic C8 records and
    state this in results. A page that starts in ABC and is published later
    keeps the same bytes; aliases within a DP share bytes and a Put key.
    """
    def __init__(self, sequence):
        parents = {}
        def root(key):
            parents.setdefault(key, key)
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key
        def join(left, right):
            left, right = root(left), root(right)
            if left != right:
                parents[right] = left
        identities = None
        geometry = None
        for inputs in sequence:
            if identities is None:
                identities, geometry = inputs.identities, inputs.geometry
            if inputs.identities != identities or inputs.geometry != geometry:
                raise ValueError('Replay page plan requires stable request geometry')
            for identity, pages in zip(inputs.identities, inputs.page_keys):
                for logical_page, shared in enumerate(pages):
                    key = ('logical', *identity, logical_page)
                    root(key)
                    if shared is not None:
                        join(key, ('published', *shared))
        if identities is None:
            raise ValueError('Replay page plan requires at least one round')
        self.identities, self.geometry = identities, geometry
        groups = {}
        rows = []
        for identity in identities:
            row = []
            for page in range(geometry['history'] // 128):
                representative = root(('logical', *identity, page))
                if representative not in groups:
                    groups[representative] = len(groups)
                row.append(groups[representative])
            rows.append(row)
        self.page_indices = np.asarray(rows, dtype=np.int32)
        self.page_count = len(groups)

    def page_payload(self, page_indices):
        from contract_payload import payload
        indices = np.asarray(page_indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.page_count):
            raise ValueError('Synthetic KV page identity outside replay plan')
        tokens = (indices[:, None] * 128 + np.arange(128)).reshape(-1)
        width = self.geometry['token_bytes']
        if width not in (656, 2624):
            raise ValueError('Replay requires supported packed C8 token width')
        return np.concatenate([payload(layer, tokens) for layer in range(width // 656)], axis=1)

    def token_payload(self, row, logical_ids):
        from contract_payload import payload
        logical_ids = np.asarray(logical_ids, dtype=np.int64)
        if not 0 <= row < len(self.identities) or np.any(logical_ids < 0) or np.any(logical_ids >= self.geometry['history']):
            raise ValueError('Replay payload request outside captured history')
        tokens = self.page_indices[row, logical_ids // 128] * 128 + logical_ids % 128
        width = self.geometry['token_bytes']
        return np.concatenate([payload(layer, tokens) for layer in range(width // 656)], axis=1)
