# SPDX-License-Identifier: Apache-2.0
"""Host-side input restore for the three single-round feature ablation arms."""
from time import perf_counter

import numpy as np


class CheckedHotPayloads:
    """Serve expected KV bytes from the independently verified immutable host fixture."""

    def __init__(self, sample, pages, hot, stats):
        self.sample = sample
        self.pages = pages
        self.hot = hot
        self.stats = stats
        self.slots = {}

    def token_payload(self, row, ids):
        ids = np.asarray(ids)
        layout = self.sample.captured_layout
        if (row not in self.hot or ids.ndim != 1 or ids.dtype.kind not in 'iu' or
                np.any(ids < 0) or np.any(ids >= layout.shape[1])):
            return self.pages.token_payload(row, ids)
        values = self.hot[row]
        if row not in self.slots:
            hot_ids = layout[row, :self.sample.geometry['resident']]
            # Recompute the original reference, separately from the reset that
            # produced values. Never read expected bytes from device storage.
            if not np.array_equal(values, self.pages.token_payload(row, hot_ids)):
                raise ValueError(f'Initial hot KV reference differs from the page plan: row={row}')
            slots = np.full(layout.shape[1], -1, dtype=np.int32)
            slots[hot_ids] = np.arange(len(hot_ids), dtype=np.int32)
            self.slots[row] = slots
        offsets = self.slots[row][ids]
        missing = offsets < 0
        count = int(np.count_nonzero(missing))
        self.stats['verified_hot_tokens_reused'] += len(ids) - count
        self.stats['fresh_expected_tokens'] += count
        if not count:
            return values[offsets]
        if count == len(ids):
            return self.pages.token_payload(row, ids)
        expected = np.empty((len(ids), *values.shape[1:]), dtype=values.dtype)
        expected[~missing] = values[offsets[~missing]]
        expected[missing] = self.pages.token_payload(row, ids[missing])
        return expected


class FeatureInputRestore:
    def __init__(self, sample, pages):
        self.sample = sample
        self.pages = pages
        self.index = None
        self.hot = {}
        self.stats = dict(full_resets=0, fresh_replica_resets_skipped=0, full_prepares=0,
                          duplicate_prepares_skipped=0, skipped_upload_bytes_lower_bound=0,
                          initial_index_gathers=0, initial_index_gathers_reused=0,
                          initial_hot_payloads_built=0, initial_hot_payloads_reused=0,
                          verified_hot_tokens_reused=0, fresh_expected_tokens=0,
                          full_reset_seconds=0.0, full_prepare_seconds=0.0)
        self.checked_pages = CheckedHotPayloads(sample, pages, self.hot, self.stats)

    def index_for_layout(self, layout):
        # Native prepare has read back the actual per-state device layout. Only
        # use the host cache for the exact captured permutation just restored.
        if not np.array_equal(layout, self.sample.captured_layout):
            return self.sample.index_for_layout(layout)
        if self.index is None:
            self.index = self.sample.index_for_layout(layout)
            self.stats['initial_index_gathers'] += 1
        else:
            self.stats['initial_index_gathers_reused'] += 1
        return self.index

    def __getattr__(self, name):
        return getattr(self.sample, name)

    def token_payload(self, row, ids):
        resident = self.sample.geometry['resident']
        if not np.array_equal(ids, self.sample.captured_layout[row, :resident]):
            return self.pages.token_payload(row, ids)
        if row not in self.hot:
            self.hot[row] = self.pages.token_payload(row, ids)
            self.stats['initial_hot_payloads_built'] += 1
        else:
            self.stats['initial_hot_payloads_reused'] += 1
        return self.hot[row]

    def install(self, execution):
        fixture = self
        stats = self.stats

        class FeatureForward(execution.StreamingForward):
            clean_round = None

            def __call__(self, *args, **kwargs):
                # Eager and graph capture run this hook. Every captured graph
                # replay follows an explicit prepare, which consumes the marker.
                self.clean_round = None
                return super().__call__(*args, **kwargs)

        class FeatureState(execution.FusedReplayState):
            _inside_reset = False
            _count_uploads = False

            def upload(self, target, array):
                if self._count_uploads:
                    self._upload_bytes += array.nbytes
                return super().upload(target, array)

            def reset(self):
                if self.initial is not fixture.sample or self.pages is not fixture.pages:
                    self.forward.clean_round = None
                    return super().reset()
                if self.forward.clean_round is fixture.sample:
                    # The first batch reset visits freshly constructed replicas.
                    # Their constructor already completed this same full reset.
                    stats['fresh_replica_resets_skipped'] += 1
                    stats['skipped_upload_bytes_lower_bound'] += self._reset_upload_bytes
                    return
                self.forward.clean_round = None
                self._inside_reset = self._count_uploads = True
                self._upload_bytes = 0
                start = perf_counter()
                try:
                    # Native reset uses pages only for token_payload. Restore
                    # the original page plan before any launch or validation.
                    self.pages = fixture
                    super().reset()
                finally:
                    self.pages = fixture.pages
                    self._inside_reset = self._count_uploads = False
                self._reset_upload_bytes = self._upload_bytes
                self.forward.clean_round = fixture.sample
                stats['full_resets'] += 1
                stats['full_reset_seconds'] += perf_counter() - start

            def prepare(self, inputs):
                previous = self.forward.clean_round
                self.forward.clean_round = None
                if inputs is fixture.sample and previous is inputs and not self._inside_reset:
                    # Native reset ends in prepare(initial). The next call, when
                    # adjacent, imports exactly the same immutable single round.
                    stats['duplicate_prepares_skipped'] += 1
                    stats['skipped_upload_bytes_lower_bound'] += self._prepare_upload_bytes
                    return
                start = perf_counter()
                if self._inside_reset and inputs is fixture.sample:
                    before = self._upload_bytes
                    super().prepare(fixture)
                    self._prepare_upload_bytes = self._upload_bytes - before
                else:
                    super().prepare(inputs)
                stats['full_prepares'] += 1
                stats['full_prepare_seconds'] += perf_counter() - start

            def check_completion(self):
                self.forward.clean_round = None
                return super().check_completion()

            def validate(self, inputs):
                self.forward.clean_round = None
                if inputs is not fixture.sample or self.pages is not fixture.pages:
                    return super().validate(inputs)
                try:
                    self.pages = fixture.checked_pages
                    return super().validate(inputs)
                finally:
                    self.pages = fixture.pages

        execution.StreamingForward = FeatureForward
        execution.FusedReplayState = FeatureState
