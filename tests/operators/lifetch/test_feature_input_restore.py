# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / 'support'))
from feature_input_restore import FeatureInputRestore


class Sample:
    geometry = {'resident': 3}

    def __init__(self, bias=0):
        self.captured_layout = np.array([[3, 1, 2, 0]], dtype=np.int32)
        self.keys = np.arange(4, dtype=np.int32)[None, :] + bias

    def index_for_layout(self, layout):
        return np.take_along_axis(self.keys, layout, axis=1)


class Pages:
    def token_payload(self, row, ids):
        return ids + 100 * row


def test_expected_hot_kv_matches_official_pages_for_every_selected_byte():
    class BytePages:
        def token_payload(self, row, ids):
            return ((np.asarray(ids)[:, None]*17 + np.arange(656) + row) % 251).astype(np.uint8)

    fixture = FeatureInputRestore(Sample(), BytePages())
    fixture.token_payload(0, np.array([3, 1, 2]))
    for ids in ([2, 0, 3, 0, 1], [0], [3, 2], [3, 1, 0]):
        ids = np.array(ids)
        expected = fixture.checked_pages.token_payload(0, ids)
        actual = fixture.pages.token_payload(0, ids)
        np.testing.assert_array_equal(expected, actual)
        actual[-1, 655] ^= 1
        assert not np.array_equal(expected, actual)
    assert fixture.stats['verified_hot_tokens_reused'] == 7
    assert fixture.stats['fresh_expected_tokens'] == 4

    damaged = FeatureInputRestore(Sample(), BytePages())
    damaged.token_payload(0, np.array([3, 1, 2]))[-1, 655] ^= 1
    with pytest.raises(ValueError, match='reference differs'):
        damaged.checked_pages.token_payload(0, np.array([1]))


def setup():
    class Forward:
        def __init__(self, state):
            self.state = state

        def __call__(self):
            self.state.layout[:] = self.state.layout[:, ::-1].copy()
            self.state.hot.fill(999)
            self.state.keys.fill(-1)

    class State:
        def __init__(self, initial, pages):
            self.initial, self.pages = initial, pages
            self.forward = module.StreamingForward(self)
            self.layout = np.zeros((1, 4), dtype=np.int32)
            self.hot = np.zeros(3, dtype=np.int32)
            self.keys = np.zeros((1, 4), dtype=np.int32)
            self.reset()

        def upload(self, target, array):
            target[:] = array

        def reset(self):
            self.upload(self.layout, self.initial.captured_layout)
            self.upload(self.hot, self.pages.token_payload(0, self.layout[0, :3]))
            self.prepare(self.initial)

        def prepare(self, sample):
            self.upload(self.keys, sample.index_for_layout(self.layout))

        def check_completion(self):
            return True

        def validate(self, inputs):
            return inputs

    module = SimpleNamespace(StreamingForward=Forward, FusedReplayState=State)
    sample, pages = Sample(), Pages()
    fixture = FeatureInputRestore(sample, pages)
    fixture.install(module)
    return fixture, module.FusedReplayState


def test_only_adjacent_identical_round_can_skip_and_device_state_is_private():
    fixture, cls = setup()
    s, p = fixture.sample, fixture.pages
    a, b = cls(s, p), cls(s, p)
    a.reset()
    a.prepare(s)
    assert fixture.stats['fresh_replica_resets_skipped'] == 1
    assert fixture.stats['duplicate_prepares_skipped'] == 1
    a.prepare(s)
    assert fixture.stats['full_prepares'] == 3
    a.forward()
    a.prepare(s)
    np.testing.assert_array_equal(a.keys, s.index_for_layout(a.layout))
    np.testing.assert_array_equal(b.layout, s.captured_layout)
    a.reset()
    np.testing.assert_array_equal(a.hot, s.captured_layout[0, :3])
    a.prepare(Sample(71))
    np.testing.assert_array_equal(a.keys, s.index_for_layout(s.captured_layout) + 71)
    a.validate(s)
    a.prepare(s)
    np.testing.assert_array_equal(a.keys, s.index_for_layout(s.captured_layout))
    assert fixture.stats['duplicate_prepares_skipped'] == 1
    assert fixture.stats['initial_index_gathers'] == fixture.stats['initial_hot_payloads_built'] == 1
    assert fixture.stats['initial_index_gathers_reused'] == fixture.stats['initial_hot_payloads_reused'] == 2
    assert fixture.stats['skipped_upload_bytes_lower_bound'] == (4 + 3 + 4 + 4) * 4
    assert a.pages is b.pages is p


def test_cache_falls_back_on_different_layout_tokens_page_plan_and_sample():
    fixture, cls = setup()
    s = fixture.sample
    cls(s, fixture.pages)
    changed = s.captured_layout[:, ::-1].copy()
    np.testing.assert_array_equal(fixture.index_for_layout(changed), s.index_for_layout(changed))
    np.testing.assert_array_equal(fixture.token_payload(0, changed[0, :3]), changed[0, :3])
    for other_sample, pages in ((Sample(), fixture.pages), (s, Pages())):
        state = cls(other_sample, pages)
        state.prepare(other_sample)
        state.reset()
        assert state.pages is pages
    assert fixture.stats['duplicate_prepares_skipped'] == 0
    assert fixture.stats['fresh_replica_resets_skipped'] == 0
    assert fixture.stats['initial_index_gathers_reused'] == 0


@pytest.mark.parametrize('boundary', ['forward', 'check_completion', 'validate'])
def test_any_launch_or_validation_boundary_invalidates_fresh_input(boundary):
    fixture, cls = setup()
    state = cls(fixture.sample, fixture.pages)
    method = getattr(state, boundary)
    method(fixture.sample) if boundary == 'validate' else method()
    state.prepare(fixture.sample)
    assert fixture.stats['duplicate_prepares_skipped'] == 0


def test_native_diagnose_protocol_recovers_every_mutation_each_replay():
    fixture, cls = setup()
    s, p = fixture.sample, fixture.pages
    states = [cls(s, p)]
    states[0].forward()
    states[0].validate(s)
    states.extend(cls(s, p) for _ in range(9))
    for state in states:
        state.reset()
    for state in states:
        state.forward()
        state.check_completion()
        state.reset()
    for state in states:
        state.forward()  # capture
    for stage in ['warmup'] * 4 + ['measured', 'detail']:
        for state in states:
            state.reset()
        for state in states:
            state.prepare(s)
            np.testing.assert_array_equal(state.hot, s.captured_layout[0, :3])
            np.testing.assert_array_equal(state.keys, s.index_for_layout(s.captured_layout))
            # Captured graph writes devices without a Python forward call.
            state.keys.fill(-1)
            state.hot.fill(-1)
            state.layout[:] = state.layout[:, ::-1].copy()
        for state in states:
            state.check_completion() if stage == 'warmup' else state.validate(s)
    assert fixture.stats['duplicate_prepares_skipped'] == 60
    assert fixture.stats['fresh_replica_resets_skipped'] == 9
    assert fixture.stats['full_resets'] == 81
    assert fixture.stats['full_prepares'] == 81
