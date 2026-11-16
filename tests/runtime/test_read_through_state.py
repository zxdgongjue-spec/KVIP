# SPDX-License-Identifier: Apache-2.0
"""CPU specification for non-promoting, low-route KV reads.

This models the proposed cache policy, not kernel execution or NPU timing.
Physical Index rows are a permutation; their resident prefix continues to
describe hot KV slots. Low rounds read cold KV into per-round scratch and do
not change this permutation. The normal promoting path may follow immediately.
"""

from copy import deepcopy
from dataclasses import dataclass
import random

import pytest


@dataclass(frozen=True)
class ReadPlan:
    generation: int
    slots: tuple
    reads: tuple
    scratch: dict


class ReadThroughOracle:
    """One stable owner's Index/hot/ABC state with a separate scratch span."""

    def __init__(self, history=128, resident=16, abc_slots=8, generation=1):
        self.generation = generation
        self.logical_ids = list(range(history))
        self.resident = resident
        self.abc_slots = abc_slots
        self.hot_valid = [True] * resident
        self.hot = [self.payload(i) for i in range(resident)]
        self.tail_begin = history - abc_slots
        self.tail_end = history

    def payload(self, logical):
        # Generation makes accidentally reusing another request's KV visible.
        return self.generation, logical

    @property
    def scratch_begin(self):
        return self.resident + self.abc_slots

    def protected_state(self):
        return deepcopy((self.logical_ids, self.hot, self.hot_valid,
                         self.tail_begin, self.tail_end, self.generation))

    def selected(self, logical_rows):
        inverse = {logical: physical for physical, logical in enumerate(self.logical_ids)}
        return [[None if token is None else inverse[token] for token in row]
                for row in logical_rows]

    def known_slot(self, physical):
        logical = self.logical_ids[physical]
        # ABC wins over a stale hot copy; modulo uses ABC capacity, never total
        # output stride. Adding scratch must not change this address mapping.
        if self.tail_begin <= logical < self.tail_end:
            if logical - self.tail_begin < self.abc_slots:
                return self.resident + logical % self.abc_slots
            return None
        if physical < self.resident and self.hot_valid[physical]:
            return physical
        return None

    def plan(self, selected, capacity=8192):
        if len(selected) > 4 or any(len(row) > 2048 for row in selected):
            raise ValueError("Q4/Top2048 contract exceeded")
        if any(token is not None and not 0 <= token < len(self.logical_ids)
               for row in selected for token in row):
            raise ValueError("Index position outside history")
        reads, scratch, assigned, slots = [], {}, {}, []
        for row in selected:
            output = []
            for physical in row:
                if physical is None:
                    output.append(None)
                    continue
                slot = self.known_slot(physical)
                if slot is None:
                    if physical not in assigned:
                        if len(reads) == capacity:
                            # An implementation may fall back before submitting
                            # writes; it must never wrap into hot/ABC slots.
                            raise OverflowError("Read-through scratch exhausted")
                        slot = self.scratch_begin + len(reads)
                        assigned[physical] = slot
                        reads.append(physical)
                        scratch[slot] = self.payload(self.logical_ids[physical])
                    slot = assigned[physical]
                output.append(slot)
            slots.append(tuple(output))
        return ReadPlan(self.generation, tuple(slots), tuple(reads), scratch)

    def materialize(self, plan):
        if plan.generation != self.generation:
            raise ValueError("Stale request generation")
        abc = {self.resident + logical % self.abc_slots: self.payload(logical)
               for logical in range(self.tail_begin,
                                    min(self.tail_end, self.tail_begin + self.abc_slots))}
        def value(slot):
            if slot is None:
                return None
            if slot < self.resident:
                assert self.hot_valid[slot]
                return self.hot[slot]
            if slot < self.scratch_begin:
                return abc[slot]
            return plan.scratch[slot]
        return tuple(tuple(value(slot) for slot in row) for row in plan.slots)

    def promote_high(self, logical_rows):
        """Reference high-round state transition; not its scheduling model."""
        selected = self.selected(logical_rows)
        plan = self.plan(selected)
        selected_hot = {slot for row in plan.slots for slot in row
                        if slot is not None and slot < self.resident}
        free = [slot for slot in range(self.resident) if slot not in selected_hot]
        if len(plan.reads) > len(free):
            raise OverflowError("Oracle high policy needs more hot slots")
        logical_reads = [self.logical_ids[source] for source in plan.reads]
        for logical, destination in zip(logical_reads, free):
            source = self.logical_ids.index(logical)
            self.logical_ids[source], self.logical_ids[destination] = (
                self.logical_ids[destination], self.logical_ids[source])
            self.hot[destination] = self.payload(logical)
            self.hot_valid[destination] = True
        self.assert_hot_index_consistent()
        return len(plan.reads)

    def assert_hot_index_consistent(self):
        assert sorted(self.logical_ids) == list(range(len(self.logical_ids)))
        assert all(not valid or self.hot[slot] == self.payload(self.logical_ids[slot])
                   for slot, valid in enumerate(self.hot_valid))


def assert_outputs(state, plan, logical_rows):
    expected = tuple(tuple(None if token is None else state.payload(token)
                           for token in row) for row in logical_rows)
    assert state.materialize(plan) == expected


def test_reserved_hot_stays_hit_abc_precedence_and_invalid_hot_reads():
    state = ReadThroughOracle()
    # A stale hot copy of an ABC token may coexist with the active ABC ring.
    state.logical_ids[3], state.logical_ids[123] = 123, 3
    state.hot[3] = (state.generation, -123)
    state.hot_valid[5] = False
    before = state.protected_state()
    rows = [[0, 5, 123, 30], [30, 5, None], [123, 30], [0, 5]]
    plan = state.plan(state.selected(rows))
    assert_outputs(state, plan, rows)
    assert plan.slots[0][0] == 0  # All valid hot, including LRU candidates, hit.
    assert plan.slots[0][2] == state.resident + 123 % state.abc_slots
    assert len(plan.reads) == 2  # Invalid hot5 and cold30, once across all Qs.
    assert state.protected_state() == before


def test_abc_modulo_does_not_include_scratch_and_retired_tail_is_cold():
    state = ReadThroughOracle(history=129)
    rows = [[120, 121, 128]]
    plan = state.plan(state.selected(rows))
    assert_outputs(state, plan, rows)
    assert plan.slots[0] == (24, 17, 16)
    assert len(plan.reads) == 1
    # Enlarging the total output to hold 8192 misses must not change ring slots.
    assert plan.slots[0][1:] != (16 + 121 % 8200, 16 + 128 % 8200)


def test_all_cold_wrong_low_prediction_has_q4_capacity_without_overwrite():
    state = ReadThroughOracle(history=10000, resident=16, abc_slots=512)
    rows = [list(range(16 + q * 2048, 16 + (q + 1) * 2048)) for q in range(4)]
    before = state.protected_state()
    plan = state.plan(state.selected(rows))
    assert len(plan.reads) == 8192
    assert len(plan.scratch) == 8192
    assert min(plan.scratch) == state.scratch_begin
    assert max(plan.scratch) == state.scratch_begin + 8191
    assert_outputs(state, plan, rows)
    assert state.protected_state() == before
    with pytest.raises(OverflowError, match="exhausted"):
        state.plan(state.selected(rows), capacity=512)
    assert state.protected_state() == before


def test_low_high_low_has_no_index_repair_or_stale_scratch_hit():
    state = ReadThroughOracle()
    low = [[0, 30, 121], [2, 30, 31], [30], [31]]
    before = state.protected_state()
    first = state.plan(state.selected(low))
    assert_outputs(state, first, low)
    assert state.protected_state() == before
    state.assert_hot_index_consistent()
    # Same IDs miss again: scratch is per-round storage, not a promoted cache.
    second = state.plan(state.selected(low))
    assert len(second.reads) == len(first.reads) == 2
    assert state.promote_high(low) == 2
    after_high = state.protected_state()
    third = state.plan(state.selected(low))
    assert not third.reads
    assert_outputs(state, third, low)
    assert state.protected_state() == after_high


def test_request_generation_invalidates_plan_and_new_payload():
    state = ReadThroughOracle()
    rows = [[30, 31], [30], [], []]
    old_plan = state.plan(state.selected(rows))
    replacement = ReadThroughOracle(generation=state.generation + 1)
    with pytest.raises(ValueError, match="generation"):
        replacement.materialize(old_plan)
    new_plan = replacement.plan(replacement.selected(rows))
    assert len(new_plan.reads) == 2
    assert old_plan.scratch != new_plan.scratch
    assert_outputs(replacement, new_plan, rows)


def test_lifetime_requires_previous_attention_to_finish_before_scratch_reuse():
    state = ReadThroughOracle()
    first = state.plan(state.selected([[30]]))
    second = state.plan(state.selected([[31]]))
    assert first.slots == second.slots
    assert first.scratch != second.scratch
    # Identical graph addresses are intentional. A real shared arena must be
    # ordered after first-round attention, not replayed concurrently.
    assert first.scratch[state.scratch_begin] == (1, 30)
    assert second.scratch[state.scratch_begin] == (1, 31)


def test_temporal_traffic_is_counted_not_inferred_from_one_snapshot():
    nonpromoting, promoting = ReadThroughOracle(), ReadThroughOracle()
    rows = [[0, 30, 31], [30, 32], [31], [32]]
    retained_reads = sum(len(nonpromoting.plan(nonpromoting.selected(rows)).reads)
                         for _ in range(8))
    promoted_reads = sum(promoting.promote_high(rows) for _ in range(8))
    assert retained_reads == 24
    assert promoted_reads == 3
    # Nonpromotion can also preserve an older hot token which promotion
    # evicts. It is a policy change, not a universally free optimization.
    assert not nonpromoting.plan(nonpromoting.selected([[1]])).reads
    assert promoting.plan(promoting.selected([[1]])).reads


def test_randomized_permuted_index_and_q4_union_are_exact():
    randomizer = random.Random(40716)
    state = ReadThroughOracle(history=256, resident=64, abc_slots=16)
    for step in range(80):
        hot = randomizer.sample(state.logical_ids[:state.resident], 12)
        cold = randomizer.sample(state.logical_ids[state.resident:240], 8)
        rows = [randomizer.sample(hot + cold + [250], 16) for _ in range(4)]
        before = state.protected_state()
        selected = state.selected(rows)
        plan = state.plan(selected)
        assert_outputs(state, plan, rows)
        expected_cold = {token for row in selected for token in row
                         if state.known_slot(token) is None}
        assert set(plan.reads) == expected_cold
        assert len(plan.reads) == len(expected_cold)
        assert state.protected_state() == before
        if step % 3 == 2:
            state.promote_high(rows)
        state.assert_hot_index_consistent()


def test_padding_and_invalid_positions_never_write_state():
    state = ReadThroughOracle()
    before = state.protected_state()
    plan = state.plan([[None] * 2048] * 4)
    assert not plan.reads and not plan.scratch
    assert all(slot is None for row in plan.slots for slot in row)
    for token in (-1, len(state.logical_ids)):
        with pytest.raises(ValueError, match="outside history"):
            state.plan([[token]])
    assert state.protected_state() == before
