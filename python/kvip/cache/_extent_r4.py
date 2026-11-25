# SPDX-License-Identifier: Apache-2.0
"""Physical-slot frequency scoring for R4 extent eviction."""

import numpy as np
from numba import njit

HALF_LIFE = 64
GRACE_ROUNDS = 2
DECAY_LOOKUP_ROUNDS = 1024
DECAY_LOOKUP = np.exp2(-np.arange(DECAY_LOOKUP_ROUNDS, dtype=np.float64) / HALF_LIFE)
DECAY_LOOKUP.flags.writeable = False


@njit(inline="always")
def decay(age):
    if age < DECAY_LOOKUP_ROUNDS:
        return DECAY_LOOKUP[age]
    return np.exp2(-age / HALF_LIFE)


@njit(nogil=True)
def touch(feedback, count, last, born, victims, extent, round_index, seen):
    seen.fill(0)
    for slot in feedback:
        if seen[slot]:
            continue
        seen[slot] = 1
        count[slot] = count[slot] * decay(round_index - last[slot]) + 1
        last[slot] = round_index
        if (np.uint32(victims[slot >> 5]) >> (slot & 31)) & 1:
            born[slot // extent] = round_index


@njit(nogil=True)
def scores(count, last, born, active, extent, round_index):
    blocks = active // extent
    score = np.zeros(blocks, dtype=np.float64)
    recency = np.full(blocks, -1, dtype=np.int64)
    for block in range(blocks):
        start = block * extent
        for slot in range(start, start + extent):
            stamp = last[slot]
            if stamp >= 0:
                score[block] = max(score[block], count[slot] * decay(round_index - stamp))
                recency[block] = max(recency[block], stamp)
        if round_index - born[block] < GRACE_ROUNDS:
            score[block] = np.inf
    return score, recency


@njit(nogil=True)
def write_victims(order, extent, slots, bitmap):
    bitmap.fill(0)
    for index, block in enumerate(order):
        for offset in range(extent):
            slot = block * extent + offset
            slots[index * extent + offset] = slot
            bitmap[slot >> 5] |= np.int32(np.uint32(1) << (slot & 31))


class ExtentR4:
    def __init__(self, extent):
        self.extent = extent
        self.round = 0
        self.count = np.empty(0, dtype=np.float32)
        self.last = np.empty(0, dtype=np.int64)
        self.born = np.empty(0, dtype=np.int64)
        self.seen = np.empty(0, dtype=np.uint8)

    def resize(self, capacity):
        extra = capacity - self.count.size
        if extra <= 0:
            return
        blocks = (capacity + self.extent - 1) // self.extent
        self.count = np.pad(self.count, (0, extra))
        self.last = np.pad(self.last, (0, extra), constant_values=-1)
        self.born = np.pad(self.born, (0, blocks - self.born.size), constant_values=-GRACE_ROUNDS)
        self.seen = np.empty(capacity, dtype=np.uint8)

    def reset(self):
        self.round = 0
        self.count.fill(0)
        self.last.fill(-1)
        self.born.fill(-GRACE_ROUNDS)

    def commit(self, feedback, bitmap, aligned):
        self.round += 1
        touch(
            feedback,
            self.count,
            self.last,
            self.born,
            bitmap,
            self.extent,
            self.round,
            self.seen,
        )
        if not aligned:
            self.born.fill(-GRACE_ROUNDS)

    def select(self, active, count, slots, bitmap):
        score, recency = scores(self.count, self.last, self.born, active, self.extent, self.round)
        order = np.lexsort((recency, score))[: count // self.extent]
        write_victims(order, self.extent, slots, bitmap)
