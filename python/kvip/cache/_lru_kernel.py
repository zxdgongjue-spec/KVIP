# SPDX-License-Identifier: Apache-2.0
"""Unmodified Numba kernels from mtp_xzw_0903 pre_victim_lru.py."""

import numpy as np
from numba import njit, prange

BITMAP_WORD_BITS = 32


@njit(inline="always")
def _single_bit_index(bit_value):
    """返回一个正 32-bit 整数中唯一置位 bit 的下标。"""

    index = 0
    if bit_value >= (1 << 16):
        bit_value >>= 16
        index += 16
    if bit_value >= (1 << 8):
        bit_value >>= 8
        index += 8
    if bit_value >= (1 << 4):
        bit_value >>= 4
        index += 4
    if bit_value >= (1 << 2):
        bit_value >>= 2
        index += 2
    if bit_value >= (1 << 1):
        index += 1
    return index


@njit(inline="always")
def _write_oldest_victims(
    row,
    next_slot,
    queue_head,
    victim_slots,
    victim_bitmap,
    victim_count,
    bitmap_words,
):
    """清空完整 ABI bitmap，并缓存链头开始的 victim。"""

    for word in range(bitmap_words):
        victim_bitmap[row, word] = 0

    slot = int(queue_head[row])
    for index in range(victim_count):
        victim_slots[row, index] = slot
        word = slot >> 5
        bit = slot & 31
        victim_bitmap[row, word] = np.int32(np.uint32(victim_bitmap[row, word]) | (np.uint32(1) << np.uint32(bit)))
        slot = int(next_slot[row, slot])


@njit(nogil=True)
def _rebuild_extent_order(next_slot, head, extent, seen, links, ends):
    """Project slot recency onto the newest member of each complete extent."""
    seen.fill(0)
    slot = head[0]
    previous = -1
    ends[0] = -1
    while slot >= 0:
        block = slot // extent
        seen[block] += 1
        if seen[block] == extent:
            links[1, block] = previous
            if previous >= 0:
                links[0, previous] = block
            else:
                ends[0] = block
            previous = block
        slot = next_slot[0, slot]
    ends[1] = previous
    if previous >= 0:
        links[0, previous] = -1


@njit(nogil=True)
def _write_extent_victims(row, slots, bitmap, count, extent, links, ends):
    bitmap[row].fill(0)
    block = ends[0]
    for written in range(0, count, extent):
        start = block * extent
        for offset in range(extent):
            slots[row, written + offset] = start + offset
        bitmap[row, start >> 5] |= np.int32(((np.uint32(1) << extent) - 1) << (start & 31))
        block = links[0, block]


@njit(inline="always")
def _touch_extent(block, links, ends):
    tail = ends[1]
    if block == tail:
        return
    previous = links[1, block]
    following = links[0, block]
    if previous < 0:
        ends[0] = following
    else:
        links[0, previous] = following
    links[1, following] = previous
    links[0, tail] = block
    links[1, block] = tail
    links[0, block] = -1
    ends[1] = block


@njit(parallel=True, nogil=True)
def _reset_lru_rows(
    flat_rows,
    active_capacities,
    next_slot,
    previous_slot,
    queue_head,
    queue_tail,
    victim_slots,
    victim_bitmap,
    unique_counts,
    victim_count,
    bitmap_words,
):
    """把指定稳定行恢复为按 slot ID 升序的冷 LRU。"""

    capacity = next_slot.shape[1]
    for selected_index in prange(flat_rows.shape[0]):
        row = int(flat_rows[selected_index])
        active = int(active_capacities[row])
        for slot in range(capacity):
            if slot < active:
                previous_slot[row, slot] = slot - 1
                if slot + 1 < active:
                    next_slot[row, slot] = slot + 1
                else:
                    next_slot[row, slot] = -1
            else:
                previous_slot[row, slot] = -1
                next_slot[row, slot] = -1
        queue_head[row] = 0
        queue_tail[row] = active - 1
        unique_counts[row] = 0
        # victim_slots 和 victim_bitmap 也要清空，避免 commit 时误用旧数据。
        _write_oldest_victims(
            row,
            next_slot,
            queue_head,
            victim_slots,
            victim_bitmap,
            victim_count,
            bitmap_words,
        )


@njit(nogil=True)
def _feedback_slots_in_range(destination_slots, feedback_offsets, active_capacities):
    """在修改任意 LRU 行之前检查所有 packed feedback，保证失败原子性。"""

    for selected_index in range(active_capacities.size):
        active = int(active_capacities[selected_index])
        for index in range(feedback_offsets[selected_index], feedback_offsets[selected_index + 1]):
            slot = int(destination_slots[index])
            if slot < 0 or slot >= active:
                return False
    return True


@njit(parallel=True, nogil=True)
def _commit_lru_rows(
    flat_rows,
    destination_slots,
    feedback_offsets,
    active_capacities,
    next_slot,
    previous_slot,
    queue_head,
    queue_tail,
    victim_slots,
    victim_bitmap,
    unique_counts,
    victim_count,
    bitmap_words,
    extent,
    extent_shift,
    extent_links,
    extent_ends,
):
    """提交指定稳定行的 Final slot，并缓存这些行的下一代 victim。"""
    # 负责把 LIDU 返回的 destination slot 移动到 LRU 尾部

    # 每行可以并行计算
    for selected_index in prange(flat_rows.shape[0]):
        row = int(flat_rows[selected_index])
        active = int(active_capacities[row])
        active_words = (active + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS

        # 上一代 victim bitmap 已消费完，临时复用为 Final slot 去重表。
        for word in range(bitmap_words):
            victim_bitmap[row, word] = 0

        unique_count = 0
        # offsets 对应 layer-major 的稳定行顺序；混合 MTP 不做固定宽度 reshape。
        for index in range(feedback_offsets[selected_index], feedback_offsets[selected_index + 1]):
            slot = int(destination_slots[index])
            word = slot >> 5
            bit = slot & 31
            bit_mask = np.uint32(1) << np.uint32(bit)
            if np.uint32(victim_bitmap[row, word]) & bit_mask:
                continue
            # bitmap 表示本轮所有不重复的 destination slot 集合
            victim_bitmap[row, word] = np.int32(np.uint32(victim_bitmap[row, word]) | bit_mask)
            unique_count += 1
        unique_counts[row] = unique_count

        head = int(queue_head[row])
        tail = int(queue_tail[row])

        # slot-ID 升序扫描，确保同一步的 LRU 更新顺序确定。slot ID 从小到大扫描。
        last_block = -1
        for word in range(active_words):
            remaining_bits = np.int64(np.uint32(victim_bitmap[row, word]))
            while remaining_bits != 0:
                # 提取当前最低的置位 bit
                lowest_bit = remaining_bits & (np.int64(0) - remaining_bits)
                # 得到 bit 下标
                bit = _single_bit_index(lowest_bit)
                slot = (word << 5) + bit

                if extent > 1:
                    block = slot >> extent_shift
                    if block != last_block:
                        _touch_extent(block, extent_links, extent_ends)
                        last_block = block

                # 更新链表，把节点接到 tail 后面
                if slot != tail:
                    old_previous = int(previous_slot[row, slot])
                    old_next = int(next_slot[row, slot])
                    if old_previous >= 0:
                        next_slot[row, old_previous] = old_next
                    else:
                        head = old_next
                    if old_next >= 0:
                        previous_slot[row, old_next] = old_previous

                    next_slot[row, tail] = slot
                    previous_slot[row, slot] = tail
                    next_slot[row, slot] = -1
                    tail = slot

                remaining_bits &= remaining_bits - 1

        queue_head[row] = head
        queue_tail[row] = tail
        # 完成slot移动后，重新生成victim
        if extent > 1:
            _write_extent_victims(row, victim_slots, victim_bitmap, victim_count, extent, extent_links, extent_ends)
        else:
            _write_oldest_victims(
                row,
                next_slot,
                queue_head,
                victim_slots,
                victim_bitmap,
                victim_count,
                bitmap_words,
            )
