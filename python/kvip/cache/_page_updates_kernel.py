# SPDX-License-Identifier: Apache-2.0
"""One program per appended page; no historical-page upload or KV payload copy."""

import triton
import triton.language as tl


@triton.jit
def publish_pages_kernel(
    ROWS,
    ADDRESSES,
    VALID,
    MAP,
    FUSED_TABLE,
    GROUPS: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    FUSED: tl.constexpr,
    HISTORY: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ROWS + row * 3)
    address = tl.load(ROWS + row * 3 + 1)
    clear = tl.load(ROWS + row * 3 + 2)
    lane = tl.arange(0, BLOCK)
    tl.store(ADDRESSES + index, address)
    tl.store(VALID + index, 1)
    tl.store(MAP + tl.maximum(clear, 0) + lane, -1, clear >= 0)
    if FUSED:
        owner = index // (GROUPS * PAGES)
        group = index // PAGES % GROUPS
        page = index % PAGES
        pointer = tl.load(FUSED_TABLE + group * 2).to(tl.pointer_type(tl.int64))
        token_bytes = tl.load(FUSED_TABLE + group * 2 + 1)
        token = page * BLOCK + lane
        tl.store(pointer + owner * HISTORY + token, address + lane * token_bytes, token < HISTORY)
