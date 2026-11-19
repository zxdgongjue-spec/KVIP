# SPDX-License-Identifier: Apache-2.0
"""Startup-only checks for the experimental score producer + original T2 tail."""


def validate_score_t2(shape, parts):
    if not isinstance(parts, int) or isinstance(parts, bool) or not 1 <= parts <= 32:
        raise ValueError('score_parts must be an integer in 1..32')
    b, h, r, w = shape.batch, shape.history, shape.resident, shape.rounds
    if not (1 <= b <= 16 and h % 256 == 0 and h // 256 >= parts and b * parts <= shape.aicCount):
        raise ValueError('score tasks must fit the effective AIC budget with nonempty whole-page slices')
    if (shape.indexHeads != 32 or not shape.splitQ or w <= 1 or r % 512 or
            h <= r or (h-r) % 256 or h-r < 512*(w-1) or shape.liCores != 2*b):
        raise ValueError('score_t2 requires the original Q4/head32 two-history-owner T2 schedule')
    half_pages = (h-r)//256
    half_blocks = (((half_pages+1)//2) + w-2)//(w-1)
    if half_blocks*(2*w-3) >= half_pages:
        raise ValueError('T2 final window must contain a tile on both shards')
    return parts
