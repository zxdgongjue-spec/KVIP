# SPDX-License-Identifier: Apache-2.0
"""Startup checks and window metadata for direct score producers + MR23 T2 tail."""


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


def score_window_schedule(shape, parts, query_parallel=False):
    """Describe direct MR14 producers on the exact MR23 half-window geometry."""
    validate_score_t2(shape, parts)
    half = (((shape.history-shape.resident+511)//512+shape.rounds-2)//(shape.rounds-1))*256
    waves = []
    for epoch in range(shape.rounds):
        begin = shape.resident+2*half*(epoch-1) if epoch else 0
        end = min(shape.history, shape.resident+2*half*epoch) if epoch else shape.resident
        p = min(parts, (end-begin)//256)
        waves.append(dict(epoch=epoch, begin=begin, end=end, parts=p,
                          predicate=epoch>0, score_aics=shape.batch*p, score_aivs=shape.batch*p*2,
                          ready_end0=min(shape.history, begin+half) if epoch else shape.resident,
                          ready_end1=end))
    return dict(mode='window_query' if query_parallel else 'window', score_post='direct', cold_predicate='producer',
                service='mr23_request_post' if query_parallel else 'mr23_ranked_window',
                topk_aivs=(shape.batch*4 if query_parallel else
                           min(shape.batch,shape.serviceLanes) if hasattr(shape,'serviceLanes') else None),
                half_window=half, waves=waves,
                li_launches=shape.rounds*(3 if query_parallel else 2)+2, scatter_launches=0,
                final_topk='mr23_incremental', resolve_fetch='mr23')
