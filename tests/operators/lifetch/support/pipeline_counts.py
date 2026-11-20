# SPDX-License-Identifier: Apache-2.0
"""Read existing completed-replay counters outside the measured graph event."""
import numpy as np


def new_pipeline_count_report():
    return dict(status='PASS', unit='token Get requests',
                collection='after measured event and completion validation; no kernel instrumentation',
                sources=dict(prefetch='Row.requests / uint32 rows[:,9]',
                             final_get='sum FinalPost.count / final_posts[:,:,0]',
                             final_crosscheck='Row.finalRequests / uint32 rows[:,247]'),
                records=[])


def collect_pipeline_counts(states, source_round):
    """Return one record per measured input/request; reject missing evidence.

    The caller must finish the measured event and validate device completion
    before calling. This function performs CPU readback only, with no launches,
    reset, stamps, or approximation of counts from the configured budget.
    """
    records = []
    if not states:
        raise ValueError('Pipeline counts require completed measured states')
    for input_index, state in enumerate(states):
        rotating = state.rotating
        shape = rotating.runtime_shape
        batch, workers = int(shape.batch), int(shape.finalWorkers)
        if not 1 <= batch <= 16 or workers not in (1,2,4):
            raise ValueError(f'Pipeline runtime shape mismatch: input={input_index}')
        raw_rows = rotating.rows.cpu().numpy()
        raw_posts = rotating.final_posts.cpu().numpy()
        if (raw_rows.shape != (batch,256) or raw_rows.dtype.kind not in 'iu'
                or raw_rows.dtype.itemsize != 4):
            raise ValueError(f'Pipeline Row uint32 ABI mismatch: input={input_index}')
        if (raw_posts.shape != (batch,workers,32) or raw_posts.dtype.kind not in 'iu'
                or raw_posts.dtype.itemsize != 4 or np.any(raw_posts[:,:,0] < 0)):
            raise ValueError(f'Pipeline FinalPost uint32 ABI mismatch: input={input_index}')
        rows = raw_rows.view(np.uint32)
        posts = raw_posts.view(np.uint32)
        for request in range(batch):
            prefetch = int(rows[request,9])
            final_counts = [int(count) for count in posts[request,:,0]]
            final_get = sum(final_counts)
            row_final = int(rows[request,247])
            if final_get != row_final:
                raise ValueError('Pipeline final Get counters disagree: '
                    f'source_round={source_round} input={input_index} request={request} '
                    f'FinalPost={final_get} Row.finalRequests={row_final}')
            records.append(dict(source_round=int(source_round), input=input_index,
                request=request, prefetch_requests=prefetch, final_get_requests=final_get,
                total_get_requests=prefetch+final_get, final_get_by_worker=final_counts,
                row_final_requests=row_final))
    return records
