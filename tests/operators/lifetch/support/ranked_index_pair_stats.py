# SPDX-License-Identifier: Apache-2.0
"""Index and CQ endpoints from the separate instrumented HIGH replay."""

import numpy as np

from ranked_five_report import final_q_workers
from ranked_help_report import tail_endpoints
from ranked_joint_report import joint_endpoints


def quantile(ticks, clock, pct=50):
    return float(np.percentile(ticks, pct) / clock) if np.size(ticks) else None


def stage_windows(raw, *, candidate):
    workers = final_q_workers(raw)
    queues = 4 * workers
    times = np.asarray(raw['times'], dtype=np.int64)
    clock = float(raw['cycles_per_us'])
    if not np.isfinite(clock) or clock <= 0 or times.ndim != 2 or times.shape[1] != 6656:
        raise ValueError('Invalid clock or HIGH team trace geometry')
    n = len(times)
    cores = times[:, 4096:6144].reshape(n, 64, 32)
    spare = cores[:, 48+queues:]
    plans = times[:, :1024].reshape(n, 16, 64)[..., 32+22]
    q4 = cores[:, 2:48:3, 27]
    fields = times[:, 6144:].reshape(n, 64, 8)[:, 48:]
    observed = np.zeros((n, 16), dtype=bool)
    samples = []
    for slot in range(2 if candidate else 1):
        offset = 16 + 5 * slot
        begin, end, scatter, moves, targets = (spare[..., offset+j] for j in range(5))
        call, lane = np.nonzero(moves)
        row = targets[call, lane] - 1
        home = queues + lane
        peer = np.where(home < 2*queues, home-queues, home)
        if (np.any(moves < 0) or np.any(moves > 1024) or np.any((row != home) & (row != peer))
                or np.any(row < 0) or np.any(row >= 16)
                or np.any(observed[call, row]) or len(set(zip(call, row))) != len(call)):
            raise ValueError('Repeated, illegal or oversized staged fourth-field chunk')
        observed[call, row] = True
        a, b, c, size = (v[call, lane] for v in (begin, end, scatter, moves))
        physical = 48 + home
        if (np.any(a <= 0) or np.any(a < spare[call, lane, 0]) or np.any(a > b) or np.any(b > c)
                or np.any(fields[call, row, 2] >> 32 != physical)
                or np.any(fields[call, row, 3] < size)
                or np.any(plans[call, row] <= 0) or np.any(q4[call, row] <= 0)):
            raise ValueError('Invalid staged timing, unpublished row, or final fourth-field owner')
        # Both producers stamp immediately AFTER publication; clipping tolerates
        # the reader observing the flag before the producer writes its timestamp.
        ready = np.maximum(plans[call, row], q4[call, row])
        samples.append(dict(calls=len(call), moves=int(size.sum()),
                            after_read=c-b, ready_delay=np.maximum(0,c-ready)))
        if slot:
            largest, longest, pieces, busy = (spare[call, lane, j] for j in range(26, 30))
            first_moves = spare[call, lane, 19]
            if (np.any(first_moves <= 0) or np.any(largest <= 0) or np.any(largest > 256)
                    or np.any(size > largest*pieces) or np.any(largest > size)
                    or np.any(pieces < 1) or np.any(longest <= 0) or np.any(busy < longest)
                    or np.any(busy > b-a)):
                raise ValueError('Partner read violates its slice budget or cycle accounting')
            # A second read can pause to serve the first row. Only charge
            # time that MUST have been spent reading inside its observed wait.
            own_end = spare[call, lane, 17]
            own_scatter = spare[call, lane, 18]
            common = np.maximum(0, np.minimum(b, own_scatter)-np.maximum(a, own_end))
            covered = np.maximum(0, busy-((b-a)-common))
            samples[-1].update(longest=longest, busy=busy, covered=covered)
    first = samples[0]
    out = dict(counts=dict(first_reads=first['calls']),
               p50_us=dict(first_read_to_scatter=quantile(first['after_read'],clock),
                           first_ready_to_scatter=quantile(first['ready_delay'],clock)),
               p95_us=dict(first_ready_to_scatter=quantile(first['ready_delay'],clock,95)))
    if candidate:
        other = samples[1]
        out['counts'].update(second_reads=other['calls'], second_moves=other['moves'])
        out['p50_us'].update(second_busy_read=quantile(other['busy'],clock),
                             second_read_in_first_wait_min=quantile(other['covered'],clock),
                             second_longest_slice=quantile(other['longest'],clock))
        out['p95_us']['second_ready_to_scatter'] = quantile(other['ready_delay'],clock,95)
    return out


def describe_high(raw, *, candidate):
    report = stage_windows(raw, candidate=candidate)
    end = joint_endpoints(raw)
    tails = tail_endpoints(raw, 'joint')
    report['p50_us'].update(
        fourth_minus_primary=tails['last_fourth_minus_last_primary_us']['p50'],
        last_Index=end['last_index']['p50'], last_CQ=end['last_cq']['p50'],
        Index_minus_CQ=end['last_index_minus_last_cq']['p50'],
        last_status=end['last_status']['p50'])
    return report
