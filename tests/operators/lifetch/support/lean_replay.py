# SPDX-License-Identifier: Apache-2.0
"""G2 matched route and timing checks. All host reads run outside headline timing."""

from collections import defaultdict
from statistics import median

import numpy as np


def request_step(ordinal, warmup):
    """Change request generations within the existing warmup budget."""
    first_generation = max(2, warmup - 2)
    if warmup >= 3 and ordinal > first_generation:
        return dict(generation=2, step=ordinal - first_generation)
    return dict(generation=1, step=ordinal)


def set_step(route, step, generation):
    count = route.requests.shape[0]
    rows = np.empty((count, 4), dtype=np.int64)
    rows[:, 0] = np.arange(1, count + 1)
    rows[:, 1:] = [generation, step, 4]
    route.set_requests(rows)


def check_route(states, step, generation, *, low):
    counts = []
    for state in states:
        route = state.rotating.adaptive_route
        mode = int(route.mode.cpu().item())
        feedback = route.feedback.cpu().numpy()[1:]
        demand = route.demand.cpu().numpy().reshape(-1)
        expected_mode = (2 if state.route_policy == 'early' else 1) if low else 0
        if (mode != expected_mode or np.any(feedback[:, 0] != generation) or np.any(feedback[:, 1] != step)
                or np.any(feedback[:, 3] != 4) or np.any(demand < 0)
                or not np.array_equal(feedback[:, 2], demand)):
            raise ValueError(f'Incorrect route or uncommitted feedback at generation {generation}, step {step}, mode={mode}')
        counts.append(int(demand.sum()))
    return dict(generation=generation, step=step, low=len(states) if low else 0, high=0 if low else len(states),
                feedback_demand_per_call=counts)


def measured_range(values):
    return dict(min=min(values), p50=median(values), max=max(values))


def zero_traffic(state, before):
    """Read the current final descriptors, independently of any older prefetch map."""
    rotating = state.rotating
    rows = rotating.rows.cpu().numpy().view(np.uint32)
    posts = rotating.final_posts.cpu().numpy()
    ids = state.update.logical_output.cpu().numpy().reshape(state.batch, 4, 2048)
    physical = rotating.adaptive_route.demand.cpu().numpy()
    if np.any(rows[:, 9] != 0) or np.any(rows[:, 16:16 + rotating.rounds] != 0):
        raise ValueError('Zero-window route issued a speculative Get')
    counts = posts[:, :, 0].astype(np.int64).sum(axis=1)
    if not np.array_equal(counts, physical) or not np.array_equal(counts, rows[:, 247]):
        raise ValueError('Zero-window final descriptors disagree with completed Get count')
    selected = cold = 0
    for r in range(state.batch):
        demand = np.unique(ids[r])
        local = before['logical_ids'][r, :state.resident][before['hot_valid'][r] != 0]
        begin, end = before['tail_bounds'][r]
        missing = ~np.isin(demand, local) & ((demand < begin) | (demand >= end))
        actual = int(missing.sum())
        if actual != counts[r]:
            raise ValueError(f'Zero-window request {r}: completed {counts[r]}, exactly {actual} cold IDs')
        selected += len(demand)
        cold += actual
    return dict(totals=dict(prefetch=0, final=int(counts.sum()), unused_prefetch=0,
                            initial_cold=cold, selected_unique=selected,
                            KV_read_bytes=int(counts.sum())*state.token_bytes),
                window_posted=[0]*rotating.rounds)


def clipped_union(start, end, intervals):
    """Length covered inside [start, end] by the union of all intervals."""
    finish = start
    total = 0
    for a, z in sorted(intervals):
        if a >= end:
            break
        begin = max(finish, a)
        covered = min(end, z)
        if begin < covered:
            total += covered - begin
            finish = covered
    return total


def window_stages(events, mhz):
    """Decode full 128-byte retired Event snapshots. An unexecuted window has no time."""
    invocations, epochs, lanes, _ = events.shape
    reports = []
    for epoch in range(epochs):
        eligible = invocations * lanes
        raw = events[:, epoch].reshape(-1, 16)
        words = raw.view('<u4')
        complete = ((words[:, 6] == 1) & (words[:, 5] == epoch)
                    & (raw[:, 0] > 0) & (raw[:, 1] >= raw[:, 0]))
        e = raw[complete]
        def metric(values):
            return dict(n=len(values), eligible=eligible,
                        p50_us=float(np.median(values))/mhz if len(values) else None,
                        max_us=int(np.max(values))/mhz if len(values) else None)

        def interval(begin, end):
            valid = (e[:, begin] > 0) & (e[:, end] >= e[:, begin])
            return metric(e[valid, end] - e[valid, begin])

        reports.append(dict(epoch=epoch, service_us=interval(0, 1),
                            predecessor_wait_us=interval(4, 13), incremental_us=interval(13, 14),
                            scan_us=metric(e[:, 5]), candidate_us=metric(e[:, 6]),
                            post_us=metric(e[:, 7]), active_post_us=metric(e[e[:, 10] > 0, 7]),
                            posted_tokens=measured_range(e[:, 10].astype(int).tolist()) if len(e) else None))
    return reports


def stage_report(states, profile, samples, mhz, policy, directory):
    # Each summary keeps the relevant unit of observation. No per-request sum
    # of duplicate Get stamps and no zero for a fast path missing a probe.
    if not mhz or mhz <= 0:
        raise ValueError('Missing calibrated device cycles per microsecond')
    buckets = defaultdict(list)
    eligible = defaultdict(int)
    direct = dict(compact=0, overflow=0, other=0)
    raw_times, completed_events, route_probes, group_counts = [], [], [], []
    diagnostic_fingerprints = []
    for state in states:
        b = state.batch
        raw = state.forward.times.cpu().numpy().reshape(-1)
        raw_times.append(raw)
        flexible = False
        if getattr(state, 'verify_ranked_events', False):
            from correctness import topk_fingerprint
            trailer = b*256 + 64*32
            flexible = raw.size == trailer + 64*8 and raw[trailer] in (0x49445835, 0x49445836)
            ids = state.update.logical_output.cpu().numpy().reshape(b, 4, 2048)
            diagnostic_fingerprints.append(topk_fingerprint(ids))
        rotating = state.rotating
        event_bytes = rotating.events.cpu().numpy().view(np.uint8)
        event_count = rotating.rounds*rotating.service_lanes
        event_words = event_bytes.reshape(-1, 128)[:event_count].view('<u8')
        completed_events.append(event_words.reshape(rotating.rounds, rotating.service_lanes, 16))
        rows = rotating.rows.cpu().numpy().view(np.uint32).reshape(b, 256)
        producer = raw[:b*64].reshape(b, 64)
        li = raw[b*64:b*128].reshape(b, 4, 16)
        fetch = raw[b*128:b*256].reshape(b, 128)
        workers = state.forward.q_workers
        if policy != 'main':
            probe = state.rotating.adaptive_route.probe_ticks.cpu().numpy()
            route_probes.append(probe)
            for name, cores in (('routing_probe_aic', probe[:32]), ('routing_probe_aiv', probe[32:])):
                eligible[name] += len(cores)
                buckets[name].extend(int(c)/mhz for c in cores if c > 0)
        for r in range(b):
            p, u = producer[r, :32], producer[r, 32:]
            def bracket(name, arr, start, stop):
                eligible[name] += 1
                first, last = int(arr[start]), int(arr[stop])
                if first > 0 and last >= first:
                    buckets[name].append((last-first) / mhz)
            bracket('planning', p, 1, 8)
            bracket('consumer_including_waits', u, 10, 24)
            if not flexible:
                bracket('index_metadata', u, 22, 23)
            if policy != 'main':
                mark = int(p[9])
                key = {1: 'compact', 2: 'overflow'}.get(mark, 'other')
                direct[key] += 1
                eligible['direct_plan_and_describe'] += 1
                if key != 'other' and p[30]:
                    buckets['direct_plan_and_describe'].append(int(p[30])/mhz)
                eligible['index_active_work'] += 1
                if u[26]:
                    buckets['index_active_work'].append(int(u[25])/mhz)
            bracket('finalize_including_waits', fetch[r], 96, 99)
            if rows[r, 9] > 0:
                eligible['prefetch_cq_drain'] += 1
                # Finalizer ft[6] times DrainQueue; ft[7] confirms its call.
                if fetch[r, 103]:
                    buckets['prefetch_cq_drain'].append(int(fetch[r, 102])/mhz)
            # Nonempty hardware Get spans include post, transfer and draining.
            # The consumer span includes waits too. Their intersection bounds
            # time in common, without treating every such cycle as saved.
            windows = []
            for q in range(4):
                if policy != 'main':
                    bracket('final_topk', li[r,q], 14, 15)
                bracket('li_final_classification', li[r,q], 0, 7)
                f = fetch[r, q*24:(q+1)*24]
                for groups, start, end in ((16, 3, 4), (17, 7, 8)):
                    if f[groups] and 0 < f[start] <= f[end]:
                        windows.append((int(f[start]), int(f[end])))
            name = 'consumer_final_get_overlap_upper'
            eligible[name] += 1
            if windows and u[10] and u[24] >= u[10]:
                common = clipped_union(int(u[10]), int(u[24]), windows)
                buckets[name].append(common/mhz)
            # Main stamps only general-consumer phases. Sum all five or flag
            # this row as missing, since the disjoint shortcut skips them.
            spans = [(int(u[12+2*i]), int(u[13+2*i])) for i in range(5)]
            eligible['index_apply_general'] += 1
            if all(a > 0 and z >= a for a, z in spans):
                buckets['index_apply_general'].append(sum(z-a for a, z in spans)/mhz)
        # The fetch-group stamps are replicated on every row in its lane.
        # Use just one row for each (query,worker), and exclude empty groups.
        submitted_groups = 0
        for lane in range(min(workers, b)):
            for q in range(4):
                f = fetch[lane, q*24:(q+1)*24]
                for phase in range(2):
                    if int(f[16+phase]) <= 0:
                        continue
                    submitted_groups += int(f[16+phase])
                    for name, off in (('final_get_post', 12), ('final_get_cq_wait', 13)):
                        eligible[name] += 1
                        buckets[name].append(int(f[off+2*phase])/mhz)
        group_counts.append(submitted_groups)
    scopes = dict(planning='producer entry to completed per-row plan, includes waits',
                  routing_probe_aic='elapsed route choice per first-LI AIC, inside LI headline',
                  routing_probe_aiv='elapsed route choice per first-LI AIV, inside LI headline',
                  consumer_including_waits='Index consumer entry to metadata publication, per row',
                  consumer_final_get_overlap_upper='overlap of consumer including waits with its workers’ nonempty final Gets including Post, per eligible row; upper bound only',
                  index_apply_general='five instrumented general Index phase bodies, per row; disjoint path excluded',
                  index_active_work='nonempty row, measured Index work including attempts or rollback; excludes publication waits and final metadata',
                  direct_plan_and_describe='candidate direct resolver including descriptor generation, excludes input staging, per marked row',
                  index_metadata='per-row metadata interval after final selection',
                  finalize_including_waits='per-row finalizer entry through status publication, includes joins',
                  final_topk='per-query final score read and TopK sort before location classification, candidate diagnostic only',
                  li_final_classification='per-query final classification including its waits',
                  final_get_post='KVAF post-active time per nonempty final Get worker/phase',
                  prefetch_cq_drain='per request that submitted prefetch, software DrainQueue duration across all its windows',
                  final_get_cq_wait='KVAF drain-wait time per nonempty final Get worker/phase')
    stages = {}
    def metric(name, values, n, scope):
        stages[name] = dict(p50_us=median(values) if values else None, max_us=max(values) if values else None,
                            n=len(values), eligible=n, scope=scope)
    for name, desc in scopes.items():
        metric(name, buckets[name], eligible[name], desc)
    for stage in profile['timing']['stage_mean_us']:
        values = [sample['stages_us'][stage] for sample in profile['samples']]
        metric('kernel_'+stage, values, len(profile['samples']), 'headline device kernel per invocation')
    traffic_rows = []
    windows, occupied = [], []
    for sample in samples:
        algorithm = sample['algorithm']
        if 'totals' not in algorithm:
            raise ValueError(f'Missing headline traffic evidence: {algorithm.get("error")}')
        traffic_rows.append(algorithm['totals'])
        if 'window_posted' in algorithm:
            windows.append(algorithm['window_posted'])
            occupied.append([0]*len(algorithm['window_posted']))
        else:
            requests = algorithm['requests']
            if policy == 'early':
                # Observations were copied with the headline sample, before
                # the independent counter replay restored the fixture.
                completed = sample['route_physical_reads']
                if completed != [r['prefetch'] + r['final'] for r in requests]:
                    raise ValueError('Early-route physical reads disagree with completed Get counters')
                if any(w['posted'] for r in requests for epoch, w in enumerate(r['windows']) if epoch != 1):
                    raise ValueError('Early route prefetched outside its first cold window')
            windows.append([sum(req['windows'][w]['posted'] for req in requests)
                            for w in range(len(requests[0]['windows']))])
            occupied.append([sum(req['windows'][w]['posted'] > 0 for req in requests)
                             for w in range(len(requests[0]['windows']))])
    traffic = {k: measured_range([row[k] for row in traffic_rows]) for k in
               ('prefetch', 'final', 'unused_prefetch', 'initial_cold', 'selected_unique', 'KV_read_bytes')}
    events = np.stack(completed_events)
    np.savez_compressed(directory / 'lean_ticks.npz', times=np.stack(raw_times), prefetch_events=events, cycles_per_us=mhz,
                        q_workers=states[0].forward.q_workers,
                        **({'route_probe': np.stack(route_probes)} if route_probes else {}))
    return dict(stages=stages, traffic=traffic, diagnostic_topk_set_digests=diagnostic_fingerprints,
                window_posted=[measured_range(col) for col in zip(*windows)],
                window_nonempty_requests=[measured_range(col) for col in zip(*occupied)],
                diagnostic_final_get_submitted_groups=measured_range(group_counts),
                route_policy=policy, direct_plan_rows=direct, cycles_per_us=mhz,
                window_stages=window_stages(events, mhz),
                measured_graph_calls=len(samples), diagnostic_graph_replays=1, diagnostic_graph_calls=len(states),
                standalone_init_us=None, index_cq_hidden_us=None,
                scope='Auxiliary counters exclude headline. Individual role durations run in parallel. '
                      'Main has no isolated final TopK probe. Counterfactual hidden Index time is unmeasured. '
                      'Stock producer can choose a different sparse helper under detailed instrumentation.')
