# SPDX-License-Identifier: Apache-2.0
"""Normalize existing device captures, keeping repeat and clock boundaries."""
import csv
import json
from pathlib import Path

import numpy as np

from reporting import atomic_json
from summarize_core_timeline import decode


def distribution(values):
    data = np.asarray(values, dtype=np.float64)
    return dict(mean_us=float(data.mean()), p50_us=float(np.median(data)),
                min_us=float(data.min()), max_us=float(data.max()), samples=len(data))


def final_transfer_scope(raw, requests, transport):
    """Same-replay byte/span metric from existing Get entry/return markers.

    This is not first-WQE-to-last-CQ latency. URMA records KVAF calls; UBmem
    records its query worker's copy interval including dependency gaps.
    """
    batch, workers = raw['batch'], raw['workers_per_query']
    width, mhz = raw['token_bytes'], raw['clock_mhz']
    role_cores, cores = raw['resolver_cores'], raw['cores']
    if width not in (656, 2624) or not mhz or mhz <= 0 or len(requests) != batch:
        raise ValueError('Invalid Fetch scope geometry/clock/count records')
    if len(cores) != role_cores * batch + 4 * workers:
        raise ValueError('Fetch scope core records do not match active workers')
    intervals, groups = [], []
    for q in range(4):
        for lane in range(workers):
            assigned = list(range(lane, batch, workers))
            counts = [requests[b]['final_counts'][q] for b in assigned]
            total = sum(row['total'] for row in counts)
            if any(row['safe'] < 0 or row['conflict'] < 0 or row['total'] != row['safe'] + row['conflict']
                   for row in counts):
                raise ValueError('Fetch safe/conflict counts do not match total')
            block = role_cores * batch + q * workers + lane
            core = cores[block]
            phases = ((total, 5, 6),) if transport == 'ubmem' else (
                (sum(row['safe'] for row in counts), 5, 6),
                (sum(row['conflict'] for row in counts), 10, 11))
            for count, begin_index, end_index in phases:
                if not count:
                    continue
                begin, end = int(core[begin_index]), int(core[end_index])
                if not 0 < core[0] <= begin < end <= core[1]:
                    raise ValueError(f'block={block}: missing/nonmonotonic Get timestamps with {count} entries')
                intervals.append((begin, end))
            groups.append(dict(block=block, query=q + 1, lane=lane, entries=total, bytes=total * width))
    entries = sum(group['entries'] for group in groups)
    result = dict(entries=entries, bytes=entries * width, groups=groups,
                  scope='KVAF call entry/return' if transport == 'urma' else 'UBmem copy worker interval',
                  note='Global elapsed span includes gaps/dependencies; not first WQE to hardware completion',
                  first_cycles=None, last_cycles=None, span_us=None, GBps=None)
    if intervals:
        first, last = min(item[0] for item in intervals), max(item[1] for item in intervals)
        span = (last - first) / mhz
        result.update(first_cycles=first, last_cycles=last, span_us=span, GBps=entries * width / span / 1000)
    return result


def summarize(directory, config):
    directory = Path(directory)
    paths = sorted([*directory.rglob('full_attribution.json'), *directory.rglob('attribution.json')])
    result = dict(level=config.level, stages=list(config.stages), transport=config.transport,
                  trace='COMPLETE', captures=[], missing=[], unavailable=[])
    timeline_rows = []
    if not paths:
        result['missing'].append('device attribution files')
    for path in paths:
        content = json.loads(path.read_text())
        samples = content['samples'] if isinstance(content, dict) else content
        capture = dict(source=str(path), samples=[], stage_summary={})
        stage_values = {}
        if len(samples) != config.iterations:
            result['missing'].append(f'{path}: expected {config.iterations} repeats, got {len(samples)}')
        for sample in samples:
            item = dict(repeat=sample['round'], accounting=sample.get('accounting', {}),
                        prefetch=sample.get('prefetch'), final=sample.get('final'))
            if 'algorithm' in sample:
                item['algorithm'] = sample['algorithm']
                if 'error' in sample['algorithm']:
                    result['missing'].append(f'{path} repeat={sample["round"]}: algorithm: {sample["algorithm"]["error"]}')
            else:
                result['missing'].append(f'{path} repeat={sample["round"]}: algorithm counters')
            for key in ('source_round', 'scheduler_round', 'identities', 'correctness', 'diagnostic_error'):
                if key in sample:
                    item[key] = sample[key]
            if sample.get('diagnostic_error'):
                result['missing'].append(f'{path}: {sample["diagnostic_error"]}')
            accounting = item['accounting']
            if not accounting or 'error' in accounting:
                result['missing'].append(f"{path} repeat={sample['round']}: {accounting.get('error', 'missing accounting')}")
            for stage, duration in accounting.get('stages_us', {}).items():
                if stage in config.stages or ('resolve_update' in stage and
                                               set(config.stages) & {'resolve', 'update', 'fetch', 'finalize'}):
                    stage_values.setdefault(stage, []).append(duration)
            details = sample.get('streaming_cycles', sample.get('streaming', []))
            if details:
                scope = details[0].get('prefetch_scope')
                raw = details[0].get('core_timeline_raw')
                mhz = raw.get('clock_mhz') if raw else None
                if scope and mhz and scope.get('interval_cycles', 0) > 0:
                    span = scope['interval_cycles'] / mhz
                    item['prefetch_bandwidth'] = dict(bytes=scope['bytes'], span_us=span,
                        GBps=scope['bytes'] / span / 1000, completion='software_observed',
                        note=scope.get('note', 'Prefetch start to software-observed completion'),
                        first_submit_cycles=scope['first_doorbell'], last_complete_cycles=scope['last_cq_observed'])
                if raw:
                    try:
                        decode(raw)  # Validate core bounds and cross-core publication ordering before global spans.
                        item['final_get_bandwidth'] = final_transfer_scope(raw, details, config.transport)
                    except (KeyError, TypeError, ValueError) as error:
                        item.pop('prefetch_bandwidth', None)
                        result['missing'].append(f'{path} repeat={sample["round"]}: final Get scope: {error}')
                    else:
                        final = item['final_get_bandwidth']
                        if sample.get('final') is not None and final['entries'] != sample['final']:
                            result['missing'].append(f'{path}: final scope entries differ from device total')
                        if scope and mhz and final['entries'] and final['span_us']:
                            begin = min(scope['first_doorbell'], final['first_cycles'])
                            end = max(scope['last_cq_observed'], final['last_cycles'])
                            if end > begin > 0:
                                span = (end - begin) / mhz
                                byte_count = scope['bytes'] + final['bytes']
                                item['combined_observed_bandwidth'] = dict(
                                    bytes=byte_count, span_us=span, GBps=byte_count / span / 1000,
                                    first_cycles=begin, last_cycles=end,
                                    note='Prefetch doorbell/CQ observation plus final Get call interval; includes control gaps')
                if config.level >= 2:
                    # Counts/durations retain their request and query association.
                    item['requests'] = details
                    if raw:
                        try:
                            decoded = decode(raw)
                        except ValueError as error:
                            result['missing'].append(f'{path} repeat={sample["round"]}: {error}')
                        else:
                            if config.transport == 'ubmem':
                                for core in decoded['cores']:
                                    core.pop('KVAF_counters', None)  # UBmem fields are not URMA CQ cycles.
                                    for stage in core['stages']:
                                        stage['name'] = stage['name'].replace('prefetch_CQ_join', 'prefetch_completion_join')
                            item['timeline'] = decoded
                            for core in decoded['cores']:
                                timeline_rows.append(dict(capture=str(path), repeat=sample['round'],
                                    block=core['block'], role=core['role'], step='whole_core',
                                    start_us=core['start_us'], end_us=core['end_us'], duration_us=core['duration_us']))
                                if config.level == 3:
                                    for stage in core['stages']:
                                        timeline_rows.append(dict(capture=str(path), repeat=sample['round'],
                                            block=core['block'], role=core['role'], step=stage['name'],
                                            start_us=stage['start_us'], end_us=stage['end_us'], duration_us=stage['duration_us']))
                    else:
                        result['missing'].append(f'{path} repeat={sample["round"]}: per-core records')
            elif config.level >= 2:
                result['missing'].append(f'{path} repeat={sample["round"]}: detailed counters')
            capture['samples'].append(item)
        capture['stage_summary'] = {stage: distribution(values) for stage, values in stage_values.items()}
        result['captures'].append(capture)
    native_paths = sorted(directory.rglob('native_attribution.json'))
    for path in native_paths:
        content = json.loads(path.read_text())
        capture = dict(source=str(path), path='serial', samples=content['samples'], stage_summary={})
        values = {}
        if len(content['samples']) != config.iterations:
            result['missing'].append(f'{path}: native repeat count differs from iterations')
        for sample in content['samples']:
            accounting = sample.get('accounting', {})
            if not accounting or 'error' in accounting:
                result['missing'].append(f'{path}: {accounting.get("error", "missing native accounting")}')
            for stage, duration in accounting.get('stages_us', {}).items():
                selected = stage in config.stages or (stage == 'lidu_prepare' and
                    set(config.stages) & {'resolve', 'update'})
                if selected:
                    values.setdefault(stage, []).append(duration)
        capture['stage_summary'] = {stage: distribution(samples) for stage, samples in values.items()}
        result['captures'].append(capture)
    # These are distinct from missing expected timestamps. Existing operators
    # do not provide token identity at each admission/consumption boundary.
    result['unavailable'] = [
        'useful/redundant/refetched copy attribution: consumption identity not recorded by existing operators',
        'exact final/combined transfer span: first submission and matching completion not recorded on every transport',
        'window candidates/threshold/budget suppression: native threshold log not enabled by the current public operator',
        *([] if native_paths else ['serial stage attribution: no native stage capture in this run']),
    ]
    if config.level == 1:
        result['unavailable'] = [value for value in result['unavailable']
                                 if not value.startswith(('useful/', 'window candidates/'))]
    if 'fetch' in config.stages:
        for capture in result['captures']:
            if capture.get('path') == 'serial':
                continue
            for sample in capture['samples']:
                if 'combined_observed_bandwidth' not in sample:
                    only = ('final_get_bandwidth' if sample.get('prefetch') == 0 else
                            'prefetch_bandwidth' if sample.get('final') == 0 else None)
                    if only and only in sample:
                        sample['combined_observed_bandwidth'] = dict(sample[only])
                for metric in ('prefetch_bandwidth', 'final_get_bandwidth', 'combined_observed_bandwidth'):
                    if metric not in sample and ((metric == 'prefetch_bandwidth' and sample.get('prefetch') == 0)
                            or (metric == 'final_get_bandwidth' and sample.get('final') == 0)
                            or (metric == 'combined_observed_bandwidth' and sample.get('prefetch') == 0 and sample.get('final') == 0)):
                        sample[metric] = dict(bytes=0, span_us=None, GBps=None, note='No transfers; bandwidth N/A')
                    if metric not in sample:
                        result['missing'].append(
                            f"{capture['source']} repeat={sample['repeat']}: {metric} unavailable")
    if config.level == 3 and 'fetch' in config.stages:
        fetch_paths = sorted(directory.rglob('fetch_only.json'))
        result['fetch_only'] = [dict(source=str(path), **json.loads(path.read_text()))
                                for path in fetch_paths]
        if config.transport == 'ubmem':
            result['unavailable'].append('Fetch-only: UNSUPPORTED for UBmem standalone worker')
        elif not fetch_paths:
            result['missing'].append('Fetch-only comparison results')
        for comparison in result['fetch_only']:
            expected = {(query, version) for query in ('Q1', 'Q2', 'Q3', 'Q4', 'Q4_all')
                        for version in ('native', 'fused')}
            observed = {(row['query'], row['version']) for row in comparison.get('rows', [])}
            if comparison.get('status') != 'PASS' or observed != expected:
                result['missing'].append(f"{comparison['source']}: incomplete Fetch-only comparison")
    if result['missing']:
        result['trace'] = 'INCOMPLETE'
    atomic_json(directory / 'diagnostic.json', result)
    lines = [f"DIAG level={config.level} transport={config.transport} trace={result['trace']}",
             'Separate profiled pass; times are not added across overlapping cores.',
             'Synthetic repeats restore the same state; they are not consecutive decode rounds.']
    for capture in result['captures']:
        lines.append(f"source={capture['source']}")
        for stage, values in capture['stage_summary'].items():
            lines.append(f"{stage}: mean={values['mean_us']:.3f} p50={values['p50_us']:.3f} "
                         f"min={values['min_us']:.3f} max={values['max_us']:.3f} us n={values['samples']}")
    for capture in result['captures']:
        for sample in capture['samples']:
            repeat = sample.get('repeat', sample.get('round'))
            accounting = sample.get('accounting', {})
            intervals = accounting.get('intervals_us', {})
            origin = min((start for spans in intervals.values() for start, end in spans), default=0)
            for stage, spans in intervals.items():
                if stage not in capture['stage_summary']:
                    continue
                for start, end in spans:
                    lines.append(f'STAGE repeat={repeat} stage={stage} start_us={start-origin:.3f} '
                                 f'end_us={end-origin:.3f} duration_us={end-start:.3f}')
            lines.append(f"ACCOUNTING repeat={repeat} gap_us={accounting.get('inter_kernel_gap_us', 'N/A')} "
                         f"boundary_us={accounting.get('boundary_residual_us', 'N/A')}")
            for metric in ('prefetch_bandwidth', 'final_get_bandwidth', 'combined_observed_bandwidth'):
                if metric in sample:
                    bandwidth = sample[metric]
                    lines.append(f"BANDWIDTH repeat={sample['repeat']} metric={metric} "
                                 f"bytes={bandwidth['bytes']} span_us={bandwidth['span_us'] if bandwidth['span_us'] is not None else 'N/A'} GB/s={bandwidth['GBps'] if bandwidth['GBps'] is not None else 'N/A'}")
                    if bandwidth.get('note'):
                        lines.append(f"  scope: {bandwidth['note']}")
            algorithm = sample.get('algorithm', {})
            totals = algorithm.get('totals', {})
            if totals:
                lines.append(f"ALGORITHM repeat={sample['repeat']} source_round={sample.get('source_round', 'synthetic')} " + ' '.join(
                    f'{key}={value}' for key, value in totals.items()))
                for request in algorithm['requests']:
                    fields = ('initial_hot_valid', 'selected_unique', 'initial_cold', 'prefetch', 'final', 'KV_read_bytes')
                    lines.append(f"  request={request['request']} " + ' '.join(
                        f"{key}={request.get(key) if request.get(key) is not None else 'N/A'}" for key in fields))
                if config.level >= 2:
                    for request in algorithm['requests']:
                        lines.append(f"  request={request['request']} "
                                     f"window_posted={[row['posted'] for row in request['windows']]} "
                                     f"window_unused={[row['unused'] for row in request['windows']]} "
                                     f"Q_final={[row['final_posted'] for row in request['queries']]}")
    for comparison in result.get('fetch_only', []):
        for row in comparison.get('rows', []):
            lines.append(f"FETCH-ONLY {row['query']} {row['version']} "
                         f"aiv={row['aiv']} jettys/aiv={row['jettys_per_aiv']} "
                         f"entries={row['count']} bytes={row['bytes']} "
                         f"mean/p50/p99_us={row['mean_us']:.3f}/{row['p50_us']:.3f}/{row['p99_us']:.3f} "
                         f"GB/s={row['effective_GBps']}")
    lines.extend('INCOMPLETE: ' + value for value in result['missing'])
    lines.extend('N/A: ' + value for value in result['unavailable'])
    (directory / 'diagnostic_summary.log').write_text('\n'.join(lines) + '\n')
    if timeline_rows:
        with (directory / 'timeline.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(timeline_rows[0]))
            writer.writeheader()
            writer.writerows(timeline_rows)
    if config.level == 1:
        rendered = level_one_tables(result)
        (directory / 'level1_summary.log').write_text(rendered + '\n')
        print(rendered, flush=True)
    else:
        print('\n'.join(lines), flush=True)
    return result


def algorithm_counts(mapping, selected, rows, posts, resident, windows, *,
                     hot_valid=None, lru_slots=None, logical_ids=None, tail_bounds=None, token_bytes=656):
    """Reconcile existing admission records, not hypothetical transfer savings.

    The prefetch map stores resident+admission_ordinal; ABC uses a separate
    range. Record membership proves unused transfers, but does not prove which
    selected token's copy was consumed. useful/redundant stay unclassified.
    """
    mapping, selected, rows, posts = map(np.asarray, (mapping, selected, rows, posts))
    batch, history = mapping.shape
    if selected.shape != (batch, 4, 2048) or rows.shape != (batch, 256) or posts.shape[:2] != (batch, 4):
        raise ValueError('Algorithm capture shape differs from Q4 ABI')
    if not 1 <= windows <= 32 or token_bytes not in (656, 2624):
        raise ValueError('Algorithm capture window/token width is invalid')
    if logical_ids is None:
        logical_ids = np.broadcast_to(np.arange(history), (batch, history))
    logical_ids = np.asarray(logical_ids)
    if logical_ids.shape != mapping.shape:
        raise ValueError('Algorithm capture logical mapping shape differs')
    requests = []
    for row in range(batch):
        positions = selected[row]
        if np.any(positions < 0) or np.any(positions >= history):
            raise ValueError('Algorithm capture contains invalid TopK positions')
        ids = logical_ids[row, positions]
        demand = np.unique(ids)
        admitted = np.flatnonzero((mapping[row] >= resident) & (mapping[row] < resident + 2048))
        ordinals = mapping[row, admitted] - resident
        posted = int(rows[row, 9])
        if len(admitted) != posted or not np.array_equal(np.sort(ordinals), np.arange(posted)):
            raise ValueError('Prefetch identity records do not match posted counter')
        epochs = rows[row, 16:16 + windows].astype(np.int64)
        if np.any(epochs < 0) or int(epochs.sum()) != posted:
            raise ValueError('Window counters do not match total prefetch')
        admitted = admitted[np.argsort(ordinals)]
        fetched_ids = logical_ids[row, admitted]
        final_counts = posts[row, :, 0].astype(np.int64)
        final = int(rows[row, 247])
        if np.any(final_counts < 0) or int(final_counts.sum()) != final:
            raise ValueError('Per-Q final counts do not match final total')
        unused = ~np.isin(fetched_ids, demand)
        window_rows, begin = [], 0
        for epoch, count in enumerate(epochs):
            end = begin + int(count)
            unused_count = int(unused[begin:end].sum())
            window_rows.append(dict(window=epoch + 1, posted=int(count),
                                    bytes=int(count) * token_bytes, unused=unused_count,
                                    selected_members=int(count) - unused_count,
                                    cumulative_posted=end, cumulative_unused=int(unused[:end].sum())))
            begin = end
        local = None
        if hot_valid is not None:
            valid = np.asarray(hot_valid)[row]
            if valid.shape != (resident,) or np.any((valid != 0) & (valid != 1)):
                raise ValueError('Invalid initial hot validity')
            local = logical_ids[row, :resident][valid != 0]
            initial_hot_count = len(local)
            if tail_bounds is not None:
                begin_tail, end_tail = map(int, np.asarray(tail_bounds)[row])
                if not 0 <= begin_tail <= end_tail <= history:
                    raise ValueError('Invalid initial ABC bounds')
                local = np.union1d(local, np.arange(begin_tail, end_tail))
        query_rows, seen = [], set()
        for query in range(4):
            unique = set(map(int, ids[query]))
            added = np.array(sorted(unique - seen), dtype=np.int64)
            query_rows.append(dict(query=query + 1, topk=len(ids[query]), unique=len(unique),
                                   cross_q_duplicate=len(unique & seen), new_unique=len(added),
                                   initial_local=None if local is None else int(np.isin(added, local).sum()),
                                   final_posted=int(final_counts[query])))
            seen.update(unique)
        cold = None if local is None else int((~np.isin(demand, local)).sum())
        candidate_hot = None
        if local is not None and lru_slots is not None:
            slots = np.asarray(lru_slots)[row]
            slots = slots[(slots >= 0) & (slots < resident)]
            candidate_hot = int(np.isin(demand, np.intersect1d(local, logical_ids[row, slots])).sum())
        requests.append(dict(request=row, selected_unique=len(demand), initial_cold=cold,
                             initial_hot_valid=None if local is None else initial_hot_count,
                             candidate_hot=candidate_hot, prefetch=posted, final=final,
                             KV_read_bytes=(posted + final) * token_bytes,
                             unused_prefetch=int(unused.sum()),
                             selected_prefetch=posted - int(unused.sum()),
                             useful=None, redundant=None, refetched=None,
                             windows=window_rows, queries=query_rows))
    totals = {key: sum(item[key] for item in requests) for key in
              ('selected_unique', 'prefetch', 'final', 'KV_read_bytes', 'unused_prefetch', 'selected_prefetch')}
    totals['initial_cold'] = None if hot_valid is None else sum(item['initial_cold'] for item in requests)
    totals['cold_miss_rate'] = (None if totals['initial_cold'] is None or not totals['selected_unique']
                               else totals['initial_cold'] / totals['selected_unique'])
    return dict(requests=requests, totals=totals,
                attribution='unused is exact membership; useful/redundant/refetched lack consumption identity')


def algorithm_before(rotating):
    """Snapshot initial state outside profiled replay, without a Get-time fence."""
    batch = rotating.runtime_shape.batch
    owners = (np.arange(batch) if rotating.row_owners is None
              else rotating.row_owners.cpu().numpy())
    validity = rotating.hot_valid.cpu().numpy()
    if np.any(owners < 0) or np.any(owners >= len(validity)):
        raise ValueError('Diagnostic owner is outside hot validity storage')
    return dict(hot_valid=validity[owners].copy(), lru_slots=rotating.lru_slots.cpu().numpy().copy(),
                tail_bounds=None if rotating.tail_bounds is None else rotating.tail_bounds.cpu().numpy().copy())


def algorithm_after(rotating, before):
    shape = rotating.runtime_shape
    try:
        return algorithm_counts(
            rotating.mapping.cpu().numpy(),
            rotating.arguments[9].cpu().numpy().view(np.int32).reshape(shape.batch, 4, 2048),
            rotating.rows.cpu().numpy().view(np.uint32), rotating.final_posts.cpu().numpy(),
            shape.resident, rotating.rounds, token_bytes=shape.tokenBytes, **before)
    except ValueError as error:
        return {'error': str(error)}


def level_one_tables(result):
    """Compact terminal view; raw intervals and scopes remain in diagnostic.json."""
    def cell(value):
        return 'N/A' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)

    def table(title, headers, rows):
        values = [headers, *[[cell(value) for value in row] for row in rows]]
        widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
        def line(row):
            return ' | '.join(value.ljust(width) for value, width in zip(row, widths))
        return [title, line(values[0]), '-+-'.join('-' * width for width in widths),
                *[line(row) for row in values[1:]], '']

    lines = [f"DIAG level=1 transport={result['transport']} trace={result['trace']}",
             'Times: us. Separate profiled replays; synthetic repeats are not decode rounds.', '']
    stages, counts, bandwidth = [], [], []
    scopes = {}
    for index, capture in enumerate(result['captures'], 1):
        path = capture.get('path', 'fused')
        label = f'{index}:{path}'
        lines.append(f"[{label}] {capture['source']}")
        for stage, values in capture['stage_summary'].items():
            stages.append([label, stage, values['mean_us'], values['p50_us'],
                           values['min_us'], values['max_us'], values['samples']])
        for sample in capture['samples']:
            repeat = sample.get('repeat', sample.get('round'))
            for request in sample.get('algorithm', {}).get('requests', []):
                counts.append([label, repeat, request['request'], *[request.get(key) for key in
                    ('initial_hot_valid', 'selected_unique', 'initial_cold', 'prefetch', 'final', 'KV_read_bytes')]])
            for key, name in (('prefetch_bandwidth', 'prefetch'), ('final_get_bandwidth', 'final Get'),
                              ('combined_observed_bandwidth', 'combined')):
                metric = sample.get(key)
                if metric:
                    bandwidth.append([label, repeat, name, metric['bytes'], metric['span_us'], metric['GBps']])
                    scopes[name] = metric.get('note', metric.get('scope', 'software-observed'))
    lines += table('Stage distribution (overlapping stages must not be added)',
                   ['Path', 'Stage', 'Mean', 'p50', 'Min', 'Max', 'N'], stages)
    lines += table('Per-request counts', ['Path', 'Repeat', 'Request', 'Hot', 'Demand', 'Cold',
                                          'Prefetch', 'Final', 'KV bytes'], counts)
    lines += table('Effective bandwidth (decimal GB/s)',
                   ['Path', 'Repeat', 'Scope', 'Bytes', 'Span us', 'GB/s'], bandwidth)
    lines += [f'{name}: {note}' for name, note in scopes.items()]
    lines += ['INCOMPLETE: ' + value for value in dict.fromkeys(result['missing'])]
    lines += ['N/A: ' + value for value in dict.fromkeys(result['unavailable'])]
    return '\n'.join(lines)
