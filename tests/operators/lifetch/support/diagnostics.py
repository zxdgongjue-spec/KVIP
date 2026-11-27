# SPDX-License-Identifier: Apache-2.0
"""Stage timing, algorithm statistics and transfer evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from decimal import Decimal
from pathlib import Path

import numpy as np
from reporting import atomic_json
from workloads import TOPK, load_score_capture, score_values, select_series

DEVICE_STAGES = ("init", "li", "resolve", "fetch")


QUANTILES = (0, .01, .1, .25, .5, .75, .9, .99, 1)

POSITION_BINS = 16


def format_metric(value):
    return 'N/A' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)


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
        capture = dict(source=str(path), samples=[], stage_summary={},
                       graph_accounting=content.get('graph_accounting') if isinstance(content, dict) else None,
                       graph_calls=content.get('graph_calls') if isinstance(content, dict) else None)
        detail_path = path.parent / 'detail_samples.json'
        detail_samples = json.loads(detail_path.read_text()) if detail_path.exists() and config.level >= 2 else []
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
            for key in ('source_round', 'scheduler_round', 'identities', 'correctness', 'diagnostic_error', 'rank_progress'):
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
            detail_sample = next((row for row in detail_samples if row['round'] == sample['round']), None)
            details = (detail_sample['streaming_cycles'] if detail_sample else
                       sample.get('streaming_cycles', sample.get('streaming', [])))
            if detail_sample:
                item['detail_source'] = str(detail_path)
                item['detail_repeat'] = detail_sample['round']
                if 'algorithm' in detail_sample:
                    item['algorithm'] = detail_sample['algorithm']
                    item['algorithm_source'] = str(detail_path)
                    totals = item['algorithm'].get('totals', {})
                    item.update(prefetch=totals.get('prefetch'), final=totals.get('final'))
                    if 'error' in item['algorithm']:
                        result['missing'].append(f'{detail_path}: {item["algorithm"]["error"]}')
                if 'rank_progress' in detail_sample:
                    item['rank_progress'] = detail_sample['rank_progress']
            if details:
                scope = details[0].get('prefetch_scope')
                raw = details[0].get('core_timeline_raw')
                mhz = raw.get('clock_mhz') if raw else details[0].get('terminal_timeline', {}).get('clock_mhz')
                for metric in ('prefetch_bandwidth', 'final_get_bandwidth', 'combined_observed_bandwidth'):
                    if metric in details[0]:
                        item[metric] = details[0][metric]
                if scope and mhz and scope.get('interval_cycles', 0) > 0:
                    span = scope['interval_cycles'] / mhz
                    item['prefetch_bandwidth'] = dict(bytes=scope['bytes'], span_us=span,
                        GBps=scope['bytes'] / span / 1000, completion='software_observed',
                        note=scope.get('note', 'Prefetch start to software-observed completion'),
                        first_submit_cycles=scope['first_doorbell'], last_complete_cycles=scope['last_cq_observed'])
                if raw:
                    try:
                        decode_timeline(raw)  # Validate core bounds and cross-core publication ordering before global spans.
                        item['final_get_bandwidth'] = final_transfer_scope(raw, details, config.transport)
                    except (KeyError, TypeError, ValueError) as error:
                        result['missing'].append(f'{path} repeat={sample["round"]}: final Get scope: {error}')
                    else:
                        final = item['final_get_bandwidth']
                        expected_final = (sum(row['total'] for request in details
                                              for row in (request.get('final_counts') or []))
                                          if detail_sample else sample.get('final'))
                        if expected_final is not None and final['entries'] != expected_final:
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
                    if raw or details[0].get('terminal_timeline'):
                        try:
                            decoded = details[0].get('terminal_timeline') or decode_timeline(raw)
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
                                if config.level >= 2:
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
        capture = dict(source=str(path), path='serial', samples=content['samples'], stage_summary={},
                       graph_accounting=content.get('graph_accounting') if isinstance(content, dict) else None,
                       graph_calls=content.get('graph_calls') if isinstance(content, dict) else None)
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
    parallel_terminal = any('topk_resolve_update' in sample.get('accounting', {}).get('stages_us', {})
                            for capture in result['captures'] for sample in capture['samples'])
    if parallel_terminal:
        result['unavailable'].append('Parallel terminal: legacy standalone Fetch-only comparison is not applicable; transfer scopes use submission-to-CQ observation')
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
                    if metric not in sample and (config.level >= 2 or 'topk_resolve_update' not in sample.get('accounting', {}).get('stages_us', {})):
                        result['missing'].append(
                            f"{capture['source']} repeat={sample['repeat']}: {metric} unavailable")
    if config.level == 3 and 'fetch' in config.stages and not parallel_terminal:
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
             'Event and profiler share the measured graph; detailed core stamps use an additional pass.',
             'Synthetic repeats restore the same state; they are not consecutive decode rounds.']
    for capture in result['captures']:
        lines.append(f"source={capture['source']}")
        for stage, values in capture['stage_summary'].items():
            lines.append(f"{stage}: mean={values['mean_us']:.2f} p50={values['p50_us']:.2f} "
                         f"min={values['min_us']:.2f} max={values['max_us']:.2f} us n={values['samples']}")
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
                    lines.append(f'STAGE repeat={repeat} stage={stage} start_us={start-origin:.2f} '
                                 f'end_us={end-origin:.2f} duration_us={end-start:.2f}')
            lines.append(f"ACCOUNTING repeat={repeat} gap_us={accounting.get('inter_kernel_gap_us', 'N/A')} "
                         f"boundary_us={accounting.get('boundary_residual_us', 'N/A')}")
            for metric in ('prefetch_bandwidth', 'final_get_bandwidth', 'combined_observed_bandwidth'):
                if metric in sample:
                    bandwidth = sample[metric]
                    lines.append(f"BANDWIDTH repeat={sample['repeat']} metric={metric} "
                                 f"bytes={bandwidth['bytes']} span_us={format_metric(bandwidth['span_us'])} GB/s={format_metric(bandwidth['GBps'])}")
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
                         f"mean/p50/p99_us={row['mean_us']:.2f}/{row['p50_us']:.2f}/{row['p99_us']:.2f} "
                         f"GB/s={format_metric(row['effective_GBps'])}")
    lines.extend('INCOMPLETE: ' + value for value in result['missing'])
    lines.extend('N/A: ' + value for value in result['unavailable'])
    (directory / 'diagnostic_replays.log').write_text('\n'.join(lines) + '\n')
    if timeline_rows:
        with (directory / 'timeline.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(timeline_rows[0]))
            writer.writeheader()
            writer.writerows(timeline_rows)
    if config.level == 1:
        rendered = level_one_tables(result)
        (directory / 'level1_summary.log').write_text(rendered + '\n')
        (directory / 'diagnostic_summary.log').write_text(rendered + '\n')
        print(rendered, flush=True)
    elif config.level >= 2:
        rendered = level_two_tables(result)
        (directory / f'level{config.level}_summary.log').write_text(rendered + '\n')
        (directory / 'diagnostic_summary.log').write_text(rendered + '\n')
        print(rendered, flush=True)
    else:
        print('\n'.join(lines), flush=True)
    return result


def algorithm_counts(mapping, selected, rows, posts, resident, windows, *,
                     hot_valid=None, lru_slots=None, reserved_hot=None, logical_ids=None, tail_bounds=None, token_bytes=656, admission_records=True):
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
        if admission_records and (len(admitted) != posted or not np.array_equal(np.sort(ordinals), np.arange(posted))):
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
            unused_count = int(unused[begin:end].sum()) if admission_records else None
            window_rows.append(dict(window=epoch + 1, posted=int(count),
                                    bytes=int(count) * token_bytes, unused=unused_count,
                                    selected_members=int(count) - unused_count if admission_records else None,
                                    cumulative_posted=end, cumulative_unused=int(unused[:end].sum()) if admission_records else None))
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
        # Preserve every admission identity outside graph timing. Physical IDs
        # refer to the pre-call Index; hot validity and reservations must come
        # from that same snapshot, before Index update changes ownership.
        transfers = []
        transfer_counts = dict(stable_hot=0, reserved_hot=0, cold=0, abc=0, repeated=0)
        if admission_records and hot_valid is not None and reserved_hot is not None:
            reservations = np.asarray(reserved_hot)[row]
            if reservations.shape != (resident,):
                raise ValueError('Invalid initial reserved-hot shape')
            window_ids = np.repeat(np.arange(1, windows + 1), epochs)
            seen_tokens = set()
            for ordinal, (position, token) in enumerate(zip(admitted, fetched_ids)):
                position, token = int(position), int(token)
                live = bool(position < resident and valid[position])
                reserved = int(reservations[position]) if position < resident else -1
                abc = tail_bounds is not None and begin_tail <= token < end_tail
                category = ('abc' if abc else 'reserved_hot' if live and reserved >= 0
                            else 'stable_hot' if live else 'cold')
                repeated = token in seen_tokens
                seen_tokens.add(token)
                transfer_counts[category] += 1
                transfer_counts['repeated'] += int(repeated)
                transfers.append(dict(ordinal=ordinal, window=int(window_ids[ordinal]),
                                      position=position, token=token, hot_valid=int(live),
                                      reserved=reserved, category=category,
                                      repeated=repeated, unused=bool(unused[ordinal])))
        requests.append(dict(request=row, selected_unique=len(demand), initial_cold=cold,
                             initial_hot_valid=None if local is None else initial_hot_count,
                             candidate_hot=candidate_hot, prefetch=posted, final=final,
                             KV_read_bytes=(posted + final) * token_bytes,
                             unused_prefetch=int(unused.sum()) if admission_records else None,
                             selected_prefetch=posted - int(unused.sum()) if admission_records else None,
                             useful=None, redundant=None, refetched=None,
                             windows=window_rows, queries=query_rows,
                             prefetch_transfers=transfers, prefetch_classes=transfer_counts))
    totals = {key: (sum(item[key] for item in requests) if all(item[key] is not None for item in requests) else None) for key in
              ('selected_unique', 'prefetch', 'final', 'KV_read_bytes', 'unused_prefetch', 'selected_prefetch')}
    totals['initial_cold'] = None if hot_valid is None else sum(item['initial_cold'] for item in requests)
    totals['cold_miss_rate'] = (None if totals['initial_cold'] is None or not totals['selected_unique']
                               else totals['initial_cold'] / totals['selected_unique'])
    return dict(requests=requests, totals=totals,
                attribution=('unused is exact membership; useful/redundant/refetched lack consumption identity' if admission_records else
                             'Parallel terminal retires admission identities; unused/conflict classification is unavailable'))


def algorithm_before(rotating):
    """Snapshot initial state outside profiled replay, without a Get-time fence."""
    batch = rotating.runtime_shape.batch
    owners = (np.arange(batch) if rotating.row_owners is None
              else rotating.row_owners.cpu().numpy())
    validity = rotating.hot_valid.cpu().numpy()
    if np.any(owners < 0) or np.any(owners >= len(validity)):
        raise ValueError('Diagnostic owner is outside hot validity storage')
    return dict(hot_valid=validity[owners].copy(), lru_slots=rotating.lru_slots.cpu().numpy().copy(),
                reserved_hot=rotating.reserved_hot.cpu().numpy().copy(),
                tail_bounds=None if rotating.tail_bounds is None else rotating.tail_bounds.cpu().numpy().copy())


def prefetch_tag_audit(ids, before, resident):
    """Check tags against the original residency, before Index exchanges."""
    result = []
    for row in range(ids.shape[0]):
        queries = []
        for query in range(4):
            encoded = ids[row, query].astype(np.uint32)
            raw = encoded & np.uint32(0x3fffffff)
            bounded = raw < resident
            safe = np.minimum(raw, resident - 1)
            live = np.asarray(before['hot_valid'])[row, safe] != 0
            reserved = np.asarray(before['reserved_hot'])[row, safe]
            stable = bounded & live & (reserved < 0)
            if before.get('tail_bounds') is not None:
                begin, end = before['tail_bounds'][row]
                stable &= ~((raw >= begin) & (raw < end))
            missing = stable & ((encoded & np.uint32(0x80000000)) == 0)
            queries.append(dict(query=query + 1, stable_hot=int(stable.sum()),
                                missing_hot_tags=int(missing.sum()),
                                missing_positions=raw[missing].astype(int).tolist()))
        result.append(queries)
    return result


def algorithm_after(rotating, before, *, admissions=None):
    shape = rotating.runtime_shape
    try:
        mapping = rotating.mapping.cpu().numpy()
        if admissions is not None:
            mapping = np.full_like(mapping, -1)
            counts = rotating.rows[:, 9].cpu().tolist()
            for b, count in enumerate(counts):
                ids = np.asarray(admissions[b, :count], dtype=np.int64)
                if not 0 <= count <= 2048 or np.any(ids < 0) or np.any(ids >= shape.history) or len(np.unique(ids)) != count:
                    raise ValueError('Invalid terminal prefetch identity snapshot')
                mapping[b, ids] = shape.resident + np.arange(count)
        result = algorithm_counts(
            mapping,
            rotating.arguments[9].cpu().numpy().view(np.int32).reshape(shape.batch, 4, 2048),
            rotating.rows.cpu().numpy().view(np.uint32), rotating.final_posts.cpu().numpy(),
            shape.resident, rotating.rounds, token_bytes=shape.tokenBytes,
            admission_records=admissions is not None or not hasattr(rotating, "parallel_terminal"), **before)
        if hasattr(rotating, 'window_topk_storage'):
            prefix_bytes = 4 * 2048 * 6 + 1024
            # The public prefix is immutable after SaveResident; the private
            # prefix retains tagged IDs after the final incremental merge.
            public = rotating.arguments[13][:shape.batch * prefix_bytes].cpu().numpy().view(np.uint8)
            private = rotating.window_topk_storage[shape.batch * 512:
                       shape.batch * (512 + prefix_bytes)].cpu().numpy().view(np.uint8)
            def decode_ids(buffer):
                return np.stack([buffer[row * prefix_bytes + 4 * 2048 * 2:
                                        row * prefix_bytes + 4 * 2048 * 6].copy().view(np.uint32).reshape(4, 2048)
                                 for row in range(shape.batch)])
            for name, buffer in (('prefix_tag_audit', public), ('final_tag_audit', private)):
                audits = prefetch_tag_audit(decode_ids(buffer), before, shape.resident)
                for request, audit in zip(result['requests'], audits):
                    request[name] = audit
        return result
    except ValueError as error:
        return {'error': str(error)}


def level_one_tables(result):
    return level_two_tables(dict(result, level=1))


def diagnostic_table(title, headers, rows):
    def cell(value):
        return format_metric(value)
    values = [headers, *[[cell(value) for value in row] for row in rows]]
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    def line(row):
        return ' | '.join(value.ljust(width) for value, width in zip(row, widths))
    return '\n'.join([title, line(values[0]), '-+-'.join('-' * width for width in widths),
                      *[line(row) for row in values[1:]]])


def resolver_phase_rows(timeline):
    """Partition each resolver core's existing stamps; never add other cores."""
    rows = []
    for core in timeline['cores']:
        if not core['role'].startswith('resolver'):
            continue
        buckets = [0.] * 10
        for stage in core['stages']:
            name = stage['name']
            if name in ('input_and_bitmap', 'initialize'):
                index = 0
            elif name.startswith('Q') and name.endswith('.resolve'):
                index = 1
            elif name.startswith('Q') and name.endswith(('.output', '.descriptor', '.publish')):
                index = 2
            elif name in ('prefetch_CQ_join', 'prefetch_completion_join'):
                index = 3
            elif name == 'wait.index_update':
                index = 4
            elif name == 'wait.prefetch_ready':
                index = 5
            elif name.startswith('wait.Q') and name.endswith('_fetch'):
                index = 6
            elif name in ('prefetch_residency', 'residency'):
                index = 7
            elif name == 'status':
                index = 8
            else:
                # Includes bookkeeping between stamps; not assumed idle or CQ wait.
                index = 9
            buckets[index] += stage['duration_us']
        rows.append([core['block'], *buckets, core['duration_us'],
                     sum(buckets) - core['duration_us']])
    return rows


def metric_distribution(values):
    """Unavailable observations are not zero; never infer quantiles from summaries."""
    valid = [float(v) for v in values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
    if not valid:
        return [None] * 6 + [0, len(values)]
    quantiles = np.percentile(valid, [50, 90, 99]).tolist() if len(valid) > 1 else [None] * 3
    return [float(np.mean(valid)), *quantiles, min(valid), max(valid), len(valid), len(values)]


def level_two_tables(result):
    """Stable category/table order at every level; retain raw replays in JSON."""
    level = result.get('level', 2)
    tables = {}
    notes = set()
    current = ()
    expected = 1

    def add(title, identities, headers, rows, *, samples=None):
        if not rows:
            return
        table = tables.setdefault(title, dict(identities=identities, groups={}))
        for row in rows:
            identity = (*current, *row[:len(identities)])
            for name, value in zip(headers, row[len(identities):]):
                values, count = table['groups'].setdefault((*identity, name), ([], expected if samples is None else samples))
                values.append(value)

    for index, capture in enumerate(result['captures'], 1):
        path = capture.get('path', 'fused')
        samples = capture['samples']
        round_counts = {}
        for sample in samples:
            key = sample.get('source_round', '-')
            round_counts[key] = round_counts.get(key, 0) + 1
        for sample in samples:
            source_round = sample.get('source_round', '-')
            current = (index, path, source_round)
            expected = round_counts[source_round]
            accounting = sample.get('accounting', {})
            add('Stage distribution', ['Stage'], ['Duration us'],
                [[name, value] for name, value in accounting.get('stages_us', {}).items()])
            add('Stage distribution', ['Stage'], ['Duration us'],
                [[name, accounting[key]] for name, key in (('within-call gaps', 'inter_kernel_gap_us'),
                  ('overlap', 'overlap_us')) if key in accounting])
            requests = sample.get('requests', [])
            traffic, windows, queries = [], [], []
            for request in sample.get('algorithm', {}).get('requests', []):
                demand, cold = request.get('selected_unique'), request.get('initial_cold')
                prefetch, final, unused = (request.get(k) for k in ('prefetch', 'final', 'unused_prefetch'))
                traffic.append([request['request'], request.get('initial_hot_valid'), demand, cold,
                    100*cold/demand if demand and cold is not None else None,
                    prefetch, final, request.get('KV_read_bytes'),
                    (prefetch+final)/cold if cold and prefetch is not None and final is not None else None,
                    unused, 100*unused/prefetch if prefetch and unused is not None else None])
                for window in request.get('windows', []):
                    windows.append([request['request'], window['window'], *[window.get(k) for k in
                        ('posted', 'bytes', 'unused', 'cumulative_posted', 'cumulative_unused')]])
                for query in request.get('queries', []):
                    owner = next((r for r in requests if r['batch'] == request['request']), {})
                    counts = owner.get('final_counts') or []
                    conflict = counts[query['query']-1].get('conflict') if len(counts) >= query['query'] else None
                    local = query.get('initial_local')
                    queries.append([request['request'], query['query'], query['unique'], query['cross_q_duplicate'],
                        query['new_unique'], local, query['new_unique']-local if local is not None else None,
                        query['final_posted'], conflict])
            add('Demand and actual reads', ['Request'], ['Hot', 'Demand', 'Cold', 'Cold miss %',
                'Prefetch', 'Final', 'KV bytes', 'Read amplification', 'Unused', 'Unused %'], traffic)
            if level >= 2:
                add('Window prefetch / unused', ['Request', 'Window'], ['Posted', 'Bytes', 'Unused',
                    'Cumulative posted', 'Cumulative unused'], windows)
                add('Per-Q demand / repair', ['Request', 'Q'], ['Unique demand', 'Seen in earlier Qs',
                    'New demand', 'New: already local', 'New: initially cold', 'Final reads', 'Conflict reads'], queries)
            if level >= 3:
                add('Per-Q prefetch rank / progress', ['Request', 'Q', 'Window'],
                    ['Score ready %', 'TopK covered %', 'Rank'], sample.get('rank_progress', []))
            add('Effective bandwidth', ['Scope'], ['Bytes', 'Span us', 'GB/s'],
                [[name, *[sample.get(key, {}).get(k) for k in ('bytes', 'span_us', 'GBps')]]
                 for key, name in (('prefetch_bandwidth', 'prefetch'), ('final_get_bandwidth', 'final Get'),
                                   ('combined_observed_bandwidth', 'combined'))] if path != 'serial' else [])
            if level < 2:
                continue
            timeline = sample.get('timeline')
            if timeline:
                add('Completion distribution', [], ['Observed span us'], [[timeline['observed_span_us']]])
                cores = timeline['cores']
                add('Active cores', ['Block', 'Role'], ['Start us', 'End us', 'Duration us', 'Last frequency'],
                    [[c['block'], c['role'], c['start_us'], c['end_us'], c['duration_us'],
                      c['block'] == timeline['last_block']] for c in cores])
                legacy = resolver_phase_rows(timeline)
                if legacy:
                    add('Resolver phase breakdown', ['Block'], ['Input/init us', 'Q1-Q4 resolve us',
                        'Q output/publish us', 'Prefetch completion us', 'Wait update us', 'Wait prefetch flag us',
                        'Wait Q Fetch us', 'Residency us', 'Status us', 'Other/control us', 'Total us', 'Delta us'], legacy)
                elif any(c['role'].startswith('terminal') for c in cores):
                    notes.add('Parallel terminal has no separate legacy resolver/Fetch workers: their empty phase, classification and URMA-counter tables are omitted; actual work is shown in Terminal phase breakdown / Core work / wait intervals.')
                    add('Terminal phase breakdown', ['Block', 'Role', 'Phase'], ['Duration us'],
                        [[c['block'], c['role'], p['name'], p['duration_us']] for c in cores
                         for p in c['stages'] if not p['name'].startswith('control_gap')])
                groups = sample.get('final_get_bandwidth', {}).get('groups', [])
                rates = []
                for group in groups:
                    core = next((c for c in cores if c['block'] == group['block']), None)
                    if core is None:
                        continue
                    spans = [p for p in core['stages'] if p['name'].endswith('.KVAF_Get') and p['duration_us'] > 0]
                    duration = group.get('span_us') or (max(p['end_us'] for p in spans)-min(p['start_us'] for p in spans)
                                if spans and group['entries'] else None)
                    rates.append([group['block'], group['query'], group['lane'], group['entries'], group['bytes'],
                                  duration, group['bytes']/duration/1000 if duration else None])
                add('Per-block effective bandwidth', ['Block', 'Q', 'Lane'], ['Reads', 'Bytes', 'Span us', 'GB/s'], rates)
                if level >= 3:
                    phases, counters = [], []
                    for core in cores:
                        gaps = [p['duration_us'] for p in core['stages'] if p['name'].startswith('control_gap')]
                        if gaps:
                            phases.extend([[core['block'], core['role'], name, None, None, value, None]
                                for name, value in (('control gaps: total', sum(gaps)), ('control gaps: max', max(gaps)))])
                        for phase in core['stages']:
                            if phase['name'].startswith('control_gap'):
                                continue
                            cycles = (phase['end_cycles']-phase['start_cycles']
                                      if 'end_cycles' in phase and 'start_cycles' in phase else None)
                            phases.append([core['block'], core['role'], phase['name'], phase['start_us'],
                                           phase['end_us'], phase['duration_us'], cycles])
                        values = core.get('KVAF_counters')
                        if values:
                            counters.append([core['block'], core['role'], *[values[k]/timeline['clock_mhz']
                                if values.get(k) is not None else None for k in
                                ('safe_post_cycles', 'safe_CQ_cycles', 'conflict_post_cycles', 'conflict_CQ_cycles')]])
                    add('Core work / wait intervals', ['Block', 'Role', 'Phase'], ['Start us', 'End us', 'Duration us', 'Cycles'], phases)
                    add('URMA group counters', ['Block', 'Role'], ['Safe submit us', 'Safe CQ us', 'Conflict submit us', 'Conflict CQ us'], counters)
            if requests:
                first = requests[0]
                mhz = first.get('core_timeline_raw', {}).get('clock_mhz') or first.get('terminal_timeline', {}).get('clock_mhz')
                if level >= 3:
                    work, steps = [], []
                    for event in first.get('prefetch_services', []):
                        identity = [event['event'], event['core'], event['epoch'], event.get('lane')]
                        work.append([*identity, event['bitmap_bytes'], event['candidate_count'], event['posted_count']])
                        for name, key in (('predecessor wait', 'predecessor_wait_cycles'), ('incremental', 'incremental_cycles'),
                                          ('scan', 'scan_cycles'), ('candidates', 'candidate_cycles'), ('submit/copy', 'post_cycles')):
                            value = event.get(key)
                            steps.append([*identity, name, value, value/mhz if value is not None and mhz else None])
                    add('Prefetch service work', ['Event', 'Core pair', 'Epoch', 'Lane'], ['Bitmap bytes', 'Candidates', 'Posted'], work)
                    add('Prefetch service steps', ['Event', 'Core pair', 'Epoch', 'Lane', 'Step'], ['Cycles', 'Duration us'], steps)
                for request in requests:
                    def us(value):
                        return value/mhz if value is not None and mhz else None
                    if level >= 3:
                        add('LI TopK classification', ['Request', 'Q'],
                            ['Map wait us', 'Barrier us', 'Load1 us', 'Classify1+pipe wait us', 'Wait2 us',
                             'Classify2+pipe wait us', 'Bitmap us', 'Publish us'],
                            [[request['batch'], q, *[us(values.get(k)) for k in ('mapping_wait', 'barrier', 'load_first',
                              'classify_first', 'wait_second', 'classify_second', 'bitmap_merge', 'publish')]]
                             for q, values in enumerate(request.get('li', []), 1)])
                    update, waits = request.get('update', []), request.get('update_wait', [])
                    add('Per-Q resolver / update', ['Request', 'Q'], ['Resolve us', 'Output us', 'Pack us',
                        'Publish us', 'Update wait us', 'Update apply us'],
                        [[request['batch'], q+1, *[us(p.get(k)) for k in ('resolve', 'output', 'pack', 'publish')],
                          us(waits[q+1]) if len(waits)>q+1 else None, us(update[q+1]) if len(update)>q+1 else None]
                         for q, p in enumerate(request.get('producer', []))])
        # Single graph event is one observation, not one per invocation.
        accounting_samples = ([dict(accounting=capture['graph_accounting'])] if capture.get('graph_accounting') else samples)
        for sample in accounting_samples:
            current = (index, path, sample.get('source_round', '-'))
            expected = len(accounting_samples)
            a = sample.get('accounting', {})
            if not a or 'error' in a:
                continue
            stages = a['stages_us']
            gap, boundary = a['inter_kernel_gap_us'], a['boundary_residual_us']
            overlap = a.get('overlap_us', sum(stages.values())-(a['kernel_span_us']-gap))
            total = sum(stages.values())-overlap+gap+boundary
            add('Same-replay graph accounting', [], [*[f'{stage} us' for stage in stages], 'Overlap us', 'Gap us', 'Boundary us',
                'Reconstructed us', 'Graph event us', 'Delta us'],
                [[*stages.values(), overlap, gap, boundary, total, a['event_us'], total-a['event_us']]])

    if level >= 3:
        for index, comparison in enumerate(result.get('fetch_only', []), 1):
            for row in comparison.get('rows', []):
                current = (f'fetch{index}', row['version'], '-')
                values = row.get('samples_us', [])
                expected = len(values)
                if not values:
                    notes.add('Fetch-only raw samples unavailable: quantiles cannot be reconstructed from summary values.')
                    continue
                for value in values:
                    add('Fetch-only comparison', ['Q'], ['Duration us', 'Bytes', 'GB/s', 'AIVs', 'Jettys/AIV', 'Reads',
                        'Safe reads', 'Conflict reads'], [[row['query'], value, row['bytes'],
                        row['bytes']/value/1000 if value and row['bytes'] else None, row['aiv'], row['jettys_per_aiv'],
                        row['count'], row.get('safe_count'), row.get('conflict_count')]])

    categories = (
        ('ALGORITHM / KV TRAFFIC', ('Demand and actual reads', 'Window prefetch / unused', 'Per-Q demand / repair',
          'Per-Q prefetch rank / progress', 'Prefetch service work')),
        ('PERFORMANCE | same measured graph; detailed stamps OFF', ('Stage distribution', 'Same-replay graph accounting')),
        ('TRANSFER | separate instrumented replays', ('Effective bandwidth', 'Per-block effective bandwidth', 'URMA group counters', 'Fetch-only comparison')),
        ('CORE WORK / WAIT | separate instrumented replays', ('Completion distribution', 'Active cores',
          'Resolver phase breakdown', 'Terminal phase breakdown', 'Per-Q resolver / update',
          'Core work / wait intervals', 'LI TopK classification', 'Prefetch service steps')))
    lines = [f"DIAG level={level} transport={result['transport']} trace={result['trace']}",
        'Across-replay statistics: avg / p50 / p90 / p99 / min / max. N = valid samples; Total = expected samples. Missing = -. Single observations have no percentiles.',
        'Source rounds, requests, queries, windows and logical blocks remain separate. Quantiles of different metrics must not be added.',
        'Ratios/bandwidth are calculated per replay before aggregation. Zero denominator = -. Last frequency is count/Total.',
        'Terminal bandwidth: submission to software-observed CQ completion, including overlapping Index/output work; not isolated Get latency.']
    for index, capture in enumerate(result['captures'], 1):
        lines.append(f"Capture {index} ({capture.get('path', 'fused')}): performance={capture['source']}")
        details = sorted({s.get('algorithm_source', s.get('detail_source')) for s in capture['samples']
                          if s.get('algorithm_source', s.get('detail_source'))})
        if details:
            lines.append('  Detailed algorithm/core/transfer source: ' + ', '.join(details))
    for category, names in categories:
        available = [name for name in names if name in tables]
        if not available:
            continue
        lines.append(category)
        for title in available:
            table = tables[title]
            rows = []
            for key, (values, count) in table['groups'].items():
                if key[-1] == 'Last frequency':
                    stats = [f'{sum(values)}/{count}', None, None, None, None, None, len(values), count]
                else:
                    stats = metric_distribution(values)
                    stats[-1] = count
                rows.append([*key, *['-' if v is None else v for v in stats]])
            lines.append(diagnostic_table(title, ['Capture', 'Path', 'Source round', *table['identities'],
                'Metric', 'avg', 'p50', 'p90', 'p99', 'min', 'max', 'N', 'Total'], rows))
    lines.extend(sorted(notes))
    lines.extend('INCOMPLETE: ' + value for value in result.get('missing', []))
    lines.extend('Unavailable: ' + value for value in result.get('unavailable', []))
    return '\n\n'.join(lines)


def footprint(addresses, token_bytes=656):
    a = np.asarray(addresses, dtype=np.uint64)
    unique = np.unique(a)
    result = dict(
        requests=len(a),
        unique=len(unique),
        address_set_sha256=hashlib.sha256(unique.tobytes()).hexdigest(),
        span_bytes=int(unique[-1] - unique[0]) + token_bytes if len(unique) else 0,
    )
    for size in (4096, 2097152):
        # Include straddling tokens; these are virtual regions, not physical pages.
        pages = np.unique(np.concatenate((a // size, (a + token_bytes - 1) // size)))
        result[f"regions_{size}"] = len(pages)
    gaps = np.diff(unique)
    result["sorted_gap_p50_bytes"] = float(np.median(gaps)) if len(gaps) else 0
    return result


def address_snapshot(rotating, remote, block_bases):
    rows = rotating.rows.cpu().numpy().view(np.uint32)
    mapping = rotating.mapping.cpu().numpy()
    prefetch = remote[mapping >= 0]
    blocks = rotating.blocks.cpu().numpy().reshape(len(rows), -1)
    offsets = rotating.offsets.cpu().numpy().reshape(len(rows), -1)
    final = []
    for row in range(len(rows)):
        n = int(rows[row, 247])
        ids = blocks[row, :n]
        if np.any(ids < 0) or np.any(ids >= len(block_bases)):
            raise ValueError("Final descriptor block out of range")
        final.extend((block_bases[ids] + offsets[row, :n].astype(np.uint64)).tolist())
    if len(prefetch) != int(rows[:, 9].astype(np.uint64).sum()):
        raise ValueError("Prefetch mapping count differs from requests; footprint inference invalid")
    return {
        "prefetch": footprint(prefetch),
        "final": footprint(final),
        "all": footprint(np.concatenate((prefetch, np.asarray(final, dtype=np.uint64)))),
    }


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def account(events, name, elapsed, *, full=False, fused_final=False):
    scopes = [e for e in events if e.get("name") == name and e.get("ph") == "X"]
    if len(scopes) != 1:
        return dict(error=f"Expected one CPU replay scope, found {len(scopes)}")
    scope = scopes[0]
    origin = Decimal(str(scope["ts"]))
    start, end = 0.0, float(scope["dur"])
    kernels = (dict(init="v9_init_kernel", li="v9_li_kernel", resolve="v9_resolve_kernel",
                    publish="kvop_direct_update_outputs_early_kernel",
                    apply="kvop_direct_apply_owned_kernel", fetch="v9_fetch_update_kernel")
               if full else {stage: f"v9_{stage}_kernel" for stage in DEVICE_STAGES})
    if full and any(e.get("name", "").lower().startswith("v9_streaming_resolve_update") for e in events):
        # This kernel contains both resolver and index update; reporting separate
        # kernel durations would invent a boundary that does not exist.
        kernels = dict(init="v9_init_kernel", li="v9_li_kernel",
                       resolve_update="v9_streaming_resolve_update")
        if not fused_final:
            kernels["fetch"] = "v9_fetch_update_kernel"
    parallel_terminal = full and any(
        e.get("name", "").lower().startswith("v9_parallel_terminal") for e in events
    )
    if parallel_terminal:
        kernels = dict(li="v9_li_kernel", topk_resolve_update="v9_parallel_terminal")
    if any(e.get("name", "").lower().startswith("v9_early_fetch_kernel") for e in events):
        kernels["early_fetch"] = "v9_early_fetch_kernel"
    suffix = r"(?:(?:_\d+)|(?:_mix_ai[cv])|(?:_ai[cv]))*" if parallel_terminal else r"(?:_\d+)*"
    names = {stage: re.compile(re.escape(prefix) + suffix) for stage, prefix in kernels.items()}
    if parallel_terminal:
        names["li"] = re.compile(r"(?:v9_li_kernel|v9_li_fixed_hot|v9_li_fixed_cold)" + suffix)
    groups = {stage: [] for stage in names}
    for event in events:
        n = event.get("name", "").lower()
        cat = (event.get("cat") or "").lower()
        # CANN may omit cat. Accept exact kernel names with recognized dispatch
        # suffixes, while excluding launch APIs and explicitly tagged CPU ops.
        stage = next((key for key, pattern in names.items() if pattern.fullmatch(n)), None)
        if event.get("ph") != "X" or stage is None or (cat and "kernel" not in cat):
            continue
        timestamp = float(Decimal(str(event["ts"])) - origin)
        finish = timestamp + float(event.get("dur", 0))
        if not start <= timestamp or finish > end:
            continue
        groups[stage].append((timestamp, finish))
    # Current URMA streaming starts with hot LI; older captures have Init.
    if fused_final and 'resolve_update' in groups and not groups.get('init'):
        groups.pop('init', None)
    if any(not value for value in groups.values()):
        return dict(
            error="Missing named device kernels; inspect raw trace, no ordinal guessing",
            found={k: len(v) for k, v in groups.items()},
        )
    durations = {k: sum(b - a for a, b in union(v)) for k, v in groups.items()}
    intervals = union([interval for values in groups.values() for interval in values])
    busy = sum(b - a for a, b in intervals)
    overlap = sum(durations.values()) - busy
    if overlap > 0.1:
        return dict(error="Cross-stage overlap prevents additive stage accounting", overlap_us=overlap)
    span = intervals[-1][1] - intervals[0][0]
    residual = elapsed - span if elapsed is not None else None
    if residual is not None and residual < -2:
        return dict(error="Event/trace interval mismatch", event_us=elapsed, kernel_span_us=span)
    return dict(
        event_us=elapsed,
        kernel_span_us=span,
        stages_us=durations,
        intervals_us=groups,
        inter_kernel_gap_us=span - busy,
        boundary_residual_us=residual,
        note="Boundary residual includes event-to-kernel edges and clock measurement error; not proven CPU idle",
    )


def decode_final_detail(owner_words, status_words):
    """Decode per-request diagnostic counters after synchronization."""
    detail = []
    for owner, status in zip(owner_words, status_words):
        if owner[4] and owner[11]:
            detail.append(dict(ticks=[int(x) for x in owner[4:12]],
                               safe=int(owner[12]), conflict=int(owner[13]), pending=int(owner[14]),
                               kvaf_ticks=[int(x) for x in status[8:16].view(np.uint64)]))
    return detail


def collect_details(torch, rotating, graph, reset, verify, rounds, folder, pipeline, *, ubmem=False):
    """Separate instrumented replays; never contribute to kernel-stage timing."""
    stream = torch.npu.current_stream(rotating.device)
    pipeline.counter_clock_mhz = pipeline.calibrate_counter_mhz()
    samples = []
    try:
        pipeline.set_diagnostics(True)
        for step in range(rounds):
            reset()
            pipeline.times.zero_()
            initial = algorithm_before(rotating) if getattr(pipeline, 'terminal', None) is not None else None
            if hasattr(rotating, 'rank_progress'):
                rotating.rank_progress.zero_()
            stream.synchronize()
            if ubmem:
                graph.replay(stream)
            else:
                graph.replay()
            stream.synchronize()
            verify()
            sample = dict(round=step, streaming_cycles=pipeline.diagnostic_cycles())
            if initial is not None:
                sample['algorithm'] = algorithm_after(rotating, initial, admissions=pipeline.events_src.cpu().numpy())
            if hasattr(rotating, 'rank_progress'):
                records = rotating.rank_progress.cpu().tolist()
                bounds = rotating.query_ends.cpu().tolist()
                sample['rank_progress'] = [
                    [b, q + 1, epoch, *decode_rank_progress(word, bounds[b][q])]
                    for b, queries in enumerate(records) for q, epochs in enumerate(queries)
                    for epoch, word in enumerate(epochs[:rotating.rounds]) if word]
            samples.append(sample)
    finally:
        pipeline.set_diagnostics(False)
        stream.synchronize()
    (folder / 'detail_samples.json').write_text(json.dumps(samples, indent=2))


def decode_rank_progress(word, bound):
    # Accept both the original 18-bit capture and extended long-history capture.
    if word & (1 << 63):
        mask = (1 << 26) - 1
        ready, covered, rank = word & mask, (word >> 26) & mask, ((word >> 52) & 2047) + 1
    else:
        mask = (1 << 18) - 1
        ready, covered, rank = word & mask, (word >> 18) & mask, (word >> 36) & 4095
    return [100 * ready / max(bound, 1), 100 * covered / max(bound, 1), rank]


def collect_core(torch, rotating, graph, reset, verify, rounds, folder, *, full=False, diagnostics=None, fused_final=False):
    """Profile the existing core graph separately from primary timing."""
    import torch_npu
    from torch.profiler import record_function

    stream = torch.npu.current_stream(rotating.device)
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    with profile:
        for step in range(rounds):
            reset()
            if diagnostics is not None and hasattr(rotating, "rank_progress"):
                rotating.rank_progress.zero_()
            stream.synchronize()
            algorithm_initial = algorithm_before(rotating)
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvop_{'full' if full else 'core'}_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            # Read diagnostic counters after the timed graph and correctness checks.
            owner_words = rotating.owners[:len(rows) * 32].cpu().numpy().view(np.uint64).reshape(len(rows), 16)
            status_words = rotating.row_status.cpu().numpy().view(np.uint32).reshape(len(rows), 32)
            detail = decode_final_detail(owner_words, status_words)
            samples.append(
                dict(
                    name=name,
                    detail=detail,
                    round=step,
                    us=begin.elapsed_time(end) * 1000,
                    prefetch=int(rows[:, 9].astype(np.uint64).sum()),
                    final=int(rows[:, 247].astype(np.uint64).sum()),
                    epochs=rows[:, 16 : 16 + rotating.rounds].astype(np.uint64).sum(axis=0).tolist(),
                )
            )
            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)
            if diagnostics is not None and hasattr(rotating, "rank_progress"):
                records = rotating.rank_progress.cpu().tolist()
                bounds = rotating.query_ends.cpu().tolist()
                samples[-1]["rank_progress"] = [
                    [b, q + 1, epoch, *decode_rank_progress(word, bounds[b][q])]
                    for b, queries in enumerate(records) for q, epochs in enumerate(queries)
                    for epoch, word in enumerate(epochs[:rotating.rounds]) if word
                ]
            if diagnostics is not None:
                samples[-1]["streaming_cycles"] = diagnostics()
    # Save events/counters before parsing, so collection never needs repeating
    # merely because a vendor trace schema differs.
    label = "full" if full else "core"
    (folder / f"{label}_samples.json").write_text(json.dumps(samples, indent=2))
    path = folder / f"{label}_trace.json"
    profile.export_chrome_trace(str(path.resolve()))
    trace = json.loads(path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account(events, sample["name"], sample["us"], full=full, fused_final=fused_final)
    result = dict(samples=samples, trace=str(path))
    (folder / f"{label}_attribution.json").write_text(json.dumps(result, indent=2))
    (folder / f"{label}_summary.log").write_text(core_summary(result, label=label) + "\n")
    return result


def core_summary(result, label="core"):
    lines = [f"{label.upper()} DEVICE TRACE: separate profiled pass; all values below are from the same replay."]
    for row in result["samples"]:
        a = row["accounting"]
        if "error" in a:
            lines.append(f"round={row['round']} attribution unavailable: {a}")
            continue
        values = dict(a["stages_us"], gap=a["inter_kernel_gap_us"], boundary=a["boundary_residual_us"])
        lines.append(
            f"round={row['round']} total_us={row['us']:.3f} "
            + " ".join(f"{key}_us={value:.3f}" for key, value in values.items())
            + f" sum_us={sum(values.values()):.3f} prefetch={row['prefetch']} final={row['final']}"
            + f" epochs={row['epochs']}"
        )
    return "\n".join(lines)


def account_ubmem(events, name, elapsed):
    """Attribute exact UBmem device names within a single profiled replay."""
    import re

    scopes = [event for event in events if event.get("name") == name and event.get("ph") == "X"]
    if len(scopes) != 1:
        return {"error": f"Expected one replay scope, got {len(scopes)}"}
    scope = scopes[0]
    origin = Decimal(str(scope["ts"]))
    duration = float(scope["dur"])
    stages = {stage: [] for stage in ("init", "li", "resolve", "fetch", "complete",
                                     "update_owned", "update_outputs", "apply_owned", "resolve_update_fetch")}
    examples = []
    for event in events:
        if event.get("ph") != "X":
            continue
        kernel = str(event.get("name", "")).lower()
        category = str(event.get("cat") or "").lower()
        match = re.fullmatch(r"kvip_ubmem_(?:656|2624)_(init|li|resolve|fetch|complete)_kernel(?:_\d+)*", kernel)
        update = re.fullmatch(r"kvip_ubmem_direct_(update_owned|update_outputs|apply_owned)_kernel(?:_\d+)*", kernel)
        streaming = re.fullmatch(r"v9_streaming_resolve_update_(?:656|2624)(?:_\d+)?", kernel)
        if not match and not update and not streaming:
            if "kvip_ubmem" in kernel and len(examples) < 12:
                examples.append((kernel, category))
            continue
        if category and any(value in category for value in ("cpu", "python", "runtime")):
            continue
        start = float(Decimal(str(event["ts"])) - origin)
        end = start + float(event.get("dur", 0))
        if start < 0 or end > duration:
            continue
        stages["resolve_update_fetch" if streaming else (match.group(1) if match else update.group(1))].append((start, end))
    required = ("init", "li", "resolve", "complete", "update_owned", "update_outputs", "apply_owned")
    if stages["resolve_update_fetch"]:
        required = ("init", "li", "resolve_update_fetch")
    if any(not stages[key] for key in required):
        return {"error": "Missing named UBmem device kernels", "found": {key: len(value) for key, value in stages.items()},
                "examples": examples}
    timings = {key: sum(end - start for start, end in union(value)) for key, value in stages.items()}
    intervals = union([interval for value in stages.values() for interval in value])
    span = intervals[-1][1] - intervals[0][0]
    busy = sum(end - start for start, end in intervals)
    overlap = sum(timings.values()) - busy
    timings["update"] = sum(timings[key] for key in ("update_owned", "update_outputs", "apply_owned"))
    return {"stages_us": timings, "intervals_us": stages, "kernel_span_us": span, "inter_kernel_gap_us": span - busy,
            "boundary_residual_us": elapsed - span, "overlap_us": overlap,
            "fetch_separate": bool(stages["fetch"]),
            "note": "resolve includes inline fetch when fetch_separate is false; boundary includes measurement edges"}


def collect_ubmem(torch, rotating, graph, reset, verify, rounds, folder, streaming=None, level=1):
    """Collect a separately profiled graph replay; never included in event timing."""
    import torch_npu
    from torch.profiler import record_function

    folder.mkdir(parents=True, exist_ok=True)
    stream = torch.npu.current_stream(rotating.device)
    samples = []
    trace_path = folder / "trace.json"
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=rounds, repeat=1),
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace_path.resolve())),
    )
    if streaming is not None:
        streaming.set_diagnostics(False)
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            algorithm_initial = algorithm_before(rotating)
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f"kvip_ubmem_trace_{step}"
            with record_function(name):
                begin.record(stream)
                graph.replay(stream)
                end.record(stream)
                end.synchronize()
                torch.npu.synchronize()
            verify()
            rows = rotating.rows.cpu().numpy().view(np.uint32)
            samples.append({"name": name, "round": step, "us": begin.elapsed_time(end) * 1000,
                            "prefetch": int(rows[:, 9].astype(np.uint64).sum()),
                            "final": int(rows[:, 247].astype(np.uint64).sum())})
            samples[-1]["algorithm"] = algorithm_after(rotating, algorithm_initial)
            profile.step()
    if streaming is not None:
        streaming.set_diagnostics(False)
    (folder / "samples.json").write_text(json.dumps(samples, indent=2))
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    for sample in samples:
        sample["accounting"] = account_ubmem(events, sample["name"], sample["us"])
        account = sample["accounting"]
        if "error" not in account and hasattr(rotating.library, "v9_fetch_workers") and not account["fetch_separate"] and not account["stages_us"].get("resolve_update_fetch"):
            account["error"] = "Separate fetch kernel expected but absent from trace"
    (folder / "attribution.json").write_text(json.dumps(samples, indent=2))
    if streaming is not None and level >= 2:
        collect_details(torch, rotating, graph, reset, verify, rounds, folder, streaming, ubmem=True)
    for sample in samples:
        account = sample["accounting"]
        if "error" in account:
            print(f"UBMEM_TRACE_ERROR round={sample['round']} {account}; raw={trace_path}", flush=True)
            continue  # Invalid attribution is retained, never reported as stage timing.
        print(f"UBMEM_TRACE round={sample['round']} total_us={sample['us']:.3f} "
              + " ".join(f"{key}_us={value:.3f}" for key, value in account["stages_us"].items())
              + f" span_us={account['kernel_span_us']:.3f} gap_us={account['inter_kernel_gap_us']:.3f}"
              + f" boundary_us={account['boundary_residual_us']:.3f} overlap_us={account['overlap_us']:.3f}"
              + f" fetch_separate={account['fetch_separate']} prefetch={sample['prefetch']} final={sample['final']}",
              flush=True)
    print(f"UBMEM_TRACE_FILES {folder}", flush=True)


def collect_replay(torch, state, graph, sequence, validate, config, folder):
    """Profile a continuous sequence; never restore captured state per round."""
    import torch_npu
    from reporting import CorrectnessRecorder, atomic_json
    from torch.profiler import record_function

    folder.mkdir(parents=True, exist_ok=True)
    pipeline, rotating = state.forward, state.rotating
    stream = torch.npu.current_stream(state.device)
    checks = CorrectnessRecorder()
    # A complete sequence is the unit of reset/warmup, as in unprofiled bench.
    for repeat in range(config.warmup):
        state.reset()
        for inputs in sequence:
            state.prepare(inputs)
            stream.synchronize()
            graph.replay()
            stream.synchronize()
            checks.check(f'warmup={repeat} source_round={inputs.source_round}',
                         lambda: validate(inputs))
    pipeline.set_diagnostics(False)
    trace_path = folder / 'trace.json'
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0,
                                            active=config.iterations * len(sequence.rounds), repeat=1),
        on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(trace_path.resolve())))
    try:
        with profile:
            for repeat in range(config.iterations):
                state.reset()
                for inputs in sequence:
                    state.prepare(inputs)
                    stream.synchronize()
                    before = algorithm_before(rotating)
                    # TopK and prefetch positions refer to the *pre-call* Index
                    # layout, which update mutates during this replay.
                    before['logical_ids'] = state.storage.logical_ids[1:].cpu().numpy().copy()
                    begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
                    name = f'real_repeat_{repeat}_source_{inputs.source_round}'
                    with record_function(name):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                        end.synchronize()
                    good = checks.check(name, lambda: validate(inputs))
                    algorithm = algorithm_after(rotating, before)
                    totals = algorithm.get('totals', {})
                    sample = dict(name=name, round=repeat, source_round=inputs.source_round,
                                  scheduler_round=inputs.scheduler_round,
                                  identities=inputs.identities, us=begin.elapsed_time(end) * 1000,
                                  correctness='PASS' if good else 'FAIL', algorithm=algorithm,
                                  prefetch=totals.get('prefetch'), final=totals.get('final'))
                    samples.append(sample)
                    atomic_json(folder / 'samples.json', samples)
                    profile.step()
    finally:
        pipeline.set_diagnostics(False)
    trace = json.loads(trace_path.read_text())
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    for sample in samples:
        sample['accounting'] = (account_ubmem(events, sample['name'], sample['us'])
                               if config.transport == 'ubmem' else
                               account(events, sample['name'], sample['us'], full=True, fused_final=True))
    detail_samples = []
    if config.level >= 2:
        pipeline.counter_clock_mhz = pipeline.calibrate_counter_mhz()
        try:
            pipeline.set_diagnostics(True)
            for repeat in range(config.iterations):
                state.reset()
                for inputs in sequence:
                    state.prepare(inputs)
                    pipeline.times.zero_()
                    initial = None
                    if getattr(pipeline, 'terminal', None) is not None:
                        initial = algorithm_before(rotating)
                        initial['logical_ids'] = state.storage.logical_ids[1:].cpu().numpy().copy()
                    stream.synchronize()
                    graph.replay()
                    stream.synchronize()
                    checks.check(f'detail={repeat} source={inputs.source_round}', lambda: validate(inputs))
                    detail = dict(round=repeat, source_round=inputs.source_round,
                                  streaming_cycles=pipeline.diagnostic_cycles())
                    if getattr(pipeline, 'terminal', None) is not None:
                        detail['algorithm'] = algorithm_after(rotating, initial, admissions=pipeline.events_src.cpu().numpy())
                    detail_samples.append(detail)
        finally:
            pipeline.set_diagnostics(False)
            stream.synchronize()
    # Each source round has its own repeat distribution. Never pool different
    # decode rounds under the same synthetic-case summary.
    for source_round in sorted({sample['source_round'] for sample in samples}):
        if detail_samples:
            atomic_json(folder / f'round-{source_round}' / 'detail_samples.json',
                        [sample for sample in detail_samples if sample['source_round'] == source_round])
        atomic_json(folder / f'round-{source_round}' / 'full_attribution.json', dict(
            samples=[sample for sample in samples if sample['source_round'] == source_round]))
    return dict(correctness=checks.status, failures=checks.failures,
                rounds=len(sequence.rounds), repeats=config.iterations, samples=len(samples),
                trace_path=str(trace_path))


def account_native(events, name, elapsed):
    """Attribute native graph kernels by explicit names, never launch order.

    LIDU already includes descriptor/address preparation. Unknown kernels remain
    explicit and make coverage incomplete instead of being charged to Get.
    """
    scopes = [event for event in events if event.get('name') == name and event.get('ph') == 'X']
    if len(scopes) != 1:
        return dict(error=f'Expected one native replay scope, found {len(scopes)}')
    scope = scopes[0]
    origin = Decimal(str(scope['ts']))
    groups, unknown = {}, []
    aliases = {
        'li': ('quantlightningindexer',),
        'lidu_prepare': ('a5lidufusedbatchgetpreparemtp',),
        'fetch': (
            'kvafexecutebatchgetaivkernelv3',
            'kvafexecutebatchgetubmemkernelv2',
            'kvafexecutebatchgetaivkernelv2',
        ),
    }
    for event in events:
        if event.get('ph') != 'X':
            continue
        category = (event.get('cat') or '').lower()
        normalized = re.sub('[^a-z0-9]', '', event.get('name', '').lower())
        stage = next((stage for stage, names in aliases.items()
                      if any(normalized.startswith(prefix) for prefix in names)), None)
        if category and 'kernel' not in category:
            # The dynamically loaded Get can appear under the device task
            # category instead of the kernel category used by torch ops.
            if stage != 'fetch' or any(tag in category for tag in ('cpu', 'python', 'runtime')):
                continue
        begin = float(Decimal(str(event['ts'])) - origin)
        end = begin + float(event.get('dur', 0))
        if begin < 0 or end > float(scope['dur']):
            continue
        if stage is None:
            if not event.get('cat'):
                continue  # Untagged CPU scopes are not evidence of device execution.
            unknown.append(dict(name=event['name'], start_us=begin, end_us=end))
        else:
            groups.setdefault(stage, []).append((begin, end))
    missing = sorted(set(aliases) - groups.keys())
    stages = {stage: sum(end - begin for begin, end in union(intervals))
              for stage, intervals in groups.items()}
    intervals = union([interval for values in groups.values() for interval in values]
                      + [(row['start_us'], row['end_us']) for row in unknown])
    result = dict(stages_us=stages, intervals_us=groups, unclassified=unknown,
                  event_us=elapsed, missing_stages=missing,
                  note='LIDU includes descriptor/address preparation; no separate mapping boundary inferred')
    if intervals:
        busy = sum(end - begin for begin, end in intervals)
        span = intervals[-1][1] - intervals[0][0]
        result.update(kernel_span_us=span, inter_kernel_gap_us=span-busy,
                      boundary_residual_us=elapsed-span)
    if missing or unknown or not intervals:
        result['error'] = 'Native stage coverage incomplete; raw names retained for attribution'
    elif sum(stages.values()) - busy > .1 or elapsed - span < -2:
        result['error'] = 'Native stage overlap or event/trace interval mismatch'
    return result


def collect_native(torch, graph, reset, verify, rounds, folder):
    """Profile the unchanged native graph; synchronize only around whole replays."""
    import torch_npu
    from reporting import atomic_json
    from torch.profiler import record_function

    folder.mkdir(parents=True, exist_ok=True)
    stream = torch.npu.current_stream()
    samples = []
    profile = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False, profile_memory=False, with_stack=False)
    with profile:
        for step in range(rounds):
            reset()
            stream.synchronize()
            begin, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
            name = f'kvop_native_{step}'
            with record_function(name):
                begin.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
            samples.append(dict(name=name, round=step, us=begin.elapsed_time(end)*1000))
            atomic_json(folder / 'native_samples.json', samples)
            verify()
    trace_path = folder / 'native_trace.json'
    profile.export_chrome_trace(str(trace_path.resolve()))
    trace = json.loads(trace_path.read_text())
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    for sample in samples:
        sample['accounting'] = account_native(events, sample['name'], sample['us'])
    result = dict(path='serial', samples=samples, trace=str(trace_path))
    atomic_json(folder / 'native_attribution.json', result)
    return result


def selected_bounds(selected, history):
    """Summarize invalid LI outputs; UINT32_MAX is the padding sentinel."""
    values = np.asarray(selected).astype(np.uint32, copy=False)
    invalid = (values >= history) & (values != np.uint32(0xFFFFFFFF))
    examples = []
    for position in np.argwhere(invalid)[:8]:
        index = tuple(int(x) for x in position)
        examples.append(dict(index=index, value=int(values[index])))
    return dict(shape=list(values.shape), history=int(history),
                invalid_count=int(invalid.sum()), examples=examples)


def score_rows(raw, history):
    """Describe the first request's four uint16 score rows, without interpreting dtype."""
    values = np.asarray(raw, dtype=np.uint8).view('<u2').reshape(4, history)
    return [dict(query=q, nonzero=int(np.count_nonzero(row)),
                 zero=int(np.count_nonzero(row == 0)),
                 all_ones=int(np.count_nonzero(row == 0xffff)),
                 minimum=int(row.min()), maximum=int(row.max()))
            for q, row in enumerate(values)]


def actual_reads(mapping, selected, rows, resident):
    """Inspect completed replay outputs outside timing; count unique prefetches."""
    result = dict(
        prefetch_hot=0, prefetch_cold=0, unused_hot=0, unused_cold=0, final=int(rows[:, 247].astype(np.uint64).sum())
    )
    for row, tokens in zip(mapping, selected):
        fetched = np.flatnonzero(row >= 0)
        used = np.isin(fetched, tokens.reshape(-1))
        hot = fetched < resident
        result["prefetch_hot"] += int(hot.sum())
        result["prefetch_cold"] += int((~hot).sum())
        result["unused_hot"] += int((hot & ~used).sum())
        result["unused_cold"] += int((~hot & ~used).sum())
    prefetch = result["prefetch_hot"] + result["prefetch_cold"]
    if prefetch != int(rows[:, 9].astype(np.uint64).sum()):
        raise ValueError("Prefetch mapping and device request counts disagree")
    result["total"] = prefetch + result["final"]
    result["unused"] = result["unused_hot"] + result["unused_cold"]
    return result


def completion_failures(rows, posts, statuses, prefetch, final):
    """Decode existing device status buffers; no device work or state changes."""
    failures = []
    for b in range(len(rows)):
        cq = {
            name: words[2 * b : 2 * b + 2, :8].tolist()
            for name, words in (("prefetch_cq", prefetch), ("final_cq", final))
        }
        cq_bad = any(any(w[0] != 0x4B5641464247444F or w[2] for w in pair) for pair in cq.values())
        if not (rows[b, 11] or rows[b, 211] or rows[b, 248] or np.any(posts[b, :, 1:4]) or statuses[b, 0] or cq_bad):
            continue
        failures.append(
            dict(
                request=b,
                prefetch_error=int(rows[b, 11]),
                prefetch_pending=int(rows[b, 211]),
                final_pending=int(rows[b, 248]),
                final_requests=int(rows[b, 247]),
                resolver_status=statuses[b, :5].tolist(),
                posts_count_pending_error_deferred_descriptor=posts[b, :, :5].tolist(),
                **cq,
            )
        )
    return failures


def summarize_mismatches(row, selected, ids, slots, output, payload, destinations,
                         addresses, posts, lru, mapping, stride, resident, remote_addresses, limit=8):
    selected = np.asarray(selected).reshape(-1)
    ids = np.asarray(ids).reshape(-1)
    slots = np.asarray(slots).reshape(-1)
    out = np.asarray(output)
    reference = np.asarray(payload)
    destinations = np.asarray(destinations).reshape(-1)
    addresses = np.asarray(addresses).reshape(-1)
    remote_addresses = np.asarray(remote_addresses).reshape(-1).view(np.uint64)
    posts = np.asarray(posts)
    mapping = np.asarray(mapping)
    lru = np.asarray(lru)
    bad = np.flatnonzero(np.any(out[slots] != reference[ids], axis=1))
    entries = []
    for p in bad[:limit]:
        slot = int(slots[p])
        actual = out[slot]
        candidates = np.flatnonzero(np.all(reference[:, :16] == actual[:16], axis=1))
        actual_ids = [int(i) for i in candidates if np.array_equal(reference[i], actual)][:8]
        writers = []
        for q, post in enumerate(posts):
            count, start = int(post[0]), int(post[4])
            for i in range(start, start + count):
                if 0 <= i < len(destinations) and int(destinations[i]) == row * stride + slot:
                    remote = int(addresses[i]) & ((1 << 64) - 1)
                    source_ids = np.flatnonzero(remote_addresses == remote)
                    writers.append(dict(q=q + 1, descriptor=i, remote_address=hex(remote),
                                        source_tokens=[int(x) for x in source_ids[:4]]))
        virtual = np.flatnonzero(mapping >= 0)
        prefetched = [int(t) for t in virtual
                      if 0 <= int(mapping[t]) - resident < len(lru)
                      and int(lru[int(mapping[t]) - resident]) == slot]
        entries.append(dict(query=int(p // 2048 + 1), position=int(p % 2048),
                            selected_physical=int(selected[p]), expected_token=int(ids[p]), slot=slot,
                            actual_matching_tokens=actual_ids, prefetch_writers=prefetched, actual_prefix=bytes(actual[:16]).hex(),
                            expected_prefix=bytes(reference[ids[p], :16]).hex(), final_writers=writers,
                            lru_ordinal=next((int(i) for i in np.flatnonzero(lru == slot)), None)))
    return dict(batch=int(row), mismatch_positions=int(len(bad)), examples=entries,
                posts=posts[:, :6].astype(np.int64).tolist())


def split_native_jetty_descriptors(phases, jettys=2, inflight=2048):
    """Match KVAF native grouping and 128-WQE rotation for SQ audit."""
    result = [[] for _ in range(jettys)]
    for entries in phases:
        # Native BatchGet uses an unbounded group when the entire call fits.
        group_size = inflight // 2 if inflight and len(entries) > inflight else len(entries)
        next_jetty = 0
        for start in range(0, len(entries), max(group_size, 1)):
            group = entries[start:start + group_size]
            for post in range((len(group) + 127) // 128):
                jetty = (next_jetty + post) % jettys
                result[jetty].extend(group[post * 128:(post + 1) * 128])
            next_jetty = (next_jetty + (len(group) + 127) // 128) % jettys
    return result


def audit_queue_writes(owner, rows, posts, row, bad_slots, destinations, remote_addresses, q_workers=None):
    """Read already-completed SQ entries on failure; no device work is submitted."""
    import ctypes

    torch = owner.torch
    snapshot = owner.abi_library.kvip_transport_snapshot
    snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    snapshot.restype = ctypes.c_uint32

    def read(address, size):
        if not address or address % 32 or not 0 < size <= 2 * 1024 * 1024 or size % 32:
            raise ValueError(f"invalid queue snapshot: {address:#x}, {size}")
        result = torch.empty(size, dtype=torch.uint8, device=owner.device)
        stream = torch.npu.current_stream(owner.device)
        code = snapshot(address, result.data_ptr(), size, stream.npu_stream)
        if code:
            raise RuntimeError(f"queue snapshot launch returned {code}")
        return result.cpu().numpy()

    contexts = owner.device_args.cpu().numpy().view('<u8')
    workers = posts.shape[1]
    batch = posts.shape[0]
    queue_specs = []
    for owner_row in range(batch):
        initial = int(rows[owner_row, 212]) | (int(rows[owner_row, 213]) << 32)
        queue_specs.append((f'b{owner_row}_prefetch', owner_row * 2, initial, int(rows[owner_row, 9])))
        if q_workers:
            continue
        for q, post in enumerate(posts[owner_row]):
            view = owner_row * 2 + 1 if q == 0 else batch * 2 + owner_row * (workers - 1) + q - 1
            count = int(post[0])
            if not count:
                continue
            queue = read(int(contexts[view, 9]), 64).view('<u8')
            queue_specs.append((f'b{owner_row}_final_q{q+1}', view, int(queue[1]) - count, count))
    # Each Q worker owns two contiguous Jetty views after all prefetch views.
    group_indices = {}
    if q_workers:
        for q in range(workers):
            for lane in range(min(q_workers, batch)):
                phases = []
                for phase in range(2):
                    indices = []
                    for b in range(lane, batch, q_workers):
                        post = posts[b, q]
                        count, begin, safe = int(post[0]), int(post[4]), int(post[5])
                        start = begin if phase == 0 else begin + safe
                        length = safe if phase == 0 else count - safe
                        indices.extend(range(start, start + length))
                    phases.append(indices)
                for jetty, indices in enumerate(split_native_jetty_descriptors(phases)):
                    if not indices:
                        continue
                    view = 2 * batch + 2 * (q * q_workers + lane) + jetty
                    queue = read(int(contexts[view, 9]), 64).view('<u8')
                    name = f'b{lane}_final_q{q+1}_jetty{jetty}'
                    group_indices[name] = np.asarray(indices, dtype=np.int64)
                    queue_specs.append((name, view, int(queue[1]) - len(indices), len(indices)))
    result = []
    final_checks = []
    for name, view, start, count in queue_specs:
        if not count:
            continue
        entity = read(int(contexts[view, 8]), 192).view('<u4')
        meta = entity.view('<u8')
        depth = int(entity[19])
        if start < 0 or not 0 < count < depth <= 1048576:
            raise ValueError(f'invalid SQ range {name}: {start}, {count}, {depth}')
        first = min(count, depth - start % depth)
        chunks = [read(int(meta[7]) + (start % depth) * 64, first * 64)]
        if first < count:
            chunks.append(read(int(meta[7]), (count-first) * 64))
        sq = np.concatenate(chunks).view('<u4').reshape(count, 16)
        local = sq[:, 14].astype(np.uint64) | (sq[:, 15].astype(np.uint64) << np.uint64(32))
        remote = sq[:, 10].astype(np.uint64) | (sq[:, 11].astype(np.uint64) << np.uint64(32))
        if '_final_q' in name:
            owner_row = int(name.split('_')[0][1:])
            query = int(name.rsplit('q', 1)[1].split('_', 1)[0]) - 1
            begin = int(posts[owner_row, query, 4])
            index = group_indices.get(name, slice(begin, begin + count))
            wanted_local = np.uint64(owner.output.data_ptr()) + destinations[index].astype(np.uint64) * np.uint64(656)
            wanted_remote = remote_addresses[index].astype(np.uint64)
            different = np.flatnonzero((local != wanted_local) | (remote != wanted_remote))
            final_checks.append(dict(queue=name, checked=count, mismatches=int(len(different)),
                examples=[dict(index=int(i), actual_local=hex(int(local[i])), expected_local=hex(int(wanted_local[i])),
                               actual_remote=hex(int(remote[i])), expected_remote=hex(int(wanted_remote[i])))
                          for i in different[:3]]))
        for slot in bad_slots:
            expected = int(owner.output.data_ptr()) + (row * owner.output.shape[1] + int(slot)) * 656
            lengths = sq[:, 12].astype(np.uint64)
            for i in np.flatnonzero((local < expected + 656) & (local + lengths > expected)):
                result.append(dict(queue=name, sq_index=int((start + int(i)) % depth), slot=int(slot),
                                   local=hex(int(local[i])), remote=hex(int(remote[i])),
                                   length=int(sq[i, 12])))
    return result, final_checks


def decode_timeline(raw):
    batch, workers = raw['batch'], raw['workers_per_query']
    resolver_cores = raw.get('resolver_cores', 2)
    cores, pu, fetch = raw['cores'], raw['producer_update'], raw['fetch']
    mhz = raw['clock_mhz']
    if not mhz or mhz <= 0 or len(cores) != resolver_cores * batch + 4 * workers:
        raise ValueError('missing clock calibration or core records')
    if any(not c[0] or c[1] < c[0] for c in cores):
        raise ValueError('missing/nonmonotonic core entry or exit marker')
    origin = min(c[0] for c in cores)
    result = []
    for block, c in enumerate(cores):
        stages = []
        def add(name, start, end):
            start, end = int(start), int(end)
            if not start or not end or end < start or start < c[0] or end > c[1]:
                raise ValueError(f'block={block} invalid {name}: {start}..{end}, core={c[:2]}')
            stages.append(dict(name=name, start_cycles=start, end_cycles=end,
                               start_us=(start-origin)/mhz, end_us=(end-origin)/mhz,
                               duration_us=(end-start)/mhz))
        if block < resolver_cores * batch:
            b = block // resolver_cores
            p, u, f = pu[b][:32], pu[b][32:], fetch[b]
            if block % resolver_cores == 0:
                role = f'resolver+finalize b{b}'
                add('input_and_bitmap', p[1], p[2])
                add('initialize', p[2], p[3])
                for q in range(4):
                    t = p[10+q*5:15+q*5]
                    for i, name in enumerate(('resolve','output','descriptor','publish')):
                        add(f'Q{q+1}.{name}', t[i], t[i+1])
                add('prefetch_CQ_join', f[96], f[100])
                for i, name in enumerate(('index_update','prefetch_ready','Q1_fetch','Q2_fetch','Q3_fetch','Q4_fetch')):
                    add('wait.'+name, f[104+2*i], f[105+2*i])
                if f[116]:
                    add('prefetch_residency', f[116], f[117])
                add('residency', f[97], f[98])
                add('status', f[98], f[99])
            elif block % resolver_cores == 1:
                role = f'index_update b{b}'
                add('initialize_and_wait_prefetch', u[10], u[12])
                for q in range(5):
                    if q:
                        add(f'wait.Q{q}', u[11+2*q], u[12+2*q])
                    add('apply.prefetch' if q == 0 else f'apply.Q{q}', u[12+2*q], u[13+2*q])
                add('metadata', u[22], u[23])
            else:
                role = f'output_pack b{b}'
                add('wait.prefetch_publication', c[0], c[2])
                add('input_and_output_masks', c[2], c[3])
                for q in range(4):
                    t = c[4+q*6:10+q*6]
                    for i, name in enumerate(('logical_lookup', 'wait.resolver', 'locations_and_pack', 'output_and_sync', 'publish')):
                        add(f'Q{q+1}.{name}', t[i], t[i+1])
        else:
            q, lane = divmod(block-resolver_cores*batch, workers)
            role = f'fetch Q{q+1} lane{lane}'
            for phase, name in enumerate(('safe','conflict')):
                t = c[2+phase*5:7+phase*5]
                if not t[0]:
                    if phase == 0 and lane < batch:
                        raise ValueError(f'block {block}: missing safe phase')
                    continue
                for i, step in enumerate(('prepare','input_layout','queue_setup','KVAF_Get')):
                    add(f'{name}.{step}', t[i], t[i+1])
            if c[12]:
                add('finalize.wait_index', c[12], c[13])
                add('finalize.residency_and_ack', c[13], c[14])
                add('finalize.diagnostic_write', c[14], c[15])
            for b in range(lane, batch, workers):
                ft = fetch[b][q*24:q*24+24]
                add(f'request{b}.wait_Q_ready', ft[0], ft[1])
                add(f'request{b}.metadata', ft[1], ft[2])
                published = pu[b][14+q*5]
                if ft[1] < published:
                    raise ValueError('cross-core timestamps violate producer publication order')
            # Shared group counters are copied to each assigned request; report once.
            if lane < batch:
                f = fetch[lane][q*24:q*24+24]
                counters = dict(safe_post_cycles=f[12], safe_CQ_cycles=f[13],
                                conflict_post_cycles=f[14], conflict_CQ_cycles=f[15])
            else:
                counters = {}
        intervals = sorted((s['start_cycles'], s['end_cycles']) for s in stages)
        cursor = c[0]
        gaps = []
        for start, end in intervals:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < c[1]:
            gaps.append((cursor, c[1]))
        for i, (start, end) in enumerate(gaps):
            add(f'control_gap{i}', start, end)
        stages.sort(key=lambda x: (x['start_cycles'], x['end_cycles']))
        item = dict(block=block, role=role, start_cycles=c[0], end_cycles=c[1],
                    start_us=(c[0]-origin)/mhz, end_us=(c[1]-origin)/mhz,
                    duration_us=(c[1]-c[0])/mhz, stages=stages)
        if block >= resolver_cores*batch:
            item['KVAF_counters'] = counters
            item['requests'] = list(range(lane, batch, workers))
            item['published_misses'] = [raw['flags'][b][q+1][1] for b in item['requests']]
        result.append(item)
    return dict(origin_cycles=origin, clock_mhz=mhz, cores=result,
                last_block=max(result, key=lambda x:x['end_cycles'])['block'],
                observed_span_us=(max(c[1] for c in cores)-origin)/mhz)


def save_transport_snapshot(path, ids, slots, output, mapping, lru, rows, addresses):
    ids = np.asarray(ids).reshape(len(mapping), 4, 2048)
    slots = np.asarray(slots).reshape(ids.shape)
    order = np.argsort(ids, axis=-1, kind="stable")
    sorted_ids = np.take_along_axis(ids, order, axis=-1)
    sorted_slots = np.take_along_axis(slots, order, axis=-1)
    # Canonicalize by logical token, because TopK order and physical hot slots
    # are not required to match between independent concurrent executions.
    payload = np.stack([output[b][sorted_slots[b]] for b in range(len(mapping))])
    np.savez_compressed(path, ids=sorted_ids, slots=sorted_slots, payload=payload,
                        mapping=mapping, lru=lru, epochs=rows[:, 16:48], orders=rows[:, 214:246], addresses=addresses)


def compare_transport_snapshots(urma_path, ubmem_path, resident):
    with np.load(urma_path) as a, np.load(ubmem_path) as b:
        if not np.array_equal(a['ids'], b['ids']):
            raise AssertionError('cross-transport logical TopK sets differ')
        if not np.array_equal(a['payload'], b['payload']):
            raise AssertionError('cross-transport selected KV bytes differ')
        lines = ['logical_topk=identical selected_kv_bytes=identical']
        if a['lru'].shape != b['lru'].shape:
            raise ValueError('cross-transport LRU candidate shapes differ')
        changed_rows = np.flatnonzero(np.any(a['lru'] != b['lru'], axis=1))
        lines.append(f'lru_candidate_order_different_rows={len(changed_rows)}/{len(a["lru"])} '
                     f'differing_positions={np.count_nonzero(a["lru"] != b["lru"])}')
        for row in range(len(a['mapping'])):
            am, bm = a['mapping'][row], b['mapping'][row]
            at, bt = np.flatnonzero(am >= 0), np.flatnonzero(bm >= 0)
            common = np.intersect1d(at, bt, assume_unique=True)
            def slot(m, lru, tokens):
                virtual = m[tokens].astype(np.int64) - resident
                if np.any((virtual < 0) | (virtual >= lru.shape[0])):
                    raise ValueError('prefetch mapping outside candidate slots')
                return lru[virtual]
            sa, sb = slot(am, a['lru'][row], common), slot(bm, b['lru'][row], common)
            def epoch_of(mapping, tokens, counts, orders):
                used = np.flatnonzero(counts)
                if len(np.unique(orders[used])) != len(used) or np.any(orders[used] == 0):
                    raise ValueError("invalid prefetch epoch execution order")
                used = used[np.argsort(orders[used])]
                by_ordinal = np.repeat(used, counts[used].astype(np.int64))
                all_ordinals = mapping[mapping >= 0].astype(np.int64) - resident
                ordinals = mapping[tokens].astype(np.int64) - resident
                if (len(by_ordinal) != len(all_ordinals)
                        or not np.array_equal(np.sort(all_ordinals), np.arange(len(all_ordinals)))
                        or np.any((ordinals < 0) | (ordinals >= len(by_ordinal)))):
                    raise ValueError("prefetch counters do not identify each candidate epoch")
                return by_ordinal[ordinals]
            ea = epoch_of(am, common, a['epochs'][row], a['orders'][row])
            eb = epoch_of(bm, common, b['epochs'][row], b['orders'][row])
            for name, snapshot, mapping, tokens in (("urma", a, am, at), ("ubmem", b, bm, bt)):
                epochs = epoch_of(mapping, tokens, snapshot['epochs'][row], snapshot['orders'][row])
                used = np.isin(tokens, snapshot['ids'][row].reshape(-1))
                breakdown = []
                for epoch in np.unique(epochs):
                    mask = epochs == epoch
                    breakdown.append(dict(epoch=int(epoch), hot=int(np.count_nonzero(mask & (tokens < resident))),
                                          cold=int(np.count_nonzero(mask & (tokens >= resident))),
                                          unused=int(np.count_nonzero(mask & ~used))))
                lines.append(f"row={row} {name}_window_breakdown={breakdown}")
            lines.append(f'row={row} urma={len(at)} ubmem={len(bt)} common={len(common)} '
                         f'urma_only={len(at)-len(common)} ubmem_only={len(bt)-len(common)} '
                         f'common_slot_differences={np.count_nonzero(sa != sb)} common_epoch_differences={np.count_nonzero(ea != eb)} '
                         f'epochs_urma={a["epochs"][row].tolist()} epochs_ubmem={b["epochs"][row].tolist()}')
            changed = np.flatnonzero((sa != sb) | (ea != eb))[:3]
            for i in changed:
                token = int(common[i])
                lines.append(f"  token={token} block={token//128} block_offset={token%128} "
                             f"urma_epoch={ea[i]} ubmem_epoch={eb[i]} urma_slot={sa[i]} ubmem_slot={sb[i]} "
                             f"urma_addr=0x{int(a['addresses'][row, token]):x} "
                             f"ubmem_addr=0x{int(b['addresses'][row, token]):x}")
        lines.append('raw DDR VAs differ across allocations; token identity gives logical block=token//128 and offset=token%128')
        lines.append('per-token epoch reconstructed from allocation ordinal and recorded epoch execution order')
        return lines


def calibration_distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return dict(count=len(values), finite=len(finite),
                negative_inf=int(np.isneginf(values).sum()), positive_inf=int(np.isposinf(values).sum()),
                mean=float(finite.mean()) if len(finite) else None,
                quantiles=dict(zip(map(str, QUANTILES), np.quantile(finite, QUANTILES).tolist())) if len(finite) else {})


def position_histogram(positions, length):
    return np.histogram(np.asarray(positions) / length, bins=np.linspace(0, 1, POSITION_BINS + 1))[0].tolist()


def capture_rows(capture):
    """Return statistics plus private token sets used only for temporal matching."""
    a, g = capture.arrays, capture.manifest['geometry']
    batch, history, resident = (g[key] for key in ('batch', 'history', 'resident'))
    posted, processed = a['post.window_posted'], a['post.window_processed_ends']
    records, ordinals = a['post.prefetch_records'], a['post.prefetch_ordinals']
    final, thresholds = a['post.final_records'], a['post.threshold_records']
    if (posted.ndim != 2 or posted.shape[0] != batch or posted.dtype.kind != 'i'
            or processed.shape != (*posted.shape, 4) or processed.dtype.kind != 'i'
            or records.ndim != 2 or records.shape[1] != 3 or records.dtype.kind != 'i'
            or ordinals.shape != (len(records),) or ordinals.dtype.kind != 'i'
            or final.ndim != 2 or final.shape[1] != 4 or final.dtype.kind != 'i'
            or thresholds.ndim != 4 or thresholds.shape[:2] != (batch, 4) or thresholds.shape[-1] != 4
            or thresholds.dtype.kind != 'i'):
        raise ValueError('Calibration requires valid window, threshold and transfer record shapes')
    if (np.any(posted < 0) or np.any(processed < 0) or np.any(processed > history)
            or np.any(np.diff(processed, axis=1) < 0)
            or np.any(records[:, 0] < 0) or np.any(records[:, 0] >= batch)
            or np.any(final[:, 0] < 0) or np.any(final[:, 0] >= batch)):
        raise ValueError('Invalid calibration window/transfer bounds')
    result = []
    for row in range(batch):
        layout = a['pre.logical_ids'][row]
        topk = a['post.topk'][row]
        sets = [set(map(int, layout[query])) for query in topk]
        union = set.union(*sets)
        hot = set(map(int, layout[:resident][a['pre.hot_valid'][row] != 0]))
        begin_tail, end_tail = map(int, a['pre.tail_bounds'][row])
        if not 0 <= begin_tail <= end_tail <= history:
            raise ValueError('Invalid captured ABC bounds')
        local = hot | set(range(begin_tail, end_tail))
        fetched = records[records[:, 0] == row]
        order = ordinals[records[:, 0] == row]
        if (len(fetched) != int(posted[row].sum())
                or not np.array_equal(np.sort(order), np.arange(len(fetched)))):
            raise ValueError('Prefetch ordinals/window counts disagree')
        fetched = fetched[np.argsort(order)]
        if (np.any(fetched[:, 1] < 0) or np.any(fetched[:, 1] >= history)
                or not np.array_equal(layout[fetched[:, 1]], fetched[:, 2])):
            raise ValueError('Prefetch logical identity differs from pre-Index position')
        unused = np.array([int(token) not in union for token in fetched[:, 2]], dtype=bool)
        window_rows, offset = [], 0
        for window, count in enumerate(posted[row]):
            count = int(count)
            window_rows.append(dict(window=window + 1, posted=count,
                unused=int(unused[offset:offset + count].sum()),
                processed_ends=processed[row, window].astype(int).tolist()))
            offset += count
        queries, seen = [], set()
        for q in range(4):
            end = int(a['pre.query_ends'][row, q])
            keys = a['post.score_keys'][row, q, :end]
            scores = score_values(keys)
            cutoff = np.partition(keys, end - TOPK)[end - TOPK]
            hot_end = min(resident, end)
            hot_cutoff = np.partition(keys[:hot_end], hot_end - TOPK)[hot_end - TOPK]
            valid_thresholds = []
            for valid, applies, key, kind in thresholds[row, q]:
                if not kind:
                    continue
                if (not 0 <= valid <= end or not valid <= applies <= history
                        or not 0 <= key <= 65535 or kind not in (1, 2, 3, 4)):
                    raise ValueError('Invalid threshold record')
                expected = np.partition(keys[:valid], valid - TOPK)[valid - TOPK] if valid >= TOPK else 0
                if key != expected:
                    raise ValueError('Threshold record differs from captured scores')
                valid_thresholds.append(dict(valid=int(valid), applies=int(applies), key=int(key), kind=int(kind)))
            queries.append(dict(query=q + 1, selected=TOPK, unique_new=len(sets[q] - seen),
                cross_query_duplicates=len(sets[q] & seen), initial_cold=len(sets[q] - local),
                cutoff_key=int(cutoff), resident_cutoff_key=int(hot_cutoff),
                cutoff_tie_population=int(np.count_nonzero(keys == cutoff)),
                scores=calibration_distribution(scores), selected_scores=calibration_distribution(scores[topk[q]]),
                selected_cutoff_margin=(calibration_distribution(scores[topk[q]] - float(score_values([cutoff])[0]))
                    if np.isfinite(score_values([cutoff])[0]) else None),
                beyond_resident_above_threshold=int(np.count_nonzero(keys[resident:] > hot_cutoff)),
                physical_position_hist=position_histogram(topk[q], end),
                logical_position_hist=position_histogram(list(sets[q]), history),
                thresholds=valid_thresholds))
            seen.update(sets[q])
        overlap = [[len(left & right) / len(left | right) for right in sets] for left in sets]
        repairs = final[final[:, 0] == row]
        if np.any(repairs[:, 1] < 0) or np.any(repairs[:, 1] >= history):
            raise ValueError('Final transfer logical token outside history')
        result.append((dict(history=history, resident=resident, heads=g['heads'],
            initial_hot_valid=len(hot), selected_unique=len(union), initial_cold=len(union-local),
            prefetch=len(fetched), unused_prefetch=int(unused.sum()), final=len(repairs),
            q_jaccard=overlap, queries=queries, windows=window_rows), union))
    return result


def calibrate_captures(root, group, rounds, output):
    references = select_series(root, group, rounds)
    if not references:
        raise ValueError('No captures selected for calibration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous, rows, source_checksums = {}, [], {}
    for ref in references:
        capture = load_score_capture(ref.manifest_path, require_real=False)
        source_checksums[str(ref.manifest_path)] = capture.manifest['archive']['sha256']
        for (request_id, owner), (row, selected) in zip(ref.request_signature, capture_rows(capture), strict=True):
            identity = (ref.dp_rank, request_id, owner)
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            row.update(dp=ref.dp_rank, request_key=key, source_round=ref.source_round,
                       scheduler_round=ref.scheduler_round, temporal=None)
            prior = previous.get(identity)
            if prior is not None and prior[0] + 1 == ref.scheduler_round:
                old = prior[1]
                row['temporal'] = dict(retained=len(old & selected), previous=len(old), current=len(selected),
                                      jaccard=len(old & selected) / len(old | selected))
            previous[identity] = (ref.scheduler_round, selected)
            rows.append(row)
        atomic_json(output / 'request_rounds.json', rows)
    totals = {key: sum(row[key] for row in rows) for key in
              ('selected_unique', 'initial_cold', 'prefetch', 'unused_prefetch', 'final')}
    pairwise = np.asarray([row['q_jaccard'] for row in rows])
    targets = dict(schema_version=1, kind='lifetch_empirical_fit_targets', fitted_generator=False,
        captures=len(references), request_rounds=len(rows), totals=totals,
        cold_miss_rate=totals['initial_cold'] / totals['selected_unique'],
        unused_rate=totals['unused_prefetch'] / totals['prefetch'] if totals['prefetch'] else None,
        q_jaccard_mean=pairwise.mean(axis=0).tolist(),
        temporal_jaccard=calibration_distribution([row['temporal']['jaccard'] for row in rows if row['temporal']]),
        metrics={key: calibration_distribution([row[key] for row in rows]) for key in totals},
        physical_position_hist=np.sum([q['physical_position_hist'] for row in rows for q in row['queries']], axis=0).tolist(),
        logical_position_hist=np.sum([q['logical_position_hist'] for row in rows for q in row['queries']], axis=0).tolist(),
        scope='Captured cache trajectory; selected token identity at cutoff ties is capture-specific. Not TPOT or kernel timing.',
        model_plan='Fit Q-overlap, score margins and position locality jointly; validate generated Q/Index through actual LI.')
    atomic_json(output / 'fit_targets.json', targets)
    atomic_json(output / 'sources.json', dict(group=group, round_range=rounds, archives=source_checksums))
    lines = [f"CALIBRATION captures={len(references)} request_rounds={len(rows)}",
             f"cold/selected={totals['initial_cold']}/{totals['selected_unique']} rate={targets['cold_miss_rate']:.6f}",
             f"prefetch={totals['prefetch']} unused={totals['unused_prefetch']} final={totals['final']}",
             f"adjacent_round_pairs={targets['temporal_jaccard']['count']}",
             'Q Jaccard mean:']
    lines.extend(' '.join(f'{value:.4f}' for value in row) for row in targets['q_jaccard_mean'])
    lines.append('Empirical targets only; no claim of fitted generator or device performance.')
    (output / 'summary.log').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return targets


def read_range(samples, key):
    values = [s[key] for s in samples]
    return f'{min(values)}..{max(values)}'


def analyze_prefetch_pressure(root, batch, misses):
    lines = ['OFFLINE: no device execution; counts cover the entire batch.',
             'Graph p50 is unprofiled. All per-core times are same-replay calibrated us.',
             'CQ counters include polling/processing; they are NOT pure network latency.',
             'No prefetch bandwidth inferred: these timelines start after LI, not at first prefetch post.']
    details, records = [], []
    for miss in misses:
        folder = root / f'current-b{batch}-m{miss}'
        reports = list(folder.glob('case*/report.json'))
        if len(reports) != 1:
            raise ValueError(f'Expected one report: {folder}')
        path = reports[0]
        report = json.loads(path.read_text())
        if report['status'] != 'pass':
            raise ValueError(f'Unchecked result: {path}')
        reads = report['read_samples']['prototype_with_update']
        if not reads:
            raise ValueError('Missing read samples')
        native = sum(report['native_misses_per_request'])
        for s in reads:
            if s['total'] != s['prefetch_hot'] + s['prefetch_cold'] + s['final']:
                raise ValueError('Read accounting mismatch')
        timing = report['timing']
        lines += [f'\nBS={batch} miss={miss}% serial_graph={timing["native_chain"]["p50_us"]:.3f} current_graph={timing["prototype_with_update"]["p50_us"]:.3f} us',
                  f'native_reads={native}; independent read samples (min..max, do not add range endpoints):',
                  ' | '.join(f'{k}={read_range(reads,k)}' for k in
                             ('prefetch_hot','prefetch_cold','unused_hot','unused_cold','final','total'))]
        main_reports = list((root / f'main-b{batch}-m{miss}').glob('case*/report.json'))
        if len(main_reports) == 1:
            main = json.loads(main_reports[0].read_text())
            lines.append(f'main_graph={main["timing"]["prototype_with_update"]["p50_us"]:.3f} us; main_total_reads=' +
                         read_range(main['read_samples']['prototype_with_update'], 'total'))
        samples = json.loads(path.with_name('full_attribution.json').read_text())['samples']
        if not samples:
            raise ValueError('Missing full pipeline profiling samples')
        for sample in samples:
            if 'error' in sample['accounting']:
                raise ValueError(f'Invalid full attribution: {path}')
            raw = sample['streaming_cycles'][0]['core_timeline_raw']
            if raw['batch'] != batch:
                raise ValueError('Trace batch differs')
            tl = decode_timeline(raw)
            cores = tl['cores']
            last = next(c for c in cores if c['block'] == tl['last_block'])
            fetch = [c for c in cores if c['role'].startswith('fetch ')]
            updates = [c for c in cores if c['role'].startswith('index_update ')]
            end_fetch = max(c['end_us'] for c in fetch)
            record = dict(miss=miss, round=sample['round'], timeline=tl,
                          profiled_stages=sample['accounting']['stages_us'],
                          prefetch=sample['prefetch'], final=sample['final'])
            records.append(record)
            lines += [f'ROUND {sample["round"]}: stages_us={json.dumps(record["profiled_stages"], sort_keys=True)}',
                      f'  reads prefetch={sample["prefetch"]} final={sample["final"]}; update_end={max(c["end_us"] for c in updates):.3f} fetch_end={end_fetch:.3f} kernel_last={last["block"]}({last["role"]}) end={last["end_us"]:.3f} tail_after_fetch={last["end_us"]-end_fetch:.3f}',
                      '  Fetch block | Q/lane | start..end | published_entries | Get intervals | post_us | CQ_us']
            for c in fetch:
                stages = c['stages']
                gets = [s for s in stages if s['name'].endswith('.KVAF_Get')]
                counters = c.get('KVAF_counters', {})
                post = sum(counters.get(k,0) for k in ('safe_post_cycles','conflict_post_cycles')) / tl['clock_mhz']
                cq = sum(counters.get(k,0) for k in ('safe_CQ_cycles','conflict_CQ_cycles')) / tl['clock_mhz']
                intervals = ','.join(f'{s["name"]}:{s["start_us"]:.2f}..{s["end_us"]:.2f}' for s in gets)
                lines.append(f'  {c["block"]} | {c["role"]} | {c["start_us"]:.2f}..{c["end_us"]:.2f} | {sum(c["published_misses"])} | {intervals or "empty"} | {post:.2f} | {cq:.2f}')
            lines.append('  Last core waits/finalization: stage start..end (duration)')
            for s in last['stages']:
                if s['name'] in ('prefetch_CQ_join','residency','status') or s['name'].startswith('wait.'):
                    lines.append(f'  {s["name"]} {s["start_us"]:.2f}..{s["end_us"]:.2f} ({s["duration_us"]:.2f})')
            details.append(f'BS={batch} miss={miss} ROUND={sample["round"]}')
            for c in cores:
                details.append(f'block={c["block"]} {c["role"]} {c["start_us"]:.3f}..{c["end_us"]:.3f}')
                for s in c['stages']:
                    details.append(f'  {s["name"]}: {s["start_us"]:.3f}..{s["end_us"]:.3f} duration={s["duration_us"]:.3f}')
    lines += ['\nMEDIANS across profiled rounds (not additive to unprofiled graph p50):',
              'miss | prefetch | final | fused_kernel_us | update_end | fetch_end | kernel_end']
    for miss in misses:
        group = [r for r in records if r['miss'] == miss]
        values = [[r['prefetch'], r['final'], r['profiled_stages']['resolve_update'],
                   max(c['end_us'] for c in r['timeline']['cores'] if c['role'].startswith('index_update ')),
                   max(c['end_us'] for c in r['timeline']['cores'] if c['role'].startswith('fetch ')),
                   r['timeline']['observed_span_us']] for r in group]
        lines.append(str(miss) + ' | ' + ' | '.join(f'{statistics.median(v):.3f}' for v in zip(*values)))
    return '\n'.join(lines)+'\n', '\n'.join(details)+'\n', records


def read_cases(root, transport):
    pattern = 'full_attribution.json' if transport == 'urma' else 'attribution.json'
    cases = {}
    for path in root.rglob(pattern):
        match = re.search(r'(?:case\d+-b|/b)(\d+)', str(path))
        if not match:
            continue
        batch = int(match[1])
        if batch in cases:
            raise ValueError(f'Multiple {transport} BS{batch} runs; pass one exact run directory')
        data = json.loads(path.read_text())
        samples = data['samples'] if isinstance(data, dict) else data
        cases[batch] = (path, samples)
    if not cases:
        raise ValueError(f'No {transport} full attribution under {root}')
    return cases


def summarize_transport_tail(root, transport):
    result = {}
    for batch, (path, samples) in sorted(read_cases(root, transport).items()):
        valid = [s for s in samples if 'error' not in s['accounting']]
        if not valid:
            print(f'{transport} BS={batch} INVALID_TRACE source={path}')
            continue
        stage = 'resolve_update' if transport == 'urma' else 'resolve_update_fetch'
        times = [s['accounting']['stages_us'][stage] for s in valid]
        result[batch] = statistics.median(times)
        print(f'{transport} BS={batch} fused_us={result[batch]:.3f} rounds={len(valid)}/{len(samples)} source={path}')
        for sample in valid:
            details = sample.get('streaming_cycles', sample.get('streaming'))
            if not details:
                print(f"  round={sample['round']} missing per-core diagnostics")
                continue
            raw = details[0]['core_timeline_raw']
            mhz = raw['clock_mhz']
            if not mhz or mhz <= 0:
                raise ValueError(f'Invalid clock calibration: {path}')
            cores = raw['cores']
            origin = min(c[0] for c in cores if c[0])
            last = max(range(len(cores)), key=lambda i: cores[i][1])
            def role(i):
                n = raw['resolver_cores']
                if i < batch * n:
                    return f'{("resolver", "update", "output")[i % n]} request={i // n}'
                q, lane = divmod(i-batch*n, raw['workers_per_query'])
                return f'fetch Q{q+1} lane={lane}'
            print(f"  round={sample['round']} prefetch={sample['prefetch']} final={sample['final']} last=block{last} {role(last)} end_us={(cores[last][1]-origin)/mhz:.3f}")
            # One real representative per round: last core, plus all active Fetch cores.
            selected = sorted(set([last] + list(range(batch*raw['resolver_cores'], len(cores)))))
            for i in selected:
                c = cores[i]
                print(f'    block={i} {role(i)} start={(c[0]-origin)/mhz:.3f} end={(c[1]-origin)/mhz:.3f} span={(c[1]-c[0])/mhz:.3f}')
            if transport == 'ubmem':
                for q in range(4):
                    fields = (('query_ready_wait', 'dependency_classify', 'safe_get', 'dependency_wait', 'conflict_get', 'completion_and_index_join') if 'safe_get' in details[0]['fetch']['ubmem_queries'][q] else ('query_ready_wait', 'previous_query_copy_wait', 'kvaf_mte_get', 'completion_and_index_join'))
                    text = []
                    for field in fields:
                        values = [d['fetch']['ubmem_queries'][q][field]/mhz for d in details]
                        text.append(f'{field}=median:{statistics.median(values):.3f}/max:{max(values):.3f}')
                    print(f'    Q{q+1} per_request_us ' + ' '.join(text))
                    query = details[0]['fetch']['ubmem_queries'][q]
                    if 'safe_entries' in query:
                        for field in ('safe_entries', 'conflict_entries', 'safe_calls', 'conflict_calls'):
                            print(f'      {field}=' + str(sum(d['fetch']['ubmem_queries'][q][field] for d in details)))
    return result


def decode_terminal_timeline(rows, clock_mhz):
    """Decode the four-core terminal's own stamps, without inventing old stages."""
    if clock_mhz <= 0:
        raise ValueError('Invalid terminal counter frequency')
    active = [(b, q, t) for b, queries in enumerate(rows) for q, t in enumerate(queries) if t[10]]
    if not active:
        raise ValueError('Missing terminal timestamps')
    origin = min(t[0] for _, _, t in active)
    phases = (('initialize', 0, 1), ('TopK', 1, 2), ('publish.TopK', 2, 3),
              ('deduplicate_and_slots', 3, 11), ('output', 11, 7),
              ('WQE_and_publish', 7, 4), ('release_scratch', 4, 5),
              ('diagnostic_snapshot', 8, 26), ('index_update_and_join', 26, 16),
              ('final_CQ_join', 16, 24), ('prefetch_CQ_join', 24, 17), ('retire', 17, 10))
    cores = []
    for b, q, t in active:
        stages = []
        cursor = t[0]
        for name, a, z in phases:
            begin, end = t[a], t[z]
            if begin < cursor or end < begin:
                raise ValueError(f'Invalid terminal stamp: request={b} Q={q+1} phase={name}')
            if begin > cursor:
                stages.append(dict(name='control_gap', start_cycles=cursor, end_cycles=begin,
                    start_us=(cursor-origin)/clock_mhz, end_us=(begin-origin)/clock_mhz,
                    duration_us=(begin-cursor)/clock_mhz))
            stages.append(dict(name=name, start_cycles=begin, end_cycles=end,
                start_us=(begin-origin)/clock_mhz, end_us=(end-origin)/clock_mhz,
                duration_us=(end-begin)/clock_mhz))
            cursor = end
        cores.append(dict(block=b*4+q, role=f'terminal b{b} Q{q+1}',
            start_us=(t[0]-origin)/clock_mhz, end_us=(t[10]-origin)/clock_mhz,
            duration_us=(t[10]-t[0])/clock_mhz, stages=stages))
    last = max(cores, key=lambda core: core['end_us'])
    return dict(clock_mhz=clock_mhz, cores=cores, last_block=last['block'],
                observed_span_us=last['end_us'])


def terminal_diagnostics(stamps, rows, posts, events, destinations, service_lanes, token_bytes, clock_mhz):
    """All counts and scopes here belong to this instrumented replay only."""
    timeline = decode_terminal_timeline(stamps, clock_mhz)
    result, groups = [], []
    for b, queries in enumerate(stamps):
        prior = int(rows[b, 9])
        total = int(rows[b, 8])
        prefix_slots = set(map(int, destinations[b, :prior]))
        counts = []
        for q, t in enumerate(queries):
            count, start = int(posts[b, q, 0]), int(t[22])
            if count != t[20] or (count and (start < prior or start + count > total)):
                raise ValueError(f'Terminal assignment count mismatch: request={b} Q={q+1}')
            slots = destinations[b, start:start+count]
            conflict = sum(int(slot) in prefix_slots for slot in slots)
            if conflict:
                raise ValueError('Terminal final allocations overlap prior prefetch destinations')
            counts.append(dict(total=count, safe=count, conflict=conflict))
            if count:
                if not 0 < t[15] <= t[4] <= t[24]:
                    raise ValueError('Missing terminal submission/CQ timestamps')
                groups.append(dict(block=b*4+q, query=q+1, lane=b, entries=count,
                    bytes=count*token_bytes, first_cycles=t[15], last_cycles=t[24],
                    span_us=(t[24]-t[15])/clock_mhz))
        result.append(dict(batch=b, final_counts=counts))
    head = result[0]
    head['terminal_timeline'] = timeline
    events = np.asarray(events).reshape(-1, 128).copy().view('<u8')
    services = []
    for index, event in enumerate(events):
        if not event[0] or event[1] < event[0]:
            continue
        words = event.view('<u4')
        services.append(dict(event=index, core=int(words[4]), epoch=int(words[5]), lane=index % service_lanes,
            begin_cycles=int(event[0]), end_cycles=int(event[1]),
            predecessor_wait_cycles=int(event[13])-int(event[4]) if event[4] and event[13] >= event[4] else None,
            incremental_cycles=int(event[14])-int(event[13]) if event[13] and event[14] >= event[13] else None,
            scan_cycles=int(event[5]), candidate_cycles=int(event[6]), post_cycles=int(event[7]),
            bitmap_bytes=int(event[8]), candidate_count=int(event[9]), posted_count=int(event[10])))
    head['prefetch_services'] = services

    def scope(entries, begin, end, note):
        if entries and not 0 < begin < end:
            raise ValueError(f'Invalid transfer scope: {note}')
        span = (end-begin)/clock_mhz if entries else None
        return dict(entries=entries, bytes=entries*token_bytes, span_us=span,
                    GBps=entries*token_bytes/span/1000 if span else None,
                    first_cycles=begin, last_cycles=end, note=note)

    final_count = sum(g['entries'] for g in groups)
    head['final_get_bandwidth'] = scope(final_count,
        min((g['first_cycles'] for g in groups), default=0),
        max((g['last_cycles'] for g in groups), default=0),
        'Terminal submission to CQ observation; includes overlapped Index/output work, not isolated Get latency')
    head['final_get_bandwidth']['groups'] = groups
    prior_count = int(rows[:, 9].astype(np.uint64).sum())
    starts = [int(e[11]) for e in events if e[11]]
    fallback = not starts
    if fallback:
        starts = [int(e[0]) for e in events if e[0]]
    head['prefetch_bandwidth'] = scope(prior_count, min(starts, default=0),
        max((stamps[b][1][17] for b in range(len(stamps)) if rows[b, 9]), default=0),
        'Prefetch service entry to CQ join return' if fallback else 'Prefetch first doorbell to CQ join return')
    nonempty = [head[k] for k in ('prefetch_bandwidth', 'final_get_bandwidth') if head[k]['entries']]
    head['combined_observed_bandwidth'] = scope(prior_count+final_count,
        min((s['first_cycles'] for s in nonempty), default=0), max((s['last_cycles'] for s in nonempty), default=0),
        'Union envelope of prefetch and terminal submissions through software-observed completion')
    return result
