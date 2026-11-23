# SPDX-License-Identifier: Apache-2.0
"""Structured results and concise terminal tables."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def timing_summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError('timing samples must be a nonempty vector')
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('timing samples must be finite positive microseconds')
    return dict(_timing_distribution(values), samples_us=values.tolist())


def _timing_distribution(values):
    return dict(mean_us=float(values.mean()), p50_us=float(np.median(values)),
                p99_us=float(np.percentile(values, 99)), samples=len(values))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CorrectnessRecorder:
    """Only numerical/state assertions are recoverable; execution errors propagate."""

    def __init__(self):
        self.failures = []
        self.checked = 0

    def check(self, label, callback):
        self.checked += 1
        try:
            callback()
        except AssertionError as error:
            self.failures.append(dict(check=label, error=str(error)))
            return False
        return True

    @property
    def status(self):
        return 'FAIL' if self.failures else 'PASS' if self.checked else 'NOT_RUN'


def comparison_row(case_id, serial, fused, correctness, execution='COMPLETED'):
    """Keep measured deltas on incorrect results; expose acceptance separately."""
    row = dict(case=case_id, execution=execution, correctness=correctness,
               timing='COMPLETE' if serial and fused else 'INCOMPLETE')
    for name, timing in [('serial', serial), ('fused', fused)]:
        for metric in ('mean_us', 'profiler_mean_us', 'p50_us', 'p99_us', 'samples', 'timing_protocol', 'stage_mean_us', 'inter_kernel_gap_mean_us', 'overlap_mean_us', 'outside_invocations_mean_us'):
            row[f'{name}_{metric}'] = timing.get(metric) if timing else None
    if serial and fused:
        row['reduction'] = 1 - fused['mean_us'] / serial['mean_us']
        row['speedup'] = serial['mean_us'] / fused['mean_us']
    else:
        row.update(reduction=None, speedup=None)
    row['accepted'] = (execution == 'COMPLETED' and correctness == 'PASS'
                       and row['timing'] == 'COMPLETE')
    return row


def save_summary(directory, rows):
    """Rewrite the accumulated summary after each completed case."""
    directory = Path(directory)
    atomic_json(directory / 'summary.json', dict(cases=rows))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = directory / 'results.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / 'results.csv')
    batched = any(row.get('serial_timing_protocol') == 'one_profiled_graph' for row in rows)
    headers = ['#', 'Transport', 'BS', 'Miss%', 'Execution', 'Check',
               'Serial mean', 'Fused mean', 'Gain', 'Serial p50/p99', 'Fused p50/p99']
    if batched:
        headers[6:8] = ['Serial event mean', 'Fused event mean']
        headers += ['Serial prof mean', 'Fused prof mean']
        headers[9:11] = ['Serial prof p50/p99', 'Fused prof p50/p99']
    get_only = any(row.get('serial_get_only_mean_us') is not None for row in rows)
    if get_only:
        headers += ['Serial Get mean', 'Serial Get p50/p99']
    cells = []
    for index, row in enumerate(rows, 1):
        def number(key):
            value = row.get(key)
            return '—' if value is None else f'{value:.2f}'
        def percentiles(prefix):
            return f"{number(prefix + '_p50_us')}/{number(prefix + '_p99_us')}"
        miss = row.get('miss_percent')
        reduction = row.get('reduction')
        cells.append([str(index), row.get('transport') or '—', str(row.get('batch') or '—'),
                      '—' if miss is None else f'{miss:g}', row['execution'], row['correctness'],
                      number('serial_mean_us'), number('fused_mean_us'),
                      '—' if reduction is None else f'{reduction:+.2%}',
                      percentiles('serial'), percentiles('fused')])
        if batched:
            cells[-1] += [number('serial_profiler_mean_us'), number('fused_profiler_mean_us')]
        if get_only:
            cells[-1] += [number('serial_get_only_mean_us'), percentiles('serial_get_only')]
    widths = [max(len(line[col]) for line in [headers, *cells]) for col in range(len(headers))]
    def formatted(line):
        return ' | '.join(value.ljust(width) if col in (1, 4, 5) else value.rjust(width)
                          for col, (value, width) in enumerate(zip(line, widths)))
    separator = '-+-'.join('-' * width for width in widths)
    lines = ['LIFetch comparison | time: us | Gain: reduction in mean time; positive is faster',
             formatted(headers), separator, *(formatted(line) for line in cells), separator,
             f"Completed: {sum(row['execution'] == 'COMPLETED' for row in rows)}/{len(rows)}; "
             f"correctness PASS: {sum(row['correctness'] == 'PASS' for row in rows)}/{len(rows)}"]
    if batched:
        stage_rows = []
        for index, row in enumerate(rows, 1):
            for path in ('serial', 'fused'):
                for stage, value in (row.get(f'{path}_stage_mean_us') or {}).items():
                    stage_rows.append([str(index), path, stage, f'{value:.2f}'])
                value = row.get(f'{path}_inter_kernel_gap_mean_us')
                if value is not None:
                    stage_rows.append([str(index), path, 'within-call gaps', f'{value:.2f}'])
                value = row.get(f'{path}_overlap_mean_us')
                if value:
                    stage_rows.append([str(index), path, 'subtract overlap', f'{-value:.2f}'])
                value = row.get(f'{path}_outside_invocations_mean_us')
                if value is not None:
                    stage_rows.append([str(index), path, 'between calls + event edges', f'{value:.2f}'])
        stage_headers = ['#', 'Path', 'Same-graph stage', 'Mean us/call']
        stage_widths = [max(len(r[i]) for r in [stage_headers, *stage_rows]) for i in range(4)]
        lines += ['', 'Profiler stages from the same measured graph; per-call spans include internal gaps.',
                  ' | '.join(v.ljust(w) for v, w in zip(stage_headers, stage_widths)),
                  '-+-'.join('-'*w for w in stage_widths)]
        lines += [' | '.join(v.ljust(w) for v, w in zip(r, stage_widths)) for r in stage_rows]
    for index, row in enumerate(rows, 1):
        if row.get('mode') == 'check' or row['execution'] != 'COMPLETED' or row['correctness'] != 'PASS':
            lines.append(f"[{index}] {row['case']}" + (f": {row['reason']}" if row.get('reason') else ''))
    (directory / 'summary.log').write_text('\n'.join(lines) + '\n')


def add_transfer_counts(row, report):
    serial_misses = report.get('native_misses_per_request')
    fused_counts = report.get('fused_read_counts', [])
    if not fused_counts:
        # Older synthetic URMA reports already contain measured read samples.
        fused_counts = [dict(prefetch=sample['total'] - sample['final'], final=sample['final'])
                        for sample in report.get('read_samples', {}).get('prototype_with_update', [])]
    if serial_misses is None and not fused_counts:
        return
    width = row['token_bytes'] = report.get('token_bytes', 656)
    if serial_misses is not None:
        row['serial_read_entries'] = sum(serial_misses)
        row['serial_read_bytes'] = row['serial_read_entries'] * width
    if fused_counts:
        row['fused_read_count_samples'] = len(fused_counts)
        phases = dict(prefetch=[sample['prefetch'] for sample in fused_counts],
                      final=[sample['final'] for sample in fused_counts],
                      total=[sample['prefetch'] + sample['final'] for sample in fused_counts])
        for phase, counts in phases.items():
            prefix = f'fused_{phase}'
            mean = sum(counts) / len(counts)
            row[f'{prefix}_entries_mean'] = mean
            row[f'{prefix}_entries_min'] = min(counts)
            row[f'{prefix}_entries_max'] = max(counts)
            row[f'{prefix}_bytes_mean'] = mean * width


def add_profile_metrics(rows, diagnostic):
    """Join each case to its own profiled replays, independently of timing and correctness."""
    captures = {Path(capture['source']).resolve(): capture['samples'] for capture in diagnostic['captures']}
    stage_order = dict(
        serial=('total', 'li', 'lidu_prepare', 'get'),
        fused=('total', 'init', 'li', 'resolve_update', 'resolve_update_fetch', 'resolver_span',
               'output_pack_span', 'index_apply_span', 'prefetch_span', 'final_get_span',
               'transfer_span', 'finalize_span'),
    )
    transfer_phases = (('prefetch_bandwidth', 'prefetch'), ('final_get_bandwidth', 'final_get'),
                       ('combined_observed_bandwidth', 'transfer'))
    for row in rows:
        if 'report' not in row:
            continue
        folder = Path(row['report']).parent
        paths = dict(serial=folder / 'native_attribution.json',
                     fused=Path(row.get('fused_attribution', folder / 'full_attribution.json')))
        for path, source in paths.items():
            samples = captures.get(source.resolve())
            if samples is None:
                root = (folder / path / 'diagnose').resolve()
                samples = [sample for capture, records in captures.items()
                           if capture.is_relative_to(root) for sample in records]
            if not samples:
                continue
            times = defaultdict(list)
            transfers = defaultdict(list)
            for sample in samples:
                accounting = sample.get('accounting', {})
                stages = dict(accounting.get('stages_us', {}), **sample.get('core_phase_spans_us', {}))
                if path == 'serial' and 'fetch' in stages:
                    stages['get'] = stages.pop('fetch')
                if not accounting.get('error'):
                    for name in ('inter_kernel_gap', 'boundary_residual'):
                        stages[name] = accounting.get(f'{name}_us')
                stages['total'] = sample.get('us', accounting.get('event_us'))
                for name, value in stages.items():
                    if value is not None and np.isfinite(value):
                        times[name].append(value)
                for metric, phase in transfer_phases:
                    scope = sample.get(metric, {})
                    if scope.get('span_us') and scope.get('GBps') is not None:
                        times[f'{phase}_span'].append(scope['span_us'])
                        transfers[phase].append(scope)
            prefix = f'{path}_profiled'
            stages = dict.fromkeys((*stage_order[path], 'inter_kernel_gap', 'boundary_residual', *times))
            for stage in stages:
                values = times[stage]
                distribution = (_timing_distribution(np.asarray(values)) if values else
                                dict(mean_us=None, p50_us=None, p99_us=None, samples=0))
                for key, value in distribution.items():
                    row[f'{prefix}_{stage}_{key}'] = value
            if path == 'fused':
                for _, phase in transfer_phases:
                    scopes = transfers[phase]
                    row[f'{prefix}_{phase}_bytes_mean'] = (
                        sum(scope['bytes'] for scope in scopes) / len(scopes) if scopes else None)
                    row[f'{prefix}_{phase}_GBps_p50'] = (
                        float(np.median([scope['GBps'] for scope in scopes])) if scopes else None)


def collect_results(directory, config, exit_code):
    """Read structured worker results; log PASS strings are not evidence."""
    directory = Path(directory)
    rows = []
    observed = set()
    reports = list(directory.rglob('report.json'))
    reports += [path for path in directory.rglob('completed_timings.json')
                if not (path.parent / 'report.json').exists()]
    for path in sorted(reports):
        report = json.loads(path.read_text())
        batch = report.get('batch')
        miss = report.get('miss_percent', getattr(config, 'miss_percent', None))
        observed.add((batch, miss))
        timing = report.get('timing', {})
        correct = report.get('correctness', report.get('status', 'NOT_RUN').upper())
        if report.get('serial_correctness', 'PASS') != 'PASS':
            correct = 'FAIL'
        row = comparison_row(str(path.parent.relative_to(directory)), timing.get('native_chain'),
                             timing.get('prototype_with_update'), correct,
                             execution=report.get('execution', 'INCOMPLETE' if path.name == 'completed_timings.json' else 'COMPLETED'))
        row.update(batch=batch, transport=config.transport, mode=config.mode, report=str(path),
                   seed=getattr(config, 'seed', None), miss_percent=report.get('miss_percent', getattr(config, 'miss_percent', None)),
                   history=getattr(config, 'history', None), resident=getattr(config, 'resident', None),
                   prefetch_tokens=getattr(config, 'prefetch_tokens', None))
        if config.transport == 'urma':
            fetch_jettys = report.get('fetch_jettys_per_aiv', getattr(config, 'fetch_jettys_per_aiv', None))
            if fetch_jettys is not None:
                row['fetch_jettys_per_aiv'] = fetch_jettys
        if report.get('fused_attribution'):
            row['fused_attribution'] = str((path.parent / report['fused_attribution']).resolve())
        if timing.get('native_get_only'):
            for key in ('mean_us', 'p50_us', 'p99_us', 'samples'):
                row[f'serial_get_only_{key}'] = timing['native_get_only'].get(key)
        add_transfer_counts(row, report)
        rows.append(row)
    misses = getattr(config, 'miss_percent', None)
    misses = misses if isinstance(misses, (tuple, list)) else (misses,)
    for miss in misses:
        for batch in (getattr(config, 'batches', ()) or ()):
            if (batch, miss) not in observed:
                row = comparison_row(f'BS{batch}/miss{miss}', None, None, 'NOT_RUN',
                                     execution='FAILED' if exit_code else 'INCOMPLETE')
                row.update(batch=batch, miss_percent=miss, transport=config.transport,
                           reason='missing structured case result')
                rows.append(row)
    if exit_code and rows and all(row['execution'] == 'COMPLETED' for row in rows):
        row = comparison_row('worker', None, None, 'NOT_RUN', execution='FAILED')
        row['reason'] = f'worker exit={exit_code}; completed case timings retained'
        rows.append(row)
    if not rows:
        rows.append(comparison_row(directory.name, None, None, 'NOT_RUN',
                                   execution='FAILED' if exit_code else 'INCOMPLETE'))
    miss_order = {miss: index for index, miss in enumerate(misses)}
    batch_order = {batch: index for index, batch in enumerate(getattr(config, 'batches', ()) or ())}
    rows.sort(key=lambda row: (miss_order.get(row.get('miss_percent'), len(miss_order)),
                               batch_order.get(row.get('batch'), len(batch_order))))
    return rows


def record_case_failure(directory, error):
    """Keep completed timing samples without carrying an obsolete overall PASS."""
    path = Path(directory) / 'completed_timings.json'
    if not path.exists():
        return
    report = json.loads(path.read_text())
    report['prior_correctness'] = report.get('correctness', 'NOT_RUN')
    report['correctness'] = 'FAIL' if isinstance(error, AssertionError) else 'INCOMPLETE'
    report['execution'] = 'FAILED'
    report['failure'] = f'{type(error).__name__}: {error}'
    atomic_json(path, report)
