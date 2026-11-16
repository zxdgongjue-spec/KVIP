# SPDX-License-Identifier: Apache-2.0
"""Structured results and concise terminal tables."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


def timing_summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError('timing samples must be a nonempty vector')
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('timing samples must be finite positive microseconds')
    return dict(mean_us=float(values.mean()), p50_us=float(np.median(values)),
                p99_us=float(np.percentile(values, 99)), samples=len(values),
                samples_us=values.tolist())


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
        for metric in ('mean_us', 'p50_us', 'p99_us', 'samples'):
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
    headers = ['#', 'Transport', 'BS', 'Miss%', 'Execution', 'Check',
               'Serial mean', 'Fused mean', 'Gain', 'Serial p50/p99', 'Fused p50/p99']
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
    widths = [max(len(line[col]) for line in [headers, *cells]) for col in range(len(headers))]
    def formatted(line):
        return ' | '.join(value.ljust(width) if col in (1, 4, 5) else value.rjust(width)
                          for col, (value, width) in enumerate(zip(line, widths)))
    separator = '-+-'.join('-' * width for width in widths)
    lines = ['LIFetch comparison | time: us | Gain: reduction in mean time; positive is faster',
             formatted(headers), separator, *(formatted(line) for line in cells), separator,
             f"Completed: {sum(row['execution'] == 'COMPLETED' for row in rows)}/{len(rows)}; "
             f"correctness PASS: {sum(row['correctness'] == 'PASS' for row in rows)}/{len(rows)}"]
    for index, row in enumerate(rows, 1):
        if row.get('mode') == 'check' or row['execution'] != 'COMPLETED' or row['correctness'] != 'PASS':
            lines.append(f"[{index}] {row['case']}" + (f": {row['reason']}" if row.get('reason') else ''))
    (directory / 'summary.log').write_text('\n'.join(lines) + '\n')


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
