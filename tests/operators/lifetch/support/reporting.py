"""Shared result accounting; correctness never hides completed timing samples."""
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
    lines = ['case | transport | BS | miss% | execution | correctness | serial mean us | fused mean us | reduction | serial p50/p99 us | fused p50/p99 us']
    for row in rows:
        def number(key):
            value = row.get(key)
            return 'N/A' if value is None else f'{value:.3f}'
        reduction = row.get('reduction')
        delta = 'N/A' if reduction is None else f'{reduction:.2%}'
        lines.append(f"{row['case']} | {row.get('transport', 'N/A')} | {row.get('batch', 'N/A')} | "
                     f"{row.get('miss_percent', 'N/A')} | {row['execution']} | {row['correctness']} | "
                     f"{number('serial_mean_us')} | {number('fused_mean_us')} | {delta} | "
                     f"{number('serial_p50_us')}/{number('serial_p99_us')} | "
                     f"{number('fused_p50_us')}/{number('fused_p99_us')}")
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
        observed.add(batch)
        timing = report.get('timing', {})
        correct = report.get('correctness', report.get('status', 'NOT_RUN').upper())
        if report.get('serial_correctness', 'PASS') != 'PASS':
            correct = 'FAIL'
        row = comparison_row(str(path.parent.relative_to(directory)), timing.get('native_chain'),
                             timing.get('prototype_with_update'), correct,
                             execution=report.get('execution', 'INCOMPLETE' if path.name == 'completed_timings.json' else 'COMPLETED'))
        row.update(batch=batch, transport=config.transport, mode=config.mode, report=str(path),
                   seed=getattr(config, 'seed', None), miss_percent=getattr(config, 'miss_percent', None),
                   history=getattr(config, 'history', None), resident=getattr(config, 'resident', None),
                   prefetch_tokens=getattr(config, 'prefetch_tokens', None))
        rows.append(row)
    for batch in (getattr(config, 'batches', ()) or ()):
        if batch not in observed:
            row = comparison_row(f'BS{batch}', None, None, 'NOT_RUN',
                                 execution='FAILED' if exit_code else 'INCOMPLETE')
            row.update(batch=batch, reason='missing structured case result')
            rows.append(row)
    if exit_code and rows and all(row['execution'] == 'COMPLETED' for row in rows):
        row = comparison_row('worker', None, None, 'NOT_RUN', execution='FAILED')
        row['reason'] = f'worker exit={exit_code}; completed case timings retained'
        rows.append(row)
    if not rows:
        rows.append(comparison_row(directory.name, None, None, 'NOT_RUN',
                                   execution='FAILED' if exit_code else 'INCOMPLETE'))
    return rows
