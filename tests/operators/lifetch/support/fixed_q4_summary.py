# SPDX-License-Identifier: Apache-2.0
"""Summarize matched single-event measurements, retaining each small-batch donor cohort."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from large_batch_input import FIXED_BATCHES, GROUPS
from reporting import atomic_json


def read_arm(path, group, batch, arm, physical, host):
    data = json.loads((path / 'ARM_RESULT.json').read_text())
    identity = data['identity']
    calls = identity['calls']
    if (data['arm'] != arm or (identity['group'], identity['batch']) != (group, batch)
            or calls != 10 or identity['physical_npu'] != physical or identity['host'] != host or len(data['samples']) != calls
            or data['checked_queries'] != calls * batch * 4 or data['checked_rows'] != calls * batch
            or data['full_KV_Index_checked_rows'] != batch
            or (arm == 'fused' and data['ordered_Index_diagnostic_rows'] != batch)):
        raise ValueError(f"Incomplete {arm} evidence: {path}")
    for stage in ('graph', 'LI', 'tail'):
        values = [s[stage] for s in data['samples']]
        if (any(not math.isfinite(v) or v <= 0 for v in values)
                or [mean(values), median(values)] != data['mean_p50_us'][stage]):
            raise ValueError(f"Invalid {stage} samples: {path}")
    return data


def summarize_cell(root, group, batch, physical, host):
    pieces = []
    samples = {arm: [] for arm in ('serial', 'fused')}
    n = max(1, 16 // batch)
    for i in range(n):
        case_dir = root / f'g{group}/bs{batch}'
        if n > 1:
            case_dir = case_dir / f'cohort{i}'
        fused = read_arm(case_dir / 'fused', group, batch, 'fused', physical, host)
        serial_dir = Path(fused['comparison_archive']) if 'comparison_archive' in fused else case_dir
        serial = read_arm(serial_dir / 'serial', group, batch, 'serial', physical, host)
        fi, si = fused['identity'], serial['identity']
        if dict(si, native=fi['native']) != fi:
            raise ValueError(f"Different operands, hardware or timing protocol at {case_dir}")
        donors = fi['donor_rows']
        if batch <= 16 and donors != list(range(i, 16, n)):
            raise ValueError(f"Unexpected small-batch donor cohort at {case_dir}: {donors}")
        for arm, result in (('serial', serial), ('fused', fused)):
            samples[arm].extend(result['samples'])
        s, f = serial['mean_p50_us']['graph'][0], fused['mean_p50_us']['graph'][0]
        pieces.append(dict(cohort=i, donor_rows=donors, serial_graph_us=s, fused_graph_us=f,
                           graph_gain_pct=100 * (s - f) / s, passed=f <= 0.8 * s))
    times = {arm: {stage: [mean(v[stage] for v in items), median(v[stage] for v in items)]
                   for stage in items[0]} for arm, items in samples.items()}
    s, f = times['serial']['graph'][0], times['fused']['graph'][0]
    return dict(group=group, batch=batch, serial_graph_us=s, fused_graph_us=f,
                graph_gain_pct=100 * (s - f) / s, graph_saved_us=s - f,
                passed=f <= 0.8 * s, cohorts=pieces, mean_p50_us=times)


def report(root, groups, batches):
    physical = int((root / 'physical_npu.txt').read_text())
    host = (root / 'host.txt').read_text().strip()
    if physical not in (3, 4) or not host:
        raise ValueError(f"Missing or unsupported matrix measurement identity at {root}")
    cells = [summarize_cell(root, group, batch, physical, host) for group in groups for batch in batches]
    full = set(groups) == set(GROUPS) and set(batches) == set(FIXED_BATCHES)
    for cell in cells:
        cohort_pass = sum(v['passed'] for v in cell['cohorts'])
        print(f"G{cell['group']} BS{cell['batch']} SINGLE_EVENT "
              f"serial_graph_mean_us={cell['serial_graph_us']:.2f} fused_graph_mean_us={cell['fused_graph_us']:.2f} "
              f"saved_us={cell['graph_saved_us']:+.2f} gain_pct={cell['graph_gain_pct']:+.2f} "
              f"target20={'PASS' if cell['passed'] else 'MISS'} cohorts_at_target={cohort_pass}/{len(cell['cohorts'])}")
    passed = sum(c['passed'] for c in cells)
    out = dict(protocol='one_restored_graph_per_device_event_with_profiler', physical_npu=physical, host=host,
               complete=full, passed=full and passed == len(cells), cells=cells)
    atomic_json(root / 'fixed_q4_summary.json', out)
    print(f"SINGLE_EVENT_TARGET cells_at_target={passed}/{len(cells)} full_matrix={int(full)} "
          f"SUMMARY={root / 'fixed_q4_summary.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--groups', type=int, choices=GROUPS, nargs='+', required=True)
    parser.add_argument('--batches', type=int, choices=FIXED_BATCHES, nargs='+', required=True)
    args = parser.parse_args()
    report(args.root, args.groups, args.batches)
