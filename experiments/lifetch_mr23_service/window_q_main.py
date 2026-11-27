# SPDX-License-Identifier: Apache-2.0
"""Run and verify the unmodified main baseline for the W1 query matrix."""

import statistics
import subprocess
import sys

from run_mr25_handoff import apply_group_timing, read
from diagnostics import account
from terminal_handoff import COUNT_COLUMNS, STAGES, fixture_hash, terminal_stages


NATIVE_BENCH = """
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'tests/operators/lifetch/support'), str(root / 'python')]
from workloads import configuration
from execution import launch

raise SystemExit(launch(configuration(sys.argv[3:]), sys.argv[2]))
"""


def measure_main(root, output, case, commit):
    logical_batch, waves = case['batch'], case['waves']
    batch, miss = case['physical_batch'], case['miss_percent']
    worker = output / f'bs{logical_batch}' / 'main_p2'
    arguments = list(case['arguments'])
    # Native main derives the 128-token window budget from 512 / requested W4.
    option = arguments.index('--prefetch-window-limit')
    del arguments[option:option + 2]
    arguments[0] = 'diagnose'
    arguments.extend(('--level', '1'))
    subprocess.run([sys.executable, '-B', '-c', NATIVE_BENCH, str(root), str(worker), *arguments],
                   cwd=root, check=True)

    fixture = worker / f'miss{miss:g}' / 'fixture'
    folder = fixture.parent / f'case-b{batch}'
    report = read(folder / 'report.json')
    result = read(folder / 'fused/path_result.json')
    timing = result['all_rounds']
    calls = int(arguments[arguments.index('--iterations') + 1])
    if (report['execution'] != 'COMPLETED' or report['correctness'] != 'PASS'
            or result['correctness'] != 'PASS' or timing['event_mean_us'] <= 0
            or timing['graph_calls'] != calls or timing['measured_graphs'] != 1):
        raise ValueError(f'Main BS{batch}: incomplete checked graph run in {folder}')

    diagnostic = folder / 'fused/diagnose/round-1'
    profile = read(diagnostic / 'graph_profile.json')
    trace = read(diagnostic / 'trace.json')
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    checked = account(events, 'lifetch_measured_graph', profile['timing']['graph_event_us'],
                      full=True, fused_final=True)
    if 'error' in checked:
        raise ValueError(f'Main BS{batch}: cannot verify native kernel sequence: {checked}')
    profile['accounting'] = checked
    stages = terminal_stages(profile, kernel_route='p2')
    samples = read(diagnostic / 'full_attribution.json')['samples']
    if len(stages) != calls or len(samples) != calls:
        raise ValueError(f'Main BS{batch}: profiler or request count coverage differs')
    counts = []
    for invocation, sample in enumerate(samples):
        requests = sample['algorithm'].get('requests', [])
        if (sample['source_round'] != 1 or sample['accounting']['invocation'] != invocation
                or [request['request'] for request in requests] != list(range(batch))):
            raise ValueError(f'Main BS{batch}: unexpected request or graph invocation identity')
        for request in requests:
            early, final = request['prefetch'], request['final']
            windows = [row['posted'] for row in request['windows']]
            final_by_query = [row['final_posted'] for row in request['queries']]
            if (len(windows) != case['effective_windows'] or len(final_by_query) != 4
                    or sum(windows) != early or sum(final_by_query) != final
                    or not 0 <= early <= 512 or any(not 0 <= value <= 128 for value in windows)
                    or min(final_by_query) < 0):
                raise ValueError(f'Main BS{batch}: invalid completed Get counters')
            counts.append(dict(source_round=1, invocation=invocation, request=request['request'],
                               **dict(zip(COUNT_COLUMNS, (early, final, early + final)))))

    row = dict(variant='p2', source_commit=commit, batch=logical_batch, accepted=True, score_parts=2,
               cold_score_parts=2, windows=case['effective_windows'],
               service_lanes=2 * case['service_aics'], fixture_sha256=fixture_hash(fixture),
               kv_ready_us=timing['event_mean_us'], fused_event_us=timing['event_mean_us'],
               profiler_p50_us=timing['p50_us'],
               per_call_requests=counts, report=str(folder / 'report.json'),
               **{key: statistics.mean(sample[key] for sample in stages) for key in STAGES},
               **{key + '_per_call': sum(record[key] for record in counts) / calls for key in COUNT_COLUMNS})
    if waves > 1:
        apply_group_timing(row, profile['timing'], waves)
        timing_label, get_unit = f'{waves}xBS{batch} graph KV ready', 'group'
    else:
        timing_label, get_unit = 'KV ready', 'call'
    early, final = (row[f'{field}_per_{get_unit}'] for field in COUNT_COLUMNS[:2])
    print(f'BS{logical_batch} main P2: {timing_label} {row["kv_ready_us"]:.2f} µs, '
          f'prefetch / final Get per {get_unit} {early:.2f} / {final:.2f}', flush=True)
    return row
