#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run graph-mode LIFetch correctness and performance for one transport."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / 'tests/operators/lifetch/support'
sys.path.insert(0, str(SUPPORT))
from compare_transports import execute  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', required=True, choices=('urma', 'ubmem'))
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--kvaf-server', required=True)
    p.add_argument('--kvaf-root', type=Path)
    p.add_argument('--soc')
    p.add_argument('--cann', default='/usr/local/Ascend/cann')
    p.add_argument('--batches', default='16', help='comma-separated request counts')
    p.add_argument('--history', type=int, default=131072)
    p.add_argument('--resident', type=int, default=16384)
    p.add_argument('--index-heads', type=int, choices=(32, 64), default=32)
    p.add_argument('--intermediate-fetches', type=int, default=8)
    p.add_argument('--prefetch-tokens', type=int, default=512)
    p.add_argument('--miss-percent', type=float, default=10)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--trace-rounds', type=int, default=3, help='URMA device profiling replays; UBmem reports graph timing')
    p.add_argument('--runtime-shapes', action='store_true')
    p.add_argument('--resolver-compare', action='store_true', help='URMA resolver comparison')
    p.add_argument('--comparison-output', type=Path)
    p.add_argument('--comparison-sfa', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'tests/results/lifetch')
    p.add_argument('--timeout-seconds', type=int, default=1800)
    a = p.parse_args(argv)
    try:
        batches = [int(x) for x in a.batches.split(',')]
    except ValueError:
        p.error('batches must be comma-separated integers')
    if not batches or any(not 1 <= x <= 32 for x in batches):
        p.error('batches must be in 1..32')
    if (a.history <= a.resident + 4 or not 0 <= a.miss_percent <= 100 or a.warmup < 0
            or a.iterations < 1 or a.trace_rounds < 0 or not 1 <= a.intermediate_fetches <= 32
            or not 0 <= a.prefetch_tokens <= 2048 or a.prefetch_tokens % a.intermediate_fetches
            or a.timeout_seconds < 20):
        p.error('invalid geometry, prefetch budget, timing or timeout')
    if (a.comparison_output or a.comparison_sfa) and (len(batches) != 1 or a.runtime_shapes):
        p.error('comparison snapshots require one batch and no runtime-shapes')
    state = ROOT / 'build/dependencies.json'
    kvaf_root = (a.kvaf_root or (Path(json.loads(state.read_text())['kvaf']) if state.exists()
                                 else ROOT / 'third_party/KVAF')).resolve()
    common = ['--device', str(a.device), '--kvaf-server', a.kvaf_server, '--history', str(a.history),
              '--resident', str(a.resident), '--index-heads', str(a.index_heads), '--miss-percent',
              str(a.miss_percent), '--prefetch-tokens', str(a.prefetch_tokens), '--warmup', str(a.warmup),
              '--iterations', str(a.iterations)]
    if a.transport == 'urma':
        command = [sys.executable, '-u', str(SUPPORT / 'test_urma_runner.py'), *common,
                   '--kvaf-root', str(kvaf_root), '--cann', a.cann, '--batches', a.batches,
                   '--intermediate-fetches', str(a.intermediate_fetches), '--trace-rounds', str(a.trace_rounds),
                   '--output', str(a.output)]
        if a.soc:
            command += ['--soc', a.soc]
        if a.runtime_shapes:
            command.append('--runtime-shapes')
        if a.resolver_compare:
            command.append('--resolver-compare')
        if a.comparison_output:
            command += ['--comparison-output', str(a.comparison_output)]
        if a.comparison_sfa:
            command += ['--comparison-sfa', str(a.comparison_sfa)]
        return subprocess.call(command)
    if a.intermediate_fetches != 8:
        p.error('UBmem fixture currently supports exactly 8 windows')
    if not a.soc:
        p.error('--soc is required for UBmem, e.g. Ascend950PR_9599')
    if a.resolver_compare:
        p.error('resolver comparison is currently supported only by URMA')
    run = a.output.resolve() / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    run.mkdir(parents=True)
    log, summary = run / 'worker.log', run / 'summary.log'
    print(f'Summary: {summary}', flush=True)
    command = [sys.executable, '-u', str(SUPPORT / 'ubmem_smoke.py'), *common,
               '--build-dir', str(kvaf_root / 'build-npu'), '--soc', a.soc,
               '--batches', *map(str, batches), '--token-bytes', '656']
    if a.runtime_shapes:
        command.append('--runtime-shapes')
    if a.comparison_output:
        command += ['--comparison-output', str(a.comparison_output)]
    if a.comparison_sfa:
        command += ['--comparison-sfa', str(a.comparison_sfa)]
    print('UBmem reports full-graph timing; per-stage trace is not collected.', flush=True)
    code = execute(command, log, a.timeout_seconds, 'ubmem')
    lines = log.read_text(errors='replace').splitlines()
    selected = [line for line in lines if line.startswith(('UBMEM_LIFETCH_PASS', 'UBMEM_SFA_PASS',
                'UBMEM_GRAPH_EVENT_US', 'UBMEM_FETCH_COUNTS', 'UBMEM_WINDOW_TRIGGERS', 'UBMEM_RUNTIME_BINARY'))]
    expected = len(batches) + int(a.runtime_shapes and (a.history, a.resident) != (33024, 8192))
    ok = (code == 0 and sum(line.startswith('UBMEM_LIFETCH_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_SFA_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_GRAPH_EVENT_US') for line in selected) == expected)
    result = [f'{"PASS" if ok else "FAIL"}: transport=ubmem exit={code}', *selected]
    if not ok:
        result += ['--- worker.log tail ---', *lines[-25:]]
    summary.write_text('\n'.join(result) + '\n')
    print('\n'.join(result), flush=True)
    print(f'Full logs: {run}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
