#!/usr/bin/env python3
"""BS1 Resolver/Index-update NPU correctness and eager event comparison."""
import argparse
import datetime
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def call(command, log, timeout):
    with log.open('w') as out:
        try:
            result = subprocess.run([str(x) for x in command], cwd=REPO,
                                    stdout=out, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'timeout after {timeout}s: {log}') from exc
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-35:])
        raise RuntimeError(f'exit={result.returncode}: {log}\n{tail}')
    return log.read_text(errors='replace')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', type=int, default=5)
    p.add_argument('--kvaf-root', required=True, type=Path)
    p.add_argument('--soc', default='Ascend950PR_9599')
    p.add_argument('--cann', type=Path, default=Path('/usr/local/Ascend/cann'))
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--timeout', type=int, default=90, help='seconds per invocation')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.device < 0 or args.warmup < 0 or args.iterations < 1 or args.timeout < 1:
        p.error('invalid device/warmup/iterations/timeout')
    kvaf = args.kvaf_root.resolve() / 'access_operator/demo/aiv_batch_get/include'
    if not kvaf.is_dir():
        p.error(f'KVAF include missing: {kvaf}')
    if not (args.cann / 'compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake').is_file():
        p.error(f'CANN compiler not found under {args.cann}')
    output = (args.output or REPO / 'tests/results/resolver-index-npu' /
              datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = output / 'summary.log'
    lines = ['RUNNING: BS1 Q4; event timing is eager, each sample uses fresh process and inputs.',
             f'Output: {output}', f'device={args.device} warmup={args.warmup} iterations={args.iterations}',
             'Excluded: LI, KV Fetch/CQ, SFA, graph capture/replay and fixture setup.']
    def emit(message):
        lines.append(message)
        summary.write_text('\n'.join(lines) + '\n')
        print(message, flush=True)
    summary.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    build = output / 'build'
    try:
        emit('Building isolated A/B operators...')
        call(['cmake', '-S', HERE, '-B', build, '-DRUN_MODE=npu',
              f'-DSOC_VERSION={args.soc}', f'-DASCEND_CANN_PACKAGE_PATH={args.cann}',
              f'-DKVAF_INCLUDE={kvaf}',
              f'-DOP_ROOT={REPO / "csrc/operators/lightning_indexer_with_fetch_safe"}'],
             output / 'configure.log', 120)
        call(['cmake', '--build', build, '--target', 'pipeline_ab_parallel_sim', '-j2'],
             output / 'build.log', 600)
        exe = build / 'pipeline_ab_parallel_sim'
        pattern = re.compile(r'^PASS mode=(\w+) fixture=([\w-]+) final_miss=(\d+) q_counts=\[([^]]+)\] event_status=(-?\d+) event_ms=([0-9.eE+-]+)', re.M)
        def run(mode, fixture, name):
            text = call([exe, mode, fixture, str(args.device)], output / (name + '.log'), args.timeout)
            match = pattern.search(text)
            if not match or match.group(1, 2) != (mode, fixture):
                raise RuntimeError(f'missing verified PASS: {output / (name + ".log")}')
            if int(match[5]) != 0 or float(match[6]) <= 0:
                raise RuntimeError(f'ACL event time unavailable (status={match[5]}, ms={match[6]}): {output / (name + ".log")}')
            return {'miss': int(match[3]), 'q_counts': match[4], 'us': float(match[6]) * 1000}
        for fixture in ('normal', 'collision', 'cross-query'):
            a = run('serial', fixture, f'check-serial-{fixture}')
            b = run('pipeline', fixture, f'check-pipeline-{fixture}')
            if (a['miss'], a['q_counts']) != (b['miss'], b['q_counts']):
                raise RuntimeError(f'{fixture}: Q counts differ: A={a} B={b}')
            emit(f'CORRECT {fixture}: final_miss={a["miss"]} Q=[{a["q_counts"]}] A/B row checks passed')
        samples = {'serial': [], 'pipeline': []}
        for i in range(args.warmup + args.iterations):
            order = ('serial', 'pipeline') if i % 2 == 0 else ('pipeline', 'serial')
            for mode in order:
                row = run(mode, 'normal', f'{"warmup" if i < args.warmup else "sample"}-{i:03d}-{mode}')
                if i >= args.warmup:
                    samples[mode].append(row['us'])
            emit(f'RUNNING: completed {i + 1}/{args.warmup + args.iterations} paired rounds')
        a = statistics.median(samples['serial']); b = statistics.median(samples['pipeline'])
        emit(f'PASS eager event p50_us: serial={a:.3f} pipeline={b:.3f} change={(b/a-1)*100:+.2f}%')
        emit(f'Ranges us: serial={min(samples["serial"]):.3f}..{max(samples["serial"]):.3f} pipeline={min(samples["pipeline"]):.3f}..{max(samples["pipeline"]):.3f}')
        emit('A uses 1 resolver + 1 planner + 4 apply AIVs; B uses 1 resolver + 1 update AIV concurrently.')
        emit('This is not graph replay or end-to-end speedup; NPU device cycles are not inferred from event time.')
        (output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
    except (RuntimeError, OSError) as exc:
        emit(f'FAIL: {exc}')
        return 1
    finally:
        print(f'Summary: {summary}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
