#!/usr/bin/env python3
"""Real URMA snapshot AB: unchanged prefix, experimental four-core terminal."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'csrc/operators/common/lifetch').is_dir())


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Snapshot runner layout changed: {old[:80]}')
    return text.replace(old, new)


def runner_source(source, candidate, output):
    source = replace_once(source,
        'r"v9_(?:li_kernel|streaming_resolve_update)(?:_\\d+)*"',
        'r"v9_(?:li_kernel|streaming_resolve_update|parallel_terminal)(?:_\\d+)*"')
    if candidate:
        injection = f'''
    import ctypes
    sys.path.insert(0, {str(HERE)!r})
    from build import build as build_terminal
    from native_forward import ParallelTerminalForward
    products = build_terminal(Path({str(output / 'native-build')!r}), headers, cann, acl.get_soc_name())
    forward = ParallelTerminalForward(forward,
        ctypes.CDLL(products['prefix']['path']), ctypes.CDLL(products['terminal']['path']))
'''
        source = replace_once(source, '    forward = LIFetchForward(rotating, update)',
                              '    forward = LIFetchForward(rotating, update)' + injection)
    if candidate:
        source = replace_once(source, '        if any(status) or row["invalid_topk"]:',
            '        if any(status) or row["invalid_topk"]:\n'
            '            from failure_report import save_failure\n'
            '            save_failure(output, forward, rotating, status)')
    marker = '        selected = topk[valid].contiguous().numpy().tobytes()'
    audit = '''        if measured and not any(status) and not int(bad.sum()):
            logical = update.logical_output.cpu().reshape(shape["batch"],4,2048)[valid]
            slots = update.slot_output.cpu().reshape(shape["batch"],4,2048)[valid].long()
            if bool(((slots < 0) | (slots >= snap.meta["output_stride"])).any()):
                raise RuntimeError("Output slot outside owner allocation")
            owners = inputs.row_owners.cpu().long()[:,None].expand(-1,4)[valid]
            payload = hbm.cpu()[owners[:,None]*snap.meta["output_stride"]+slots]
            order = logical.argsort(dim=1)
            torch.save(dict(logical=logical.gather(1,order),
                kv=payload.gather(1,order[:,:,None].expand(-1,-1,payload.shape[-1]))),
                output / f"selected-{iteration-args.warmup}.pt")
'''
    return replace_once(source, marker, audit + marker)



def validate_kernel_chain(values, candidate, iterations):
    """Reject incomplete profiles rather than treating a missing stage as zero."""
    patterns = [rf'v9_li_kernel_{stage}(?:_\d+)*' for stage in range(3)]
    patterns += ([r'v9_parallel_terminal(?:_\d+)*'] if candidate else
                 [r'v9_li_kernel_3(?:_\d+)*', r'v9_streaming_resolve_update(?:_\d+)*'])
    matched = []
    for pattern in patterns:
        names = [name for name in values if re.fullmatch(pattern, name)]
        if len(names) != 1:
            raise ValueError(f'Expected exactly one stage {pattern}; found {names}')
        matched.extend(names)
    if set(matched) != set(values):
        raise ValueError(f'Unexpected profiler kernels: {set(values) - set(matched)}')
    if iterations <= 0 or any(len(values[name]) != iterations for name in matched):
        raise ValueError('Profiler sample count differs from measured iterations')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--kvaf-root', required=True, type=Path)
    p.add_argument('--kvaf-server', default='127.0.0.1:28885')
    p.add_argument('--device', type=int, default=4)
    p.add_argument('--windows', type=int, choices=(4,8), default=4)
    p.add_argument('--prefetch-tokens', type=int, default=1024)
    p.add_argument('--prefetch-rank-start', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--timeout-seconds', type=int, default=1200)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        p.error('iterations must be positive and warmup nonnegative')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source = (HERE / 'snapshot_runner_base.py').read_text()
    manifest = json.loads((args.snapshot / 'manifest.json').read_text())
    groups = manifest.get('groups', [])
    if len(groups) != 1 or groups[0]['shape']['splitQ'] != 1:
        raise ValueError('This experiment requires one sequence-split snapshot group')
    print('A: current checkout original tail. B: real URMA four-core terminal; no old resolver. Experimental, not performance-qualified.', flush=True)
    results = {}
    configurations = {}
    for name in ('A', 'B'):
        folder = args.output / name
        folder.mkdir()
        runner = folder / 'runner.py'
        runner.write_text(runner_source(source, name == 'B', folder))
        command = [sys.executable, str(runner), str(args.snapshot.resolve()),
                   '--case', 'graph-captured', '--profile-only', '--output', str(folder),
                   '--kvip-root', str(ROOT), '--kvaf-root', str(args.kvaf_root.resolve()),
                   '--kvaf-server', args.kvaf_server, '--device', str(args.device),
                   '--windows', str(args.windows), '--prefetch-tokens', str(args.prefetch_tokens),
                   '--prefetch-rank-start', str(args.prefetch_rank_start),
                   '--warmup', str(args.warmup), '--iterations', str(args.iterations)]
        print(f'{name}: {folder / "worker.log"}', flush=True)
        with (folder / 'worker.log').open('w') as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout_seconds).returncode
            except subprocess.TimeoutExpired:
                code = 124
        if code:
            print((folder / 'worker.log').read_text()[-10000:])
            raise SystemExit(f'{name} failed ({code}); no timing acceptance; see {folder}')
        configurations[name] = json.loads((folder / 'effective_config.json').read_text())
        if name == 'B' and configurations['A'] != configurations['B']:
            raise RuntimeError('A/B effective geometry or prefetch configuration differs')
        values = json.loads((folder / 'kernel_times.json').read_text())
        validate_kernel_chain(values, name == 'B', args.iterations)
        means = {k: sum(v)/len(v) for k,v in values.items() if v}
        terminal = [k for k in means if k.startswith('v9_parallel_terminal')]
        legacy = [k for k in means if k.startswith('v9_streaming_resolve_update') or
                  re.fullmatch(r'v9_li_kernel_3(?:_\d+)*', k)]
        if (name == 'B' and (len(terminal) != 1 or legacy)) or (name == 'A' and not legacy):
            raise RuntimeError(f'Unexpected kernel chain: {means}')
        if any(len(v) != args.iterations for v in values.values()):
            raise RuntimeError('Profiler sample count differs from measured iterations')
        results[name] = dict(kernels_us=means, total_kernel_us=sum(means.values()),
                             tail_us=sum(means[k] for k in terminal + legacy))
    import torch
    checks = []
    for i in range(args.iterations):
        a = torch.load(args.output / 'A' / f'selected-{i}.pt', weights_only=True)
        b = torch.load(args.output / 'B' / f'selected-{i}.pt', weights_only=True)
        checks.append({k: torch.equal(a[k], b[k]) for k in ('logical', 'kv')})
    results['correctness'] = checks
    results['effective_config'] = configurations['A']
    results['checkout'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    results['source_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    results['prefix_delta_us'] = {}
    for stage in range(3):
        pattern = rf'v9_li_kernel_{stage}(?:_\d+)*'
        a = sum(v for k,v in results['A']['kernels_us'].items() if re.fullmatch(pattern, k))
        b = sum(v for k,v in results['B']['kernels_us'].items() if re.fullmatch(pattern, k))
        results['prefix_delta_us'][str(stage)] = dict(A=a, B=b, delta=b-a)
    results['scope'] = 'snapshot operator graph; real URMA; kernel sums exclude graph gaps; not vLLM throughput'
    (args.output / 'ab_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print('Path | tail us | all kernels us')
    for name in ('A','B'):
        print(f"{name} | {results[name]['tail_us']:.2f} | {results[name]['total_kernel_us']:.2f}")
        print(json.dumps(results[name]['kernels_us'], indent=2))
    print('Unchanged prefix | A us | B us | delta us')
    for stage, row in results['prefix_delta_us'].items():
        print(f"kernel {stage} | {row['A']:.2f} | {row['B']:.2f} | {row['delta']:+.2f}")
    print('Every measured replay logical/KV equality:', checks)
    print('Summary:', args.output / 'ab_summary.json')
    if not all(all(c.values()) for c in checks):
        raise SystemExit('AB correctness FAIL; timings not accepted')


if __name__ == '__main__':
    main()
