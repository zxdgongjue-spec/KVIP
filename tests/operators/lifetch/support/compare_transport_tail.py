"""Compare saved optimized URMA/UBmem full traces; does not execute a device."""
import argparse
import json
import re
import statistics
from pathlib import Path


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


def summarize(root, transport):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urma', type=Path, required=True)
    parser.add_argument('--ubmem', type=Path, required=True)
    args = parser.parse_args()
    urma, ubmem = summarize(args.urma, 'urma'), summarize(args.ubmem, 'ubmem')
    print('BS | URMA fused us | UBmem fused us | UBmem minus URMA us')
    for batch in sorted(urma.keys() & ubmem.keys()):
        print(f'{batch} | {urma[batch]:.3f} | {ubmem[batch]:.3f} | {ubmem[batch]-urma[batch]:+.3f}')
    print('Separate profiled runs; compare counts too. Kernel span includes concurrent work and waits.')


if __name__ == '__main__':
    main()
