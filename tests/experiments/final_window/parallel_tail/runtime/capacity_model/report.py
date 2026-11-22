"""Reject incomplete native-model runs before reporting simulator timings."""
import argparse
import json
from pathlib import Path
import re


AUDITS = ('wqe', 'topk', 'counts', 'output', 'index', 'errors',
          'transport_status')


def parse_run(text):
    if re.findall(r'^verified_launch=(\d+)$', text, re.M) != ['1', '2']:
        raise ValueError('Expected two verified launches')
    if re.findall(r'^mode=9 posted=\d+ bad=(\d+)$', text, re.M) != ['0', '0']:
        raise ValueError('Missing or failed complete-state audit')
    for audit in AUDITS:
        if re.findall(rf'^audit_{audit}=(\d+)$', text, re.M) != ['0', '0']:
            raise ValueError(f'Missing or failed {audit} audit')
    rows = re.findall(r'^core(\d+)\.veccore(\d+)\s+([\d.]+)\s+[\d.]+\s*$',
                      text, re.M)
    if {(a, b) for a, b, _ in rows} != {('0', '0'), ('0', '1'),
                                                     ('1', '0'), ('1', '1')} or len(rows) != 4:
        raise ValueError('Expected one complete four-core timing table')
    if 'Op profiling finish.' not in text:
        raise ValueError('Profiler has not finished')
    times = {f'{a}.{b}': float(value) for a, b, value in rows}
    return {'verified_launches': 2, 'audit_bad': 0,
            'per_core_us': times, 'duration_us': max(times.values()),
            'hardware_measurement': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()
    result = []
    for path in args.logs:
        try:
            result.append({'log': str(path), **parse_run(path.read_text())})
        except (OSError, ValueError) as error:
            result.append({'log': str(path), 'unqualified': str(error)})
    print(json.dumps(result, indent=2))
    return int(any('unqualified' in row for row in result))


if __name__ == '__main__':
    raise SystemExit(main())
