"""Print compact results from the isolated terminal URMA comparison."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    summary = args.run / 'ab_summary.json'
    if not summary.exists():
        for path in sorted(args.run.glob('*/worker.log')):
            print(f'--- {path} (last 30 lines) ---')
            print('\n'.join(path.read_text(errors='replace').splitlines()[-30:]))
        raise SystemExit('No completed AB summary')
    data = json.loads(summary.read_text())
    print('Source:', summary)
    print('Commit:', data.get('checkout'))
    print('Config:', json.dumps(data['effective_config'], ensure_ascii=False))
    print('Scope:', data.get('scope'))
    print('Path | Tail us | All kernels us')
    for name in ('A', 'B'):
        item = data[name]
        print(f"{name} | {item['tail_us']:.2f} | {item['total_kernel_us']:.2f}")
        for kernel, value in item['kernels_us'].items():
            print(f'  {kernel}: {value:.2f} us')
    print('B-A:', {key: round(data['B'][key]-data['A'][key], 3)
                   for key in ('tail_us', 'total_kernel_us')})
    print('Prefix deltas:', data['prefix_delta_us'])
    print('Per-replay equality:', data['correctness'])
    if not all(all(row.values()) for row in data['correctness']):
        raise SystemExit('AB correctness FAIL')


if __name__ == '__main__':
    main()
