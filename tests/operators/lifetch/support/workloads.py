"""Public test configuration, independent of ACL/PyTorch and device imports."""
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BATCHES = (1, 4, 8, 16)
CHECK_SUITES = ('init', 'basic', 'boundary', 'stateful', 'contract', 'real')
STAGES = ('init', 'li', 'resolve', 'update', 'fetch', 'finalize')


def csv_values(value):
    values = value.split(',')
    if any(not item or item.strip() != item for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('expected distinct comma-separated values')
    return tuple(values)


def miss_values(value):
    try:
        values = tuple(float(item) for item in csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError('miss must be comma-separated percentages') from error
    if any(not math.isfinite(item) or not 0 <= item <= 100 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('miss percentages must be distinct finite values in 0..100')
    return values[0] if len(values) == 1 else values


def batch_values(value):
    try:
        batches = tuple(int(item) for item in csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError('BS must be comma-separated integers') from error
    if any(not 1 <= batch <= 16 for batch in batches):
        raise argparse.ArgumentTypeError('LIFetch supports BS1..16')
    return batches


def round_range(value):
    try:
        start, end = map(int, value.split(':'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('round range must be START:END') from error
    if not 0 <= start <= end:
        raise argparse.ArgumentTypeError('round range must be increasing and nonnegative')
    return start, end


def configuration(argv=None):
    parser = argparse.ArgumentParser(description='Unified LIFetch tests')
    commands = parser.add_subparsers(dest='mode', required=True)
    for mode in ('all', 'check', 'bench', 'diagnose'):
        command = commands.add_parser(mode)
        command.add_argument('--transport', choices=('urma', 'ubmem'), default='urma')
        command.add_argument('--device', type=int, default=0)
        command.add_argument('--kvaf-root', type=Path, default=ROOT / 'third_party/KVAF')
        command.add_argument('--kvaf-server')
        command.add_argument('--output', type=Path, default=ROOT / 'tests/results/lifetch')
        command.add_argument('--timeout-seconds', type=int, default=300)
        if mode == 'check':
            command.add_argument('--suite', type=csv_values, default=('all',))
            command.add_argument('--cases', type=csv_values)
            command.add_argument('--snapshot', type=Path)
        if mode in ('bench', 'diagnose'):
            command.add_argument('--input', choices=('synthetic', 'real'), default='synthetic')
            # None distinguishes absent arguments from explicitly supplied defaults on real input.
            command.add_argument('--miss-percent', type=miss_values if mode == 'bench' else float)
            command.add_argument('--queries', type=int)
            command.add_argument('--batches', type=batch_values)
            command.add_argument('--seed', type=int)
            command.add_argument('--history', type=int)
            command.add_argument('--resident', type=int)
            command.add_argument('--index-heads', type=int)
            command.add_argument('--trace-root', type=Path)
            command.add_argument('--group', type=int)
            command.add_argument('--round-range', type=round_range)
            command.add_argument('--prefetch-tokens', type=int, default=512)
            command.add_argument('--intermediate-fetches', type=int, default=8)
            command.add_argument('--warmup', type=int, default=4)
            command.add_argument('--iterations', type=int, default=20)
        if mode == 'diagnose':
            command.add_argument('--level', type=int, choices=(1, 2, 3), default=1)
            command.add_argument('--stages', type=csv_values, default=('all',))
    args = parser.parse_args(argv)
    if args.device < 0 or args.timeout_seconds <= 0:
        parser.error('device must be nonnegative and timeout positive')
    if args.mode == 'check':
        if 'all' in args.suite and len(args.suite) != 1:
            parser.error('suite all cannot be combined with individual suites')
        if set(args.suite) - {*CHECK_SUITES, 'all'}:
            parser.error('unknown check suite')
        if 'real' in args.suite and args.snapshot is None:
            parser.error('suite real requires --snapshot')
        if args.suite == ('all',):
            args.suite = CHECK_SUITES if args.snapshot else CHECK_SUITES[:-1]
    if args.mode in ('bench', 'diagnose'):
        synthetic = dict(miss_percent=10., queries=4, batches=DEFAULT_BATCHES, seed=43,
                         history=131072, resident=16384, index_heads=32)
        real = ('trace_root', 'group', 'round_range')
        if args.input == 'real':
            if any(getattr(args, key) is not None for key in synthetic):
                parser.error('real input geometry comes from trace; synthetic options are not allowed')
            if any(getattr(args, key) is None for key in real):
                parser.error('real input requires --trace-root, --group, --round-range')
            if args.group < 0:
                parser.error('group must be nonnegative')
        else:
            if any(getattr(args, key) is not None for key in real):
                parser.error('trace options require --input real')
            for key, default in synthetic.items():
                if getattr(args, key) is None:
                    setattr(args, key, default)
            misses = args.miss_percent if isinstance(args.miss_percent, tuple) else (args.miss_percent,)
            if any(not math.isfinite(miss) or not 0 <= miss <= 100 for miss in misses) or args.queries != 4:
                parser.error('current full-chain fixture supports Q4 and miss in 0..100')
            if (args.history % 256 or args.resident % 256 or args.resident < 2048
                    or args.history <= args.resident + 4 or args.index_heads not in (32, 64)
                    or args.seed < 0):
                parser.error('invalid synthetic geometry or seed')
        if (args.warmup < 0 or args.iterations <= 0 or not 1 <= args.intermediate_fetches <= 32
                or not 0 <= args.prefetch_tokens <= 2048
                or args.prefetch_tokens % args.intermediate_fetches):
            parser.error('invalid timing or prefetch configuration')
    if args.mode == 'diagnose':
        if 'all' in args.stages and len(args.stages) != 1:
            parser.error('stages all cannot be combined with individual stages')
        if set(args.stages) - {*STAGES, 'all'}:
            parser.error('unknown diagnostic stage')
        if args.stages == ('all',):
            args.stages = STAGES
    return args
