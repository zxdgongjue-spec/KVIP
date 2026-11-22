"""Generate a complete parallel-overlap candidate with the existing oracle."""
import argparse
from pathlib import Path

from prepare import generate
from overlap_codegen import adapt_parallel_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pattern', choices=('isolated', 'chain', 'cycle', 'cross_cycle'), default='cycle')
    args = parser.parse_args()
    generate(args.output.resolve(), args.pattern)
    adapt_parallel_overlap(args.output.resolve())
