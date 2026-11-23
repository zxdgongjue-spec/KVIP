"""Read the two warmed, counterbalanced measurements of each production variant."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

LINE = re.compile(r'PASS row=(\d+) pass=(\d+) variant=(\d+) count=(\d+) ticks=(\d+)')
ORDER = (0, 1, 0, 1, 1, 0)


def summarize(log, fixture, handoff=False):
    cases = json.loads(fixture.read_text())
    samples = {}
    pattern = re.compile(LINE.pattern.replace('count=', 'rows=')) if handoff else LINE
    for match in pattern.finditer(log.read_text()):
        row, run, version, count, ticks = map(int, match.groups())
        if (run, row) in samples or ORDER[run] != version or count != (len(cases[row]['rows']) if handoff else cases[row]['posted']) or ticks <= 0:
            raise ValueError(f'invalid sample {log}, row={row}, run={run}')
        samples[run, row] = ticks
    if len(samples) != len(ORDER) * len(cases):
        raise ValueError(f'incomplete results in {log}')
    by_variant = [[mean([samples[run, row] for run in range(2, 6) if ORDER[run] == v])
                   for row in range(len(cases))] for v in (0, 1)]
    old, new = map(mean, by_variant)
    return dict(group=cases[0]['group'], measured_lists=len(cases), old_ticks=old, new_ticks=new,
                reduction_pct=100*(old-new)/old, ticks_by_window=by_variant)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('logs', type=Path)
    args.add_argument('fixtures', type=Path)
    args.add_argument('--handoff', action='store_true')
    cfg = args.parse_args()
    for group in (10, 4, 17, 7, 2):
        print(json.dumps(summarize(cfg.logs/f'model-g{group}.log', cfg.fixtures/f'g{group}.json', cfg.handoff)))
