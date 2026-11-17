#!/usr/bin/env python3
"""Inspect schema-4 captures and export one replayable round (CPU/NumPy only)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/operators/lifetch/support'))
from workloads import (ReplayPagePlan, ReplaySequence, checked_path, load_score_capture,
                       select_series)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def inventory(root):
    """Read metadata only; do not decompress thousands of full score arrays."""
    samples = {}
    paths = sorted(root.glob('dp[0-9]*/series.json'))
    if not paths:
        roots = sorted({path.parent.parent for path in root.rglob('series.json')
                        if path.parent.name.startswith('dp') and path.parent.name[2:].isdigit()})
        if not roots:
            raise ValueError(f'No dp*/series.json found under {root}')
        for candidate in roots:
            print(f'\nTRACE_ROOT={candidate}')
            inventory(candidate)
        return
    for path in paths:
        series = json.loads(path.read_text())
        for entry in series['rounds']:
            for group in entry['groups']:
                manifest = json.loads(checked_path(path.parent, group['manifest']).read_text())
                if not manifest.get('real_inputs_saved'):
                    continue
                key = entry['round_index'], group['group']
                samples.setdefault(key, []).append((series['dp_rank'], manifest['geometry']['batch'],
                                                    manifest['geometry']['token_bytes']))
    print('Real Q/Index samples (metadata only; use --round/--group for integrity and miss checks)')
    print('Round | Group | DP ranks | Total BS | Source bytes/token')
    for (number, group), rows in sorted(samples.items()):
        print(f'{number:5} | {group:5} | {",".join(str(x[0]) for x in sorted(rows))} | '
              f'{sum(x[1] for x in rows):8} | {sorted(set(x[2] for x in rows))}')
    if not samples:
        print('No real Q/Index samples found; score-only captures cannot execute LI.')


def request_counts(capture):
    a, geometry = capture.arrays, capture.manifest['geometry']
    rows = []
    resident = geometry['resident']
    for row in range(geometry['batch']):
        selected = np.unique(a['post.topk'][row])
        logical = a['pre.logical_ids'][row, selected]
        hot = selected < resident
        hot[hot] = a['pre.hot_valid'][row, selected[hot]] != 0
        begin, end = map(int, a['pre.tail_bounds'][row])
        if not 0 <= begin <= end <= geometry['history']:
            raise ValueError('Invalid ABC tail bounds')
        abc = (logical >= begin) & (logical < end)
        cold = ~(hot | abc)
        missing_ddr = cold & (a['pre.address_valid'][row, selected] == 0)
        candidate_hot = hot & np.isin(selected, a['pre.lru_slots'][row])
        rows.append(dict(dp=capture.manifest['metadata']['dp_rank'], request=row,
                         demand=len(selected), hot=int(hot.sum()), abc_only=int((abc & ~hot).sum()),
                         cold=int(cold.sum()), candidate_hot=int(candidate_hot.sum()),
                         cold_without_ddr=int(missing_ddr.sum())))
    return rows


def display(rows):
    print('Captured Q4 union; cold = demand absent from valid hot AND ABC. Counts are not Get counts.')
    print('DP | Request | Demand | Hot | ABC-only | Cold | Cold miss % | Candidate hot | Cold w/o DDR')
    for row in rows:
        pct = 100 * row['cold'] / row['demand'] if row['demand'] else 0
        print(f"{row['dp']:2} | {row['request']:7} | {row['demand']:6} | {row['hot']:4} | "
              f"{row['abc_only']:8} | {row['cold']:4} | {pct:11.2f} | {row['candidate_hot']:13} | "
              f"{row['cold_without_ddr']:12}")
    total = {key: sum(row[key] for row in rows) for key in
             ('demand', 'hot', 'abc_only', 'cold', 'candidate_hot', 'cold_without_ddr')}
    total['cold_miss_percent'] = 100 * total['cold'] / total['demand'] if total['demand'] else None
    print(f"TOTAL BS={len(rows)} demand={total['demand']} cold={total['cold']} "
          f"cold_miss={total['cold_miss_percent']:.2f}% candidate_hot={total['candidate_hot']}")
    return total


def run(args):
    root = args.root.resolve()
    if args.group is None or args.round is None:
        if args.group is not None or args.round is not None or args.output is not None:
            raise ValueError('--round and --group must be supplied together')
        inventory(root)
        return
    references = select_series(root, args.group, (args.round, args.round))
    if args.dp_ranks is not None:
        available = {ref.dp_rank for ref in references}
        if len(set(args.dp_ranks)) != len(args.dp_ranks) or not set(args.dp_ranks) <= available:
            raise ValueError('--dp-ranks contains duplicates or unavailable ranks')
        references = [ref for ref in references if ref.dp_rank in args.dp_ranks]
    output = args.output.resolve() if args.output else None
    if output and output.exists():
        raise ValueError(f'Output already exists: {output}; use a new directory')
    if output:
        output.mkdir(parents=True)
        write_json(output / 'conversion.json', dict(status='RUNNING'))
    rows, sources, widths = [], [], set()
    try:
        for ref in references:
            capture = load_score_capture(ref.manifest_path, require_real=output is not None)
            rows.extend(request_counts(capture))
            sources.append(str(ref.manifest_path))
            widths.add(capture.manifest['geometry']['token_bytes'])
            if not output:
                continue
            manifest = copy.deepcopy(capture.manifest)
            width = manifest['geometry']['token_bytes']
            if width not in (656, 2624):
                raise ValueError(f'Unsupported source token width: {width}')
            if args.layer_index is not None:
                if not 0 <= args.layer_index < width // 656:
                    raise ValueError('--layer-index outside captured layer group')
                manifest['geometry']['token_bytes'] = 656
            target = output / f'dp{ref.dp_rank}' / 'round0001' / f'group{args.group:03d}'
            target.mkdir(parents=True)
            archive = checked_path(capture.path.parent, manifest['archive']['file'])
            shutil.copyfile(archive, target / 'scores.npz')
            manifest['archive']['file'] = 'scores.npz'
            # Arrays, scores, owners, slots and their checksum are unchanged.
            # Only synthetic KV width is projected; captured transfer records
            # remain source evidence, not projected replay byte measurements.
            manifest['replay_export'] = dict(source=str(capture.path), source_round=args.round,
                source_token_bytes=width, layer_index=args.layer_index,
                kv='synthetic C8; real Q/Index and captured residency',
                captured_transfer_records='unchanged; source group units')
            write_json(target / 'manifest.json', manifest)
            series = json.loads((root / f'dp{ref.dp_rank}' / 'series.json').read_text())
            entry = copy.deepcopy(series['rounds'][args.round - 1])
            entry['round_index'] = 1
            member = next(copy.deepcopy(g) for g in entry['groups'] if g['group'] == args.group)
            member['manifest'] = f'round0001/group{args.group:03d}/manifest.json'
            entry['groups'] = [member]
            series.update(rounds=[entry], completed_rounds=1, expected_rounds=1,
                          dp_ranks=[r.dp_rank for r in references])
            series['replay_export'] = dict(source=str(root), source_round=args.round,
                                          independent_snapshot=True)
            write_json(target.parents[1] / 'series.json', series)
        totals = display(rows)
        result = dict(status='ANALYZED', source=str(root), source_round=args.round,
                      group=args.group, source_token_bytes=sorted(widths),
                      layer_index=args.layer_index, requests=rows, totals=totals, sources=sources)
        if output:
            sequence = ReplaySequence(output, args.group, (1, 1))
            batch = next(iter(sequence))
            # Exercise the same assembly and page-alias plan as device replay.
            pages = ReplayPagePlan(sequence)
            if totals['cold_without_ddr']:
                raise ValueError('Selected cold tokens lack DDR addresses')
            result.update(status='COMPLETE', batch=batch.geometry['batch'],
                          replay_token_bytes=batch.geometry['token_bytes'], pages=pages.page_count,
                          validation='CPU integrity, captured TopK, replay assembly and page plan PASS')
            write_json(output / 'conversion.json', result)
            print(f"READY --input real --trace-root {output} --group {args.group} --round-range 1:1")
            print('Independent captured round; device correctness and performance still require NPU replay.')
    except Exception as error:
        if output:
            write_json(output / 'conversion.json', dict(status='FAILED', error=str(error), sources=sources))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Directory containing dp*/series.json')
    parser.add_argument('--round', type=int, help='Capture round, not scheduler round')
    parser.add_argument('--group', type=int)
    parser.add_argument('--dp-ranks', type=int, nargs='+', help='Default: all captured DPs')
    parser.add_argument('--output', type=Path, help='Export one verified round to a new replay directory')
    parser.add_argument('--layer-index', type=int, help='Project synthetic KV to one 656B layer; e.g. 0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
