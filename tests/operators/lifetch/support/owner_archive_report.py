"""Read the four-group level-1 owner-pack run and its original measurements."""
import argparse
import hashlib
import json
import re
import sys
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median

from peer_delta import read
from query_owner_report import SHA256, compare_layout_receipts, load

GROUPS = (7, 10, 17, 4)
ARM = 'owner-pack'
NATIVE = '36d92c3e663e5164af54e7761187f3204e76fe05'
READER = 'b9122782a566e1b5d40c366ccf8c03e50dc2d131'
# Archived host16 NPU4 run fused.2oPXZw, 6bca602a baseline (4 warmups, 20 calls).
# Event mean, device-span median, LI mean, resolve_update mean; no stage medians printed.
HOST16_FIXED = {7: (553.21, 539.13, 285.86, 261.36), 10: (356.87, 347.72, 265.23, 82.72),
                17: (387.34, 371.94, 266.85, 114.32), 4: (359.82, 352.77, 267.10, 85.89)}
GRAPH_LINE = re.compile(r'^G(\d+) owner-pack graph_mean/P50_us=(\d+\.\d{2}/\d+\.\d{2}) '
                        r'tail_mean/P50_us=(\d+\.\d{2}/\d+\.\d{2})$', re.M)
READER_PATH = re.compile(r'/reader-([0-9a-f]{40})/')


def pair(values):
    return f'{mean(values):.2f}/{median(values):.2f}'


def read_summary(text):
    entries = GRAPH_LINE.findall(text)
    if len(entries) != len(GROUPS) or {int(g) for g, _, _ in entries} != set(GROUPS):
        raise ValueError('expected exactly one owner-pack timing for each of G7/G10/G17/G4')
    return {int(g): (graph, tail) for g, graph, tail in entries}


def headers(root):
    for name, expected in [('physical_npu', '4'), ('candidate', NATIVE), (ARM, NATIVE)]:
        if (root / f'{name}.txt').read_text().strip() != expected:
            raise ValueError(f'{name}.txt must match {expected}')
    return read_summary((root / 'summary.txt').read_text())


def reader_check(root, case):
    # This runner writes no harness.txt for candidate-only runs. The NPU profiler
    # records its absolute output path relative to the detached reader checkout.
    marker = root / 'harness.txt'
    if marker.exists() and marker.read_text().strip() != READER:
        raise ValueError(f'{marker}: different harness revision')
    worker = case / 'worker.log'
    readers = set()
    if worker.is_file():
        with worker.open(errors='replace') as log:
            for line in log:
                if 'reader-' in line:
                    readers.update(READER_PATH.findall(line))
    if readers and readers != {READER}:
        raise ValueError(f'{worker}: reader path identifies {sorted(readers)}')
    if readers or marker.is_file():
        return 'archive'
    return 'unrecorded'


def inspect(root, g, summary):
    data = load(root, g, ARM, row_dependencies=True)
    inputs, digests, thp, graph_mean, graph_p50, binary = data[:6]
    if not SHA256.fullmatch(binary) or len(set(digests)) != 1 or not SHA256.fullmatch(digests[0]):
        raise ValueError(f'G{g}: invalid library hash or differing final TopK sets')
    compare_layout_receipts({ARM: data[11]}, ARM)
    if summary[g] != (f'{graph_mean:.2f}/{graph_p50:.2f}', f'{data[8]:.2f}/{data[10]:.2f}'):
        raise ValueError(f'G{g}: stored summary disagrees with measured graph or tail')
    path, = (root / f'g{g}' / ARM / 'results').glob('**/fused/path_result.json')
    reader_source = reader_check(root, path.parents[1])
    return data, path, reader_source


def timing_rows(samples):
    stages = [[] for _ in range(4)]
    li, tail, spans = [], [], []
    for i, sample in enumerate(samples):
        row = sample['accounting']
        bounds = row['intervals_us']
        if set(bounds) != {'li', 'resolve_update'} or len(bounds['li']) != 4 or len(bounds['resolve_update']) != 1:
            raise ValueError(f'call {i}: expected four LI dispatches then one tail')
        intervals = sorted(bounds['li']) + bounds['resolve_update']
        for j, (start, end) in enumerate(intervals):
            if not (isfinite(start) and isfinite(end) and end > start) or (j and start < intervals[j-1][1] - 1e-5):
                raise ValueError(f'call {i}: overlapping or invalid kernel intervals')
        durations = [end-start for start, end in intervals]
        stages_us = row['stages_us']
        if (not isclose(sum(durations[:4]), stages_us['li'], abs_tol=1e-5) or
                not isclose(durations[4], stages_us['resolve_update'], abs_tol=1e-5)):
            raise ValueError(f'call {i}: kernel intervals disagree with stage durations')
        for stage, value in zip(stages, durations[:4]):
            stage.append(value)
        li.append(stages_us['li'])
        tail.append(stages_us['resolve_update'])
        spans.append(sample['us'])
    return li, tail, spans, stages


def emit(g, data, path, reader_source):
    samples = read(path.parent / 'diagnose/round-1/full_attribution.json')['samples']
    li, tail, spans, stages = timing_rows(samples)
    inputs, _, thp, graph, p50, binary = data[:6]
    if not isclose(mean(li), data[13], abs_tol=1e-5):
        raise ValueError(f'G{g}: LI intervals disagree with path_result stage mean')
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cfg = read(path.parents[1] / 'config.json')
    print(f"G{g} input_manifest_sha256={fingerprint} trace_root={cfg['trace_root']} "
          f"reader_provenance={reader_source} KVAF={cfg['kvaf_root']} binary_sha256={binary} ub_device={thp.get('ub_device')} "
          f"numa_pages={thp.get('numa_pages_overlapping_vmas')} thp_min_pct={thp.get('coverage_min_pct')}")
    print(f'G{g} CHECK TopK=PASS calls=20/20 KV=PASS ordered_Index=PASS rows=320/320 '
          'Q_per_call=64 owners/queues=64/64 windows_per_request=4')
    print(f'G{g} graph_event_mean/span_P50_us={graph:.2f}/{p50:.2f} '
          f'LI_mean/P50_us={pair(li)} tail_mean/P50_us={pair(tail)}')
    print(f'G{g} LI_dispatch_mean/P50_us ' + ' '.join(
        f'{i}.{name}={pair(values)}' for i, (name, values) in enumerate(
            zip(('hot', 'resident_threshold', 'cold_and_prefetch', 'final_topk'), stages), 1)))
    gaps = [s - l - t for s, l, t in zip(spans, li, tail)]
    print(f'G{g} span_interkernel_gap_mean/P50_us={pair(gaps)} '
          f'event_mean_minus_span_mean_us={graph-mean(spans):+.2f} '
          f'event_mean_minus_LI_tail_mean_us={graph-mean(li)-mean(tail):+.2f}')
    algorithms = [s['algorithm'] for s in samples]
    totals = [a['totals'] for a in algorithms]
    totals_text = ' '.join(f'{k}={mean(t[k] for t in totals):.2f}'
                           for k in ('selected_unique', 'initial_cold', 'prefetch', 'unused_prefetch', 'final'))
    print(f'G{g} tokens_per_call_mean {totals_text} '
          f"cold_miss_pct={100*sum(t['initial_cold'] for t in totals)/sum(t['selected_unique'] for t in totals):.2f}")
    for q in range(4):
        entries = [r['queries'][q] for a in algorithms for r in a['requests']]
        if any(entry['query'] != q+1 for entry in entries):
            raise ValueError(f'G{g} Q{q+1}: reordered query attribution')
        posted = [r['final_posted'] for r in entries]
        safe = [r['final_safe'] for r in entries]
        print(f'G{g} Q{q+1} per_request_mean final={mean(posted):.2f} safe={mean(safe):.2f} '
              f'conflict={mean(posted)-mean(safe):.2f} empty_final={posted.count(0)}/{len(entries)} '
              f"crossQ_TopK_duplicate={mean(r['cross_q_duplicate'] for r in entries):.2f}")
    bg, bp, bl, bt = HOST16_FIXED[g]
    print(f'G{g} host20_NPU4_minus_host16_NPU4_6bca602a_us '
          f'graph_mean/P50={graph-bg:+.2f}/{p50-bp:+.2f} LI_mean={mean(li)-bl:+.2f} '
          f'tail_mean={mean(tail)-bt:+.2f}')


def measured_log_roots(work):
    roots = {}
    area = (work / 'peer-baseline').resolve()
    for path in work.glob('host20-owner-pack.*'):
        if not path.is_file():
            continue
        text = path.read_text(errors='replace')
        matches = re.findall(r'^REPLAY_COMPLETE RESULT_ROOT=(.+)$', text, re.M)
        if len(matches) == 1:
            root = Path(matches[0].strip()).resolve()
            if root.parent == area:
                roots.setdefault(root, []).append(path)
    return roots


def linked_host_log(logs, summary):
    for log in logs:
        try:
            if read_summary(log.read_text(errors='replace')) == summary:
                return True
        except ValueError:
            continue
    if logs:
        raise ValueError('completed host20 log and archive summary disagree')
    return False


def find(work, targets):
    logs = measured_log_roots(work)
    candidates = sorted((p for p in (work / 'peer-baseline').glob('fused.*') if (p / 'owner-pack.txt').is_file()),
                        key=lambda p: (p / 'summary.txt').stat().st_mtime_ns if (p/'summary.txt').is_file() else 0,
                        reverse=True)
    errors = []
    for root in candidates:
        try:
            summary = headers(root)
            if targets and {g: v[0] for g, v in summary.items()} != targets:
                raise ValueError('four-group graph mean/P50 differs from --graph')
            linked_host_log(logs.get(root.resolve(), ()), summary)
            for g in GROUPS:
                inspect(root, g, summary)
            return root.resolve()
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f'{root.name}: {exc}')
    raise ValueError('no complete matching owner-pack archive: ' + (' | '.join(errors[:8]) or str(work)))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    finder = actions.add_parser('find')
    finder.add_argument('work', type=Path)
    finder.add_argument('--graph', nargs=4, metavar='GROUP=MEAN/P50')
    actions.add_parser('report').add_argument('reference', type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    if args.action == 'find':
        targets = {}
        for value in args.graph or ():
            m = re.fullmatch(r'(7|10|17|4)=(\d+\.\d{2}/\d+\.\d{2})', value)
            if not m or int(m[1]) in targets:
                raise ValueError(f'invalid or duplicate --graph item: {value}')
            targets[int(m[1])] = m[2]
        print(find(args.work, targets))
        return
    root = args.reference.resolve()
    summary = headers(root)
    linked_host_log(measured_log_roots(root.parents[1]).get(root, ()), summary)
    checked = [(g, *inspect(root, g, summary)) for g in GROUPS]
    print(f'ARCHIVE={root} native={NATIVE} expected_reader={READER} physical_NPU=4 logical_NPU=0 '
          'BS16 Q4 token_bytes=656 fixed_route prefetch_budget=256/1024 warmup=4 measured=20')
    for g, data, path, reader_source in checked:
        emit(g, data, path, reader_source)


if __name__ == '__main__':
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        sys.exit(f'ARCHIVE_ERROR: {exc}')
