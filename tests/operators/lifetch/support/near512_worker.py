# SPDX-License-Identifier: Apache-2.0
"""Time ten restored near-512K Q4 executions on B24 and verify each result."""
import argparse
from functools import partial
import hashlib
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

import numpy as np

from cap_pair_report import cap_times, fmt_times
from main_five_report import read
from cap_pair_timing import measure
from main_five_run import verify_assignments
from native_li_reference import fused_li_output

CALLS = 10
CAPACITY = 524544


def geometry_check(inputs):
    g = inputs.geometry
    if (g['batch'], g['heads'], g['history'], g['resident'], g['token_bytes']) != (16, 32, CAPACITY, 16384, 656):
        raise ValueError(f'Unexpected near-512K workload geometry: {g}')
    ends = np.asarray(inputs.query_ends)
    if (ends.shape != (16, 4) or np.any(np.diff(ends, axis=1) != 1)
            or ends[:, 0].min() < 510 * 1024 or ends[:, 0].max() > 512 * 1024):
        raise ValueError('All 16 requests must contain at least 510K effective tokens and four causal queries')
    for row, end in enumerate(ends[:, 0]):
        if not inputs.logical_address_valid[row, 262144:end].any():
            raise ValueError(f'Row {row} has no valid DDR addresses above 256K')


def audit_journal(state, inputs, layout, valid, prefetch, final):
    counts = state.rotating.change_count.cpu().numpy()
    sources = state.forward.events_src.cpu().numpy()
    destinations = state.forward.events_dst.cpu().numpy()
    expected = prefetch + final.sum(axis=1)
    if not np.array_equal(counts, expected) or np.any(counts < 0) or np.any(counts > sources.shape[1]):
        raise ValueError('Published admission counts differ from the final Index journal')
    high_prefetch = high_final = 0
    for row, count in enumerate(counts):
        src, dst = sources[row, :count], destinations[row, :count]
        start = state.input_layout[row]
        verify_assignments(start, layout[row], inputs.captured_hot_valid[row], valid[row], src, dst)
        logical = start[src]
        high_prefetch += int(np.count_nonzero(logical[:prefetch[row]] >= 262144))
        high_final += int(np.count_nonzero(logical[prefetch[row]:] >= 262144))
    return dict(ordered_rows=len(counts), prefetch_above_256k=high_prefetch, final_above_256k=high_final)


def activate_production(main):
    reader = Path(__file__).resolve().parent
    support = main / 'tests/operators/lifetch/support'
    # The archived decoder supports more capture formats. Runtime, parser,
    # validation and kernel attribution must come from the tested checkout.
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().parent == reader and (support / Path(filename).name).is_file():
            del sys.modules[name]
    sys.path[:0] = [str(support), str(main / 'python')]
    return support


def worker(args):
    from near512_input import make_input
    from near512_oracle import serial_reference, validate_native_sets

    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '4':
        raise ValueError('Expected physical NPU4 on this worker')
    main = args.main.resolve()
    actual = subprocess.check_output(['git', '-C', main, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.revision or subprocess.check_output(
            ['git', '-C', main, 'status', '--porcelain', '-uno'], text=True).strip():
        raise ValueError('The production checkout differs from the specified clean native revision')
    args.output.mkdir(parents=True, exist_ok=True)
    sequence, inputs, provenance = make_input(args.work, args.group, args.trace, args.output)
    if len(sequence.rounds) != 1 or next(iter(sequence)) is not inputs:
        raise ValueError('The reference and all replays must share the same frozen round')
    geometry_check(inputs)
    ends = inputs.query_ends
    support = activate_production(main)
    import execution as ex
    import kvaf
    import kvip
    import torch
    import torch_npu
    from reporting import atomic_json
    from workloads import ReplayPagePlan, configuration

    for module, root in ((ex, support), (kvip, main / 'python'), (kvaf, args.kvaf_root / 'build-npu/python')):
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Unexpected {module.__name__} at {module.__file__}')
    pages = ReplayPagePlan(sequence)
    print(f"INPUT kind={provenance['source_kind']} source={provenance['source_root']} "
          f"Q1_min/max={ends[:, 0].min()}/{ends[:, 0].max()} Q4_min/max={ends[:, 3].min()}/{ends[:, 3].max()} "
          f"physical_slots={CAPACITY} registered_KV_pages={pages.page_count}", flush=True)
    print('CONFIG physical_NPU=4 BS16 Q4 hot=16384 warmup=4 measured=10 individual_events=10 '
          'restored_before_every_call=1 diagnostic_calls=1', flush=True)
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    serial = serial_reference(inputs, torch, torch_npu, device)
    captured_oracle = inputs.validate_topk if provenance['kind'] == 'capture' else None
    verified = {}
    expected_sets = np.sort(serial, axis=-1)

    def native_score_check(selected):
        if captured_oracle is not None:
            captured_oracle(selected)
        validate_native_sets(inputs, selected, serial, torch, torch_npu, device, verified)

    native_score_check(serial)
    # The production replay checks scores through this input method, outside the
    # timed graph. The stitched case has no captured scores for its new Q/K pairs.
    inputs.validate_topk = native_score_check
    torch.npu.empty_cache()
    cfg = configuration([
        'diagnose', '--transport', 'urma', '--input', 'real', '--trace-root', str(provenance['source_root']),
        '--group', str(args.group), '--round-range', '1:1', '--device', '0', '--kvaf-root', str(args.kvaf_root),
        '--kvaf-server', args.kvaf_server, '--prefetch-rank-start', '1024', '--prefetch-tokens', '1024',
        '--intermediate-fetches', '4', '--warmup', '4', '--iterations', '1', '--level', '2',
        '--output', str(args.output)])
    current = ''
    traffic = {}
    outputs = {}
    diagnostic = {}
    libraries = set()

    class Recorder(ex.CorrectnessRecorder):
        def check(self, label, callback):
            nonlocal current
            current = label
            try:
                return super().check(label, callback)
            finally:
                current = ''

    class AuditedState(ex.FusedReplayState):
        def __init__(self, *arguments, **kwargs):
            super().__init__(*arguments, **kwargs)
            if (self.history != CAPACITY or self.workspace.block_history != CAPACITY
                    or not np.all(self.workspace.key_lengths.cpu().numpy() == CAPACITY)
                    or not np.array_equal(self.workspace.inputs.query_ends.cpu().numpy().reshape(16, 4), ends)
                    or self.forward.terminal is None):
                raise ValueError('The bound capacity, active length or production terminal differs from the case')
            libraries.add(Path(self.rotating.library._name).resolve())

        def validate_index(self, sample, output):
            layout, valid = super().validate_index(sample, output)
            if current.startswith(('measured/', 'detail/')):
                prefetch = self.rotating.rows[:, 9].cpu().numpy().astype(np.int64)
                final = self.rotating.final_posts[:, :, 0].cpu().numpy().astype(np.int64)
                if np.any(prefetch < 0) or np.any(final < 0):
                    raise ValueError('Negative device request counts')
                if current.startswith('measured/'):
                    if current in traffic:
                        raise ValueError(f'Duplicate call receipt {current}')
                    traffic[current] = dict(prefetch=int(prefetch.sum()), final=int(final.sum()))
                else:
                    diagnostic[current] = audit_journal(self, sample, layout, valid, prefetch, final)
            return layout, valid

        def validate(self, sample):
            super().validate(sample)
            if current.startswith('measured/'):
                outputs[current] = fused_li_output(self).copy()

    ex.CorrectnessRecorder = Recorder
    ex.FusedReplayState = AuditedState
    ex.measure_replay_batch = partial(measure, ex, CALLS, detail_calls=1)
    fused = args.output / 'fused'
    run = ex.run_path(cfg, fused, sequence, pages, fused=True)
    labels = [f'measured/input{i}/round{inputs.source_round}' for i in range(CALLS)]
    if (run['correctness'] != 'PASS' or run['failures'] or set(traffic) != set(labels) or set(outputs) != set(labels)
            or len(libraries) != 1 or len(diagnostic) != 1):
        raise ValueError(f"Incomplete measured or diagnostic checks: {run['failures']}")
    selected = [outputs[label] for label in labels]
    matches = np.all(np.sort(np.stack(selected), axis=-1) == expected_sets, axis=-1)
    score_check = dict(calls=CALLS, queries=int(matches.size), exact_set_queries=int(matches.sum()),
                       oracle='chunked_native_LI_and_native_cutoff_verification')
    times = cap_times(fused, run['all_rounds'], calls=CALLS, source_round=inputs.source_round)
    profile = read(fused / 'diagnose' / f'round-{inputs.source_round}' / 'full_attribution.json')
    spans = [sample['us'] for sample in profile['samples']]
    times['four_kernel_span'] = (mean(spans), median(spans))
    lib, = libraries
    with lib.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    np.savez_compressed(args.output / 'near512_topk_ids.npz', serial=serial, measured=np.stack(selected))
    values = [traffic[label] for label in labels]
    detail, = diagnostic.values()
    atomic_json(args.output / 'near512_result.json', dict(
        native=actual, physical_npu=4, source_kind=provenance['source_kind'], bound_history=CAPACITY,
        scores=score_check, measured_calls=CALLS, measured_complete_KV_Index_rows=16*CALLS, diagnostic=detail,
        measured_requests=values, mean_p50_us=times, timing_protocol='restored_single_call_graph_events',
        source_root=provenance['source_root'], native_library=str(lib), library_sha256=sha))
    print(f"CHECK TopK_score_vs_native=PASS queries={score_check['queries']}/640 "
          f"KV_and_Index=PASS rows=160/160 ordered_Index_diagnostic={detail['ordered_rows']}/16 private_CQ=PASS")
    print('TIMING ' + fmt_times(times))
    print(f"TRAFFIC prefetch/final_tokens_per_call_mean={mean(v['prefetch'] for v in values):.2f}/"
          f"{mean(v['final'] for v in values):.2f} "
          f"diagnostic_events_at_or_above_256K_prefetch/final={detail['prefetch_above_256k']}/{detail['final_above_256k']}")
    print(f'REPLAY_COMPLETE measured_calls={CALLS} result={args.output / "near512_result.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'work', 'kvaf-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--kvaf-server', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--group', type=int, default=4)
    worker(parser.parse_args())
