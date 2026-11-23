"""Replay one mainline four-core terminal case; all additional checks are outside timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np


REVISION = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
CALLS = 20
BATCH = 16


def verify_assignments(start, layout, before_valid, valid, sources, destinations):
    """Complete the admission permutation in reservation order, including displaced rows."""
    src = np.asarray(sources, dtype=np.int64)
    dst = np.asarray(destinations, dtype=np.int64)
    if (src.shape != dst.shape or src.ndim != 1 or np.any(src < 0) or
            np.any(dst < 0) or np.any(src >= len(start)) or np.any(dst >= len(valid)) or
            len(np.unique(src)) != len(src) or len(np.unique(dst)) != len(dst)):
        raise ValueError('Duplicate or out-of-range terminal Index assignments')
    released = src[~np.isin(src, dst, assume_unique=True)]
    displaced = dst[~np.isin(dst, src, assume_unique=True)]
    expected = start.copy()
    expected[dst] = start[src]
    expected[released] = start[displaced]
    expected_valid = before_valid.copy()
    expected_valid[released[released < len(valid)]] = 0
    expected_valid[dst] = 1
    if not np.array_equal(expected, layout) or not np.array_equal(expected_valid, valid):
        raise ValueError('Terminal Index event replay differs from committed logical IDs or hot validity')


def case_path(args):
    return args.output / 'results/mainline/cases/diagnose'


def worker(args):
    # B24 can have either eight DP archives or one merged capture. The
    # archived reader decodes both into the same immutable ReplayRound.
    # Only the capture decoder is inherited. The runtime, profiler, public
    # binding and test code below all come from the pinned mainline checkout.
    from workloads import ReplaySequence as ArchiveSequence
    from main_five_report import archived_oracle
    from native_li_reference import digest
    sequence = ArchiveSequence(args.trace, args.group, (1, 1))
    sequence.cache_ranked_round()
    inputs = next(iter(sequence))
    _, receipt = archived_oracle(args.owner_archive, args.group)
    operand_hashes = {}
    for name in ('query', 'weights', 'query_scale', 'logical_keys', 'logical_scales', 'query_ends', 'captured_layout'):
        operand_hashes[name] = digest(getattr(inputs, name))
    if operand_hashes != receipt['native_reference']['identity']['inputs']:
        raise ValueError(f'G{args.group}: captured mainline operands do not match the archived strong serial')
    # native_li_reference imports the archived reporting module for its cache
    # writer. Start the production imports with an independent module namespace.
    for module_name in ('workloads', 'reporting'):
        sys.modules.pop(module_name, None)
    main = args.main.resolve()
    support = main / 'tests/operators/lifetch/support'
    sys.path[:0] = [str(support), str(main / 'python')]
    import execution as ex
    import kvaf
    import kvip
    import performance
    import diagnostics
    import torch
    import torch_npu
    from native_li_reference import fused_li_output
    from reporting import atomic_json
    from workloads import configuration, ReplayPagePlan

    for module, root in ((ex, support), (performance, support), (diagnostics, support),
                         (kvip, main / 'python'), (kvaf, args.kvaf_root / 'build-npu/python')):
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'Unexpected {module.__name__} loaded from {module.__file__}')
    cfg = configuration([
        'diagnose', '--transport', 'urma', '--input', 'real', '--trace-root', str(args.trace),
        '--group', str(args.group), '--round-range', '1:1', '--device', '0',
        '--kvaf-root', str(args.kvaf_root), '--kvaf-server', args.kvaf_server,
        '--prefetch-rank-start', '1024', '--prefetch-tokens', '1024', '--intermediate-fetches', '4',
        '--warmup', '4', '--iterations', str(CALLS), '--level', '2', '--output', str(args.output / 'results'),
    ])
    folder = case_path(args)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'config.json', {**{k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(cfg).items()},
                                       'physical_npu': 4, 'fused_only': True, 'production_root': str(main),
                                       'production_revision': args.revision})
    pages = ReplayPagePlan(sequence)
    paths = [str(ref.manifest_path) for refs in sequence.rounds for ref in refs]
    if (inputs.geometry['batch'], inputs.geometry['token_bytes'], len(sequence.rounds)) != (BATCH, 656, 1):
        raise ValueError('Expected one captured BS16 Q4 656B round')
    atomic_json(folder / 'replay_input.json', dict(geometry=inputs.geometry, identities=inputs.identities,
                rounds=1, page_count=pages.page_count, kv_payload='synthetic identity C8; actual KV not captured',
                source_paths=paths))
    # The official recorder gives each uninstrumented measured state and each
    # separate diagnostic state a unique label. Neither hook runs inside a graph.
    current = None
    outputs = {}
    ordered = {}
    library_paths = set()
    label_pattern = re.compile(r'(measured|detail)/input(\d+)/round1$')

    class Recorder(ex.CorrectnessRecorder):
        def check(self, label, callback):
            nonlocal current
            current = label_pattern.fullmatch(label)
            try:
                return super().check(label, callback)
            finally:
                current = None

    class AuditedState(ex.FusedReplayState):
        def __init__(self, *arguments, **kwargs):
            super().__init__(*arguments, **kwargs)
            if self.forward.terminal is None:
                raise RuntimeError('Production operator selected the legacy tail')
            library_paths.add(Path(self.rotating.library._name).resolve())

        def validate_index(self, sample, output):
            layout, valid = super().validate_index(sample, output)
            if current and current[1] == 'detail':
                call = int(current[2])
                if call in ordered:
                    raise ValueError(f'Repeated Index diagnostic for call {call}')
                sources = self.forward.events_src.cpu().numpy()
                destinations = self.forward.events_dst.cpu().numpy()
                counts = self.rotating.change_count.cpu().numpy()
                rows = self.rotating.rows.cpu().numpy()
                posts = self.rotating.final_posts.cpu().numpy()
                for row in range(self.batch):
                    count = int(counts[row])
                    if count != int(rows[row, 9]) + int(posts[row, :, 0].sum()) or not 0 <= count <= sources.shape[1]:
                        raise ValueError(f'G{cfg.group} call {call} row {row}: inconsistent Index admission count')
                    verify_assignments(self.input_layout[row], layout[row], sample.captured_hot_valid[row], valid[row],
                                       sources[row, :count], destinations[row, :count])
                ordered[call] = self.batch
            return layout, valid

        def validate(self, sample):
            super().validate(sample)
            if current and current[1] == 'measured':
                call = int(current[2])
                if call in outputs:
                    raise ValueError(f'Repeated measured verification for call {call}')
                # Decode the device's selected physical positions using the old
                # layout and require the public logical output to select the same IDs.
                outputs[call] = fused_li_output(self)

    ex.CorrectnessRecorder = Recorder
    ex.FusedReplayState = AuditedState
    fused = folder / 'fused'
    result = ex.run_path(cfg, fused, sequence, pages, fused=True)
    if result['correctness'] != 'PASS' or result['failures']:
        raise ValueError(f'Official mainline replay failed: {result["failures"]}')
    if set(outputs) != set(range(CALLS)) or set(ordered) != set(outputs) or len(library_paths) != 1:
        raise ValueError('Incomplete measured or diagnostic verification')
    samples = json.loads((fused / 'diagnose/round-1/full_attribution.json').read_text())['samples']
    for i, sample in enumerate(samples):
        entry = sample['accounting']
        intervals = entry['intervals_us']
        if (entry['invocation'] != i or set(intervals) != {'li', 'topk_resolve_update'} or
                len(intervals['li']) != 3 or len(intervals['topk_resolve_update']) != 1):
            raise ValueError(f'Unexpected mainline profiler kernels in invocation {i}')
    if len(samples) != CALLS:
        raise ValueError('Incomplete mainline profiler capture')
    library, = library_paths
    runtime = dict(host=platform.node(), visible=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                   npu=torch.npu.get_device_name(torch.device('npu:0')), torch=str(torch.__version__),
                   torch_npu=str(torch_npu.__version__), cann=str(getattr(torch.version, 'cann', '')))
    atomic_json(fused / 'mainline_checks.json', dict(
        runtime=runtime, capture_operand_hashes=operand_hashes, production_revision=args.revision,
        production_python=str(Path(kvip.__file__).resolve()),
        loaded_library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        production_terminal=True, profiled_terminal_kernel='v9_parallel_terminal',
        measured_calls=len(outputs), measured_kv_index_rows=len(outputs) * BATCH,
        ordered_diagnostic_calls=len(ordered), ordered_diagnostic_rows=sum(ordered.values())))
    np.save(fused / 'measured_ids.npy', np.stack([outputs[i] for i in range(CALLS)]), allow_pickle=False)


def main(args):
    from correctness import topk_fingerprint
    from main_five_report import archived_oracle
    from reporting import atomic_json
    from score_equivalence import save_checked
    from workloads import ReplaySequence

    if (os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != '4' or
            subprocess.check_output(['git', '-C', args.main, 'rev-parse', 'HEAD'], text=True).strip() != args.revision or
            subprocess.check_output(['git', '-C', args.main, 'status', '--porcelain', '-uno'], text=True).strip()):
        raise ValueError('Expected clean pinned production mainline on physical NPU4')
    serial, receipt = archived_oracle(args.owner_archive, args.group)
    source = receipt['native_reference']
    # The reader and mainline have different modules named execution,
    # performance, correctness, etc. Keep the production process independent.
    subprocess.run([sys.executable, '-B', __file__, *sys.argv[1:], '--worker'], check=True)
    fused = case_path(args) / 'fused'
    ids_file = fused / 'measured_ids.npy'
    outputs = np.load(ids_file, allow_pickle=False)
    inputs = next(iter(ReplaySequence(args.trace, args.group, (1, 1))))
    save_checked(fused, args.group, inputs, serial,
                 {**source, 'origin': 'completed-owner-archive', 'archive_path': str(args.owner_archive)}, outputs)
    result_file = fused / 'path_result.json'
    result = json.loads(result_file.read_text())
    result['topk_set_digests'] = [topk_fingerprint(value) for value in outputs]
    atomic_json(result_file, result)
    ids_file.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'trace', 'owner-archive', 'kvaf-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--kvaf-server', required=True)
    parser.add_argument('--group', type=int, choices=(2, 4, 7, 10, 17), required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parsed = parser.parse_args()
    (worker if parsed.worker else main)(parsed)
