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
from time import perf_counter

import numpy as np


REVISION = '609fa26fd715f9409bbe74e1b2d3685c58dfb0cf'
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


class FeatureReplayChecks:
    """Use the single-round reset boundary to reduce warmup host transfers."""

    def __init__(self, calls, warmup, policy):
        self.policy = policy
        self.setup = {f'pre_capture/input{i}' for i in range(calls)}
        self.setup.update(f'warmup{w}/input{i}/round1' for w in range(warmup) for i in range(calls))
        self.full = {'eager'} | {f'{stage}/input{i}/round1'
                                 for stage in ('measured', 'detail') for i in range(calls)}
        self.checked = set()
        self.full_seconds = self.completion_seconds = 0.

    def validate(self, label, state, inputs, full_validate):
        if label in self.checked:
            raise ValueError(f'Duplicate feature replay validation: {label}')
        start = perf_counter()
        if label in self.setup:
            # The next reset recreates hot KV, Index and CPU LRU from the capture.
            # Private queues survive reset, so check every completion before reuse.
            from terminal_batch_checks import audit_queues
            state.check_completion()
            audit_queues(state.forward.terminal, state.rotating, policy=self.policy)
            self.completion_seconds += perf_counter() - start
        elif label in self.full:
            full_validate(inputs)
            self.full_seconds += perf_counter() - start
        else:
            raise ValueError(f'Unexpected feature replay validation: {label}')
        self.checked.add(label)

    def receipt(self):
        if self.checked != self.setup | self.full:
            raise ValueError('Incomplete feature replay validation')
        return dict(full_state_checks=len(self.full), completion_and_private_CQ_checks=len(self.setup),
                    full_state_check_seconds=self.full_seconds, completion_check_seconds=self.completion_seconds)


def worker(args):
    # B24 can have either eight DP archives or one merged capture. The
    # archived reader decodes both into the same immutable ReplayRound.
    # Only the capture decoder is inherited. The runtime, profiler, public
    # binding and test code below all come from the pinned mainline checkout.
    call_ids = range(args.iterations)
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
        '--warmup', '4', '--iterations', str(args.iterations), '--level', '2', '--output', str(args.output / 'results'),
    ])
    pages = ReplayPagePlan(sequence)
    paths = [str(ref.manifest_path) for refs in sequence.rounds for ref in refs]
    capture_receipt = dict(geometry=inputs.geometry, identities=inputs.identities, rounds=1,
                           page_count=pages.page_count, kv_payload='synthetic identity C8; actual KV not captured',
                           source_paths=paths)
    history_cap = None
    if args.history_cap:
        from cap_pair_input import bind_capacity
        if (inputs.geometry['history'] != 131328 or np.max(inputs.query_ends) > 131075
                or not np.array_equal(inputs.query_ends, receipt['native_reference']['effective_native_ends'])):
            raise ValueError('Expected the original 128K capture and the same active bounds as the strong serial')
        sequence, inputs, pages, history_cap = bind_capacity(sequence, inputs, pages, args.history_cap, digest)
        cfg.history = args.history_cap
    folder = case_path(args)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'config.json', {**{k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(cfg).items()},
                                       'physical_npu': 4, 'fused_only': True, 'production_root': str(main),
                                       'production_revision': args.revision})
    if history_cap:
        atomic_json(folder / 'capture_input.json', capture_receipt)
    if (inputs.geometry['batch'], inputs.geometry['token_bytes'], len(sequence.rounds)) != (BATCH, 656, 1):
        raise ValueError('Expected one captured BS16 Q4 656B round')
    atomic_json(folder / 'replay_input.json', dict(geometry=inputs.geometry, identities=inputs.identities,
                rounds=1, page_count=pages.page_count, kv_payload='synthetic identity C8; actual KV not captured',
                source_paths=paths))
    # The official recorder gives each uninstrumented measured state and each
    # separate diagnostic state a unique label. Neither hook runs inside a graph.
    current = None
    current_label = None
    feature_checks = (FeatureReplayChecks(args.iterations, cfg.warmup, args.ablation_policy)
                      if args.ablation_policy else None)
    outputs = {}
    ordered = {}
    window_workers = {}
    team_header = main / 'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/final_window_queries.h'
    team_trace = team_header.is_file()
    spare_trace = team_trace and 'struct SpareAssignment' in team_header.read_text()
    spare_workers = {}
    library_paths = set()
    label_pattern = re.compile(r'(measured|detail)/input(\d+)/round1$')

    borrow_calls = {'measured': {}, 'detail': {}}
    cq_calls = {'measured': {}, 'detail': {}}
    measured_requests = {}
    private_buffers = {name: set() for name in ('index', 'kv', 'winners', 'window')}
    detail_ids = {}
    mailboxes = set()
    cq_status_ranges = []
    if args.ablation_policy:
        from terminal_batch_checks import ablation_receipt, check_status_binding
    if args.borrow_audit:
        import last_window_borrow as borrow
        from terminal_batch_checks import audit_queues, check_status_binding

        class BorrowForward(ex.StreamingForward):
            def set_diagnostics(self, enabled):
                borrow.set_terminal_only(self, enabled)

            def diagnostic_cycles(self):
                # Prefetch stage timestamps are explicitly disabled for the normal LI schedule.
                return [dict(terminal_timeline=diagnostics.decode_terminal_timeline(
                    self.terminal.times.cpu().tolist(), self.counter_clock_mhz))]

        ex.StreamingForward = BorrowForward

    class Recorder(ex.CorrectnessRecorder):
        def check(self, label, callback):
            nonlocal current, current_label
            current_label = label
            current = label_pattern.fullmatch(label)
            try:
                return super().check(label, callback)
            finally:
                current = None
                current_label = None

    class AuditedState(ex.FusedReplayState):
        def __init__(self, *arguments, **kwargs):
            super().__init__(*arguments, **kwargs)
            if history_cap:
                ends = self.workspace.inputs.query_ends.cpu().numpy().reshape(BATCH, 4)
                key_lengths = self.workspace.key_lengths.cpu().numpy()
                if (self.history != args.history_cap or self.workspace.block_history != args.history_cap
                        or not np.all(key_lengths == args.history_cap) or not np.array_equal(ends, inputs.query_ends)):
                    raise ValueError('Device binding did not retain full configured capacity and original Q ends')
            if args.production_graph or args.ablation_policy:
                for name, buffer in (('index', self.storage.keys), ('kv', self.output),
                                     ('winners', self.workspace.topk), ('window', self.rotating.rank_progress)):
                    address = buffer.data_ptr()
                    if address in private_buffers[name]:
                        raise ValueError(f'Two calls share mutable {name} storage')
                    private_buffers[name].add(address)
            if self.forward.terminal is None:
                raise RuntimeError('Production operator selected the legacy tail')
            lib = Path(self.rotating.library._name).resolve()
            if args.ablation_policy:
                check_status_binding(self.forward.terminal, cq_status_ranges)
            if args.borrow_audit:
                borrow.check_workspace(self.forward)
                if args.revision == borrow.COMBINED:
                    check_status_binding(self.forward.terminal, cq_status_ranges)
                address = self.rotating.rank_progress.data_ptr()
                if address in mailboxes:
                    raise ValueError('The independent calls unexpectedly share a window mailbox')
                mailboxes.add(address)
                if lib not in library_paths:
                    borrow.check_device(lib, args.revision)
            library_paths.add(lib)

        def validate_index(self, sample, output):
            layout, valid = super().validate_index(sample, output)
            if history_cap and current and current[1] == 'measured':
                call = int(current[2])
                if call in measured_requests:
                    raise ValueError(f'Repeated measured request count: {call}')
                prefetched = self.rotating.rows[:, 9].cpu().numpy()
                final = self.rotating.final_posts[:, :, 0].cpu().numpy()
                measured_requests[call] = (int(prefetched.sum()), int(final.sum()))
            if args.ablation_policy and current:
                phase, call = current[1], int(current[2])
                if call in cq_calls[phase]:
                    raise ValueError(f'Repeated queue observations for {phase}/{call}')
                cq_calls[phase][call] = ablation_receipt(self, args.ablation_policy, diagnostic=phase == 'detail')
            if args.borrow_audit and current:
                phase, call = current[1], int(current[2])
                borrow.check_flags(*borrow.options(self.forward), phase == 'detail')
                if call in borrow_calls[phase]:
                    raise ValueError(f'Repeated normal LI mailbox: {phase}/{call}')
                borrow_calls[phase][call] = borrow.decode_mailbox(
                    self.rotating.rank_progress.cpu().numpy(), self.resident, self.history)
                if args.revision == borrow.COMBINED:
                    cq_calls[phase][call] = audit_queues(self.forward.terminal, self.rotating)
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
                if team_trace:
                    # Only the separate diagnostic replay reads worker mailboxes.
                    progress = self.rotating.rank_progress.cpu().numpy()
                    ticks = progress[:, :, 24:32].tolist()
                    if spare_trace:
                        spare_workers[call] = progress[:, (0, 3), 8:20].tolist()
                    for queries in ticks:
                        if len(queries) != 4 or len({q[1] for q in queries}) != 1:
                            raise ValueError('Final-window workers reported inconsistent row snapshots')
                        for arrival, snapshot, begin, end, *_ in queries:
                            if min(arrival, snapshot) <= 0 or not max(arrival, snapshot) <= begin <= end:
                                raise ValueError('Missing or unordered final-window worker stamps')
                    window_workers[call] = ticks
                ordered[call] = self.batch
            return layout, valid

        def validate(self, sample):
            if feature_checks is None:
                super().validate(sample)
            else:
                feature_checks.validate(current_label, self, sample, super().validate)
            if args.borrow_audit and current and current[1] == 'detail':
                detail_ids[int(current[2])] = fused_li_output(self)
            if current and current[1] == ('detail' if args.availability_only else 'measured'):
                call = int(current[2])
                if call in outputs:
                    raise ValueError(f'Repeated measured verification for call {call}')
                # Decode the device's selected physical positions using the old
                # layout and require the public logical output to select the same IDs.
                outputs[call] = fused_li_output(self)

    if args.availability_only:
        from window_ready_replay import measure
        ex.measure_replay_batch = measure
    ex.CorrectnessRecorder = Recorder
    ex.FusedReplayState = AuditedState
    if history_cap and not args.production_graph:
        from cap_pair_timing import install
        install(ex, cfg)
    fused = folder / 'fused'
    result = ex.run_path(cfg, fused, sequence, pages, fused=True)
    if result['correctness'] != 'PASS' or result['failures']:
        raise ValueError(f'Official mainline replay failed: {result["failures"]}')
    if set(outputs) != set(call_ids) or set(ordered) != set(outputs) or len(library_paths) != 1:
        raise ValueError('Incomplete measured or diagnostic verification')
    if args.availability_only:
        from check_isolated_device import extract
        library, = library_paths
        device = extract(library)
        li_hashes = {key[1]: hashlib.sha256(code).hexdigest()
                     for key, code in device['v9_kernel'].items() if key[4] == '.text'}
        if li_hashes != {0: 'e3f40f2396952467d081fce63cac46752dab4dd413e5750b0a62681192b41e1c',
                         1: '7863fcf3cfc86d7cbedbc5f901f6f5b7af58493f17fa99fa1bacf9f8bf7307f9'}:
            raise ValueError(f'Loaded cold score diagnostic differs from simulator build: {li_hashes}')
        sections = device['parallel_terminal_kernel']
        tail, = [code for key, code in sections.items() if key[1] == 1 and key[4] == '.text']
        digest = hashlib.sha256(tail).hexdigest()
        if digest != '5054a60bd3822bd0790b781620f21cbfd9b8816f6d08684adc1f540a9f1083b4':
            raise ValueError(f'Loaded terminal code differs from B24 upstream: {digest} ({library})')
        atomic_json(fused / 'ready_checks.json', dict(physical_npu=4, production_revision=args.revision,
            terminal_text_sha256=digest, li_text_sha256=li_hashes,
            capture_operand_hashes=operand_hashes, loaded_library=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            kv_calls=len(outputs), ordered_index_calls=len(ordered), ordered_index_rows=sum(ordered.values())))
        np.save(fused / 'diagnostic_ids.npy', np.stack([outputs[i] for i in call_ids]), allow_pickle=False)
        return
    samples = json.loads((fused / 'diagnose/round-1/full_attribution.json').read_text())['samples']
    for i, sample in enumerate(samples):
        entry = sample['accounting']
        intervals = entry['intervals_us']
        if (entry['invocation'] != i or set(intervals) != {'li', 'topk_resolve_update'} or
                len(intervals['li']) != 3 or len(intervals['topk_resolve_update']) != 1):
            raise ValueError(f'Unexpected mainline profiler kernels in invocation {i}')
    if len(samples) != args.iterations:
        raise ValueError('Incomplete mainline profiler capture')
    library, = library_paths
    device_proof = {}
    if feature_checks is not None:
        check_times = feature_checks.receipt()
        device_proof['validation'] = check_times
        print(f'FEATURE_VALIDATION full_state_checks={check_times["full_state_checks"]} '
              f'completion_and_private_CQ_checks={check_times["completion_and_private_CQ_checks"]} '
              f'full_state_check_s={check_times["full_state_check_seconds"]:.3f} '
              f'completion_check_s={check_times["completion_check_seconds"]:.3f}', flush=True)
    if args.revision == 'd6cfd392f722dd3f96ef6e05175f503aac511dbb':
        from unified_main_device import check
        device_proof['device_text_sha256'] = check(library)
    if history_cap:
        if set(measured_requests) != set(call_ids):
            raise ValueError('Missing measured prefetch or terminal request counts')
        history_cap['measured_requests'] = {
            key: [measured_requests[i][column] for i in call_ids]
            for column, key in enumerate(('prefetch', 'final'))}
        device_proof['history_cap'] = history_cap
    if args.production_graph or args.ablation_policy:
        from main_five_report import main_times
        main_times(fused, result['all_rounds'], expected_calls=args.iterations)
        counts = {name: len(addresses) for name, addresses in private_buffers.items()}
        if set(counts.values()) != {args.iterations}:
            raise ValueError(f'Expected {args.iterations} independent states within one graph: {counts}')
        device_proof['independent_buffers'] = counts
    runtime = dict(host=platform.node(), visible=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                   npu=torch.npu.get_device_name(torch.device('npu:0')), torch=str(torch.__version__),
                   torch_npu=str(torch_npu.__version__), cann=str(getattr(torch.version, 'cann', '')))
    atomic_json(fused / 'mainline_checks.json', dict(
        runtime=runtime, **device_proof, capture_operand_hashes=operand_hashes, production_revision=args.revision,
        production_python=str(Path(kvip.__file__).resolve()),
        loaded_library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        production_terminal=True, profiled_terminal_kernel='v9_parallel_terminal',
        measured_calls=len(outputs), measured_kv_index_rows=len(outputs) * BATCH,
        ordered_diagnostic_calls=len(ordered), ordered_diagnostic_rows=sum(ordered.values())))
    if args.ablation_policy:
        if len(cq_status_ranges) != args.iterations or any(set(calls) != set(call_ids) for calls in cq_calls.values()):
            raise ValueError('Incomplete independent measured or diagnostic queue receipts')
        atomic_json(fused / 'ablation_receipts.json', dict(
            policy=args.ablation_policy, independent_status_allocations=len(cq_status_ranges),
            **{phase: [calls[i] for i in call_ids] for phase, calls in cq_calls.items()}))
    if args.borrow_audit:
        if len(mailboxes) < args.iterations:
            raise ValueError('Missing independent window mailbox storage')
        atomic_json(fused/'borrow_checks.json', dict(
            device_text_sha256=borrow.check_device(library, args.revision),
            measured=borrow.summarize(borrow_calls['measured']),
            terminal_only=borrow.summarize(borrow_calls['detail']),
            calls={phase: [values[i] for i in call_ids] for phase, values in borrow_calls.items()}))
        if args.revision == borrow.COMBINED:
            if len(cq_status_ranges) != args.iterations or any(set(calls) != set(call_ids) for calls in cq_calls.values()):
                raise ValueError('Incomplete per-query final CQ receipts')
            atomic_json(fused/'final_cq_checks.json', dict(
                independent_status_allocations=len(cq_status_ranges),
                **{phase: [calls[i] for i in call_ids] for phase, calls in cq_calls.items()}))
        if set(detail_ids) != set(call_ids):
            raise ValueError('Incomplete terminal-only TopK results')
        np.save(fused/'terminal_only_ids.npy', np.stack([detail_ids[i] for i in call_ids]), allow_pickle=False)
    if team_trace:
        if set(window_workers) != set(outputs):
            raise ValueError('Missing final-window worker diagnostic calls')
        atomic_json(fused/'window_workers.json', dict(
            fields=['arrival','snapshot','begin','end','ready_packed','topk_calls','scanned_tokens','merge_candidates'],
            calls=[window_workers[i] for i in call_ids]))
    if spare_trace:
        atomic_json(fused/'penultimate_workers.json', dict(
            queries=[0, 3],
            fields=['available','done','arrival','begin','end','topk_calls','scanned','candidates',
                    'assigned','ready_packed','covered_empty','snapshot'],
            calls=[spare_workers[i] for i in call_ids]))
    np.save(fused / 'measured_ids.npy', np.stack([outputs[i] for i in call_ids]), allow_pickle=False)


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
    origin = 'native-only-oracle' if receipt.get('reference_kind') == 'native-only' else 'completed-owner-archive'
    source = {**receipt['native_reference'], 'origin': origin, 'archive_path': str(args.owner_archive)}
    # The reader and mainline have different modules named execution,
    # performance, correctness, etc. Keep the production process independent.
    subprocess.run([sys.executable, '-B', __file__, *sys.argv[1:], '--worker'], check=True)
    fused = case_path(args) / 'fused'
    ids_file = fused / ('diagnostic_ids.npy' if args.availability_only else 'measured_ids.npy')
    outputs = np.load(ids_file, allow_pickle=False)
    inputs = next(iter(ReplaySequence(args.trace, args.group, (1, 1))))
    save_checked(fused, args.group, inputs, serial, source, outputs, expected_calls=args.iterations)
    if args.borrow_audit:
        extra_dir = fused/'terminal_only'
        extra_dir.mkdir(exist_ok=False)
        file = fused/'terminal_only_ids.npy'
        save_checked(extra_dir, args.group, inputs, serial, source, np.load(file, allow_pickle=False),
                     expected_calls=args.iterations)
        file.unlink()
    result_file = fused / 'path_result.json'
    result = json.loads(result_file.read_text())
    result['topk_set_digests'] = [topk_fingerprint(value) for value in outputs]
    atomic_json(result_file, result)
    ids_file.unlink()
    if args.availability_only:
        import shutil
        ref = json.loads((fused / 'score_topk_check.json').read_text())
        checks = json.loads((fused / 'ready_checks.json').read_text())
        if (ref['calls'], ref['queries'], checks['kv_calls'], checks['ordered_index_rows']) != (args.iterations, args.iterations*4*BATCH, args.iterations, args.iterations*BATCH):
            raise ValueError('Incomplete availability replay checks')
        output = args.output.parent
        for name in ('detail_samples.json', 'cold_publication.json', 'raw_rank_progress.npz'):
            shutil.copyfile(fused / 'diagnose/round-1' / name, output / name)
        atomic_json(output/'ready_checks.json', dict(**checks, score_check=ref))
        print(f'G{args.group} DIAGNOSTIC_CHECK TopK_score_vs_strong_serial=PASS '
              f'calls={ref["calls"]}/{args.iterations} queries={ref["queries"]}/{4*BATCH*args.iterations} '
              f'exact_sets={ref["exact_queries"]}/{ref["queries"]} '
              f'KV=PASS ordered_Index=PASS rows={checks["ordered_index_rows"]}/{args.iterations*BATCH} '
              f'workers_per_call={BATCH*4} score_publications_per_call={BATCH*4*12} terminal_text_vs_upstream=PASS', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'trace', 'owner-archive', 'kvaf-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--kvaf-server', required=True)
    parser.add_argument('--group', type=int, choices=(2, 4, 7, 10, 17), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--availability-only', action='store_true')
    modes.add_argument('--borrow-audit', action='store_true')
    modes.add_argument('--ablation-policy', choices=('single', 'paced'),
                       help='Audit each Q’s GET/CQ receipts for 10 timed calls and their separate diagnostic replay')
    modes.add_argument('--history-cap', type=int, choices=(131328, 524544),
                       help='Physical Q4 history capacity for the same 128K captured input')
    parser.add_argument('--production-graph', action='store_true',
                        help='Use the production graph containing 20 independent calls at the specified capacity')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)
    parsed.iterations = 10 if parsed.ablation_policy else 20
    if parsed.production_graph and not parsed.history_cap:
        parser.error('--production-graph requires --history-cap')
    return parsed


if __name__ == '__main__':
    parsed = parse_args()
    (worker if parsed.worker else main)(parsed)
