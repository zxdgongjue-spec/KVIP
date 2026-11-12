# SPDX-License-Identifier: Apache-2.0
"""Continuous real-input execution using production operators and KVAF Runtime."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from performance import measure_sequence
from replay_device import FusedReplayState, NativeReplayState
from replay_inputs import ReplayPagePlan, ReplaySequence
from reporting import CorrectnessRecorder, atomic_json


def run_path(config, directory, sequence, pages, *, fused, round_check=None):
    import acl
    import kvaf
    import torch
    import torch_npu
    from kvip.runtime.build import build_libraries, kvaf_build_paths
    from kvip.runtime.device import query_device_resources
    from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library
    from native_batchget_support import NativeFetchSession
    from native_support import _load_native_operators

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    initial = next(iter(sequence))
    build = config.kvaf_root / 'build-npu'
    torch.npu.set_device(config.device)
    device = torch.device(f'npu:{config.device}')
    stream = torch.npu.current_stream(device)
    cann, headers = kvaf_build_paths(build)
    soc = acl.get_soc_name()
    aics = query_device_resources(cann, config.device)['effective_aic']
    _, registration, _ = _load_native_operators('c8_offload')
    registration.require_a5_dsa_registration(('lidu_fused_batch_get_prepare_mtp_out',))
    g = initial.geometry
    width = g['token_bytes']
    workers = 40 if fused and config.transport == 'urma' else 2
    kwargs = dict(device_id=config.device, npu_core_count=workers,
                  batch_put_kernel_object_path=str(build / 'bin/kvaf_execute_batch_put_kernels.o'))
    if config.transport == 'ubmem':
        kwargs['transport'] = 'ubmem'
    else:
        kwargs['jettys_per_core'] = 2
    runtime = kvaf.NpuRuntime(str(build / 'bin/kvaf_execute_batch_get_kernels.o'), width, 128, **kwargs)
    state = graph = table = None
    pending = False
    try:
        block_bytes = 128 * width
        slot_bytes = ((block_bytes + 4095) // 4096 * 4096
                      if config.transport == 'ubmem' else block_bytes)
        runtime.link_to(config.kvaf_server)
        runtime.initialize((pages.page_count + 2) * slot_bytes, slot_bytes)
        # Reuse Runtime Put; bounded source buffers avoid a multi-GB staging copy.
        for begin in range(0, pages.page_count, 128):
            ids = list(range(begin, min(begin + 128, pages.page_count)))
            source = torch.from_numpy(pages.page_payload(ids)).to(device)
            pending = True
            runtime.batch_put(ids, source, [n * block_bytes for n in range(len(ids))],
                              [block_bytes] * len(ids))
            stream.synchronize()
            pending = False
        table = runtime.prepare_access_table(list(range(pages.page_count)))
        view = runtime.access_table_view(table, device='npu')
        indices = torch.from_numpy(pages.page_indices.reshape(-1).astype(np.int64)).to(device)
        # Native LIDU addresses logical request pages; aliases repeat addresses.
        expanded = SimpleNamespace(addresses=view.addresses[indices].contiguous(),
                                   valid=view.valid[indices].contiguous())
        page_addresses = expanded.addresses.cpu().numpy().view(np.uint64).reshape(g['batch'], -1)
        logical_addresses = (page_addresses[:, :, None] + np.arange(128) * width).reshape(g['batch'], -1)
        if fused:
            if config.transport == 'urma':
                (library,) = build_libraries(cache_dir=build / 'lifetch', cann=cann, headers=headers, soc=soc)
            else:
                shape = UbmemGeometry(g['batch'], g['history'], g['resident'], g['heads'], width,
                                      aics, config.intermediate_fetches, soc)
                library = build_ubmem_library(cache_dir=build / 'lifetch_ubmem', cann=cann, shape=shape)
            state = FusedReplayState(
                torch, runtime, view, library, initial, pages, logical_addresses,
                device=device, aics=aics, windows=config.intermediate_fetches,
                prefetch_tokens=config.prefetch_tokens, soc=soc)
        else:
            state = NativeReplayState(torch, torch_npu, runtime, expanded, initial, pages, device=device)
        checks = CorrectnessRecorder()
        def validate(inputs):
            nonlocal pending
            try:
                state.validate(inputs)
                if round_check is not None:
                    round_check(state, inputs)
            except AssertionError:
                # Numerical mismatch follows successful completion checks.
                pending = False
                raise
            else:
                pending = False
        pending = True
        state.forward()
        stream.synchronize()
        checks.check('eager', lambda: validate(initial))
        state.reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        scope = state.lease.capture_scope() if fused else nullcontext()
        pending = True
        with scope:
            with torch.npu.graph(graph, stream=capture_stream):
                state.forward()
            capture_stream.synchronize()
        pending = False
        if fused and hasattr(state.lease, 'retain_graph'):
            state.lease.retain_graph(graph)
        samples = []
        def save(sample):
            samples.append(sample)
            atomic_json(directory / 'samples.json', samples)
        # The fixed graph consumes updated buffers; use it for every real round.
        class TrackedGraph:
            def replay(self):
                nonlocal pending
                pending = True
                graph.replay()
        graphs = {refs[0].source_round: TrackedGraph() for refs in sequence.rounds}
        result = measure_sequence(torch, sequence, graphs, state.reset, state.prepare, validate,
                                  warmup=config.warmup, iterations=config.iterations, save=save)
        stream.synchronize()
        pending = False
        if checks.failures:
            result['correctness'] = 'FAIL'
            result['failures'] = checks.failures + result['failures']
        result.update(transport=config.transport, path='fused' if fused else 'serial',
                      runtime_workers=workers, kv_payload=('deterministic C8; real Q/Index; captured page aliases'
                      if config.input == 'real' else 'deterministic C8; synthetic controlled Q/Index'),
                      lru_initialization='captured 2048 oldest slots; remaining order initialized by slot ID',
                      geometry=g, input_paths=[str(ref.manifest_path) for refs in sequence.rounds for ref in refs])
        atomic_json(directory / 'path_result.json', result)
        if config.mode == 'diagnose' and fused:
            from device_trace import collect_replay

            # Full-graph timings above remain saved if profiling later fails.
            result['diagnostic'] = collect_replay(
                torch, state, TrackedGraph(), sequence, validate, config,
                directory / 'diagnose')
            if result['diagnostic']['correctness'] == 'FAIL':
                result['correctness'] = 'FAIL'
                result['failures'].extend(result['diagnostic']['failures'])
            atomic_json(directory / 'path_result.json', result)
        return result
    except BaseException as error:
        if pending:
            NativeFetchSession.quarantine('real_replay_completion_uncertain', error)
        raise
    finally:
        if not pending:
            if graph is not None:
                graph.reset()
            if state is not None:
                state.close()
            if table is not None:
                runtime.release_access_table(table)
            if runtime.initialized:
                runtime.shutdown()


def run_real(config, directory):
    sequence = (ReplaySequence.from_snapshot(config.snapshot) if getattr(config, 'operation', None) == 'real'
                else ReplaySequence(config.trace_root, config.group, config.round_range))
    pages = ReplayPagePlan(sequence)
    initial = next(iter(sequence))
    directory = Path(directory)
    atomic_json(directory / 'replay_input.json', dict(
        geometry=initial.geometry, identities=initial.identities, rounds=len(sequence.rounds),
        page_count=pages.page_count, kv_payload='synthetic identity C8; actual KV not captured',
        source_paths=[str(ref.manifest_path) for refs in sequence.rounds for ref in refs]))
    results = {}
    for name in ('serial', 'fused'):
        results[name] = run_path(config, directory / name, sequence, pages, fused=name == 'fused')
        # Preserve completed serial timing even if the fused worker later fails.
        report = dict(batch=initial.geometry['batch'], input='real', correctness='NOT_RUN',
                      execution='INCOMPLETE', timing={'native_chain': results['serial']['all_rounds']})
        if 'fused' in results:
            report.update(execution='COMPLETED',
                          correctness='PASS' if all(r['correctness'] == 'PASS' for r in results.values()) else 'FAIL')
            report['timing']['prototype_with_update'] = results['fused']['all_rounds']
        atomic_json(directory / 'completed_timings.json', report)
    report['sequence'] = results
    atomic_json(directory / 'report.json', report)
    if config.mode == 'check':
        atomic_json(directory / 'checks.json', [dict(case='real', correctness=report['correctness'],
                    detail='real Q/Index; deterministic C8 KV; serial and fused graph replay')])
    return int(report['correctness'] != 'PASS')


def run_stateful(config, directory):
    """Run the same persistent-state adapters as real replay, with fixed queries."""
    from correctness import stateful_inputs, require_stateful_round
    directory = Path(directory)
    sequence = stateful_inputs(directory)
    pages = ReplayPagePlan(sequence)
    config = SimpleNamespace(**{**vars(config), 'prefetch_tokens': 0, 'warmup': 0, 'iterations': 1})
    evidence, results = {}, {}
    for name in ('serial', 'fused'):
        evidence[name] = []
        def observe(state, inputs):
            if name == 'fused':
                layout = state.storage.logical_ids[1:].cpu().numpy()
                valid = state.storage.hot_valid[1:].cpu().numpy() != 0
                hot = [layout[row, :state.resident][valid[row]] for row in range(state.batch)]
                prefetch = state.rotating.rows[:, 9].cpu().numpy()
                final = state.rotating.rows[:, 247].cpu().numpy()
            else:
                mapping = state.native.forward.cpu().numpy()
                hot = [np.flatnonzero((row >= 0) & (row < state.resident)) for row in mapping]
                prefetch = np.zeros(state.batch, dtype=int)
                final = state.native.counts.cpu().numpy()[:-1]
            record = dict(source_round=inputs.source_round, prefetch=prefetch.astype(int).tolist(),
                          final=final.astype(int).tolist(), hot_tokens=[row.astype(int).tolist() for row in hot])
            evidence[name].append(record)
            atomic_json(directory / 'stateful_coverage.json', evidence)
            print(f"STATEFUL path={name} round={inputs.source_round} "
                  f"prefetch={record['prefetch']} final={record['final']} "
                  f"hot_valid={[len(row) for row in hot]}", flush=True)
            require_stateful_round(inputs.source_round, hot, prefetch, final)
        results[name] = run_path(config, directory / name, sequence, pages, fused=name == 'fused',
                                 round_check=observe)
        observed = {row['source_round'] for row in evidence[name]}
        if observed != {1, 2, 3, 4}:
            results[name]['correctness'] = 'FAIL'
            results[name]['failures'].append(dict(check='stateful coverage', error=f'observed rounds={sorted(observed)}'))
        atomic_json(directory / 'stateful_results.json', results)
    checks = [dict(case=f'stateful/{name}', correctness=result['correctness'],
                   detail='A load/hit; B+C eviction and fallback; A reload; state never reset between rounds')
              for name, result in results.items()]
    atomic_json(directory / 'checks.json', checks)
    atomic_json(directory / 'stateful_results.json', results)
    return int(any(row['correctness'] != 'PASS' for row in checks))
