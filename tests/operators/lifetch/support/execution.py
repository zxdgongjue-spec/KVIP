# SPDX-License-Identifier: Apache-2.0
"""Device execution and native logging shared by every public test mode.

Imports of ACL, Torch and the operator runner stay inside the worker. Reading
configuration or collecting an existing report must not require an NPU install.
"""
import contextlib
import ctypes
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from runner import progress

ROOT = Path(__file__).resolve().parents[4]


def runner_options(config):
    """Translate workload options to the existing formal URMA test runner.

    Pipeline settings are implementation details, not selectable experiments.
    This adapter disappears when the old runner's argument access is consolidated.
    """
    if config.input != 'synthetic':
        raise ValueError('real inputs require the continuous replay runner')
    values = dict(vars(config))
    values.update(
        batches=','.join(map(str, config.batches)),
        kvaf_root=Path(config.kvaf_root).resolve(),
        cann='/usr/local/Ascend/cann', soc=None,
        runtime_shapes=False, resolver_compare=False,
        optimized_resolve=True, streaming=True, streaming_fetch=True,
        fused_final=True, prepare_update_metadata=True, q_workers=2,
        final_workers=None, early_fetch=False, avoid_prefetch_slots=False,
        legacy_fetch_layout=False, parallel_fetch_finalize=False,
        fetch_only_compare=False, comparison_output=None, comparison_sfa=None,
        diagnostic_fetch_only=(config.mode == 'diagnose' and config.level == 3
                               and 'fetch' in config.stages),
        trace_rounds=config.iterations if config.mode == 'diagnose' else 0,
    )
    return SimpleNamespace(**values)


def run_worker(args, run):
    progress(run, "checking native registration and KVAF dependency")
    vendors = ROOT / "csrc/operators/lidu/_custom_opp/vendors"
    paths = sorted(str(p) for p in vendors.glob("*") if (p / "op_api/lib/libcust_opapi.so").is_file())
    if not paths:
        raise RuntimeError("Build first: bash scripts/build.sh")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join([*paths, *filter(None, [os.environ.get("ASCEND_CUSTOM_OPP_PATH")])])
    if args.soc is None:
        import acl
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        args.soc = acl.get_soc_name()
    from lifetch import run as run_lifetch

    return run_lifetch(args, run)


@contextlib.contextmanager
def native_log(path):
    """Capture C++/ACL output as well as Python diagnostics, keep console brief."""
    sys.stdout.flush()
    sys.stderr.flush()
    libc = ctypes.CDLL(None)
    libc.fflush(None)
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with path.open("w", buffering=1) as output:
            for fd in (1, 2):
                os.dup2(output.fileno(), fd)
            yield
    finally:
        libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def worker_environment(kvaf_root, inherited=None, *, require_runtime=True):
    """Select explicit dependencies before importing either transport runtime."""
    root = Path(kvaf_root).resolve()
    package = root / 'build-npu/python/kvaf/__init__.py'
    if require_runtime and not package.is_file():
        raise FileNotFoundError(f'Selected KVAF Python package is missing: {package}')
    environment = dict(os.environ if inherited is None else inherited)
    paths = [str(ROOT / 'python'), str(root / 'build-npu/python')]
    # Selected packages precede inherited paths; keep CANN and vendor setup.
    if environment.get('PYTHONPATH'):
        paths.append(environment['PYTHONPATH'])
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    return environment


def launch(config, directory):
    """Persist the job before spawning an isolated device worker."""
    from reporting import atomic_json
    from runner import supervise

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    values = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(config).items()}
    atomic_json(directory / 'config.json', values)
    environment = worker_environment(config.kvaf_root, require_runtime=getattr(config, "operation", None) != "contract")
    command = [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
               str(directory / 'config.json')]
    return supervise(command, directory, timeout_seconds=config.timeout_seconds,
                     environment=environment)


def run_config(path, shared_pool=None):
    """Internal worker protocol; the sole user-facing entry remains test_lifetch."""
    import json
    from reporting import atomic_json

    path = Path(path).resolve()
    config = SimpleNamespace(**json.loads(path.read_text()))
    config.kvaf_root = Path(config.kvaf_root).resolve()
    config.output = path.parent
    if config.mode == 'bench' and config.input == 'synthetic' and shared_pool is None:
        from bench_pool import BenchPool
        pool = BenchPool(config)
        misses = config.miss_percent if isinstance(config.miss_percent, list) else [config.miss_percent]
        failed = 0
        try:
            for miss in misses:
                child = path.parent / f'miss{miss:g}' / 'config.json'
                atomic_json(child, {**json.loads(path.read_text()), 'miss_percent': miss})
                progress(path.parent, f'BENCH START miss={miss:g} batches={config.batches}')
                failed |= run_config(child, shared_pool=pool)
                cases_log = child.parent / 'cases.log'
                if cases_log.is_file():
                    with (path.parent / 'cases.log').open('a') as output:
                        output.write(cases_log.read_text())
                progress(path.parent, f'BENCH DONE miss={miss:g}')
            pool.close()
            stats = dict(transport=config.transport, max_batch=max(config.batches),
                         cases=pool.borrows, runtime_workers=pool.workers,
                         put_bytes=max(config.batches) * config.history * 656,
                         additional_case_put_bytes=0, initialization_seconds=pool.phase_times)
            atomic_json(path.parent / 'pool.json', stats)
            pool_line = f'BENCH SHARED POOL: {stats}'
            with (path.parent / 'cases.log').open('a') as output:
                output.write(pool_line + '\n')
            print(pool_line, flush=True)
        except BaseException:
            # Unknown transfer completion: retain registered buffers until process exit.
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(70)
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=failed))
        return failed
    operation = getattr(config, 'operation', 'pipeline')
    if operation == 'contract':
        from correctness import check_contracts
        checks = check_contracts(config.transport)
        code = int(any(row['correctness'] != 'PASS' for row in checks))
        atomic_json(path.parent / 'checks.json', checks)
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
        return code
    # This worker is already isolated and has selected packages in PYTHONPATH.
    import kvaf
    expected = config.kvaf_root / 'build-npu/python/kvaf'
    actual = Path(kvaf.__file__).resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError(f'Wrong KVAF module: loaded {actual}; expected {expected}')
    if operation == 'real' or getattr(config, 'input', 'synthetic') == 'real':
        from replay_execution import run_real
        if operation == 'real':
            from workloads import configuration
            defaults = vars(configuration(['bench']))
            defaults.update(vars(config))
            defaults.update(input='real', warmup=0, iterations=1)
            config = SimpleNamespace(**defaults)
        code = run_real(config, path.parent)
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
        return code
    if getattr(config, 'check_suite', None) == 'stateful':
        from replay_execution import run_stateful
        code = run_stateful(config, path.parent)
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
        return code
    if operation == 'init':
        from init_transport import main as initialize
        argv = ['--device', str(config.device), '--kvaf-root', str(config.kvaf_root),
                '--kvaf-server', config.kvaf_server]
        for cycle in range(2):
            initialize(config.transport, argv)
        atomic_json(path.parent / 'checks.json', [dict(case='init', correctness='PASS',
                    detail='two independent create/Put/Get/lease/release cycles')])
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=0))
        return 0
    if config.transport == 'urma':
        options = runner_options(config)
        options.shared_pool = shared_pool
        code = run_worker(options, path.parent)
    else:
        from ubmem_smoke import main as run_ubmem

        argv = ['--build-dir', str(config.kvaf_root / 'build-npu'),
                '--device', str(config.device), '--kvaf-server', config.kvaf_server,
                '--batches', *map(str, config.batches), '--token-bytes', '656',
                '--history', str(config.history), '--resident', str(config.resident),
                '--index-heads', str(config.index_heads), '--seed', str(config.seed),
                '--miss-percent', str(config.miss_percent),
                '--prefetch-tokens', str(config.prefetch_tokens),
                '--intermediate-fetches', str(config.intermediate_fetches),
                '--warmup', str(config.warmup), '--iterations', str(config.iterations),
                '--result-output', str(path.parent / 'cases'), '--serial-compare']
        if getattr(config, 'check_suite', None) == 'boundary':
            argv += ['--check-case', config.check_case]
        if config.mode == 'diagnose':
            argv += ['--trace-rounds', str(config.iterations), '--trace-output', str(path.parent / 'trace')]
        code = run_ubmem(argv, shared_pool=shared_pool) or 0
    atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
    return code



def dispatch(config):
    """Execute selected workloads and always persist the accumulated outcome."""
    import json
    import time
    from correctness import check_jobs
    from reporting import atomic_json, collect_results, comparison_row, save_summary
    from workloads import configuration

    run = config.output.resolve() / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}_{time.time_ns() % 1000000:06d}')
    run.mkdir(parents=True, exist_ok=False)
    common = {key: getattr(config, key) for key in
              ('transport', 'device', 'kvaf_root', 'kvaf_server', 'output', 'timeout_seconds')}
    if config.mode == 'all':
        check = configuration(['check'])
        vars(check).update(common)
        diagnose = configuration(['diagnose'])
        vars(diagnose).update(common)
        # Diagnostic execution includes an unprofiled paired bench, once.
        jobs = [*check_jobs(check), ('bench_diagnose', diagnose)]
    elif config.mode == 'check':
        jobs = check_jobs(config)
    else:
        jobs = [(config.mode, config)]
    atomic_json(run / 'config.json', {key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(config).items()})
    rows, stopped = [], False
    for name, job in jobs:
        directory = run / 'cases' / name
        if stopped:
            row = comparison_row(name, None, None, 'NOT_RUN', execution='NOT_RUN')
            row['reason'] = 'previous worker did not reach checked completion'
            rows.append(row)
            save_summary(run, rows)
            continue
        try:
            if getattr(job, 'operation', None) != 'contract' and not job.kvaf_server:
                raise ValueError('Specify --kvaf-server HOST:PORT; no test service address is configured')
            code = launch(job, directory)
            checks = directory / 'checks.json'
            if checks.is_file():
                for check in json.loads(checks.read_text()):
                    row = comparison_row(f"{name}/{check['case']}", None, None,
                                         check['correctness'], execution='COMPLETED')
                    row.update(timing='NOT_REQUESTED', accepted=check['correctness'] == 'PASS',
                               detail=check.get('detail', ''), mode='check')
                    rows.append(row)
            else:
                records = collect_results(directory, job, code)
                for row in records:
                    row['case'] = f"{name}/{row['case']}"
                rows.extend(records)
            if job.mode == 'check':
                coverage = {str(path.relative_to(directory)): json.loads(path.read_text())
                            for path in sorted(directory.rglob('boundary_coverage.json'))}
                if coverage:
                    atomic_json(directory / 'coverage.json', coverage)
                    for source, phases in coverage.items():
                        print(f'CHECK COVERAGE {name}: padding={phases.get("padding", "not_requested")} '
                              f'transfer={phases.get("transfer", [])}; source={source}', flush=True)
            stopped = not (directory / 'worker-result.json').is_file()
            if job.mode == 'diagnose':
                from diagnostics import summarize
                diagnostic = summarize(directory, job)
                if diagnostic['trace'] != 'COMPLETE':
                    rows[-1]['accepted'] = False
                    rows[-1]['trace'] = diagnostic['trace']
        except (OSError, ValueError, RuntimeError) as error:
            row = comparison_row(name, None, None, 'NOT_RUN', execution='FAILED')
            row['reason'] = f'{type(error).__name__}: {error}'
            rows.append(row)
            stopped = True
        save_summary(run, rows)
    print((run / 'summary.log').read_text(), flush=True)
    print(f'Results: {run}', flush=True)
    return int(not rows or any(not row.get('accepted', False) for row in rows))


if __name__ == '__main__':
    sys.exit(run_config(sys.argv[1]))
