# SPDX-License-Identifier: Apache-2.0
"""Formal operator execution, shared transport resources and worker lifecycle."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import kvip
import numpy as np
from correctness import check_sfa
from diagnostics import actual_reads, collect_core, completion_failures, core_summary, score_rows, selected_bounds
from kvip.adapters.kvaf import UbmemLease
from kvip.cache.index_storage import IndexCacheStorage
from kvip.operators.lifetch.forward import LIFetchForward
from kvip.operators.lifetch.index_update import DirectIndexUpdate, UbmemIndexUpdate
from kvip.operators.lifetch.workspace import UbmemWorkspace
from kvip.operators.lifetch_ubmem.binding import UbmemBinding
from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library, kvaf_cann
from kvip.runtime.device import query_device_resources
from kvip.runtime.loader import load_lifetch_library
from performance import measure_sequence
from reporting import CorrectnessRecorder, atomic_json, record_case_failure, timing_summary
from workloads import SOURCE, ReplayPagePlan, ReplaySequence

ROOT = Path(__file__).resolve().parents[4]


TOKEN_BYTES = 656

BLOCK_SIZE = 128

TOPK = 2048

GUARD_BYTE = 0xA5


MAGIC = 0x514C495638535447

VERSION = 1

TOKENS = 131075

PADDED_TOKENS = 131200

BLOCK_TOKENS = 128

CONTEXT_WORDS = 64

DEFAULT_SERVER = "127.0.0.1:28999"

QUERIES = 4

PUT_CHUNK_BLOCKS = 128

LIFETCH_SOURCE = ROOT / "csrc/operators/lightning_indexer_with_fetch"

NATIVE_BITMAP_WORDS = 4176


def batch_inputs(common, destination, batch, history, cores):
    if batch < 1:
        raise ValueError("positive batch required")
    shutil.copytree(common, destination)
    for index in (0, 2, 3, 5, 6, 7, 9):
        path = Path(destination) / f"arg{index}.bin"
        path.write_bytes(path.read_bytes() * batch)
    path = Path(destination) / "arg13.bin"
    blob = path.read_bytes()
    prefix = 4 * 2048 * 6 + 1024
    tail = prefix + 4 * history * 2
    path.write_bytes(blob[:prefix] * batch + blob[prefix:tail] * batch + blob[tail:] * batch)
    with (Path(destination) / "arg12.bin").open("wb") as workspace:
        workspace.truncate(max(16 * 1024 * 1024, cores * 4 * history * 4 + 1024 * 1024))


def install_remote_addresses(data, addresses, batch, history):
    addresses = np.asarray(addresses, dtype=np.uint64).reshape(batch, history)
    path = Path(data) / "arg13.bin"
    blob = np.fromfile(path, np.uint8)
    prefix_bytes = 4 * 2048 * 6 + 1024 + 4 * history * 2
    tail_bytes = history * 8 + 8192 * 64 + 4 * 128
    for row in range(batch):
        begin = batch * prefix_bytes + row * tail_bytes
        blob[begin : begin + history * 8] = addresses[row].view(np.uint8)
    blob.tofile(path)


def native_fixture(torch, arguments, batch, history):
    """Zero-copy dtype/layout views; native and fused consume identical bytes."""
    device = arguments[0].device
    return {
        "query": arguments[0].view(torch.float8_e4m3fn).reshape(batch, 4, 64, 128),
        "key": arguments[1].view(torch.float8_e4m3fn).reshape(history // 128, 128, 1, 128),
        # Native MTP LIDU uses cumulative TND row boundaries. The LI adapter
        # below separately supplies BSND lengths to match the installed source.
        "actual_query": torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 4,
        "li_query_lengths": torch.full((batch,), 4, dtype=torch.int32, device=device),
        "actual_key": torch.full((batch,), history, dtype=torch.int32, device=device),
        "block_table": arguments[7].view(torch.int32).reshape(batch, history // 128),
        "native_inputs": {
            "weights": arguments[2].view(torch.bfloat16).reshape(batch, 4, 64),
            "query_scale": arguments[3].view(torch.float32).reshape(batch, 4, 64),
            "key_scale": arguments[4].view(torch.float32).reshape(history // 128, 128, 1),
        },
    }


def native_li(torch_npu, fixture):
    """Same BSND/sparse=0 contract as the accepted native Stage2 benchmark."""
    result = torch_npu.npu_quant_lightning_indexer(
        query=fixture["query"],
        key=fixture["key"],
        weights=fixture["native_inputs"]["weights"],
        query_dequant_scale=fixture["native_inputs"]["query_scale"],
        key_dequant_scale=fixture["native_inputs"]["key_scale"],
        actual_seq_lengths_query=fixture["li_query_lengths"],
        actual_seq_lengths_key=fixture["actual_key"],
        block_table=fixture["block_table"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="BSND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=fixture.get("sparse_mode", 0),
    )
    selected = result[0] if isinstance(result, (list, tuple)) else result
    return selected.reshape(-1, 1, 2048)


def runner_options(config):
    """Prepare worker-only build and diagnostic settings from public configuration."""
    if config.input != 'synthetic':
        raise ValueError('real inputs require the continuous replay runner')
    values = dict(vars(config))
    values.update(
        batches=','.join(map(str, config.batches)),
        kvaf_root=Path(config.kvaf_root).resolve(),
        cann='/usr/local/Ascend/cann', soc=None,
        q_workers=2,
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

    return run_urma(args, run)


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

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    values = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(config).items()}
    atomic_json(directory / 'config.json', values)
    environment = worker_environment(config.kvaf_root, require_runtime=getattr(config, "operation", None) != "contract")
    command = [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
               str(directory / 'config.json')]
    return supervise(command, directory, timeout_seconds=config.timeout_seconds,
                     environment=environment, show_case_details=config.mode not in ('bench', 'diagnose'))


def run_config(path, shared_pool=None):
    """Internal worker protocol; the sole user-facing entry remains test_lifetch."""
    import json

    from reporting import atomic_json

    path = Path(path).resolve()
    config = SimpleNamespace(**json.loads(path.read_text()))
    config.kvaf_root = Path(config.kvaf_root).resolve()
    config.output = path.parent
    use_pool = config.mode in ('bench', 'diagnose')
    if use_pool and config.input == 'synthetic' and shared_pool is None:
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
        code = run_stateful(config, path.parent)
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
        return code
    if operation == 'init':
        argv = ['--device', str(config.device), '--kvaf-root', str(config.kvaf_root),
                '--kvaf-server', config.kvaf_server]
        for cycle in range(2):
            initialize_transport(config.transport, argv)
        atomic_json(path.parent / 'checks.json', [dict(case='init', correctness='PASS',
                    detail='two independent create/Put/Get/lease/release cycles')])
        atomic_json(path.parent / 'worker-result.json', dict(exit_code=0))
        return 0
    if config.mode in ('bench', 'diagnose'):
        code = run_synthetic_batch(config, path.parent, shared_pool=shared_pool)
    elif config.transport == 'urma':
        options = runner_options(config)
        options.shared_pool = shared_pool
        code = run_worker(options, path.parent)
    else:

        code = run_ubmem(config, path.parent, shared_pool=shared_pool)
    atomic_json(path.parent / 'worker-result.json', dict(exit_code=code))
    return code


def dispatch(config):
    """Execute selected workloads and always persist the accumulated outcome."""
    import json
    import time

    from correctness import check_jobs
    from reporting import atomic_json, collect_results, comparison_row, save_summary
    from workloads import configuration

    if config.mode == 'convert-trace':
        from workloads import convert_replay_inputs
        return convert_replay_inputs(config.source, config.output)

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


def progress(run, stage):
    (run / "stage.log").write_text(stage + "\n")
    if run.name.startswith('miss') and (run.parent / 'config.json').is_file():
        (run.parent / 'stage.log').write_text(f'{run.name}: {stage}\n')
    print(stage, flush=True)


def log_tail(path, lines=12):
    # Bound reads even if native transport logging grows very large.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 8192))
        return stream.read().decode(errors="replace").splitlines()[-lines:]


def compiler_error(path, before=4, after=10):
    """Keep the first compiler diagnostic instead of only recursive make exits."""
    context = deque(maxlen=before)
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            entry = f"{number}: {line.rstrip()}"
            if "error:" in line or "undefined reference" in line or "CMake Error" in line:
                result = [*context, entry]
                for offset, following in enumerate(stream, 1):
                    result.append(f"{number + offset}: {following.rstrip()}")
                    if offset >= after:
                        break
                return result
            context.append(entry)
    return []


def supervise(command, run, heartbeat_seconds=20, timeout_seconds=300, environment=None, show_case_details=True):
    """Only a checked worker completion marker plus exit code zero means PASS."""
    run = Path(run)
    summary = run / "summary.log"
    started = time.monotonic()

    def status(message):
        text = f"{message}\nResults: {run}\n"
        summary.write_text(text)
        print(message, flush=True)

    status("RUNNING: starting worker")
    print(f"Summary: {summary}", flush=True)
    interrupted = False
    timed_out = False
    with (run / "worker.log").open("w") as output:
        try:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
        except OSError as error:
            status(f"FAIL: cannot start worker: {error}")
            return 1
        try:
            while True:
                try:
                    code = child.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout_seconds:
                        timed_out = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            code = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            code = child.wait()
                        break
                    stage = (
                        (run / "stage.log").read_text().strip() if (run / "stage.log").exists() else "imports/startup"
                    )
                    logs = [p for p in run.rglob("*.log") if p.name not in ("summary.log", "stage.log")]
                    latest = max(logs, key=lambda p: p.stat().st_mtime, default=None)
                    evidence = f"; latest={latest.relative_to(run)} bytes={latest.stat().st_size}" if latest else ""
                    status(f"RUNNING +{time.monotonic() - started:.0f}s: {stage}{evidence}")
        except KeyboardInterrupt:
            interrupted = True
            # Also stop compiler children, without killing unrelated processes.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    marker = run / "worker-result.json"
    try:
        result = json.loads(marker.read_text()) if marker.exists() else {}
    except (OSError, ValueError):
        result = {}
    passed = code == 0 and result.get("exit_code") == 0 and not interrupted
    detail = f"exit_code={code}"
    if timed_out:
        detail += f"; timeout={timeout_seconds}s"
    if code < 0:
        detail += f" signal={signal.Signals(-code).name}"
    if not marker.exists():
        detail += "; worker did not reach checked completion"
    stage = (run / "stage.log").read_text().strip() if (run / "stage.log").exists() else "startup"
    lines = [f"{'PASS' if passed else 'FAIL'}: {detail}; elapsed={time.monotonic() - started:.1f}s; stage={stage}"]
    cases = run / "cases.log"
    case_lines = cases.read_text().splitlines() if cases.exists() else []
    if show_case_details:
        lines.extend(case_lines)
    if not passed:
        for isolation in run.rglob("streaming_isolation.log"):
            lines.append(f"STREAMING ISOLATION: {isolation.read_text().strip()}")
        for audit in run.rglob("streaming_fetch_kv_mismatch.json"):
            try:
                data = json.loads(audit.read_text())
                compact = {key: data[key] for key in ("batch", "mismatch_positions", "posts")}
                compact["phase"] = data.get("phase", "unknown")
                compact["examples"] = data.get("examples", [])[:3]
                for key in ("actual_sq_writers", "final_wqe_comparison", "initial_output_matches", "sq_audit_error"):
                    if key in data:
                        compact[key] = data[key]
                lines.append(f"STREAMING FETCH KV AUDIT: {json.dumps(compact)}; details={audit}")
            except (OSError, ValueError, KeyError) as error:
                lines.append(f"Could not read streaming fetch audit {audit}: {error}")
        for result_path in sorted(run.rglob('path_result.json')):
            result = json.loads(result_path.read_text())
            for failure in result.get('failures', [])[:4]:
                lines.append(f"REPLAY CHECK FAILURE {result.get('path', '')}: {json.dumps(failure)}")
        for audit_path in sorted(run.rglob('score_failure.json')):
            lines.append('REPLAY SCORE AUDIT: ' + audit_path.read_text())
        # At most four short tails, including exceptions before native logging.
        logs = [p for p in run.rglob("*.log") if p.name in ("native.log", "error.log", "build.log", "configure.log")]
        logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in logs:
            if path.name in ("build.log", "configure.log"):
                diagnostic = compiler_error(path)
                if diagnostic:
                    lines.append(f"--- {path.relative_to(run)} (first compiler error) ---")
                    lines.extend(diagnostic)
                    break
        for path in [run / "worker.log", *logs[:3]]:
            lines.append(f"--- {path.relative_to(run)} (tail) ---")
            lines.extend(log_tail(path))
    lines.append(f"Full logs: {run}")
    saved_lines = lines if show_case_details else [*lines, *case_lines]
    summary.write_text("\n".join(saved_lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0 if passed else 1


class BenchPool:
    def __init__(self, config):
        self.config = config
        self.runtime = None
        self.table = None
        self.active = False
        self.borrows = 0
        self.tables = {}
        self.phase_times = {}
        self.fixtures = {}

    @contextmanager
    def phase(self, label):
        from reporting import atomic_json
        output = getattr(self.config, 'output', None)
        if output is not None:
            progress(output, f'BENCH POOL {label}')
        began = time.perf_counter()
        try:
            yield
        finally:
            self.phase_times[label] = time.perf_counter() - began
            if output is not None:
                atomic_json(output / 'pool_phases.json', self.phase_times)
                progress(output, f'BENCH POOL {label} finished {self.phase_times[label]:.3f}s')

    def borrow(self, torch, device, batch, history, workers):
        import kvaf
        import numpy as np
        from workloads import payload

        if self.active or not 1 <= batch <= max(self.config.batches) or history != self.config.history:
            raise ValueError('Invalid shared bench pool borrow')
        if self.runtime is None:
            self.torch = torch
            self.workers = 40 if self.config.transport == 'urma' else 2
            build = self.config.kvaf_root / 'build-npu'
            options = dict(device_id=int(device.index or 0), npu_core_count=self.workers,
                           batch_put_kernel_object_path=str(build / 'bin/kvaf_execute_batch_put_kernels.o'))
            if self.config.transport == 'urma':
                options['jettys_per_core'] = 2
            else:
                options['transport'] = 'ubmem'
            with self.phase('transport_create_link'):
                self.runtime = kvaf.NpuRuntime(str(build / 'bin/kvaf_execute_batch_get_kernels.o'), 656, 128, **options)
                self.runtime.link_to(self.config.kvaf_server)
            capacity = max(self.config.batches) * history // 128
            block_bytes = 128 * 656
            slot_bytes = ((block_bytes + 4095) // 4096 * 4096
                          if self.config.transport == 'ubmem' else block_bytes)
            with self.phase('ddr_allocate'):
                self.runtime.initialize((capacity + 2) * slot_bytes, slot_bytes)
            with self.phase('reference_generate_cpu'):
                values = np.empty((max(self.config.batches), history, 656), dtype=np.uint8)
                for row in range(max(self.config.batches)):
                    for begin in range(0, history, 4096):
                        end = min(begin + 4096, history)
                        values[row, begin:end] = payload(row, np.arange(begin, end))
                self.cpu_payload = torch.from_numpy(values)
            # Bound staging memory; every DDR page is written exactly once.
            with self.phase('reference_upload_hbm'):
                self.reference = self.cpu_payload.to(device)
                torch.npu.synchronize()
            flat = self.reference.reshape(capacity, block_bytes)
            with self.phase('ddr_put'):
                for begin in range(0, capacity, 128):
                    end = min(begin + 128, capacity)
                    source = flat
                    self.runtime.batch_put(list(range(begin, end)), source,
                                           [i * block_bytes for i in range(begin, end)],
                                           [block_bytes] * (end - begin))
                    torch.npu.synchronize()
                    if getattr(self.config, 'output', None) is not None:
                        progress(self.config.output, f'BENCH POOL ddr_put blocks={end}/{capacity}')
            with self.phase('access_table'):
                self.table = self.runtime.prepare_access_table(list(range(capacity)))
                self.view = self.runtime.access_table_view(self.table, device='npu')
                self.tables[max(self.config.batches)] = (self.table, self.view)
            print(f'BENCH POOL INIT transport={self.config.transport} max_batch={max(self.config.batches)} '
                  f'history={history} put_bytes={capacity * block_bytes} runtime_workers={self.workers}', flush=True)
        if workers > self.workers:
            raise ValueError('Shared bench pool has insufficient queues')
        self.active = True
        self.borrows += 1
        count = batch * history // 128
        print(f'BENCH POOL REUSE case={self.borrows} batch={batch} additional_put_bytes=0', flush=True)
        if batch not in self.tables:
            table = self.runtime.prepare_access_table(list(range(count)))
            self.tables[batch] = (table, self.runtime.access_table_view(table, device='npu'))
        table, view = self.tables[batch]
        return SimpleNamespace(runtime=self.runtime, table=table,
                               cpu_payload=self.cpu_payload[:batch], reference=self.reference[:batch],
                               device_table=view)

    def release(self):
        if not self.active:
            raise RuntimeError('Shared bench pool released without a borrower')
        self.active = False

    def close(self):
        if self.active:
            raise RuntimeError('Cannot close shared pool before verified case completion')
        if self.runtime is not None:
            self.torch.npu.synchronize()
            for table, _ in self.tables.values():
                self.runtime.release_access_table(table)
            self.runtime.shutdown()
            self.runtime = None


def command(argv, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        subprocess.run([str(arg) for arg in argv], stdout=output, stderr=subprocess.STDOUT, check=True)


def find_headers(root):
    if root is None:
        raise ValueError("--kvaf-root is required for the real KVAF backend")
    path = Path(root) / "access_operator/demo/aiv_batch_get/include"
    if not (path / "kvaf_batch_get_device.h").is_file():
        raise ValueError(f"missing KVAF device headers: {path}")
    return path


class Registration:
    @staticmethod
    def require_a5_dsa_registration(names):
        # Load only on explicit request. Fused-only tests need no LIDU build.
        import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp  # noqa: F401
        import torch

        for name in names:
            if not callable(getattr(torch.ops.nanovllm_dsa, name, None)):
                raise RuntimeError(f"native LIDU binding missing: {name}; bash build.sh native-lidu")


def _load_native_operators(mode):
    import torch
    import torch_npu  # noqa: F401

    return torch, Registration, None


def compact_mask_plan(
    indices: list[int], offsets: list[int], mask: list[int], output_indices: list[int]
) -> dict[str, list[int] | int]:
    if not (len(indices) == len(offsets) == len(mask) == len(output_indices)):
        raise ValueError("native_prepare_descriptor_lengths_differ")
    if any(type(flag) is not int or flag not in (0, 1) for flag in mask):
        raise ValueError("native_prepare_mask_not_binary")
    active = [index for index, flag in enumerate(mask) if flag]
    result = {
        "indices": [indices[index] for index in active],
        "offsets": [offsets[index] for index in active],
        "output_indices": [output_indices[index] for index in active],
        "miss_count": len(active),
    }
    if len(set(result["output_indices"])) != len(active):
        raise ValueError("native_prepare_destination_alias")
    if any(
        type(value) is not int or value < 0
        for field in ("indices", "offsets", "output_indices")
        for value in result[field]
    ):
        raise ValueError("native_prepare_negative_active_descriptor")
    return result


def native_payload(torch: Any, batch: int, tokens: int) -> Any:
    byte = torch.arange(TOKEN_BYTES, dtype=torch.int32).reshape(1, -1)
    token = torch.arange(tokens, dtype=torch.int32).reshape(-1, 1)
    rows = []
    for pool in range(batch):
        values = (token * 17 + (token >> 8) * 29 + pool * 101 + byte * 7 + (byte >> 8)).to(torch.uint8)
        for part in range(4):
            values[:, part::16] = ((token >> (part * 8)) & 255).to(torch.uint8)
        values[:, 4::16] = pool
        values[:, 5::16] = 0  # one selected layer, layer_id=0
        rows.append(values)
    return torch.stack(rows)


class NativeFetchSession:
    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> NativeFetchSession:
        # Keep a Python reference during partial construction too. An error
        # after async enqueue must reach close()/quarantine before GC can
        # destroy registered caller memory or the remote-table handle.
        session = cls.__new__(cls)
        try:
            session.__init__(*args, **kwargs)
        except BaseException:
            session.close()
            raise
        return session

    def __init__(
        self,
        torch: Any,
        device: Any,
        batch: int,
        candidate_tokens: int,
        cache_tokens: int,
        server: str,
        get_kernel: Path,
        put_kernel: Path,
        source_ids: Any,
        destination_slots: Any,
        miss_counts: Any,
        request_entries: Any,
        cache_budgets: Any,
        candidate_lens: Any,
        temporary_only: bool = False,
        put_chunk_blocks: int = 0,
        npu_core_count: int = 32,
        payload_factory: Callable[[Any, int, int], Any] | None = None,
        shared_pool=None,
    ) -> None:
        self.torch = torch
        self.graphs: list[Any] = []
        self.runtime = None
        self.table = None
        self.batch_get_attempted = False
        self.batch = batch
        self.cache_tokens = cache_tokens
        self.sources = source_ids
        self.destinations = destination_slots
        self.misses = miss_counts
        if candidate_tokens % BLOCK_SIZE or cache_tokens % BLOCK_SIZE:
            raise ValueError("native_fetch_requires_complete_128_token_blocks")
        for kernel in (get_kernel, put_kernel):
            if not kernel.is_file():
                raise ValueError(f"native_kvaf_kernel_missing:{kernel}")
        kvaf = importlib.import_module("kvaf")
        doc = str(kvaf.NpuRuntime.batch_get.__doc__)
        if "miss_count" not in doc or "mask" in doc:
            raise ValueError("native_kvaf_requires_verified_service_prefix_binding_no_abi_guess")
        if any(
            not hasattr(kvaf.NpuRuntime, method)
            for method in ("acquire_fused_batch_get_context", "release_fused_batch_get_context")
        ):
            raise ValueError("matching_kvaf_queue_health_lease_api_missing")
        qcap = candidate_tokens // BLOCK_SIZE
        self.shared_pool = shared_pool
        if shared_pool is not None:
            borrowed = shared_pool.borrow(torch, device, batch, candidate_tokens, npu_core_count)
            self.runtime, self.table = borrowed.runtime, borrowed.table
            self.cpu_payload, self.device_table = borrowed.cpu_payload, borrowed.device_table
            self.reference = borrowed.reference
        else:
            self.runtime = kvaf.NpuRuntime(
                str(get_kernel),
                batch_get_token_bytes=TOKEN_BYTES,
                batch_get_tokens_per_block=BLOCK_SIZE,
                device_id=int(device.index or 0),
                npu_core_count=npu_core_count,
                batch_put_kernel_object_path=str(put_kernel),
                jettys_per_core=2,
            )
            self.runtime.link_to(server)
            capacity = batch * qcap
            block_bytes = BLOCK_SIZE * TOKEN_BYTES
            self.runtime.initialize(capacity * block_bytes, block_bytes)
            # Select the fixture before the first PUT: PreparePut reserves new
            # remote slots and must not be used to overwrite an initialized pool.
            self.cpu_payload = (native_payload if payload_factory is None else payload_factory)(torch, batch, candidate_tokens)
            self.seed = self.cpu_payload.to(device).reshape(capacity, block_bytes)
            # Current KVAF keys are physical DDR slot IDs, not arbitrary labels.
            # Flat pool/layer/page order must match the access table below.
            keys = list(range(capacity))
            try:
                chunk = put_chunk_blocks or capacity
                if chunk < 1:
                    raise ValueError("put_chunk_blocks_must_be_nonnegative")
                for begin in range(0, capacity, chunk):
                    end = min(begin + chunk, capacity)
                    self.runtime.batch_put(
                        keys[begin:end],
                        self.seed,
                        [index * block_bytes for index in range(begin, end)],
                        [block_bytes] * (end - begin),
                    )
                    torch.npu.synchronize()
            except BaseException as error:
                # GET queue health cannot establish completion of a failed PUT.
                # Keep the registered seed alive and do not unwind its allocation.
                self.quarantine("uncertain_batch_put_completion", error)
            self.table = self.runtime.prepare_access_table(keys)
            self.device_table = self.runtime.access_table_view(self.table, device="npu", capacity=capacity)
        if temporary_only:
            if cache_tokens != TOPK:
                raise ValueError("v5_temporary_capacity_must_be_topk")
            self.block_table = torch.arange(batch * TOPK // BLOCK_SIZE, dtype=torch.int32, device=device).reshape(
                batch, -1
            )
            self.initial_cpu = torch.full((batch * TOPK, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
            self.initial = self.initial_cpu.to(device)
            self.output = self.initial.clone()
            self.slot_rows = [torch.arange(row * TOPK, (row + 1) * TOPK) for row in range(batch)]
            # The fused kernel generates each group plan. Never freeze final H
            # descriptors or launch a standalone GET during this setup.
            return
        resident_blocks = batch * cache_tokens // BLOCK_SIZE
        block_table_cpu = torch.arange(resident_blocks, 0, -1, dtype=torch.int32).reshape(batch, -1)
        self.block_table = block_table_cpu.to(device)
        output_rows = (resident_blocks + 2) * BLOCK_SIZE
        self.initial_cpu = torch.full((output_rows, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
        self.slot_rows = []
        for row in range(batch):
            slots = torch.arange(cache_tokens, dtype=torch.long)
            physical = block_table_cpu[row].long()[slots // BLOCK_SIZE] * BLOCK_SIZE + slots % BLOCK_SIZE
            self.slot_rows.append(physical)
            self.initial_cpu.index_copy_(0, physical, self.cpu_payload[row, :cache_tokens])
        self.initial = self.initial_cpu.to(device)
        self.output = self.initial.clone()
        self.fixed_sources = source_ids.cpu().reshape(batch, TOPK).tolist()
        self.fixed_destinations = destination_slots.cpu().reshape(batch, TOPK).tolist()
        self.fixed_misses = miss_counts.cpu().tolist()
        for row in range(batch):
            if not 0 <= self.fixed_misses[row] <= TOPK:
                raise ValueError("native_fetch_invalid_miss_count")
            if any(not 0 <= token < candidate_tokens for token in self.fixed_sources[row]):
                raise ValueError("native_fetch_source_out_of_range")
            if any(not 0 <= slot < cache_tokens for slot in self.fixed_destinations[row]):
                raise ValueError("native_fetch_destination_slot_out_of_range")
        descriptor_count = batch * TOPK
        prepared_indices = torch.full((descriptor_count,), -1, dtype=torch.int32, device=device)
        prepared_offsets = torch.full_like(prepared_indices, -1)
        prepared_mask = torch.full((descriptor_count,), 0xA5, dtype=torch.uint8, device=device)
        prepared_outputs = torch.full_like(prepared_indices, -1)
        resident_lengths = torch.empty((batch,), dtype=torch.int32, device=device)
        torch.ops.nanovllm_dsa.kvaf_batch_get_prepare_out(
            source_ids,
            destination_slots,
            miss_counts,
            request_entries,
            self.block_table,
            cache_budgets,
            candidate_lens,
            candidate_lens,
            prepared_indices,
            prepared_offsets,
            prepared_mask,
            prepared_outputs,
            resident_lengths,
            0,
            1,
            batch,
            qcap,
        )
        torch.npu.synchronize()
        self.plan = compact_mask_plan(
            prepared_indices.cpu().tolist(),
            prepared_offsets.cpu().tolist(),
            prepared_mask.cpu().tolist(),
            prepared_outputs.cpu().tolist(),
        )
        expected_descriptors = []
        self.expected_cpu = self.initial_cpu.clone()
        for row in range(batch):
            count = self.fixed_misses[row]
            if not 0 <= count <= TOPK:
                raise ValueError("native_fetch_invalid_miss_count")
            for index in range(count):
                token = self.fixed_sources[row][index]
                slot = self.fixed_destinations[row][index]
                output_row = int(self.slot_rows[row][slot])
                expected_descriptors.append(
                    (row * qcap + token // BLOCK_SIZE, token % BLOCK_SIZE * TOKEN_BYTES, output_row)
                )
                self.expected_cpu[output_row] = self.cpu_payload[row, token]
        actual_descriptors = list(zip(self.plan["indices"], self.plan["offsets"], self.plan["output_indices"]))
        if actual_descriptors != expected_descriptors or self.plan["miss_count"] != sum(self.fixed_misses):
            raise ValueError("original_prepare_output_does_not_match_native_miss_plan")
        # Keep nonempty caller capacity for an all-hit row/batch: count=0
        # still makes the real BatchGet a valid no-read launch.
        self.indices = torch.tensor(self.plan["indices"] or [0], dtype=torch.int32, device=device)
        self.offsets = torch.tensor(self.plan["offsets"] or [0], dtype=torch.int32, device=device)
        self.output_indices = torch.tensor(self.plan["output_indices"] or [0], dtype=torch.int32, device=device)
        self.count = torch.tensor([self.plan["miss_count"]], dtype=torch.int32, device=device)
        self.launch()
        torch.npu.synchronize()
        self.verify()

    def launch(self) -> None:
        self.batch_get_attempted = True
        self.runtime.batch_get(
            self.device_table.addresses,
            self.device_table.valid,
            self.indices,
            self.offsets,
            self.count,
            self.output_indices,
            self.output,
            self.torch.npu.current_stream(),
            False,
        )

    def reset(self) -> None:
        self.output.copy_(self.initial)

    def verify(self) -> dict[str, Any]:
        if (
            self.sources.cpu().reshape(self.batch, TOPK).tolist() != self.fixed_sources
            or self.destinations.cpu().reshape(self.batch, TOPK).tolist() != self.fixed_destinations
            or self.misses.cpu().tolist() != self.fixed_misses
        ):
            raise ValueError("native_replay_outputs_changed_frozen_fetch_plan_invalid")
        if not self.torch.equal(self.output.cpu(), self.expected_cpu):
            raise ValueError("native_batchget_exact_payload_or_guard_mismatch")
        return {
            "payload_and_guards_verified": True,
            "prepared_plan_still_matches_li_update": True,
            "effective_fetch_tokens": self.plan["miss_count"],
            "effective_fetch_bytes": self.plan["miss_count"] * TOKEN_BYTES,
            "fetch_multiset_hash": hashlib.sha256(json.dumps(self.plan, sort_keys=True).encode()).hexdigest(),
        }

    @staticmethod
    def quarantine(reason: str, error: BaseException) -> None:
        print(
            f"[QLI_V5_NATIVE] status=failed stage=fetch_quarantine reason={reason} detail={str(error)[:240]!r}",
            flush=True,
        )
        os._exit(70)

    def close(self) -> None:
        try:
            self.torch.npu.synchronize()
        except BaseException as error:
            # Identical ownership boundary to the C++ V5 smoke: completion
            # is unknown, therefore do not reset graphs, unregister HBM,
            # release the DDR table, shut down or run Python destructors.
            self.quarantine("uncertain_stream_completion", error)
        if self.runtime is not None and self.batch_get_attempted:
            try:
                # An enqueue-only BatchGet can return from its kernel after
                # CQ timeout while DMA completion remains unknown. The real
                # fused-lease acquire/release validates persistent GET queue
                # health, unlike an ACL stream fence alone. No launches occur
                # during this setup-only probe; keep all graphs/tables alive.
                stream = self.torch.npu.current_stream()
                probe = self.runtime.acquire_fused_batch_get_context(self.output, stream)
                self.runtime.release_fused_batch_get_context(probe, stream)
            except BaseException as error:
                self.quarantine("unhealthy_batch_get_queue", error)
        for graph in self.graphs:
            graph.reset()
        self.graphs.clear()
        if self.runtime is not None:
            if getattr(self, 'shared_pool', None) is not None:
                self.shared_pool.release()
                self.runtime = self.table = None
                return
            if self.table is not None:
                self.runtime.release_access_table(self.table)
                self.table = None
            self.runtime.shutdown()
            self.runtime = None


def unsigned_address_view(addresses, torch):
    """Reinterpret KVAF's address bits for ACL_UINT64; never cast or copy."""
    if addresses.dtype not in (torch.int64, torch.uint64) or not addresses.is_contiguous():
        raise ValueError(f"kvaf_address_storage_invalid:{addresses.dtype}")
    result = addresses if addresses.dtype == torch.uint64 else addresses.view(torch.uint64)
    if result.data_ptr() != addresses.data_ptr() or result.shape != addresses.shape:
        raise ValueError("kvaf_address_view_must_alias_original_storage")
    return result


def discover(
    *,
    server: str | None = None,
    kvaf_python_dir: Path | None = None,
    get_kernel: Path | None = None,
    put_kernel: Path | None = None,
    kvaf_root: Path | None = None,
) -> dict[str, Any]:
    """Read existing build artifacts/config; never build or contact a service."""
    candidates = []
    if kvaf_root is not None:
        candidates = [Path(kvaf_root) / "build-npu", Path(kvaf_root) / "build"]
    values = {
        "kvaf_python_dir": kvaf_python_dir,
        "get_kernel": get_kernel,
        "put_kernel": put_kernel,
    }
    suffixes = {
        "kvaf_python_dir": "python",
        "get_kernel": "bin/kvaf_execute_batch_get_kernels.o",
        "put_kernel": "bin/kvaf_execute_batch_put_kernels.o",
    }

    def ready(name: str, path: Path) -> bool:
        if name != "kvaf_python_dir":
            return path.is_file()
        package = path / "kvaf"
        return (package / "__init__.py").is_file() and any(package.glob("_kvaf*.so"))

    bundle = next(
        (
            base
            for base in candidates
            if all(values[name] is not None or ready(name, base / suffix) for name, suffix in suffixes.items())
        ),
        None,
    )
    for name, suffix in suffixes.items():
        if values[name] is None:
            values[name] = bundle / suffix if bundle is not None else None
        if values[name] is None or not ready(name, Path(values[name])):
            raise ValueError(
                f"v8_fetch_dependency_missing:{name}: expected a matching KVAF "
                "build-npu bundle; pass --kvaf-root to its source checkout"
            )
        values[name] = Path(values[name]).resolve()
    values["server"] = server or DEFAULT_SERVER
    return values


def quarantine_all(resources: Any, reason: str, error: BaseException | str) -> None:
    """Do not unwind Python/NPU destructors after any uncertain stage3 launch."""
    # Keep the argument referenced until process teardown; os._exit is the
    # existing KVAF safety boundary, not an exception that finally may release.
    print(
        f"[QLI_V8_NATIVE] status=failed stage=fetch_quarantine reason={reason} "
        f"detail={str(error)[:240]!r} resources_retained={resources is not None}",
        flush=True,
    )
    os._exit(70)


def output_geometry(batch: int, history: int, resident: int) -> dict[str, int]:
    if batch < 1 or history < TOPK or history % BLOCK_TOKENS:
        raise ValueError("real_fetch_requires_positive_batch_and_128_aligned_history")
    if not TOPK <= resident <= history or resident % BLOCK_TOKENS:
        raise ValueError("real_fetch_requires_aligned_resident_at_least_topk")
    row_tokens = resident + QUERIES * TOPK
    return {
        "batch": batch,
        "history": history,
        "resident": resident,
        "queries": QUERIES,
        "slots_per_query": TOPK,
        "row_tokens": row_tokens,
        "output_tokens": batch * row_tokens,
        "output_bytes": batch * row_tokens * TOKEN_BYTES,
        "blocks_per_row": history // BLOCK_TOKENS,
    }


class RealFetchSession:
    """Exclusive lease owner; never release memory on uncertain DMA completion."""

    @classmethod
    def open(
        cls,
        torch: Any,
        device: Any,
        *,
        batch: int,
        history: int,
        resident: int,
        server: str,
        get_kernel: Path | None = None,
        put_kernel: Path | None = None,
        kvaf_python_dir: Path | None = None,
        kvaf_root: Path | None = None,
        workers: int = 32,
        session_factory=None,
    ):
        owner = cls(torch, device, batch, history, resident)
        owner.session_factory = session_factory or NativeFetchSession
        try:
            owner._prepare(server, get_kernel, put_kernel, kvaf_python_dir, kvaf_root, workers)
            return owner
        except BaseException:
            owner.close()
            raise

    def __init__(self, torch, device, batch, history, resident):
        self.geometry = output_geometry(batch, history, resident)
        self.torch, self.device = torch, device
        self.batch, self.history, self.resident_budget = batch, history, resident
        self.session = None
        self.lease = None
        self.lease_stream = None
        self.in_flight = False
        self.closed = False
        self.keepalive = []
        self.graphs = []

    def _prepare(self, server, get_kernel, put_kernel, python_dir, root, workers):
        config = discover(
            server=server, get_kernel=get_kernel, put_kernel=put_kernel, kvaf_python_dir=python_dir, kvaf_root=root
        )
        loaded = sys.modules.get("kvaf")
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(config["kvaf_python_dir"]):
                raise ValueError("KVAF already imported from another runtime bundle")
        sys.path.insert(0, str(config["kvaf_python_dir"]))
        torch, device = self.torch, self.device
        placeholders = torch.zeros((self.batch, TOPK), dtype=torch.int32, device=device)
        counts = torch.zeros(self.batch, dtype=torch.int32, device=device)
        entries = torch.arange(self.batch, dtype=torch.int32, device=device)
        budgets = torch.full((self.batch,), TOPK, dtype=torch.int32, device=device)
        lengths = torch.full((self.batch,), self.history, dtype=torch.int32, device=device)
        # Existing class enforces 128-block PUT chunks and quarantines uncertain
        # PUT completion. Its Q1 temporary allocation is unused but retained.
        self.session = self.session_factory.open(
            torch,
            device,
            self.batch,
            self.history,
            TOPK,
            config["server"],
            config["get_kernel"],
            config["put_kernel"],
            placeholders,
            placeholders,
            counts,
            entries,
            budgets,
            lengths,
            temporary_only=True,
            put_chunk_blocks=PUT_CHUNK_BLOCKS,
            npu_core_count=workers,
        )
        row_tokens = self.geometry["row_tokens"]
        initial = torch.full((self.batch, row_tokens, TOKEN_BYTES), GUARD_BYTE, dtype=torch.uint8)
        initial[:, : self.resident_budget].copy_(self.session.cpu_payload[:, : self.resident_budget])
        self.initial_output = initial.to(device)
        self.output = self.initial_output.clone()
        self.cache_map = torch.full((self.batch, self.history), -1, dtype=torch.int32, device=device)
        self.cache_map[:, : self.resident_budget] = torch.arange(self.resident_budget, dtype=torch.int32, device=device)
        # Prepared access-table addresses name DDR blocks, not individual KV.
        block_addresses = unsigned_address_view(self.session.device_table.addresses, torch)
        self.ddr_block_addresses = block_addresses
        self.ddr_valid = self.session.device_table.valid
        blocks = block_addresses.detach().cpu().reshape(-1).tolist()
        valid = self.ddr_valid.detach().cpu().reshape(-1).tolist()
        expected = self.batch * self.geometry["blocks_per_row"]
        if len(blocks) < expected or len(valid) < expected or not all(valid[:expected]):
            raise ValueError("remote access table incomplete")
        addresses = [
            [
                int(blocks[row * self.geometry["blocks_per_row"] + token // BLOCK_TOKENS])
                + (token % BLOCK_TOKENS) * TOKEN_BYTES
                for token in range(self.history)
            ]
            for row in range(self.batch)
        ]
        if any(not 0 < address < 2**64 for row in addresses for address in row):
            raise ValueError("invalid remote token address")
        self.remote_token_addresses = torch.tensor(addresses, dtype=torch.uint64, device=device)
        self.keepalive.extend(
            [
                placeholders,
                counts,
                entries,
                budgets,
                lengths,
                self.initial_output,
                self.output,
                self.cache_map,
                self.remote_token_addresses,
                block_addresses,
                self.ddr_valid,
            ]
        )
        torch.npu.synchronize()

    def acquire(self):
        if self.closed or self.lease is not None or self.in_flight:
            raise ValueError("invalid real-fetch lease transition")
        self.torch.npu.synchronize()
        self.lease_stream = self.torch.npu.current_stream()
        # KVAF validates bytes / shape[0] == token_bytes. Keep the convenient
        # [B, row_slots, bytes] owner layout, but register a zero-copy slot view.
        self.leased_output = self.output.reshape(self.geometry["output_tokens"], TOKEN_BYTES)
        if self.leased_output.data_ptr() != self.output.data_ptr():
            raise ValueError("real-fetch lease view must not copy output storage")
        self.lease = self.session.runtime.acquire_fused_batch_get_context(self.leased_output, self.lease_stream)
        for name in (
            "device_channel_entity",
            "device_queue_state",
            "device_core_status",
            "output_address",
            "output_bytes",
            "worker_count",
            "jettys_per_core",
        ):
            if type(self.lease.get(name)) is not int or self.lease[name] <= 0:
                raise ValueError(f"invalid lease field: {name}")
        if (
            self.lease["output_address"] != self.output.data_ptr()
            or self.lease["output_bytes"] != self.geometry["output_bytes"]
            or self.lease.get("token_bytes") != TOKEN_BYTES
            or self.lease.get("tokens_per_block") != BLOCK_TOKENS
        ):
            raise ValueError("leased buffer geometry mismatch")
        return dict(self.lease)

    def mark_launched(self, *tensors):
        if self.closed or self.lease is None:
            raise ValueError("launch requires a live queue lease")
        self.keepalive.extend(tensors)
        self.in_flight = True


    def build_device_args(self, abi_library: Path):
        """Serialize real header ABI, one independent single-jetty view/AIV.

        The lease has two jettys per AIC worker. Each physical AIV gets exactly
        one of them and private core status; no locks or shared SQ producer.
        AIV-to-view indexing must use physical AIV identity, not batch row.
        """
        if self.lease is None or self.in_flight:
            raise ValueError("build args requires an idle live lease")
        lease = self.lease
        if lease["jettys_per_core"] != 2:
            raise ValueError("refactor requires two independent jettys per AIC worker")
        lib = ctypes.CDLL(str(Path(abi_library).resolve()))
        sizes = {}
        for name in ("args", "status", "entity", "queue"):
            fn = getattr(lib, f"refactor_fetch_{name}_bytes")
            fn.argtypes, fn.restype = [], ctypes.c_uint32
            sizes[name] = int(fn())
            if sizes[name] <= 0:
                raise ValueError(f"invalid native ABI size: {name}")
        pack = lib.refactor_fetch_pack_args
        pack.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        pack.restype = ctypes.c_uint32
        torch = self.torch
        owners = lease["worker_count"] * 2
        self.core_status = torch.zeros((owners, sizes["status"]), dtype=torch.uint8, device=self.device)
        # Not consumed by prepared-WQE submission, but valid storage keeps the
        # native args complete and available for final descriptor operations.
        self.descriptor_scratch = torch.zeros((4, QUERIES * TOPK), dtype=torch.int32, device=self.device)
        stride = QUERIES * TOPK * 4
        blobs = []
        for owner in range(owners):
            values = [
                self.ddr_block_addresses.data_ptr(),
                self.ddr_valid.data_ptr(),
                self.descriptor_scratch.data_ptr(),
                self.descriptor_scratch.data_ptr() + stride,
                self.descriptor_scratch.data_ptr() + 2 * stride,
                self.descriptor_scratch.data_ptr() + 3 * stride,
                self.output.data_ptr(),
                self.core_status.data_ptr() + owner * sizes["status"],
                lease["device_channel_entity"] + owner * sizes["entity"],
                lease["device_queue_state"] + owner * sizes["queue"],
                self.batch * self.geometry["blocks_per_row"],
                QUERIES * TOPK,
                self.geometry["output_tokens"],
                TOKEN_BYTES,
                BLOCK_TOKENS,
                1,
                1,
            ]
            fields = (ctypes.c_uint64 * len(values))(*values)
            output = ctypes.create_string_buffer(sizes["args"])
            result = pack(output, sizes["args"], fields, len(values))
            if result:
                raise ValueError(f"native ABI serialization failed: {result}")
            blobs.extend(output.raw)
        self.device_args = torch.tensor(blobs, dtype=torch.uint8, device=self.device).reshape(owners, sizes["args"])
        self.abi_library = lib
        self.abi_sizes = sizes
        self.keepalive.extend([self.core_status, self.descriptor_scratch, self.device_args, lib])
        self.torch.npu.synchronize()
        return self.device_args

    def mark_completed(self, *, completion_verified: bool):
        # Caller must inspect device CQ/drain status, not merely stream.sync.
        if not completion_verified:
            self.quarantine("real_fetch_completion_unverified", "device CQ status failed")
        self.in_flight = False

    def reset(self):
        if self.in_flight:
            self.quarantine("real_fetch_reset_inflight", "completion not verified")
        self.output.copy_(self.initial_output)


    def release(self):
        if self.in_flight:
            self.quarantine("real_fetch_release_inflight", "completion not verified")
        if self.lease is not None:
            try:
                self.torch.npu.synchronize()
                self.session.runtime.release_fused_batch_get_context(self.lease, self.lease_stream)
                self.lease = None
            except BaseException as error:
                self.quarantine("real_fetch_release_failed", error)

    def quarantine(self, reason, error):
        quarantine_all(self, reason, error)

    def close(self):
        if self.closed:
            return
        if self.in_flight:
            self.quarantine("real_fetch_close_inflight", "pending communication")
        try:
            self.release()
            if self.session is not None:
                for graph in self.graphs:
                    graph.reset()
                self.graphs.clear()
                self.session.close()
                self.session = None
        except BaseException as error:
            self.quarantine("real_fetch_close_failed", error)
        self.closed = True
        self.keepalive.clear()


class NativeMtp:
    def __init__(self, torch, device, owner, fixture, *, source_tokens=SOURCE, padded_tokens=PADDED_TOKENS, token_bytes=TOKEN_BYTES):
        if source_tokens < TOPK or padded_tokens < source_tokens or padded_tokens % 128:
            raise ValueError("native_mtp_invalid_source_geometry")
        if token_bytes not in (656, 2624):
            raise ValueError("native_mtp_invalid_token_width")
        self.token_bytes = token_bytes
        self.source_tokens, self.padded_tokens = source_tokens, padded_tokens
        self.torch, self.owner, self.fixture = torch, owner, fixture
        self.batch, self.hot = owner.batch, owner.resident_budget
        self.initial = owner.cache_map.clone()
        self.forward = self.initial.clone()
        self.reverse_initial = torch.arange(self.hot, dtype=torch.int32, device=device).repeat(self.batch, 1)
        for row, mapping in enumerate(self.initial.cpu().tolist()):
            inverse = [0] * self.hot
            for token, slot in enumerate(mapping):
                if slot >= 0:
                    inverse[slot] = token
            self.reverse_initial[row] = torch.tensor(inverse, dtype=torch.int32, device=device)
        self.reverse = self.reverse_initial.clone()
        self.cursors = torch.zeros(self.batch, dtype=torch.int32, device=device)
        self.entries = torch.arange(self.batch, dtype=torch.int32, device=device)
        self.budgets = torch.full((self.batch,), self.hot, dtype=torch.int32, device=device)
        self.preferred = torch.arange(TOPK, dtype=torch.int32, device=device).repeat(self.batch, 1)
        bitmap = torch.zeros((self.batch, 512), dtype=torch.int32)
        bitmap[:, : TOPK // 32] = -1
        self.bitmap = bitmap.to(device)
        self.block_table = torch.arange(self.batch * self.hot // 128, dtype=torch.int32, device=device).reshape(
            self.batch, -1
        )
        self.enabled = torch.zeros(1, dtype=torch.int32, device=device)
        self.destinations = torch.empty((self.batch * QUERIES, 1, TOPK), dtype=torch.int32, device=device)
        self.counts = torch.empty(self.batch + 1, dtype=torch.int32, device=device)
        self.indices = torch.empty(self.batch * QUERIES * TOPK, dtype=torch.int32, device=device)
        self.offsets = torch.empty_like(self.indices)
        self.outputs = torch.empty_like(self.indices)
        self.resident_lengths = torch.empty(self.batch, dtype=torch.int32, device=device)
        self.profiling = torch.empty((self.batch, 8, 32), dtype=torch.int64, device=device)
        self.initial_payload = (
            torch.stack(
                [owner.session.cpu_payload[row, ids] for row, ids in enumerate(self.reverse_initial.cpu().long())]
            )
            .reshape(-1, self.token_bytes)
            .to(device)
        )
        self.output = self.initial_payload.clone()
        self.total = self.counts[self.batch :]

    def reset(self):
        self.forward.copy_(self.initial)
        self.reverse.copy_(self.reverse_initial)
        self.cursors.zero_()
        self.output.copy_(self.initial_payload)

    def launch(self, selected):
        self.prepare(selected)
        self.fetch()

    def prepare(self, selected):
        if self.owner.lease is not None:
            raise ValueError("native_mtp_requires_unleased_batch_get_queues")
        kvip.lidu(
            selected,
            self.fixture["actual_query"],
            self.entries,
            self.forward,
            self.reverse,
            self.cursors,
            self.budgets,
            self.fixture["actual_key"],
            self.preferred,
            self.bitmap,
            self.block_table,
            self.fixture["actual_key"],
            self.enabled,
            self.destinations,
            self.counts,
            self.indices,
            self.offsets,
            self.outputs,
            self.resident_lengths,
            self.profiling,
            0,
            1,
            self.batch,
            self.padded_tokens // 128,
        )

    def fetch(self):
        if self.owner.lease is not None:
            raise ValueError("native_mtp_requires_unleased_batch_get_queues")
        self.owner.session.batch_get_attempted = True
        if self.token_bytes != TOKEN_BYTES:
            # Unchanged LIDU writes offsets in single-layer C8 bytes.
            self.offsets.mul_(self.token_bytes // TOKEN_BYTES)
        self.owner.session.runtime.batch_get(
            self.owner.session.device_table.addresses,
            self.owner.session.device_table.valid,
            self.indices,
            self.offsets,
            self.total,
            self.outputs,
            self.output,
            self.torch.npu.current_stream(),
            False,
        )

    def verify(self, selected):
        ids = selected.detach().cpu().reshape(self.batch, QUERIES * TOPK).long()
        # Native MTP compacts hits/misses and does not preserve query input
        # order in destinationSlots. Resolve selected tokens through its
        # updated forward map, then check the exported slot set separately.
        exported = self.destinations.cpu().reshape(self.batch, QUERIES, TOPK).long()
        forward = self.forward.cpu().long()
        reference = getattr(self.owner.session, 'reference', None)
        output = (self.output if reference is not None else self.output.cpu()).reshape(self.batch, self.hot, self.token_bytes)
        counts = self.counts.cpu().tolist()
        if any(count < 0 or count > QUERIES * TOPK for count in counts[:-1]) or counts[-1] != sum(counts[:-1]):
            raise ValueError(f"native_mtp_counts_invalid:{counts}")
        for row in range(self.batch):
            slots = forward[row, ids[row]]
            if bool(((ids[row] < 0) | (ids[row] >= self.source_tokens) | (slots < 0) | (slots >= self.hot)).any()):
                raise ValueError(f"native_mtp_location_range:row={row}")
            for q in range(QUERIES):
                expected_slots = slots[q * TOPK : (q + 1) * TOPK]
                if not self.torch.equal(expected_slots.sort().values, exported[row, q].sort().values):
                    raise ValueError(f"native_mtp_exported_slots_mismatch:row={row}:query={q}")
            if reference is not None:
                actual = output[row, slots.to(output.device)]
                expected = reference[row, ids[row].to(output.device)]
            else:
                actual = output[row, slots]
                expected = self.owner.session.cpu_payload[row, ids[row]]
            if not self.torch.equal(actual, expected):
                raise ValueError(f"native_mtp_payload_mismatch:row={row}")
        return counts


class DirectCacheUpdate(DirectIndexUpdate):
    def __init__(
        self,
        torch,
        keys,
        scales,
        addresses,
        output,
        batch,
        history,
        hot,
        *,
        topk,
        locations,
        rotating,
    ):
        self.torch, self.device = torch, keys.device
        self.batch, self.hot, self.history = batch, hot, history
        self.max_swaps = rotating.changes.shape[1]
        self.rotating = rotating
        self.identity = torch.arange(history, device=self.device).expand(batch, -1)
        self.logical = self.identity.clone()
        self.last_slots = locations.reshape(batch, -1)
        self.status = torch.zeros(batch, dtype=torch.int32, device=self.device)
        self.inputs = (
            keys,
            scales,
            addresses,
            self.logical,
            rotating.change_slots,
            rotating.changes,
            self.status,
        )
        row_owners = torch.arange(batch, dtype=torch.int32, device=self.device)
        self.unpack_indices = torch.arange(batch * 4, dtype=torch.int64, device=self.device)
        self.output_valid = torch.ones(batch * 4, dtype=torch.int64, device=self.device)
        super().__init__(
            torch,
                keys=keys.view(batch, history, 128),
            scales=scales.view(batch, history),
            addresses=addresses,
            logical_ids=self.logical,
            row_owners=row_owners,
            rotating=rotating,
            status=self.status,
            query_inputs=SimpleNamespace(unpack_indices=self.unpack_indices, output_valid=self.output_valid),
        )

    def set_order(self, orders):
        """Forward input preparation only; consume the existing service LRU."""
        candidates = np.asarray(orders, dtype=np.int32)[:, :2048].copy()
        if candidates.shape != (self.batch, 2048):
            raise ValueError("Expected 2048 existing LRU candidates per request")
        reserved = np.full((self.batch, self.hot), -1, dtype=np.int32)
        for row, slots in enumerate(candidates):
            if np.any(slots < 0) or np.any(slots >= self.hot) or len(np.unique(slots)) != 2048:
                raise ValueError("LRU candidates must be distinct hot slots")
            reserved[row, slots] = np.arange(2048, dtype=np.int32)
        self.rotating.lru_slots.copy_(self.torch.from_numpy(candidates))
        self.rotating.reserved_hot.copy_(self.torch.from_numpy(reserved))

    def check_capacity(self):
        status = self.status.cpu().tolist()
        if any(status):
            raise RuntimeError(f"Invalid producer Index permutation: {status}")


def prepare_native_bitmap(torch, native):
    """Adapt the historical fixture before graph capture and timing."""
    native.bitmap = torch.zeros((native.batch, NATIVE_BITMAP_WORDS), dtype=torch.int32, device=native.bitmap.device)


def seed_native_bitmap(torch, native, row, bitmap):
    """Copy LRU's capacity-sized bitmap into the fixed service ABI row."""
    native.bitmap[row].zero_()
    native.bitmap[row, : len(bitmap)].copy_(torch.as_tensor(bitmap, device=native.bitmap.device))


def run_urma(args, run_dir):
    from correctness import check_sfa
    from kvip.cache.lru import SlotLRU
    from kvip.operators.lifetch import binding as launch_module
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry
    from kvip.runtime.build import build_libraries
    from kvip.runtime.device import query_device_resources
    from workloads import payload, prepare_full_chain_fixture

    torch, registration, _ = _load_native_operators("c8_offload")
    import torch_npu

    registration.require_a5_dsa_registration(("lidu_fused_batch_get_prepare_mtp_out",))
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    resources = query_device_resources(args.cann, args.device, 0)
    aics = resources["effective_aic"]

    class FiniteSession(NativeFetchSession):
        @classmethod
        def open(cls, *pos, **kw):
            kw["shared_pool"] = getattr(args, "shared_pool", None)
            kw["payload_factory"] = lambda t, b, h: t.from_numpy(np.stack([payload(i, np.arange(h)) for i in range(b)]))
            return super().open(*pos, **kw)

    summaries = []
    failed = False
    original_args = args
    cases = [
        (b, args.history, args.resident, args.index_heads, args.intermediate_fetches)
        for b in map(int, args.batches.split(","))
    ]
    shared_build = run_dir / "universal-build"
    built_digest = None
    for case, (batch, history, resident, heads, windows) in enumerate(cases):
        args = SimpleNamespace(
            **{
                **vars(original_args),
                "history": history,
                "resident": resident,
                "index_heads": heads,
                "intermediate_fetches": windows,
            }
        )
        folder = run_dir / f"case{case}-b{batch}-prototype-graph"
        folder.mkdir()
        stage = "fixture preparation"
        progress(run_dir, f"BS={batch}: {stage}; logs={folder}")
        with native_log(folder / "native.log"):
            common, data, build = (folder / "fixture", folder / "input", shared_build)
            pool = getattr(args, 'shared_pool', None)
            compact = getattr(args, 'mode', None) in ('bench', 'diagnose')
            fixture_key = ('urma', compact, args.history, args.resident, args.miss_percent, args.index_heads, getattr(args, 'seed', 20261104))
            if pool is not None and fixture_key in pool.fixtures:
                cached_path, fixture_info = pool.fixtures[fixture_key]
                common.symlink_to(cached_path.resolve(), target_is_directory=True)
            else:
                from workloads import prepare_compact_fixture
                prepare_fixture = prepare_compact_fixture if compact else prepare_full_chain_fixture
                fixture_info = prepare_fixture(
                    common, args.history, args.resident, args.miss_percent,
                    index_heads=args.index_heads, seed=getattr(args, 'seed', 20261104))
                if pool is not None:
                    pool.fixtures[fixture_key] = (common, fixture_info)
            geometry = li_geometry(batch, "auto", args.intermediate_fetches, aics, 0)
            batch_inputs(common, data, batch, args.history, geometry["rotating_aic"])
            weights = np.fromfile(data / "arg2.bin", np.float32)
            (weights.view(np.uint32) >> 16).astype(np.uint16).tofile(data / "native_weights_bf16.bin")
            options = dict(
                RUN_MODE="npu",
                SOC_VERSION=args.soc,
                ASCEND_CANN_PACKAGE_PATH=args.cann,
                KVAF_INCLUDE=find_headers(args.kvaf_root),
            )
            options.update(KVIP_FUSE_QUERY_WORK="ON", KVIP_OUTPUT_WORKER="ON")
            if not shared_build.exists():
                progress(run_dir, f"BS={batch}: resolve shared LIFetch build cache; logs={folder}")
                (library,) = build_libraries(
                    cache_dir=args.kvaf_root / "build-npu/lifetch",
                    cann=args.cann, headers=options["KVAF_INCLUDE"], soc=args.soc)
                shared_build.symlink_to(library.parent.resolve(), target_is_directory=True)
            progress(run_dir, f"BS={batch}: REUSE LIFetch build={shared_build.resolve()}; logs={folder}")
            import hashlib

            # Fail before allocating/populating DDR if shared dependencies cannot load.
            load_lifetch_library(build / "libv9_bridge.so")

            digest = hashlib.sha256((build / "libv9_bridge.so").read_bytes()).hexdigest()
            if built_digest is not None and digest != built_digest:
                raise AssertionError("Binary changed between runtime shapes")
            built_digest = digest
            progress(
                run_dir,
                f"UNIVERSAL binary_sha256={digest} batch={batch} H={history} R={resident} G={heads} windows={windows}",
            )
            queue_views = batch * (final_aivs(batch, "auto", aics, 0) + 1)
            if args.q_workers:
                # Two contiguous private Jettys for each (Q, worker), after the
                # prefetch pairs; the original final views may be reused.
                queue_views = max(queue_views, 2 * batch + 8 * args.q_workers)
            workers = (queue_views + 1) // 2
            progress(run_dir, f"BS={batch}: DDR Put and transport initialization; logs={folder}")
            owner = RealFetchSession.open(
                torch,
                device,
                session_factory=FiniteSession,
                batch=batch,
                history=args.history,
                resident=args.resident,
                server=args.kvaf_server,
                kvaf_root=args.kvaf_root,
                workers=workers,
            )
            try:
                stage = "graph A/B and correctness checks" if getattr(args, "mode", None) == "bench" else "graph A/B, AIV maintenance and stateful checks"
                progress(run_dir, f"BS={batch}: {stage}; logs={folder}")
                report = run_urma_case(
                    args,
                    batch,
                    folder,
                    torch,
                    torch_npu,
                    device,
                    owner,
                    data,
                    build,
                    native_li,
                    NativeMtp,
                    launch_module,
                    SlotLRU,
                    check_sfa,
                    aics,
                )
                failed |= report.get("correctness", "PASS") != "PASS"
                report.update(
                    batch=batch, history=args.history, resident=args.resident, miss_percent=args.miss_percent,
                    index_heads=args.index_heads, seed=args.seed if hasattr(args, 'seed') else 20261104,
                    fixture=fixture_info,
                    resources=resources,
                    resource_geometry=geometry,
                    final_workers=final_aivs(batch, "auto", aics, 0),
                    build_options={k: str(v) for k, v in options.items()},
                )
                (folder / "report.json").write_text(json.dumps(report, indent=2))
            except BaseException as error:
                record_case_failure(folder, error)
                (folder / "error.log").write_text(traceback.format_exc())
                raise
            finally:
                owner.close()
        times = report["timing"]
        line = (
            f"binary_sha256={built_digest} q_workers={args.q_workers}\nBS={batch} prototype {report.get('correctness', 'PASS')} index_heads={args.index_heads} windows={geometry['intermediate_fetches']} requested_windows={args.intermediate_fetches} history={args.history} hot={args.resident} topk=2048 Q=4 graph=True\n"
            + "graph_p50_us="
            + ", ".join((f"{k}:{v['p50_us']:.2f}" for k, v in times.items()))
            + f"\nfull_speedup={times['native_chain']['p50_us'] / times['prototype_with_update']['p50_us']:.2f}x"
            + "; full=init + LI/Fetch/resolve + service output mapping + Index update; fixture restore/verification and SFA/LRU CPU excluded\n"
            + f"SFA_max_abs={report['sfa']['max_abs']:.6g}; stateful_rounds={report['stateful_rounds']} "
            + f"native_unique_misses={report['native_misses_per_request']}"
        )
        line += "\nfixture=" + str(fixture_info["hot_background"]) + "; prefetch_predicate=score>resident_threshold"
        line += "; actual_miss_pct_per_query=" + json.dumps(
            [round(count * 100 / 2048, 4) for count in fixture_info["final_misses_per_query"]]
        )
        native_reads = sum(report["native_misses_per_request"])
        line += f"\nactual_reads native: total={native_reads} unused=0 (final selected misses)"
        for label, samples in report.get("split_samples", {}).items():
            ranges = {key: [min(x[key] for x in samples), max(x[key] for x in samples)]
                      for key in ("final_reads", "nonconflicting_reads", "conflicting_reads")}
            line += f"\nfinal_split {label}: min_max=" + json.dumps(ranges)
        for label, samples in report.get("read_samples", {}).items():
            ranges = {key: [min(row[key] for row in samples), max(row[key] for row in samples)] for key in samples[0]}
            line += f"\nactual_reads {label}: min_max=" + json.dumps(ranges)
        if report.get("full_trace"):
            line += "\n" + core_summary(report["full_trace"], label="full")
        if report.get("cache_update"):
            line += "\ncache_update=" + json.dumps(report["cache_update"])
        if report.get("wall_costs"):
            line += "\nBENCH WALL COSTS seconds=" + json.dumps(report["wall_costs"])
        summaries.append(line)
        (run_dir / "cases.log").write_text("\n".join(summaries) + "\n")
        print(line, flush=True)
    progress(run_dir, "all prototype graph cases completed")
    return int(failed)


def run_urma_case(
    args,
    b,
    folder,
    torch,
    torch_npu,
    device,
    owner,
    data,
    build,
    native_li,
    NativeMtp,
    launch_module,
    SlotLRU,
    check_sfa,
    aics,
):

    h, r = (args.history, args.resident)
    install_remote_addresses(data, owner.remote_token_addresses.cpu().numpy(), b, h)
    arguments = [
        torch.from_numpy(np.fromfile(data / ("native_weights_bf16.bin" if i == 2 else f"arg{i}.bin"), np.uint8)).to(
            device
        )
        for i in range(14)
    ]
    arguments[1] = arguments[1].repeat(b)
    arguments[4] = arguments[4].repeat(b)
    arguments[7].view(torch.int32).copy_(torch.arange(b * h // 128, device=device, dtype=torch.int32))
    fixture = dict(
        query=arguments[0].view(torch.float8_e4m3fn).reshape(b, 4, args.index_heads, 128),
        key=arguments[1].view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
        actual_query=torch.arange(1, b + 1, device=device, dtype=torch.int32) * 4,
        li_query_lengths=torch.full((b,), 4, device=device, dtype=torch.int32),
        actual_key=torch.full((b,), h, device=device, dtype=torch.int32),
        block_table=arguments[7].view(torch.int32).reshape(b, -1),
    )
    fixture["native_inputs"] = dict(
        weights=arguments[2].view(torch.bfloat16).reshape(b, 4, args.index_heads),
        query_scale=arguments[3].view(torch.float32).reshape(b, 4, args.index_heads),
        key_scale=arguments[4].view(torch.float32).reshape(-1, 128, 1),
    )
    stream = torch.npu.current_stream(device)
    capture = torch.npu.Stream(device=device)
    native = NativeMtp(torch, device, owner, fixture, source_tokens=h, padded_tokens=h)
    prepare_native_bitmap(torch, native)
    state = {}
    correctness = CorrectnessRecorder()
    boundary_phases = {}
    completed_timings = {}
    initial_selected = native_li(torch_npu, fixture).cpu().reshape(b, -1).numpy()
    logical_scores = np.load(folder / "fixture/logical_scores.npy")
    physical_order = np.load(folder / "fixture/physical_to_logical.npy")
    scores = logical_scores[:, physical_order]
    compact = getattr(args, 'mode', None) in ('bench', 'diagnose')
    compact_scores = scores
    from correctness import validate_topk
    if compact:
        validate_topk(initial_selected, compact_scores)
    else:
        expected_ids = np.stack([np.flatnonzero(row >= np.partition(row, -2048)[-2048]) for row in scores])
        for row in initial_selected.reshape(b, 4, 2048):
            if not np.array_equal(np.sort(row, axis=-1), expected_ids):
                raise AssertionError("native LI differs from independent fixture score oracle")
    initial_orders = []
    policies = []
    for row in range(b):
        policy = SlotLRU()
        policy.resize(r, r)
        policy.commit(initial_selected[row][initial_selected[row] < r])
        order, slot = ([], int(policy.head[0]))
        for _ in range(r):
            order.append(slot)
            slot = int(policy.next[0, slot])
        initial_orders.append(order)
        policies.append(policy)
        native.preferred[row].copy_(torch.from_numpy(policy.slots[0]).to(device))
        seed_native_bitmap(torch, native, row, policy.bitmap[0])

    def native_call():
        state["native_topk"] = native_li(torch_npu, fixture)
        native.launch(state["native_topk"])

    def native_check():
        counts = native.verify(state["native_topk"])
        expected = [len(set(int(t) for t in ids if t >= r)) for ids in initial_selected]
        if counts[:-1] != expected:
            raise AssertionError(f"native Get work differs from logical misses: {counts[:-1]} != {expected}")
        state["native_misses"] = counts[:-1]

    bench_only = getattr(args, "mode", None) == "bench"
    wall_costs = {}
    full_trace = {}
    read_samples = {}
    split_samples = {}
    fused_read_counts = []
    sampling_protocol = dict(
        version=2, transport='urma', shared_pool=owner.session.shared_pool is not None,
        warmup=args.warmup, iterations=args.iterations,
        input=dict(batch=b, history=h, resident=r, heads=args.index_heads,
                   seed=args.seed, miss_percent=args.miss_percent,
                   windows=args.intermediate_fetches, prefetch_tokens=args.prefetch_tokens),
        timed_region='NPU events around one graph replay; reset outside',
        validation='full before sampling and final sample; completion every sample' if compact else 'full every sample',
        ordering='serial timing, fused timing, then diagnostic profiling',
    )
    atomic_json(folder / 'sampling_protocol.json', sampling_protocol)

    def capture_graph(call, reset):
        reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        owner.graphs.append(graph)
        with torch.npu.graph(graph, stream=capture):
            call()
        capture.synchronize()
        return graph

    def check_native_completion():
        runtime = owner.session.runtime
        probe = runtime.acquire_fused_batch_get_context(native.output, stream)
        runtime.release_fused_batch_get_context(probe, stream)

    def measure(call, reset, check, *, timing_name, completion_check=None, keep_graph=False):
        costs = dict(reset_s=0., validation_s=0., replay_wait_s=0., completion_s=0., reads_s=0.)
        raw_check, raw_reset = check, reset
        def check():
            began = time.perf_counter()
            try:
                return correctness.check(state["check_phase"], raw_check)
            finally:
                costs['validation_s'] += time.perf_counter() - began
        def reset():
            began = time.perf_counter()
            raw_reset()
            stream.synchronize()
            costs['reset_s'] += time.perf_counter() - began
        state["check_phase"] = f"timing {timing_name}"
        reset()
        call()
        stream.synchronize()
        check()
        graph = capture_graph(call, reset)
        graph.replay()
        stream.synchronize()
        check()
        samples = []
        count_reads = timing_name == 'prototype_with_update'
        for step in range(args.warmup + args.iterations):
            reset()
            stream.synchronize()
            start, end = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            began = time.perf_counter()
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            costs['replay_wait_s'] += time.perf_counter() - began
            if step >= args.warmup:
                samples.append(start.elapsed_time(end) * 1000)
            validate = completion_check is None or step == args.warmup + args.iterations - 1
            if validate:
                check()
            else:
                began = time.perf_counter()
                completion_check()
                costs["completion_s"] += time.perf_counter() - began
            if count_reads and step >= args.warmup:
                fused_read_counts.append(state['fused_read_count'])
            if count_reads and step >= args.warmup and validate:
                began = time.perf_counter()
                split_samples.setdefault(timing_name, []).append(state["completion"])
                read_samples.setdefault(timing_name, []).append(
                    actual_reads(
                        rotating.mapping.cpu().numpy(),
                        topk.cpu().numpy(),
                        rotating.rows.cpu().numpy().view(np.uint32),
                        r,
                    )
                )
                costs['reads_s'] += time.perf_counter() - began
        wall_costs[timing_name] = costs
        atomic_json(folder / 'bench_wall_costs.json', wall_costs)
        print(f'BENCH WALL COSTS {timing_name}: {json.dumps(costs)}', flush=True)
        completed_timings[timing_name] = timing_summary(samples)
        atomic_json(folder / 'completed_timings.json', dict(
            batch=b, miss_percent=args.miss_percent, timing=completed_timings,
            correctness=correctness.status, failures=correctness.failures, token_bytes=TOKEN_BYTES,
            native_misses_per_request=state['native_misses'], fused_read_counts=fused_read_counts))
        if keep_graph:
            return graph
        graph.reset()
        owner.graphs.remove(graph)

    native.reset()
    native_call()
    stream.synchronize()
    native_check()
    native_completion = check_native_completion if compact else None
    measure(native_call, native.reset, native_check, timing_name="native_chain", completion_check=native_completion)
    reference = state["native_topk"].cpu().reshape(b, 4, 2048).numpy()
    owner.acquire()
    rotating = launch_module.LIFetchBinding(
        torch,
        build / "libv9_bridge.so",
        arguments,
        owner,
        batch=b,
        history=h,
        resident=r,
        aic_count=aics,
        intermediate_fetches=args.intermediate_fetches,
        final_workers=0,
        prefetch_limit=args.prefetch_tokens,
        prefetch_rank_start=getattr(args, "prefetch_rank_start", 1024),
        output_stride=owner.output.shape[1],
    )
    topk = arguments[9].view(torch.int32).reshape(b, 4, 2048)
    prefix = 4 * 2048 * 6 + 1024 + 4 * h * 2
    tail = h * 8 + 8192 * 64 + 4 * 128
    addresses = arguments[13][b * prefix :].view(torch.int64).reshape(b, tail // 8)[:, :h]
    update = DirectCacheUpdate(
        torch,
        arguments[1].reshape(b, h, 128),
        arguments[4].view(torch.float32).reshape(b, h),
        addresses,
        owner.output,
        batch=b,
        history=h,
        hot=r,
        topk=topk,
        locations=rotating.locations,
        rotating=rotating,
    )
    pipeline = StreamingForward(rotating, update, torch, args.prefetch_tokens, fused_final=True, q_workers=args.q_workers)
    pipeline.legacy_layout = False
    pipeline.parallel_finalize = True
    pipeline.prepare_metadata = True
    pipeline.set_diagnostics(False)
    counter_clock_mhz = None
    saved = [arguments[i].clone() for i in (1, 4, 13)]
    update.set_order(initial_orders)
    order_snapshot = (rotating.lru_slots.clone(), rotating.reserved_hot.clone())

    def reset():
        if owner.in_flight:
            raise RuntimeError("previous replay completion not checked")
        for i, seed in zip((1, 4, 13), saved):
            arguments[i].copy_(seed)
        update.logical.copy_(update.identity)
        state.pop("before_logical", None)
        rotating.hot_valid.fill_(1)
        if compact:
            rotating.lru_slots.copy_(order_snapshot[0])
            rotating.reserved_hot.copy_(order_snapshot[1])
        else:
            update.set_order(initial_orders)
        owner.reset()
        owner.mark_launched()

    def full():
        pipeline()

    def verify(updated=False, persistent=True, completion_only=False):
        rows = rotating.rows.cpu().numpy()
        posts = rotating.final_posts.cpu().numpy()
        statuses = rotating.row_status.cpu().numpy()
        cq = [status.cpu().view(torch.uint64).numpy() for status in (rotating.prefetch_status, rotating.final_status)]
        failures = completion_failures(rows, posts, statuses, *cq)
        if failures:
            phase = state.get("check_phase", "graph replay")
            path = folder / "completion_failure.json"
            try:
                bounds = selected_bounds(topk.cpu().numpy(), h)
            except Exception as exc:
                bounds = dict(error=f"{type(exc).__name__}: {exc}")
            try:
                scores = score_rows(arguments[12][:4 * h * 2].cpu().numpy(), h)
            except Exception as exc:
                scores = dict(error=f"{type(exc).__name__}: {exc}")
            path.write_text(json.dumps(dict(phase=phase, requests=failures, selected_bounds=bounds,
                                            first_request_score_rows=scores), indent=2))
            print(f"COMPLETION FAILURE: phase={phase}; first={json.dumps(failures[0])}; full={path}", flush=True)
            print(f"LI OUTPUT BOUNDS: {json.dumps(bounds)}", flush=True)
            print(f"LI SCORE WORDS request=0: {json.dumps(scores)}", flush=True)
            raise RuntimeError(
                f"completion failed phase={phase}; bad_requests={len(failures)}; "
                f"first={json.dumps(failures[0])}; full={path}"
            )
        if not updated:
            destinations = rotating.destinations.cpu().numpy().reshape(-1)
            lru_slots = rotating.lru_slots.cpu().numpy()
            for row_index in range(b):
                # Compare physical hot slots, independent of resolver's inverse map.
                used_slots = set(lru_slots[row_index, : int(rows[row_index, 8])].tolist())
                for post in posts[row_index]:
                    count, start, safe = int(post[0]), int(post[4]), int(post[5])
                    if safe > count:
                        raise AssertionError("Invalid nonconflicting descriptor count")
                    for index, destination in enumerate(destinations[start : start + count]):
                        slot = int(destination) % owner.output.shape[1]
                        if (slot in used_slots) != (index >= safe):
                            raise AssertionError(
                                f"Final split mismatch: request={row_index} descriptor={start + index}"
                            )
        state["completion"] = dict(
            final_reads=int(statuses[:, 1].astype(np.uint64).sum()),
            nonconflicting_reads=int(posts[:, :, 5].astype(np.uint64).sum()),
            conflicting_reads=int((posts[:, :, 0].astype(np.uint64) - posts[:, :, 5].astype(np.uint64)).sum()),
            native_deferred_rows=int(np.count_nonzero(statuses[:, 3])),
            completed_checkpoints=int(statuses[:, 4].astype(np.uint64).sum()),
        )
        state['fused_read_count'] = dict(prefetch=int(rows[:, 9].astype(np.uint64).sum()),
                                         final=state['completion']['final_reads'])
        owner.mark_completed(completion_verified=True)
        if updated:
            update.check_capacity()
        if completion_only:
            return
        selected = topk.cpu().numpy()
        if np.any(selected < 0) or np.any(selected >= h):
            raise RuntimeError("TopK contains out-of-range positions")
        logical = state.get("before_logical", np.broadcast_to(np.arange(h), (b, h)))
        ids = np.take_along_axis(logical, selected.reshape(b, -1), axis=1).reshape(b, 4, 2048)
        slots = rotating.locations.reshape(b, -1).cpu().long()
        if (slots < 0).any() or (slots >= (r if persistent else owner.output.shape[1])).any():
            raise RuntimeError("Fetch slot outside cache")
        state["ids"], state["slots"] = (ids, slots.numpy())
        if compact:
            validate_topk(ids, compact_scores)
        elif not np.array_equal(np.sort(ids, axis=-1), np.sort(reference, axis=-1)):
            raise AssertionError("TopK changed")
        if updated:
            if not np.array_equal(update.logical_output.cpu().numpy().reshape(b, 4, 2048), ids):
                raise AssertionError("Service logical TopK output differs from pre-update IDs")
            if not torch.equal(update.slot_output.cpu().reshape(b, -1).long(), slots):
                raise AssertionError("Service SFA slot output differs from Fetch destinations")
        reference_device = getattr(owner.session, 'reference', None)
        output = owner.output if reference_device is not None else owner.output.cpu()
        for row in range(b):
            if slots[row].min() < 0 or slots[row].max() >= (r if persistent else owner.output.shape[1]):
                raise AssertionError("slot outside cache")
            if reference_device is not None:
                actual = output[row, slots[row].to(device)]
                expected = reference_device[row, torch.from_numpy(ids[row].reshape(-1)).to(device)]
            else:
                actual = output[row, slots[row]]
                expected = owner.session.cpu_payload[row, ids[row].reshape(-1)]
            if not torch.equal(actual, expected):
                actual, expected = actual.cpu(), expected.cpu()
                output = output.cpu()
                mismatches = (actual != expected).reshape(actual.shape[0], -1).any(dim=1)
                positions = mismatches.nonzero().flatten()[:4].tolist()
                lru = rotating.lru_slots[row].cpu().tolist()
                changes = rotating.changes[row].cpu().tolist()
                details = []
                for pos in positions:
                    token, slot = (int(selected[row].reshape(-1)[pos]), int(slots[row, pos]))
                    ordinal = lru.index(slot) if slot in lru else -1
                    details.append(
                        dict(
                            query=pos // 2048,
                            topk_pos=pos % 2048,
                            token=token,
                            slot=slot,
                            victim_ordinal=ordinal,
                            slot_owner=changes[ordinal] if ordinal >= 0 else None,
                        )
                    )
                from diagnostics import summarize_mismatches
                report = summarize_mismatches(
                    row, selected[row], ids[row], slots[row].numpy(), output[row].numpy(),
                    owner.session.cpu_payload[row].numpy(), rotating.destinations.cpu().numpy(),
                    rotating.fallback_addresses.cpu().numpy(), posts[row], lru,
                    rotating.mapping[row].cpu().numpy(), owner.output.shape[1], r,
                    owner.remote_token_addresses[row].cpu().numpy())
                report["phase"] = state.get("check_phase", "unknown")
                report["initial_output_matches"] = [
                    bool(torch.equal(owner.initial_output[row, item["slot"]].cpu(),
                                     owner.session.cpu_payload[row, item["expected_token"]]))
                    for item in report["examples"]
                ]
                report_path = folder / "streaming_fetch_kv_mismatch.json"
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                try:
                    from diagnostics import audit_queue_writes
                    destinations = rotating.destinations.cpu().numpy().reshape(-1)
                    remote = rotating.fallback_addresses.cpu().numpy().reshape(-1).view(np.uint64)
                    report["actual_sq_writers"], report["final_wqe_comparison"] = audit_queue_writes(
                        owner, rows, posts, row, [item["slot"] for item in report["examples"]],
                        destinations, remote, q_workers=args.q_workers)
                except Exception as exc:
                    report["sq_audit_error"] = f"{type(exc).__name__}: {exc}"
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                print(f"STREAMING FETCH KV AUDIT: {json.dumps(report)}; details={report_path}", flush=True)
                raise AssertionError(
                    f"KV byte mismatch: batch={row} mismatched_tokens={int(mismatches.sum())} examples={details}"
                )
        state["ids"], state["slots"] = (ids, slots.numpy())

    def verify_all_valid_hot():
        """Catch stale KV beyond the current TopK, including displaced slots."""
        valid = rotating.hot_valid.cpu().numpy().reshape(b, r)
        logical = update.logical.cpu().long()
        reference_device = getattr(owner.session, 'reference', None)
        hot_kv = owner.output[:, :r] if reference_device is not None else owner.output[:, :r].cpu()
        for row in range(b):
            mask = torch.from_numpy(valid[row] != 0)
            if reference_device is not None:
                expected = reference_device[row, logical[row, :r][mask].to(device)]
                actual = hot_kv[row, mask.to(device)]
            else:
                expected = owner.session.cpu_payload[row, logical[row, :r][mask]]
                actual = hot_kv[row, mask]
            if not torch.equal(actual, expected):
                raise AssertionError(f"valid hot slot has mismatched Index/KV owner: request={row}")

    def check_pipeline(orders, cold_start=False, label="regular"):
        state["check_phase"] = label
        reset()
        update.set_order(orders)
        if cold_start:
            rotating.hot_valid.zero_()
        before_valid = rotating.hot_valid.cpu().numpy().copy()
        before_layout = update.logical.cpu().numpy()[:, :r].copy()
        full()
        stream.synchronize()
        def validate_pipeline():
            verify(True)
            if getattr(args, "mode", None) == "check":
                for name in ("_live_count", "_live_flags", "_live_rows", "_live_events"):
                    if getattr(pipeline, name).count_nonzero().item():
                        raise AssertionError(f"Resolver left nonzero round state in {name}")
                if int(rotating.count.cpu().item()) != state["completion"]["final_reads"]:
                    raise AssertionError("Published repair count differs from completed round")
            verify_all_valid_hot()
            # Validate every Index field against its logical owner, including rows
            # outside current TopK. This checks the actual staged update, not an
            # independently executed old planner.
            logical = update.logical.cpu().long()
            seeds = [saved[0].reshape(b, h, 128).cpu(), saved[1].view(torch.float32).reshape(b, h).cpu()]
            address_seed = saved[2][b * prefix:].view(torch.int64).reshape(b, tail // 8)[:, :h].cpu()
            for row in range(b):
                if not torch.equal(torch.sort(logical[row]).values, torch.arange(h)):
                    raise AssertionError("Index logical owners are not a permutation")
                for name, tensor, seed in zip(("Index", "scales", "addresses"), update.inputs[:3], seeds + [address_seed]):
                    if not torch.equal(tensor[row].cpu(), seed[row, logical[row]]):
                        raise AssertionError(f"Staged update field mismatch: {name}, request={row}")
        passed = correctness.check(label, validate_pipeline)
        from correctness import boundary_evidence
        boundary_phases[label] = boundary_evidence(
            state["ids"], before_valid, before_layout,
            rotating.rows[:, 9].cpu().numpy(), rotating.final_posts[:, :, 0].cpu().numpy())
        atomic_json(folder / "boundary_coverage.json", boundary_phases)
        if cold_start:
            from correctness import transfer_coverage
            boundary_phases['transfer'] = transfer_coverage(
                rotating.final_posts.cpu().numpy(), rotating.destinations.cpu().numpy(),
                rotating.lru_slots.cpu().numpy(), owner.output.shape[1], r)
            atomic_json(folder / "boundary_coverage.json", boundary_phases)
        changes = rotating.changes.cpu().numpy()
        result = dict(transport=state["completion"],
            first_window_fetches=int(rotating.rows[:, 16].cpu().sum()),
            hot_changes=int(np.count_nonzero((changes >= 0) & (changes < r))),
            cold_changes=int(np.count_nonzero(changes >= r)))
        print(f"CHECK {'PASS' if passed else 'FAIL'} {label}: {json.dumps(result)}; full KV/Index/ownership verified", flush=True)
        return result

    snapshot_changes = check_pipeline(initial_orders)
    cold_start_changes = (None if bench_only or compact else
                          check_pipeline(initial_orders, cold_start=True, label="cold-start"))
    unlucky = []
    for row in range(b):
        selected_hot = list(dict.fromkeys(initial_selected[row].reshape(-1).tolist()))
        selected_hot = [x for x in selected_hot if x < r]
        seen = set(selected_hot)
        unlucky.append(selected_hot + [x for x in initial_orders[row] if x not in seen])
    hot_case = (dict(hot_changes=None, first_window_fetches=None) if bench_only or compact else
                check_pipeline(unlucky, label="hot-victim-collision"))
    restored_hot = hot_case["hot_changes"]
    if not bench_only and not compact and args.prefetch_tokens and args.miss_percent < 100:
        def check_collision_coverage():
            assert restored_hot, (
                f"Selected former hot candidates were not repaired: {hot_case}")
        correctness.check("hot-victim-collision coverage", check_collision_coverage)

    if getattr(args, "check_suite", None) == "boundary":
        reset()
        ends = rotating.query_ends.clone()
        validity = update.output_valid.clone()
        protected = (*update.inputs[:4], rotating.hot_valid, owner.output)
        snapshots = [tensor.clone() for tensor in protected]
        rotating.query_ends.zero_()
        update.output_valid.zero_()
        full()
        stream.synchronize()
        def check_padding():
            rows = rotating.rows.cpu().numpy()
            posts = rotating.final_posts.cpu().numpy()
            statuses = rotating.row_status.cpu().numpy()
            cq = [x.cpu().view(torch.uint64).numpy() for x in (rotating.prefetch_status, rotating.final_status)]
            if completion_failures(rows, posts, statuses, *cq) or update.status.cpu().count_nonzero():
                raise RuntimeError('Padding execution did not complete successfully')
            owner.mark_completed(completion_verified=True)
            assert not rows[:, 9].any() and not posts[:, :, 0].any(), 'Padding issued fetches'
            assert all(torch.equal(tensor, saved_tensor) for tensor, saved_tensor in zip(protected, snapshots)), 'Padding changed persistent state'
            assert (update.logical_output.cpu() == -1).all() and (update.slot_output.cpu() == -1).all(), 'Padding retained outputs'
        passed = correctness.check('padding', check_padding)
        boundary_phases['padding'] = 'PASS' if passed else 'FAIL'
        rotating.query_ends.copy_(ends)
        update.output_valid.copy_(validity)
        atomic_json(folder / 'boundary_coverage.json', boundary_phases)

    if getattr(args, "check_suite", None) == "boundary":
        from correctness import require_boundary
        correctness.check("boundary trigger coverage", lambda: require_boundary(args.check_case, boundary_phases))

    def init_reset():
        reset()
        for tensor in (
            rotating.rows,
            rotating.progress,
            rotating.events,
            rotating.owners,
            rotating.mapping,
            rotating.changes,
            rotating.row_status,
            rotating.count,
        ):
            tensor.fill_(1)
        header = b * (4 * 2048 * 6 + 1024)
        arguments[13][header : header + b * h].fill_(255)

    def init_check():
        owner.mark_completed(completion_verified=True)
        if (
            rotating.rows[:, [8, 9, 10, 211, 247, 248]].cpu().count_nonzero()
            or rotating.progress.cpu().count_nonzero()
            or rotating.events.cpu().count_nonzero()
            or rotating.owners.cpu().count_nonzero()
            or rotating.count.cpu().count_nonzero()
            or (not (rotating.mapping.cpu() == -1).all())
            or (not (rotating.changes.cpu() == -1).all())
        ):
            raise AssertionError("Round init retained poisoned workspace")
        header = b * (4 * 2048 * 6 + 1024)
        bitmap = arguments[13][header : header + b * h].cpu().reshape(b, 4, h // 4)
        if not (bitmap == 255).all():
            raise AssertionError("Round init unexpectedly rewrote producer-owned predicates")

    fused_completion = (lambda: verify(True, completion_only=True)) if compact else None
    fused_graph = measure(full, reset, lambda: verify(True), timing_name="prototype_with_update",
                          completion_check=fused_completion, keep_graph=bool(args.trace_rounds))
    if args.trace_rounds:
        progress(folder.parent, f"BS={b}: full graph profiler ({args.trace_rounds} replays); logs={folder}")
        pipeline.set_diagnostics(False)
        stream.synchronize()

        def trace_reset():
            reset()
            pipeline.times.zero_()

        state['check_phase'] = 'profile fused'
        try:
            full_trace.update(collect_core(
                torch, rotating, fused_graph, trace_reset,
                lambda: correctness.check('profile fused', lambda: verify(True)),
                args.trace_rounds, folder, full=True, fused_final=True))
            if args.level >= 2:
                from diagnostics import collect_details
                collect_details(torch, rotating, fused_graph, trace_reset,
                                lambda: correctness.check('detail fused', lambda: verify(True)),
                                args.trace_rounds, folder, pipeline)
                counter_clock_mhz = pipeline.counter_clock_mhz
        finally:
            pipeline.set_diagnostics(False)
            stream.synchronize()
        fused_graph.reset()
        owner.graphs.remove(fused_graph)
    if not bench_only:
        measure(lambda: rotating._call("init"), init_reset, init_check, timing_name="round_init")
        reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        owner.graphs.append(graph)
        with torch.npu.graph(graph, stream=capture):
            full()
        capture.synchronize()
        for step in range(3):
            next_orders = []
            if step:
                owner.mark_launched()
            state["before_logical"] = update.logical.cpu().numpy().copy()
            graph.replay()
            stream.synchronize()
            verify(True)
            verify_all_valid_hot()
            for row, p in enumerate(policies):
                touched = state["slots"][row]
                p.commit(touched[touched < r].astype(np.int32))
                order = []
                slot = int(p.head[0])
                for _ in range(r):
                    order.append(slot)
                    slot = int(p.next[0, slot])
                next_orders.append(order)
            update.set_order(next_orders)
        graph.reset()
        owner.graphs.remove(graph)
    forward = np.full((b, h), -1, np.int32)
    logical = update.logical.cpu().numpy()
    for row in range(b):
        forward[row, logical[row, :r]] = np.arange(r)
        forward[row, state["ids"][row].reshape(-1)] = state["slots"][row]
    after = {"topk": state["ids"].reshape(b * 4, 1, 2048), "forward": forward}
    table = torch.arange(b * (r + 8192) // 128, device=device, dtype=torch.int32).reshape(b, -1)
    tensors = dict(
        query=fixture["query"],
        resident_output=owner.output,
        forward=torch.from_numpy(forward).to(device),
        pool_rows=torch.arange(b, device=device, dtype=torch.int32),
        topk=torch.from_numpy(after["topk"]).to(device),
        resident_table=table,
        cumulative_query=fixture["actual_query"],
        logical_lens=torch.full((b,), h, device=device, dtype=torch.int32),
    )
    geom = SimpleNamespace(batch=b, queries=4, num_layers=1, layer_id=0, resident_blocks=b * (r + 8192) // 128)
    sfa = check_sfa(torch, tensors, geom, {"pool_rows": np.arange(b), "budgets": np.full(b, r + 8192)}, after,
                    output_path=None)
    fetch_only = None
    if getattr(args, "diagnostic_fetch_only", False):
        # A separate replay supplies the exact final descriptors. No standalone
        # preparation or tracing enters the full-graph performance samples.
        from performance import compare_fetch_only

        reset()
        full()
        stream.synchronize()
        verify(True)
        verify_all_valid_hot()
        fetch_only = compare_fetch_only(torch, owner, rotating, pipeline, args)
        atomic_json(folder / "fetch_only.json", dict(status="PASS", rows=fetch_only))
    if args.trace_rounds:
        from diagnostics import collect_native

        # Capture the serial graph against the unleased runtime after all fused work.
        owner.release()
        for index, initial in zip((1, 4), saved[:2]):
            arguments[index].copy_(initial)
        native.reset()
        native_call()
        stream.synchronize()
        native_check()
        measure(native.fetch, lambda: native.output.copy_(native.initial_payload), native_check,
                timing_name="native_get_only", completion_check=native_completion)
        graph = capture_graph(native_call, native.reset)
        collect_native(torch, graph, native.reset,
                       lambda: correctness.check('profile native', native_check), args.trace_rounds, folder)
        graph.reset()
        owner.graphs.remove(graph)
    return dict(
        status=correctness.status.lower(),
        correctness=correctness.status,
        correctness_failures=correctness.failures,
        boundary_coverage=boundary_phases,
        counter_clock_mhz=counter_clock_mhz,
        timing=completed_timings, wall_costs=wall_costs,
        full_trace=full_trace,
        fetch_only=fetch_only,
        read_samples=read_samples,
        split_samples=split_samples,
        fused_read_counts=fused_read_counts,
        sampling_protocol=sampling_protocol,
        sfa=sfa,
        stateful_rounds=0 if bench_only else 3,
        native_misses_per_request=state["native_misses"],
        cache_update=dict(
            swap_capacity_per_request=update.max_swaps,
            producer_cold_changes=snapshot_changes["cold_changes"],
            hot_kv_copy_payload_bytes=0,
            cold_start_changes_checked=cold_start_changes,
            prefetch_tokens=args.prefetch_tokens,
            former_hot_changes_checked=restored_hot,
            first_window_fetches_checked=hot_case["first_window_fetches"],
            scope=("Bench regular checks; boundary/stateful checks belong to check mode" if bench_only else
                   "Actual streaming pipeline verified; no temporary KV slots; zero KV copy"),
        ),
        kernel="KVIP/csrc/operators/lightning_indexer_with_fetch; preserved core tiling, direct LRU interface",
        scope="fixed Q4 shared history; graph kernel and AIV maintenance separately; no ABC growth",
    )


def initialize_transport(transport, argv=None):
    p = argparse.ArgumentParser(description=f'{transport} KVAF runtime and LIFetch lease initialization')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--kvaf-server', required=True)
    p.add_argument('--kvaf-root', type=Path)
    p.add_argument('--soc', help='CANN SoC; URMA only, detected from device when omitted')
    args = p.parse_args(argv)
    import acl
    import kvaf
    import torch
    import torch_npu  # noqa: F401
    from kvip.adapters.kvaf import KvafGetLease, UbmemLease
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    root = (args.kvaf_root or ROOT / 'third_party/KVAF').resolve()
    build = root / 'build-npu'
    if not Path(kvaf.__file__).resolve().is_relative_to(build / 'python/kvaf'):
        raise RuntimeError(f'Wrong KVAF package: {kvaf.__file__}; source scripts/env.sh for {root}')
    torch.npu.set_device(args.device)
    device = torch.device(f'npu:{args.device}')
    stream = torch.npu.current_stream(device)
    get, put = build / 'bin/kvaf_execute_batch_get_kernels.o', build / 'bin/kvaf_execute_batch_put_kernels.o'
    for path in (get, put):
        if not path.is_file():
            raise FileNotFoundError(f'KVAF kernel missing: {path}')
    kwargs = dict(device_id=args.device, npu_core_count=2, batch_put_kernel_object_path=str(put))
    if transport == 'urma':
        kwargs['jettys_per_core'] = 2
    else:
        kwargs['transport'] = 'ubmem'
    runtime = kvaf.NpuRuntime(str(get), 656, 128, **kwargs)
    lease = None
    table = None
    in_flight = False
    try:
        print(f'START {transport}: connecting and allocating four KV blocks', flush=True)
        runtime.link_to(args.kvaf_server)
        block = 128 * 656
        slot = ((block + 4095) // 4096 * 4096) if transport == 'ubmem' else block
        runtime.initialize(4 * slot, slot)
        output = torch.zeros((128, 656), dtype=torch.uint8, device=device)
        # Validate real data movement, not only resource registration.
        from workloads import payload
        expected = torch.from_numpy(payload(1, np.arange(128))).contiguous()
        source = expected.to(device)
        in_flight = True
        runtime.batch_put([0], source, [0], [block])
        stream.synchronize()
        in_flight = False
        table = runtime.prepare_access_table([0])
        view = runtime.access_table_view(table, device="npu")
        indices = torch.zeros(128, dtype=torch.int32, device=device)
        offsets = torch.arange(128, dtype=torch.int32, device=device) * 656
        count = torch.tensor([128], dtype=torch.int32, device=device)
        destinations = torch.arange(128, dtype=torch.int32, device=device)
        in_flight = True
        runtime.batch_get(view.addresses, view.valid, indices, offsets, count,
                          destinations, output, stream, False)
        stream.synchronize()
        in_flight = False
        if not torch.equal(output.cpu(), expected):
            raise AssertionError("Initialization Put/Get payload mismatch")
        runtime.release_access_table(table)
        table = None
        if transport == 'urma':
            cann, headers = kvaf_build_paths(build)
            (library,) = build_libraries(cache_dir=build / 'lifetch', cann=cann, headers=headers,
                                         soc=args.soc or acl.get_soc_name())
            lease = KvafGetLease(torch, runtime, library, output, stream)
        else:
            lease = UbmemLease(torch, runtime, output, stream)
            lease.region(output)
        lease.check_stream()
        print(f'PASS {transport}: runtime, DDR registration and lease established; releasing', flush=True)
        lease.close()
        lease = None
        runtime.shutdown()
        print(f'PASS {transport}: lease released and runtime shut down', flush=True)
        return 0
    except Exception as error:
        print(f'FAIL {transport}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        if in_flight:
            NativeFetchSession.quarantine("init_put_get_completion_uncertain", error)
        if lease is None and runtime.initialized:
            if table is not None:
                runtime.release_access_table(table)
            runtime.shutdown()
        raise


def layer_payload(owner, tokens, token_bytes):
    """Same packed C8 records used by the URMA numerical oracle."""
    from workloads import payload

    return np.concatenate(
        [payload(owner - 1 + layer * 1024, tokens) for layer in range(token_bytes // 656)], axis=1
    )


def controlled_fixture(history, resident, heads, miss_percent, seed=20261104, compact=False):
    """Use precisely the same deterministic Q4 fixture as URMA's A/B test."""
    support = Path(__file__).resolve().parent
    if str(support) not in sys.path:
        sys.path.insert(0, str(support))
    from workloads import prepare_full_chain_fixture

    with tempfile.TemporaryDirectory(prefix="kvip-ubmem-fixture-") as tmp:
        root = Path(tmp) / "fixture"
        from workloads import prepare_compact_fixture
        prepare_fixture = prepare_compact_fixture if compact else prepare_full_chain_fixture
        info = prepare_fixture(root, history, resident, miss_percent, index_heads=heads, seed=seed)
        data = {
            "info": info,
            "order": np.load(root / "physical_to_logical.npy").astype(np.int64),
            "keys": np.fromfile(root / "arg1.bin", np.uint8).reshape(history, 128),
            "scales": np.fromfile(root / "arg4.bin", np.float32),
            "query": np.fromfile(root / "arg0.bin", np.uint8).reshape(4, heads, 128),
            "weights": np.fromfile(root / "arg2.bin", np.float32).reshape(4, heads),
            "query_scale": np.fromfile(root / "arg3.bin", np.float32).reshape(4, heads),
            "scores": np.load(root / "logical_scores.npy"),
        }
    order = data["order"]
    assert len(order) == history and np.array_equal(np.sort(order), np.arange(history))
    # URMA's single-operator harness identifies tokens by physical fixture row.
    # Its source payload and initial hot slots use these same row IDs.
    physical_scores = data["scores"][:, order]
    data["expected"] = np.sort(np.argpartition(physical_scores, -2048, axis=1)[:, -2048:], axis=1)
    all_cold = np.unique(data["expected"][data["expected"] >= resident])
    data["native_miss_union"] = len(all_cold)
    return data


def mixed_query_counts(batch):
    """Exercise Q1-Q4 and one inactive row in the same graph replay."""
    counts = [1 + row % 4 for row in range(batch)]
    if batch >= 5:
        counts[-1] = 0
    return counts


def run_ubmem_case(
    kvaf,
    torch,
    build,
    device_id,
    shape,
    library,
    prefetch_tokens=512,
    args_server="127.0.0.1:28880",
    miss_percent=100,
    warmup=0,
    iterations=0,
    trace_rounds=0,
    trace_output=None,
    diagnostic_level=1,
    q_workers=2,
    seed=20261104,
    result_output=None,
    serial_compare=False,
    check_case=None,
    check_suite=None,
    shared_pool=None,
    compact=False,
):
    batch, history, resident, token_bytes = shape.batch, shape.history, shape.resident, shape.token_bytes
    checks = CorrectnessRecorder()
    fixture_key = ('ubmem', compact, history, resident, shape.heads, miss_percent, seed)
    if shared_pool is not None and fixture_key in shared_pool.fixtures:
        fixture = shared_pool.fixtures[fixture_key]
    else:
        fixture = controlled_fixture(history, resident, shape.heads, miss_percent, seed=seed, compact=compact)
        if shared_pool is not None:
            shared_pool.fixtures[fixture_key] = fixture
    expected_ids = fixture["expected"]
    cold_count = fixture["native_miss_union"]
    logical_scales = fixture["scales"]
    logical_keys = fixture["keys"]
    device = torch.device(f"npu:{device_id}")
    block_bytes = 128 * token_bytes
    slot_bytes = (block_bytes + 4095) // 4096 * 4096
    blocks = history // 128
    keys = list(range(batch * blocks))
    table = lease = graph = None
    stream = torch.npu.current_stream(device)
    if shared_pool is not None:
        borrowed = shared_pool.borrow(torch, device, batch, history, 2)
        runtime, table, device_table = borrowed.runtime, borrowed.table, borrowed.device_table
        source_cpu = borrowed.cpu_payload.numpy().reshape(-1, token_bytes)
        source = borrowed.reference.reshape(-1, token_bytes)
    else:
        source_cpu = np.concatenate([layer_payload(owner, np.arange(history), token_bytes) for owner in range(1, batch + 1)])
        source = torch.from_numpy(source_cpu).to(device)
        runtime = kvaf.NpuRuntime(
            str(build / "bin/kvaf_execute_batch_get_kernels.o"), token_bytes, 128,
            device_id=device_id, npu_core_count=2,
            batch_put_kernel_object_path=str(build / "bin/kvaf_execute_batch_put_kernels.o"), transport="ubmem")
    try:
        if shared_pool is None:
            runtime.link_to(args_server)
            runtime.initialize((len(keys) + 2) * slot_bytes, slot_bytes)
            runtime.batch_put(keys, source, [i * block_bytes for i in keys], [block_bytes] * len(keys))
            table = runtime.prepare_access_table(keys)
            device_table = runtime.access_table_view(table, device="npu")
        storage = IndexCacheStorage(torch, batch + 1, history, resident, device)
        key_tensor = torch.from_numpy(fixture["keys"]).to(device)
        scale_tensor = torch.from_numpy(fixture["scales"]).to(device)
        storage.keys[1:].copy_(key_tensor.expand(batch, -1, -1))
        storage.scales[1:].copy_(scale_tensor.expand(batch, -1))
        storage.hot_valid[1:].fill_(1)
        offsets = torch.arange(128, dtype=torch.int64, device=device) * token_bytes
        storage.addresses[1:].copy_((device_table.addresses.view(batch, blocks, 1) + offsets).reshape(batch, history))
        original_addresses = storage.addresses.cpu().numpy()
        stride = resident
        output = torch.zeros((batch + 1, stride, token_bytes), dtype=torch.uint8, device=device)
        output[1:, :resident].copy_(source.view(batch, history, token_bytes)[:, :resident])
        # Match vLLM policy.bytes_outputs: KVAF registers one token per row.
        # Keep the owner-shaped alias only for payload verification below.
        output_rows = output.view(-1, token_bytes)
        workspace = UbmemWorkspace(
            torch,
            batch=batch,
            history=history,
            resident=resident,
            aics=shape.aic_count,
            windows=shape.intermediate_fetches,
            device=device,
            index_heads=shape.heads,
        )
        workspace.inputs.prepare([4] * batch, query_owners=list(range(1, batch + 1)))
        workspace.inputs.query.view(torch.uint8).copy_(
            torch.from_numpy(fixture["query"]).to(device).repeat(batch, 1, 1)
        )
        workspace.inputs.weights.copy_(
            torch.from_numpy(fixture["weights"]).to(device, dtype=torch.bfloat16).repeat(batch, 1)
        )
        workspace.inputs.scale.copy_(torch.from_numpy(fixture["query_scale"]).to(device).repeat(batch, 1))
        workspace.inputs.query_ends.fill_(history)
        serial_result = None
        if serial_compare:
            if token_bytes != 656:
                raise ValueError("Native paired benchmark currently supports 656-byte C8 tokens")
            import torch_npu
            from performance import measure_ubmem_serial
            serial_result = measure_ubmem_serial(
                torch, torch_npu, runtime, device_table, workspace, storage, source_cpu,
                batch=batch, history=history, resident=resident, heads=shape.heads,
                warmup=warmup, iterations=iterations, trace_rounds=trace_rounds,
                reference=source.view(batch, history, token_bytes), bench_only=shared_pool is not None,
                trace_folder=(result_output / f"b{batch}-h{history}-r{resident}-w{token_bytes}-p{prefetch_tokens}"
                              if result_output else None))
            if result_output:
                from reporting import atomic_json
                serial_path = result_output / f"b{batch}-h{history}-r{resident}-w{token_bytes}-p{prefetch_tokens}"
                atomic_json(serial_path / "serial.json", serial_result)
        lease = UbmemLease(torch, runtime, output_rows, stream)
        lease.register_table(device_table)
        rotating = UbmemBinding(
            torch,
            library,
            workspace.arguments(storage),
            lease,
            output_rows,
            shape=shape,
            query_ends=workspace.inputs.query_ends.view(batch, 4),
            hot_valid=storage.hot_valid,
            row_owners=workspace.inputs.row_owners,
            token_addresses=storage.addresses,
            output_stride=stride,
            scratch=workspace.scratch,
            prefetch_limit=prefetch_tokens,
        )
        # Check the compiled UBmem schedule against the URMA runtime contract,
        # including epoch zero at the end of the hot region.
        from kvip.operators.lifetch.geometry import trigger_tokens

        trigger = rotating.library.v9_window_trigger_tile
        trigger.argtypes, trigger.restype = [ctypes.c_uint32, ctypes.c_void_p], ctypes.c_uint32
        actual_triggers = [128 * trigger(epoch, ctypes.byref(rotating.runtime_shape)) for epoch in range(rotating.rounds)]
        expected_triggers = trigger_tokens(
            batch, history, resident, intermediate_fetches=shape.intermediate_fetches, aic_count=shape.aic_count
        )
        if actual_triggers != expected_triggers:
            raise AssertionError(f"UBmem/URMA prefetch window mismatch: {actual_triggers} != {expected_triggers}")
        print(f"UBMEM_WINDOW_TRIGGERS general_tokens={actual_triggers}", flush=True)
        half = ((history - resident) // 512 + rotating.rounds - 2) // (rotating.rounds - 1) * 256
        windows = [min(history, resident + 2 * half * e) for e in range(rotating.rounds)]
        print(f"UBMEM_LI_SEQUENCE_WINDOWS tokens={windows} half_tokens={half}", flush=True)
        update = UbmemIndexUpdate(
            torch,
            keys=storage.keys,
            scales=storage.scales,
            addresses=storage.addresses,
            logical_ids=storage.logical_ids,
            row_owners=workspace.inputs.row_owners,
            rotating=rotating,
            status=torch.zeros(batch, dtype=torch.int32, device=device),
            query_inputs=workspace.inputs,
            outputs=torch.empty((2, batch * 4, 1, 2048), dtype=torch.int32, device=device),
        )
        forward = StreamingForward(rotating, update, torch, prefetch_tokens, fused_final=True, q_workers=q_workers)
        forward.ubmem_overlap_fetch = 'overlap' == "overlap"
        forward.prepare_metadata = True
        forward.parallel_finalize = True
        forward.set_diagnostics(False)

        def verify_outputs(owners, *, phase, counts=None, padding=False, completion_only=False):
            stream.synchronize()
            index_status = update.status.cpu().tolist()
            if any(index_status):
                rows = rotating.rows.cpu().tolist()
                diagnostics = dict(
                    phase=phase,
                    prefetch_tokens=prefetch_tokens,
                    index_status=index_status,
                    ubmem_status=rotating.ubmem_status.cpu().tolist(),
                    row_status=rotating.row_status.cpu().tolist(),
                    final_posts=rotating.final_posts[:, :, :8].cpu().tolist(),
                    change_count=rotating.change_count.cpu().tolist(),
                    row_fields=[{i: row[i] for i in (8, 9, 11, 208, 209, 210, 211, 247, 248)} for row in rows],
                )
                raise RuntimeError(f"Index update rejected UBmem completion: {diagnostics}")
            counters = rotating.ubmem_status.cpu().numpy()
            if (np.any(counters[:, 0]) or not np.array_equal(counters[:, 1], counters[:, 2])
                    or not np.array_equal(counters[:, 3], counters[:, 4])):
                raise RuntimeError(f"UBmem copy failed or incomplete: {counters.tolist()}")
            if completion_only:
                return
            assert np.all(counters[:, 1] <= prefetch_tokens), "Prefetch exceeded its total budget"
            epochs = rotating.rows.cpu().numpy()[:, 16 : 16 + rotating.rounds]
            assert np.all(epochs <= max(1, prefetch_tokens // shape.intermediate_fetches)), (
                "Prefetch exceeded its per-window budget"
            )
            logical = update.logical_output.cpu().numpy().reshape(batch * 4, 2048)
            slots = update.slot_output.cpu().numpy().reshape(batch * 4, 2048)
            active = sum([4] * len(owners) if counts is None else counts)
            if not padding and (np.any(logical[:active] < 0) or np.any(logical[:active] >= history)
                                or np.any(slots[:active] < 0) or np.any(slots[:active] >= resident)):
                raise RuntimeError("UBmem output token/slot outside allocated bounds")
            if padding:
                assert np.all(logical == -1) and np.all(slots == -1), "Padding inherited live outputs"
                return
            ids = storage.logical_ids.cpu().numpy()
            if np.any(ids < 0) or np.any(ids >= history):
                raise RuntimeError("UBmem Index ownership is outside allocated history")
            addresses = storage.addresses.cpu().numpy()
            scales = storage.scales.cpu().numpy()
            key_values = storage.keys.cpu().numpy()
            counts = [4] * len(owners) if counts is None else counts
            output_row = 0
            for owner, count in zip(owners, counts):
                if not count:
                    continue
                assert np.array_equal(np.sort(ids[owner]), np.arange(history)), "Index ownership is not a permutation"
                assert np.array_equal(addresses[owner], original_addresses[owner, ids[owner]]), (
                    "Index addresses detached"
                )
                assert np.array_equal(scales[owner], logical_scales[ids[owner]]), "Index scales detached"
                assert np.array_equal(key_values[owner], logical_keys[ids[owner]]), "Index keys detached"
                # Verify every valid persistent hot entry, not just selected rows.
                hot_valid = storage.hot_valid[owner].cpu().numpy() != 0
                hot_slots = np.flatnonzero(hot_valid)
                hot_indices = torch.from_numpy(hot_slots).to(device)
                actual_device = output[owner, hot_indices]
                expected_device = source.view(batch, history, token_bytes)[owner - 1,
                    torch.from_numpy(ids[owner, hot_slots]).to(device)]
                if not torch.equal(actual_device, expected_device):
                    actual_hot, expected_hot = actual_device.cpu().numpy(), expected_device.cpu().numpy()
                    difference = np.argwhere(actual_hot != expected_hot)[0]
                    raise AssertionError(
                        f"Hot slot/Index mismatch: phase={phase} owner={owner} "
                        f"slot={hot_slots[difference[0]]} byte={difference[1]}"
                    )
                for query in range(count):
                    if compact:
                        from correctness import validate_topk
                        validate_topk(logical[output_row], fixture['scores'][query, fixture['order']][None, :])
                    else:
                        assert np.array_equal(np.sort(logical[output_row]), expected_ids[query]), (
                            "LI selected the wrong TopK set")
                    selected_slots = torch.from_numpy(slots[output_row].astype(np.int64)).to(device)
                    actual_device = torch.index_select(output[owner], 0, selected_slots)
                    expected_device = source.view(batch, history, token_bytes)[owner - 1,
                        torch.from_numpy(logical[output_row].astype(np.int64)).to(device)]
                    if not torch.equal(actual_device, expected_device):
                        actual, expected = actual_device.cpu().numpy(), expected_device.cpu().numpy()
                        mismatch = np.flatnonzero(np.any(actual != expected, axis=1))
                        k = int(mismatch[0])
                        byte = int(np.flatnonzero(actual[k] != expected[k])[0])
                        raise AssertionError(
                            f"Fused Fetch bytes differ from KVAF Put: phase={phase} device={device_id} "
                            f"batch={batch} token_bytes={token_bytes} owner={owner} query={output_row} "
                            f"token={int(logical[output_row, k])} slot={int(slots[output_row, k])} "
                            f"byte={byte} expected={int(expected[k, byte])} actual={int(actual[k, byte])}"
                        )
                    output_row += 1
            assert np.all(logical[output_row:] == -1) and np.all(slots[output_row:] == -1), "TND padding is stale"

        def verify(owners, *, phase, counts=None, padding=False):
            return checks.check(phase, lambda: verify_outputs(
                owners, phase=phase, counts=counts, padding=padding))

        # Restore the exact initial mapping/cache outside the event pair.
        measured_buffers = (
            storage.keys,
            storage.scales,
            storage.addresses,
            storage.logical_ids,
            storage.hot_valid,
            output,
        )
        initial_state = tuple(t.clone() for t in measured_buffers) if iterations or check_case else ()

        # Compare LI selection against the independent torch_npu native LI on
        # exactly the same query, Index cache, scales and block table. This is
        # outside graph timing and happens before Index ownership is updated.
        import torch_npu

        native = torch_npu.npu_quant_lightning_indexer(
            query=workspace.inputs.query.reshape(batch, 4, shape.heads, 128),
            key=storage.keys.view(torch.float8_e4m3fn).reshape((batch + 1) * blocks, 128, 1, 128),
            weights=workspace.inputs.weights.reshape(batch, 4, shape.heads),
            query_dequant_scale=workspace.inputs.scale.reshape(batch, 4, shape.heads),
            key_dequant_scale=storage.scales.reshape((batch + 1) * blocks, 128, 1),
            actual_seq_lengths_query=workspace.query_lengths,
            actual_seq_lengths_key=workspace.key_lengths,
            block_table=workspace.inputs.block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=0,
        )
        native_ids = (native[0] if isinstance(native, (tuple, list)) else native).cpu().numpy().reshape(batch, 4, 2048)
        if np.any(native_ids < 0) or np.any(native_ids >= history):
            raise RuntimeError("Native LI output is outside allocated history")
        def check_native_topk(selected=native_ids):
            if compact:
                from correctness import validate_topk
                validate_topk(selected, fixture['scores'][:, fixture['order']])
                return
            assert np.all(np.sort(selected, axis=-1) == expected_ids[None, :, :]), (
                "Native LI differs from the independent controlled TopK oracle")
        checks.check('native_topk', check_native_topk)
        # The URMA test commits this same native TopK to the service LRU before
        # choosing victims. Match the exact ordered candidates and inverse map.
        from kvip.cache.lru import SlotLRU

        candidates = np.empty((batch, 2048), dtype=np.int32)
        reserved = np.full((batch, resident), -1, dtype=np.int32)
        for row in range(batch):
            policy = SlotLRU()
            policy.resize(resident, resident)
            hits = native_ids[row].reshape(-1)
            policy.commit(hits[hits < resident])
            slot = int(policy.head[0])
            for i in range(2048):
                candidates[row, i] = slot
                slot = int(policy.next[0, slot])
            reserved[row, candidates[row]] = np.arange(2048, dtype=np.int32)
        rotating.lru_slots.copy_(torch.from_numpy(candidates).to(device))
        rotating.reserved_hot.copy_(torch.from_numpy(reserved).to(device))
        del native, native_ids

        boundary_phases = {}
        before_valid = storage.hot_valid[1:].cpu().numpy().copy()
        before_layout = storage.logical_ids[1:, :resident].cpu().numpy().copy()
        forward()
        verify(list(range(1, batch + 1)), phase="eager")
        from correctness import boundary_evidence, require_boundary
        from reporting import atomic_json

        def record_boundary(label, validity, layout):
            boundary_phases[label] = boundary_evidence(
                update.logical_output.cpu().numpy().reshape(batch, 4, 2048), validity, layout,
                rotating.rows[:, 9].cpu().numpy(), rotating.final_posts[:, :, 0].cpu().numpy())
            if result_output:
                atomic_json(result_output / f"b{batch}-h{history}-r{resident}-w{token_bytes}-p{prefetch_tokens}"
                            / "boundary_coverage.json", boundary_phases)
        record_boundary('regular', before_valid, before_layout)
        # Actual service SFA with the URMA fixture's packed C8 payload, using
        # this fused path's final token-to-slot mapping. All outside timing.
        from types import SimpleNamespace

        selected_ids = update.logical_output.cpu().numpy().reshape(batch * 4, 1, 2048)
        selected_slots = update.slot_output.cpu().numpy().reshape(batch * 4, 1, 2048)
        physical_ids = storage.logical_ids.cpu().numpy()
        forward_map = np.full((batch, history), -1, np.int32)
        for row in range(batch):
            forward_map[row, physical_ids[row + 1, :resident]] = np.arange(resident)
            forward_map[row, selected_ids[row * 4 : (row + 1) * 4].reshape(-1)] = selected_slots[
                row * 4 : (row + 1) * 4
            ].reshape(-1)
        block_table = torch.arange(batch * resident // 128, device=device, dtype=torch.int32).reshape(batch, -1)
        geometry = SimpleNamespace(
            batch=batch, queries=4, num_layers=1, layer_id=0, resident_blocks=batch * resident // 128
        )
        before = {"pool_rows": np.arange(batch), "budgets": np.full(batch, resident)}
        after = {"topk": selected_ids, "forward": forward_map}
        tensors = dict(
            query=workspace.inputs.query,
            resident_output=output[1:, :, :656].contiguous(),
            forward=torch.from_numpy(forward_map).to(device),
            pool_rows=torch.arange(batch, dtype=torch.int32, device=device),
            topk=torch.from_numpy(selected_ids).to(device),
            resident_table=block_table,
            cumulative_query=torch.arange(1, batch + 1, dtype=torch.int32, device=device) * 4,
            logical_lens=torch.full((batch,), history, dtype=torch.int32, device=device),
        )
        sfa = {}
        def verify_sfa():
            sfa.update(check_sfa(torch, tensors, geometry, before, after, output_path=None))
        if checks.check('sfa', verify_sfa):
            print(f"UBMEM_SFA_PASS max_abs={sfa['max_abs']:.8f}", flush=True)
        first_counters = rotating.ubmem_status.cpu().numpy()
        print(
            "UBMEM_FETCH_COUNTS "
            f"prefetch_posted={int(first_counters[:, 1].sum())} "
            f"prefetch_completed={int(first_counters[:, 2].sum())} "
            f"final_posted={int(first_counters[:, 3].sum())} "
            f"final_completed={int(first_counters[:, 4].sum())} "
            f"native_required={batch * cold_count} "
            f"requested_miss_pct={miss_percent:g} actual_misses_per_query={fixture['info']['final_misses_per_query']} "
            f"Q4_union_per_request={cold_count} fixture={fixture['info']['model']}",
            flush=True,
        )
        def verify_cold_reads():
            required = np.full(batch, cold_count)
            if compact:
                ids = update.logical_output.cpu().numpy().reshape(batch, -1)
                required = np.array([len(np.unique(row[row >= resident])) for row in ids])
            if np.any(required):
                assert np.any(first_counters[:, 1] + first_counters[:, 3]), "Fixture did not exercise a cold DDR fetch"
            assert np.all(first_counters[:, 3] >= np.maximum(0, required - prefetch_tokens)), (
                "Final repair must fetch all missing TopK rows even beyond the speculative budget")
        checks.check('cold_read_coverage', verify_cold_reads)
        if check_case:
            for tensor, saved in zip(measured_buffers, initial_state):
                tensor.copy_(saved)
            storage.hot_valid[1:].zero_()
            forward()
            verify(list(range(1, batch + 1)), phase="cold-start")
            record_boundary('cold-start', np.zeros_like(before_valid), before_layout)
            from correctness import transfer_coverage
            boundary_phases['transfer'] = transfer_coverage(
                rotating.final_posts.cpu().numpy(), rotating.destinations.cpu().numpy(),
                rotating.lru_slots.cpu().numpy(), output.shape[1], resident)
            for tensor, saved in zip(measured_buffers, initial_state):
                tensor.copy_(saved)
        graph = kvaf.AclGraph()
        with lease.capture_scope():
            graph.capture_begin(stream)
            forward()
            graph.capture_end()
        lease.retain_graph(graph)
        if shared_pool is None:
            # Replay after changing active row ownership. Index rows and output HBM
            # belong to stable owners, not capture-time packed row positions.
            owners = list(range(batch, 0, -1))
            workspace.inputs.prepare([4] * batch, query_owners=owners)
            workspace.inputs.query_ends.fill_(history)
            graph.replay(stream)
            verify(owners, phase="graph_replay")
            # Compact TND outputs no longer have the fixed four-row stride of
            # their source request. Exercise all Q1..Q4 lengths at the requested
            # batch, including an inactive/padded row when the batch permits it.
            counts = mixed_query_counts(batch)
            packed_owners = [owner if count else 0 for owner, count in zip(owners, counts)]
            workspace.inputs.prepare(counts, query_owners=packed_owners)
            ends = torch.tensor(
                [[history if q < count else 0 for q in range(4)] for count in counts],
                dtype=torch.int32, device=device,
            )
            workspace.inputs.query_ends.view(batch, 4).copy_(ends)
            graph.replay(stream)
            verify(packed_owners, phase="packed_queries", counts=counts)
            # All-padding warmup is legal: owner zero can repeat in inactive rows.
            workspace.inputs.prepare([0] * batch, query_owners=[0] * batch)
            workspace.inputs.query_ends.zero_()
            padding_before = [tensor.clone() for tensor in measured_buffers]
            graph.replay(stream)
            verify([], phase="padding", padding=True)
            def verify_padding_unchanged():
                assert all(torch.equal(tensor, saved) for tensor, saved in zip(measured_buffers, padding_before)), "Padding changed persistent state"
                assert not rotating.rows[:, 9].cpu().count_nonzero() and not rotating.final_posts[:, :, 0].cpu().count_nonzero(), "Padding issued fetches"
            padding_ok = checks.check('padding_ownership', verify_padding_unchanged)
            boundary_phases['padding'] = 'PASS' if padding_ok else 'FAIL'
            if check_suite == 'boundary':
                checks.check('boundary trigger coverage', lambda: require_boundary(check_case, boundary_phases))
                atomic_json(result_output / f'b{batch}-h{history}-r{resident}-w{token_bytes}-p{prefetch_tokens}' / 'boundary_coverage.json', boundary_phases)
        if iterations:
            workspace.inputs.prepare([4] * batch, query_owners=list(range(1, batch + 1)))
            workspace.inputs.query_ends.fill_(history)
            elapsed = []
            wall_costs = dict(reset_s=0., replay_wait_s=0., completion_s=0., validation_s=0.)
            for turn in range(warmup + iterations):
                began = time.perf_counter()
                for tensor, saved in zip(measured_buffers, initial_state):
                    tensor.copy_(saved)
                stream.synchronize()
                wall_costs["reset_s"] += time.perf_counter() - began
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                began = time.perf_counter()
                begin.record(stream)
                graph.replay(stream)
                end.record(stream)
                end.synchronize()
                wall_costs["replay_wait_s"] += time.perf_counter() - began
                began = time.perf_counter()
                verify_outputs([], phase="bench_completion", completion_only=True)
                wall_costs["completion_s"] += time.perf_counter() - began
                if turn >= warmup:
                    elapsed.append(begin.elapsed_time(end) * 1000)
            from reporting import atomic_json, timing_summary

            measured = timing_summary(elapsed)
            result = dict(
                transport="ubmem", batch=batch, history=history, resident=resident,
                index_heads=shape.heads, token_bytes=token_bytes, seed=seed,
                prefetch_tokens=prefetch_tokens, miss_percent=miss_percent, fixture=fixture["info"],
                requested_windows=shape.intermediate_fetches, actual_windows=rotating.rounds,
                timing=dict(prototype_with_update=measured),
                execution="COMPLETED", correctness="NOT_RUN", boundary_coverage=boundary_phases,
            )
            if serial_result is not None:
                result["timing"]["native_chain"] = serial_result["timing"]
                result["serial_correctness"] = serial_result["correctness"]
                result["serial_failures"] = serial_result["failures"]
            report_path = (result_output / f"b{batch}-h{history}-r{resident}-w{token_bytes}-p{prefetch_tokens}"
                           / "report.json") if result_output else None
            if report_path:
                atomic_json(report_path, result)
            began = time.perf_counter()
            verify(list(range(1, batch + 1)), phase="timed_replay")
            wall_costs["validation_s"] += time.perf_counter() - began
            result["wall_costs"] = wall_costs
            print(f"BENCH WALL COSTS fused: {wall_costs}", flush=True)
            result['correctness'] = checks.status
            result['correctness_failures'] = checks.failures
            if report_path:
                atomic_json(report_path, result)
            print(
                f"UBMEM_GRAPH_EVENT_US p50={np.percentile(elapsed, 50):.3f} "
                f"p99={measured['p99_us']:.3f} mean={measured['mean_us']:.3f} iterations={iterations} "
                "includes=init,LI,prefetch,resolve,final_fetch,index_update "
                "excludes=fixture_restore,native_li,verification,CPU_LRU,SFA",
                flush=True,
            )
            if trace_rounds:
                from diagnostics import collect_ubmem

                def restore_fixture():
                    for tensor, saved in zip(measured_buffers, initial_state):
                        tensor.copy_(saved)

                folder = trace_output / f"b{batch}-h{history}-r{resident}-w{token_bytes}"
                collect_ubmem(torch, rotating, graph, restore_fixture,
                              lambda: verify(list(range(1, batch + 1)), phase="profiled_replay"),
                              trace_rounds, folder, streaming=forward, level=diagnostic_level)
                result['correctness'] = checks.status
                result['correctness_failures'] = checks.failures
                if report_path:
                    atomic_json(report_path, result)
        graph.reset()
        graph = None
        lease.close()
        lease = None
        if shared_pool is None:
            runtime.release_access_table(table)
            table = None
            runtime.batch_free(keys)
            runtime.shutdown()
        else:
            shared_pool.release()
            table = None
    finally:
        # A failed fence deliberately leaves the lease and its buffers retained
        # by KVAF. Never continue freeing storage after uncertain completion.
        if graph is not None:
            if graph.capturing:
                graph.capture_end()
            if graph.ready:
                graph.synchronize(stream)
            graph.reset()
        if lease is not None:
            lease.close()
        if shared_pool is None:
            if table is not None:
                runtime.release_access_table(table)
            if runtime.initialized:
                runtime.shutdown()
    print(
        f"UBMEM_LIFETCH_{checks.status} device={device_id} batch={batch} heads={shape.heads} "
        f"history={history} resident={resident} token_bytes={token_bytes} "
        f"requested_miss_pct={miss_percent:g} Q4_unique_misses={cold_count} "
        f"prefetch_tokens={prefetch_tokens} "
        "put=native_li=topk=fetch=sfa=index_update=graph=packed_queries=padding"
    )

    return checks.status == "PASS" and (serial_result is None or serial_result["correctness"] == "PASS")


def run_ubmem(config, directory, *, shared_pool=None):
    import acl
    import kvaf
    import torch
    import torch_npu  # noqa: F401

    build = Path(config.kvaf_root).resolve() / 'build-npu'
    if not Path(kvaf.__file__).resolve().is_relative_to(build / 'python/kvaf'):
        raise RuntimeError(f'Wrong KVAF module {kvaf.__file__}; expected {build}')
    if not torch.npu.is_available():
        raise RuntimeError('UBmem requires a real NPU')
    torch.npu.set_device(config.device)
    cann = kvaf_cann(build)
    aics = query_device_resources(cann, config.device)['effective_aic']
    soc = acl.get_soc_name()
    selected_library = None
    failed = False
    for batch in config.batches:
        shape = UbmemGeometry(batch, config.history, config.resident, config.index_heads,
                              TOKEN_BYTES, aics, config.intermediate_fetches, soc)
        library = build_ubmem_library(cache_dir=build / 'lifetch_ubmem', cann=cann,
                                     shape=shape, optimized=True)
        if selected_library is None:
            selected_library = library.resolve()
            print(f'UBMEM_RUNTIME_BINARY path={selected_library} '
                  f'sha256={hashlib.sha256(library.read_bytes()).hexdigest()}', flush=True)
        elif library.resolve() != selected_library:
            raise AssertionError('Runtime shape change unexpectedly rebuilt UBmem LIFetch')
        progress(directory, f'UBmem BS={batch} miss={config.miss_percent:g}: paired benchmark')
        passed = run_ubmem_case(
            kvaf, torch, build, config.device, shape, library,
            prefetch_tokens=config.prefetch_tokens, args_server=config.kvaf_server,
            miss_percent=config.miss_percent, warmup=config.warmup, iterations=config.iterations,
            trace_rounds=config.iterations if config.mode == 'diagnose' else 0,
            diagnostic_level=config.level, trace_output=directory / 'trace', seed=config.seed, result_output=directory / 'cases',
            serial_compare=True, check_case=getattr(config, 'check_case', None),
            check_suite=getattr(config, 'check_suite', None),
            shared_pool=shared_pool, compact=config.mode in ('bench', 'diagnose'),
        )
        failed |= not passed
    return int(failed)


class FusedReplayState:
    """Own one path's Index/hot/LRU state for an entire captured sequence.

    Runtime, access table and compiled library are owned by the caller. All
    input uploads and CPU checks occur outside timed graph execution. A closed
    state must not have live graphs or unfinished device work.
    """
    def __init__(self, torch, runtime, table, library, initial, pages, addresses,
                 *, transport, device, aics, windows, prefetch_tokens, soc, prefetch_rank_start=1024, lease=None):
        from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, UbmemLease
        from kvip.cache.index_storage import IndexCacheStorage
        from kvip.operators.lifetch.binding import LIFetchBinding
        from kvip.operators.lifetch.index_update import DirectIndexUpdate, UbmemIndexUpdate
        from kvip.operators.lifetch.workspace import LIFetchWorkspace, UbmemWorkspace
        from kvip.operators.lifetch_ubmem.binding import UbmemBinding
        from kvip.operators.lifetch_ubmem.build import UbmemGeometry

        self.torch, self.device, self.pages, self.initial = torch, device, pages, initial
        self.logical_addresses = addresses
        g = initial.geometry
        self.batch, self.history, self.resident = g['batch'], g['history'], g['resident']
        self.stride, self.token_bytes = g['output_stride'], g['token_bytes']
        self.storage = IndexCacheStorage(torch, self.batch + 1, self.history, self.resident, device)
        common = dict(batch=self.batch, history=self.history, aics=aics, windows=windows,
                      device=device, index_heads=g['heads'])
        ubmem = transport == 'ubmem'
        self.workspace = (UbmemWorkspace(torch, resident=self.resident, **common) if ubmem
                          else LIFetchWorkspace(torch, **common))
        self.output = torch.zeros((self.batch + 1, self.stride, self.token_bytes), dtype=torch.uint8, device=device)
        output_rows = self.output.view(-1, self.token_bytes)
        stream = torch.npu.current_stream(device)
        self.workspace.inputs.prepare([4] * self.batch, query_owners=list(range(1, self.batch + 1)))
        self.owns_lease = lease is None
        self.lease = lease or (UbmemLease(torch, runtime, output_rows, stream) if ubmem
                              else KvafGetLease(torch, runtime, library, output_rows, stream))
        try:
            if not ubmem:
                self.workspace.allocate_rotation(
                    torch, batch=self.batch, history=self.history, resident=self.resident,
                    aics=aics, windows=windows, status_bytes=self.lease.sizes['status'])
            binding = dict(
                query_ends=self.workspace.inputs.query_ends.view(self.batch, 4),
                hot_valid=self.storage.hot_valid, row_owners=self.workspace.inputs.row_owners,
                token_addresses=self.storage.addresses,
                tail_bounds=self.workspace.tail_bounds if g.get("sparse_mode", 3) else None,
                output_stride=self.stride, scratch=self.workspace.scratch,
            )
            if ubmem:
                self.lease.register_table(table)
                shape = UbmemGeometry(self.batch, self.history, self.resident, g['heads'],
                                      self.token_bytes, aics, windows, soc)
                self.rotating = UbmemBinding(torch, library, self.workspace.arguments(self.storage),
                                              self.lease, output_rows, shape=shape,
                                              prefetch_limit=prefetch_tokens, **binding)
            else:
                owner = KvafQueueView(self.lease, output_rows, table.addresses, table.valid)
                self.rotating = LIFetchBinding(
                    torch, library, self.workspace.arguments(self.storage), owner,
                    batch=self.batch, history=self.history, resident=self.resident,
                    intermediate_fetches=windows, prefetch_limit=prefetch_tokens,
                    prefetch_rank_start=prefetch_rank_start,
                    aic_count=aics, token_bytes=self.token_bytes, **binding)
            update_type = UbmemIndexUpdate if ubmem else DirectIndexUpdate
            self.update = update_type(
                torch, keys=self.storage.keys, scales=self.storage.scales,
                addresses=self.storage.addresses, logical_ids=self.storage.logical_ids,
                row_owners=self.workspace.inputs.row_owners, rotating=self.rotating,
                query_inputs=self.workspace.inputs,
                status=torch.zeros(self.batch, dtype=torch.int32, device=device),
                outputs=torch.zeros((2, self.batch * 4, 1, 2048), dtype=torch.int32, device=device))
            self.forward = StreamingForward(self.rotating, self.update, torch, prefetch_tokens)
            self.reset()
        except BaseException:
            if self.owns_lease:
                self.lease.close()
            raise

    def upload(self, target, array):
        self_value = self.torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        target.copy_(self_value.reshape(target.shape))

    def reset(self):
        """Whole-sequence reset only. Never reset KVAF queues."""
        from kvip.cache.lru import SlotLRU
        initial = self.initial
        self.upload(self.storage.logical_ids[1:], initial.captured_layout.astype(np.int64))
        self.upload(self.storage.hot_valid[1:], initial.captured_hot_valid.astype(np.int32))
        self.output.zero_()
        self.policies = []
        for row in range(self.batch):
            self.upload(self.output[row + 1, :self.resident],
                        self.pages.token_payload(row, initial.captured_layout[row, :self.resident]))
            policy = SlotLRU()
            policy.resize(self.resident, self.resident)
            # The capture exports 2048 oldest candidates, not a full LRU chain.
            # Preserve that ordered prefix; initialize unknown remainder by slot.
            oldest = initial.captured_lru[row]
            remaining = np.setdiff1d(np.arange(self.resident), oldest, assume_unique=True)
            chain = np.concatenate((oldest, remaining)).astype(np.int32)
            policy.next[0, chain] = np.r_[chain[1:], -1]
            policy.previous[0, chain] = np.r_[-1, chain[:-1]]
            policy.head[0], policy.tail[0] = chain[0], chain[-1]
            policy._select()
            self.policies.append(policy)
        self.prepare(initial)

    def prepare(self, inputs):
        """Import real inputs while retaining this path's latest cache layout."""
        if inputs.identities != self.initial.identities or inputs.geometry != self.initial.geometry:
            raise ValueError('Replay request geometry changed after binding')
        layout = self.storage.logical_ids[1:].cpu().numpy()
        self.input_layout = layout.copy()
        keys, scales, valid = inputs.index_for_layout(layout)
        self.upload(self.storage.keys[1:], keys)
        self.upload(self.storage.scales[1:], scales)
        rows = np.arange(self.batch)[:, None]
        addresses = self.logical_addresses[rows, layout].copy()
        addresses[valid == 0] = 0
        self.upload(self.storage.addresses[1:], addresses.astype(np.int64))
        self.upload(self.workspace.inputs.query.view(self.torch.uint8), inputs.query)
        # Reinterpret captured BF16 bits; numeric conversion changes the weights.
        self.upload(self.workspace.inputs.weights.view(self.torch.int16), inputs.weights.view(np.int16))
        self.upload(self.workspace.inputs.scale, inputs.query_scale)
        self.upload(self.workspace.inputs.query_ends, inputs.query_ends)
        self.upload(self.workspace.tail_bounds, inputs.tail_bounds)
        for row, (begin, end) in enumerate(inputs.tail_bounds):
            ids = np.arange(int(begin), int(end))
            if not len(ids):
                continue
            capacity = self.stride - self.resident
            if capacity <= 0 or len(ids) > capacity:
                raise ValueError('Captured ABC range exceeds replay output stride')
            slots = self.resident + ids % capacity
            payload = self.torch.from_numpy(self.pages.token_payload(row, ids)).to(self.device)
            self.output[row + 1, self.torch.from_numpy(slots).to(self.device)] = payload
        candidates = np.stack([policy.slots[0] for policy in self.policies])
        reserved = np.full((self.batch, self.resident), -1, dtype=np.int32)
        for row, slots in enumerate(candidates):
            good = slots >= 0
            reserved[row, slots[good]] = np.arange(len(slots), dtype=np.int32)[good]
        self.upload(self.rotating.lru_slots, candidates)
        self.upload(self.rotating.reserved_hot, reserved)

    def validate_index(self, inputs, output):
        """Reject damaged persistent state before the next prepare can overwrite it."""
        layout = self.storage.logical_ids[1:].cpu().numpy()
        try:
            keys, scales, valid = inputs.index_for_layout(layout)
        except (ValueError, AssertionError) as error:
            raise RuntimeError(f'Replay Index layout invalid: {error}') from error
        addresses = self.logical_addresses[np.arange(self.batch)[:, None], layout].copy()
        addresses[valid == 0] = 0
        for name, expected in (('keys', keys), ('scales', scales), ('addresses', addresses)):
            actual = getattr(self.storage, name)[1:].cpu().numpy()
            if not np.array_equal(actual, expected):
                raise RuntimeError(f'Replay persistent Index {name} mismatch')
        hot_valid = self.storage.hot_valid[1:].cpu().numpy()
        if np.any((hot_valid != 0) & (hot_valid != 1)):
            raise RuntimeError('Replay hot validity outside boolean domain')
        for row in range(self.batch):
            slots = np.flatnonzero(hot_valid[row])
            expected = self.pages.token_payload(row, layout[row, slots])
            if not np.array_equal(output[row, slots], expected):
                raise RuntimeError(f'Replay persistent hot KV mismatch: request={row}')
        return layout, hot_valid

    def check_completion(self):
        """Check device completion before reading outputs or reusing transport state."""
        status = self.update.status.cpu().numpy()
        if np.any(status):
            flags = self.forward.flags.cpu().numpy()
            workers = self.forward.worker_status.cpu().numpy()
            sources = self.forward.events_src.cpu().numpy()
            destinations = self.forward.events_dst.cpu().numpy()
            event_checks = []
            for row in range(self.batch):
                consumed = 0
                for phase in range(5):
                    publication = (12 + phase - 1 if self.forward.resolver_cores == 3 and phase else phase)
                    count = int(flags[row, publication, 1])
                    begin = 2048 + consumed if phase else 0
                    src = sources[row, begin:begin + count]
                    dst = destinations[row, begin:begin + count]
                    bad = (src < 0) | (src >= self.history) | (dst < 0) | (dst >= self.resident)
                    duplicates = []
                    for offset in range(0, count, 512):
                        for name, values in (('source', src), ('destination', dst)):
                            unique, counts = np.unique(values[offset:offset + 512], return_counts=True)
                            repeated = unique[counts > 1]
                            if len(repeated):
                                duplicates.append(dict(chunk=offset // 512, field=name, values=repeated[:8].tolist()))
                    event_checks.append(dict(request=row, phase=phase, count=count,
                                             invalid=int(bad.sum()), duplicates=duplicates))
                    if phase:
                        consumed += count
            detail = dict(completion=status.tolist(), index_update=workers[:, :12].tolist(),
                          flags=flags[:, :, :3].tolist(), event_checks=event_checks,
                          resolver=self.rotating.row_status.cpu().tolist(),
                          final_posts=self.rotating.final_posts.cpu().tolist())
            if hasattr(self, 'failure_directory'):
                np.savez_compressed(self.failure_directory / 'completion_failure.npz',
                                    flags=flags, worker_status=workers, sources=sources,
                                    destinations=destinations, completion=status)
                atomic_json(self.failure_directory / 'completion_failure.json', detail)
            print('REPLAY COMPLETION AUDIT: ' + json.dumps(detail), flush=True)
            raise RuntimeError(f'Replay Index/Fetch completion failed: {status.tolist()}')

    def validate(self, inputs):
        self.check_completion()
        ids = self.update.logical_output.cpu().numpy().reshape(self.batch, 4, 2048)
        slots = self.update.slot_output.cpu().numpy().reshape(self.batch, 4, 2048)
        invalid = (ids < 0) | (ids >= self.history) | (slots < 0) | (slots >= self.stride)
        if invalid.any():
            row, query, position = np.unravel_index(invalid.argmax(), invalid.shape)
            # Resolve output rows through the same unpack map as the output worker.
            source = int(self.workspace.inputs.unpack_indices.reshape(-1)[row * 4 + query].cpu())
            source_row, source_q = divmod(source, 4)
            raw_slot = selected = None
            if 0 <= source_row < self.batch:
                raw_slot = int(self.rotating.locations.reshape(self.batch, 4, 2048)[source_row, source_q, position].cpu())
                selected = int(self.workspace.topk.reshape(self.batch, 4, 2048)[source_row, source_q, position].cpu())
            raise RuntimeError(
                f'Replay output token/slot outside allocated bounds: request={row} Q={query + 1} '
                f'position={position} token={ids[row, query, position]} history={self.history} '
                f'slot={slots[row, query, position]} stride={self.stride} invalid={invalid.sum()} '
                f'source={source} selected={selected} resolver_slot={raw_slot}')
        output = self.output[1:].cpu().numpy()
        layout, hot_valid = self.validate_index(inputs, output)
        for row in range(self.batch):
            mask = slots[row] < self.resident
            selected_slots = slots[row][mask]
            selected_ids = ids[row][mask]
            invalid = hot_valid[row, selected_slots] == 0
            wrong_owner = layout[row, selected_slots] != selected_ids
            # Index-only exchanges invalidate freed source slots without
            # moving their KV. This round may still read the old bytes there;
            # only persistent valid slots promise the committed Index owner.
            # The full-byte check below validates every selected output slot.
            bad = (~invalid) & wrong_owner
            if np.any(bad):
                positions = np.argwhere(mask)
                examples = []
                for index in np.flatnonzero(bad)[:16]:
                    q, position = positions[index]
                    slot, token = int(selected_slots[index]), int(selected_ids[index])
                    expected = self.pages.token_payload(row, np.asarray([token]))[0]
                    examples.append(dict(query=int(q) + 1, position=int(position),
                                         token=token, slot=slot,
                                         hot_valid=int(hot_valid[row, slot]),
                                         slot_token=int(layout[row, slot]),
                                         kv_matches=bool(np.array_equal(output[row, slot], expected))))
                detail = dict(request=row, round=inputs.source_round,
                              invalid_positions=int(invalid.sum()),
                              wrong_owner_positions=int(wrong_owner.sum()), examples=examples)
                print('REPLAY HOT MAPPING AUDIT: ' + json.dumps(detail), flush=True)
                raise RuntimeError(f'Replay selected hot mapping invalid: {json.dumps(detail)}')
        failures = []
        try:
            inputs.validate_topk(ids)
        except AssertionError as error:
            failures.append(str(error))
            if hasattr(self.rotating, 'incremental'):
                meta = self.rotating.incremental.cpu().numpy().view(np.uint64)
                raw = self.workspace.compute.cpu().numpy().view(np.uint16)
                base = self.workspace.compute.data_ptr()
                audit = []
                for row in range(self.batch):
                    layout = self.input_layout[row]
                    inverse = np.argsort(layout)
                    for q in range(4):
                        offset = (int(meta[row, q, 0]) - base) // 2
                        if offset < 0 or offset + self.history > len(raw):
                            continue
                        scores = raw[offset:offset + self.history][inverse]
                        valid = inputs.logical_score_valid[row, q]
                        population = scores[valid]
                        cutoff = int(np.partition(population, len(population) - 2048)[-2048])
                        selected = ids[row, q]
                        missing = int(np.sum(population > cutoff) - np.sum(scores[selected] > cutoff))
                        below = int(np.sum(scores[selected] < cutoff))
                        reference = inputs.logical_score_keys[row, q]
                        mismatch = np.flatnonzero(valid & (scores != reference))
                        if missing or below or len(mismatch):
                            audit.append(dict(request=row, query=q + 1, device_cutoff=cutoff,
                                missing_above=missing, below_cutoff=below,
                                score_mismatches=int(len(mismatch)),
                                examples=[dict(token=int(i), device=int(scores[i]), reference=int(reference[i]))
                                          for i in mismatch[:8]]))
                print('REPLAY SCORE AUDIT: ' + json.dumps(audit), flush=True)
                if getattr(self, 'failure_directory', None):
                    atomic_json(self.failure_directory / 'score_failure.json', audit)
        for row in range(self.batch):
            expected = self.pages.token_payload(row, ids[row].reshape(-1))
            actual = output[row, slots[row].reshape(-1)]
            if not np.array_equal(actual, expected):
                failures.append(f'KV bytes: request={row}')
            hot = slots[row].reshape(-1)
            self.policies[row].commit(hot[hot < self.resident])
        assert not failures, f'Replay round={inputs.source_round}: ' + '; '.join(failures)

    def close(self):
        if self.owns_lease:
            self.lease.close()


class NativeReplayState:
    """Original LI/LIDU/BatchGet, with independent persistent logical maps."""
    upload = FusedReplayState.upload

    def __init__(self, torch, torch_npu, runtime, table, initial, pages, *, device):
        from types import SimpleNamespace

        self.torch, self.torch_npu, self.device = torch, torch_npu, device
        self.initial, self.pages, self.native_li = initial, pages, native_li
        g = initial.geometry
        self.batch, self.history, self.resident, self.stride = (
            g[name] for name in ('batch', 'history', 'resident', 'output_stride'))
        heads, width = g['heads'], g['token_bytes']
        if self.stride % 128:
            raise ValueError('Native replay output stride must align to a KV block')
        self.query = torch.empty((self.batch, 4, heads, 128), dtype=torch.float8_e4m3fn, device=device)
        self.keys = torch.empty((self.batch, self.history, 128), dtype=torch.uint8, device=device)
        self.scales = torch.empty((self.batch, self.history), dtype=torch.float32, device=device)
        self.weights = torch.empty((self.batch, 4, heads), dtype=torch.bfloat16, device=device)
        self.query_scale = torch.empty((self.batch, 4, heads), dtype=torch.float32, device=device)
        lengths = torch.full((self.batch,), self.history, dtype=torch.int32, device=device)
        self.fixture = dict(
            sparse_mode=g.get('sparse_mode', 3), query=self.query, key=self.keys.view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
            actual_query=torch.arange(1, self.batch + 1, dtype=torch.int32, device=device) * 4,
            actual_key=lengths, li_query_lengths=torch.full((self.batch,), 4, dtype=torch.int32, device=device),
            block_table=torch.arange(self.batch * self.history // 128, dtype=torch.int32, device=device).reshape(self.batch, -1),
            native_inputs=dict(weights=self.weights, query_scale=self.query_scale,
                               key_scale=self.scales.reshape(-1, 128, 1)))
        mapping = np.full((self.batch, self.history), -1, dtype=np.int32)
        for row in range(self.batch):
            hot = initial.captured_hot_valid[row] != 0
            mapping[row, initial.captured_layout[row, :self.resident][hot]] = np.flatnonzero(hot)
        if not hasattr(pages, 'cpu_payload'):
            pages.cpu_payload = torch.from_numpy(np.stack([
                pages.token_payload(row, np.arange(self.history)) for row in range(self.batch)]))
        session = SimpleNamespace(runtime=runtime, device_table=table, batch_get_attempted=False,
                                  cpu_payload=pages.cpu_payload)
        owner = SimpleNamespace(batch=self.batch, resident_budget=self.stride,
                                cache_map=torch.from_numpy(mapping).to(device), lease=None, session=session)
        self.native = NativeMtp(torch, device, owner, self.fixture, source_tokens=self.history,
                                padded_tokens=self.history, token_bytes=width)
        self.native.bitmap = torch.zeros((self.batch, 4176), dtype=torch.int32, device=device)
        self.table = table
        self.reset()

    def reset(self):
        from kvip.cache.lru import SlotLRU
        self.native.reset()
        self.policies = []
        for row in range(self.batch):
            policy = SlotLRU()
            policy.resize(self.resident, self.resident)
            oldest = self.initial.captured_lru[row]
            remainder = np.setdiff1d(np.arange(self.resident), oldest, assume_unique=True)
            order = np.concatenate((oldest, remainder)).astype(np.int32)
            policy.next[0, order] = np.r_[order[1:], -1]
            policy.previous[0, order] = np.r_[-1, order[:-1]]
            policy.head[0], policy.tail[0] = order[0], order[-1]
            policy._select()
            self.policies.append(policy)
        self.prepare(self.initial)

    def prepare(self, inputs):
        if inputs.identities != self.initial.identities or inputs.geometry != self.initial.geometry:
            raise ValueError('Native replay request geometry changed')
        self.upload(self.query.view(self.torch.uint8), inputs.query)
        self.upload(self.keys, inputs.logical_keys)
        self.upload(self.scales, inputs.logical_scales)
        self.upload(self.weights.view(self.torch.int16), inputs.weights.view(np.int16))
        self.upload(self.query_scale, inputs.query_scale)
        # Native sparse mode aligns Q4 to the final key length.
        if self.fixture['sparse_mode'] == 3 and not np.all(np.diff(inputs.query_ends, axis=1) == 1):
            raise ValueError('Native replay requires consecutive causal Q4 positions')
        self.upload(self.fixture['actual_key'], inputs.query_ends[:, -1])
        forward = self.native.forward.cpu().numpy()
        reverse = self.native.reverse.cpu().numpy()
        output = self.native.output.view(self.batch, self.stride, -1)
        tail_capacity = self.stride - self.resident
        for row, (begin, end) in enumerate(inputs.tail_bounds):
            # Expired ring entries cannot remain as false hits after wraparound.
            forward[row, forward[row] >= self.resident] = -1
            reverse[row, self.resident:] = -1
            ids = np.arange(int(begin), int(end))
            if not len(ids):
                continue
            if tail_capacity <= 0 or len(ids) > tail_capacity:
                raise ValueError('Native replay ABC exceeds bound output')
            slots = self.resident + ids % tail_capacity
            forward[row, ids] = slots
            reverse[row, slots] = ids
            output[row, self.torch.from_numpy(slots).to(self.device)] = self.torch.from_numpy(
                self.pages.token_payload(row, ids)).to(self.device)
        self.upload(self.native.forward, forward)
        self.upload(self.native.reverse, reverse)
        self.native.cursors.zero_()
        self.native.bitmap.zero_()
        for row, policy in enumerate(self.policies):
            self.upload(self.native.preferred[row], policy.slots[0])
            self.upload(self.native.bitmap[row, :policy.bitmap.shape[1]], policy.bitmap[0])
        published = inputs.logical_address_valid.reshape(self.batch, -1, 128)[:, :, 0]
        self.upload(self.table.valid, published.reshape(-1))

    def forward(self):
        self.selected = self.native_li(self.torch_npu, self.fixture)
        self.native.launch(self.selected)

    def validate(self, inputs):
        ids = self.selected.cpu().numpy().reshape(self.batch, 4, 2048)
        if np.any(ids < 0) or np.any(ids >= self.history):
            raise RuntimeError('Native replay output token outside allocated bounds')
        failures = []
        try:
            inputs.validate_topk(ids)
        except AssertionError as error:
            failures.append(str(error))
        try:
            self.native.verify(self.selected)
        except ValueError as error:
            if str(error).startswith(('native_mtp_payload_mismatch:', 'native_mtp_exported_slots_mismatch:')):
                failures.append(str(error))
            else:
                raise
        mapping = self.native.forward.cpu().numpy()
        for row, policy in enumerate(self.policies):
            slots = mapping[row, ids[row].reshape(-1)]
            policy.commit(slots[(slots >= 0) & (slots < self.resident)])
        assert not failures, '; '.join(failures)

    def close(self):
        pass  # No lease: caller owns the ordinary KVAF Runtime and table.


def measure_replay_batch(torch, states, sequence, config, directory, *, fused):
    """One graph contains iterations independent cache states, reused per round."""
    from contextlib import ExitStack
    from diagnostics import algorithm_before, algorithm_after
    from performance import profile_graph_batch
    directory = Path(directory)
    stream = torch.npu.current_stream(states[0].device)
    graph = torch.npu.NPUGraph()
    capture = torch.npu.Stream()
    checks = CorrectnessRecorder()
    pending = False
    def replay():
        nonlocal pending
        pending = True
        graph.replay()
    class TrackedGraph:
        def replay(self):
            replay()
        def completed(self):
            nonlocal pending
            pending = False
    # Multi-round validation also advances the host LRU for the next input.
    compact = config.mode == "bench" and len(sequence.rounds) == 1

    def validate_all(inputs, label):
        nonlocal pending
        pending = True
        for i, state in enumerate(states):
            if not compact or (label == 'measured' and i in (0, len(states) - 1)):
                check = f'{label}/input{i}/round{inputs.source_round}'
                try:
                    checks.check(check, lambda state=state: state.validate(inputs))
                except RuntimeError as error:
                    raise RuntimeError(f'{check}: {error}') from error
            elif fused:
                state.check_completion()
        pending = False
    try:
        for state in states:
            state.reset()
            if fused:
                state.forward.set_diagnostics(False)
        # KVAF registers each distinct caller output lazily on its first Get.
        # Registration uploads entity metadata, which cannot run inside capture.
        # Exercise every state on the normal stream, then restore it so neither
        # capture nor measurement inherits a warmed cache from this setup.
        initial = next(iter(sequence))
        for i, state in enumerate(states):
            pending = True
            state.forward()
            stream.synchronize()
            if not compact:
                checks.check(f'pre_capture/input{i}', lambda state=state: state.validate(initial))
            elif fused:
                state.check_completion()
            pending = False
            state.reset()
        stream.synchronize()
        with ExitStack() as scopes:
            if fused:
                for lease in {id(state.lease): state.lease for state in states}.values():
                    scopes.enter_context(lease.capture_scope())
            pending = True
            with torch.npu.graph(graph, stream=capture):
                for state in states:
                    state.forward()
            capture.synchronize()
            pending = False
        if fused:
            for lease in {id(state.lease): state.lease for state in states}.values():
                if hasattr(lease, "retain_graph"):
                    lease.retain_graph(graph)
        for warmup in range(config.warmup):
            for state in states:
                state.reset()
            for inputs in sequence:
                for state in states:
                    state.prepare(inputs)
                stream.synchronize()
                replay()
                stream.synchronize()
                validate_all(inputs, f'warmup{warmup}')
        for state in states:
            state.reset()
        rounds = {}
        for inputs in sequence:
            before = []
            for state in states:
                state.prepare(inputs)
                if fused and config.mode == 'diagnose':
                    initial = algorithm_before(state.rotating)
                    initial['logical_ids'] = state.input_layout.copy()
                    before.append(initial)
            stream.synchronize()
            folder = directory / 'diagnose' / f'round-{inputs.source_round}'
            measured = profile_graph_batch(torch, TrackedGraph(), calls=len(states), folder=folder,
                                           serial=not fused, transport=config.transport, stream=stream)
            validate_all(inputs, 'measured')
            rounds[inputs.source_round] = measured['timing']
            samples = []
            for i, item in enumerate(measured['samples']):
                accounting = dict(item, kernel_span_us=item['us'], event_us=None,
                                  boundary_residual_us=None, timing_source='profiler invocation')
                sample = dict(round=i, repeat=i, source_round=inputs.source_round,
                              us=item['us'], accounting=accounting)
                if before:
                    sample['algorithm'] = algorithm_after(states[i].rotating, before[i])
                    totals = sample['algorithm'].get('totals', {})
                    sample.update(prefetch=totals.get('prefetch'), final=totals.get('final'))
                samples.append(sample)
            atomic_json(folder / ('full_attribution.json' if fused else 'native_attribution.json'),
                        dict(path='fused' if fused else 'serial', samples=samples,
                             graph_accounting=measured['accounting'], graph_calls=len(states),
                             timing=measured['timing'], trace=measured['trace']))
        # Detailed stamps are additional diagnostics, never performance samples.
        if fused and config.mode == 'diagnose' and config.level >= 2:
            clock_mhz = states[0].forward.calibrate_counter_mhz()
            for state in states:
                state.reset()
                state.forward.counter_clock_mhz = clock_mhz
                state.forward.set_diagnostics(True)
            for inputs in sequence:
                for state in states:
                    state.prepare(inputs)
                    state.forward.times.zero_()
                    if hasattr(state.rotating, 'rank_progress'):
                        state.rotating.rank_progress.zero_()
                stream.synchronize()
                replay()
                stream.synchronize()
                validate_all(inputs, 'detail')
                from diagnostics import decode_rank_progress
                details = []
                for i, state in enumerate(states):
                    sample = dict(round=i, source_round=inputs.source_round,
                                  streaming_cycles=state.forward.diagnostic_cycles())
                    if hasattr(state.rotating, 'rank_progress'):
                        records = state.rotating.rank_progress.cpu().tolist()
                        bounds = state.rotating.query_ends.cpu().tolist()
                        sample['rank_progress'] = [
                            [b, q + 1, epoch, *decode_rank_progress(word, bounds[b][q])]
                            for b, queries in enumerate(records) for q, epochs in enumerate(queries)
                            for epoch, word in enumerate(epochs[:state.rotating.rounds]) if word]
                    details.append(sample)
                atomic_json(directory / 'diagnose' / f'round-{inputs.source_round}' / 'detail_samples.json', details)
        if fused and config.mode == 'diagnose' and config.level == 3 and config.transport == 'urma' and 'fetch' in config.stages:
            from performance import compare_fetch_only
            first = states[0]
            first.forward.set_diagnostics(False)
            first.reset()
            first.prepare(next(iter(sequence)))
            initial_output = first.output.clone()
            pending = True
            first.forward()
            stream.synchronize()
            first.validate(next(iter(sequence)))
            pending = False
            isolated_graphs = []
            def mark_launched():
                nonlocal pending
                pending = True
            def mark_completed(**kwargs):
                nonlocal pending
                pending = False
            owner = SimpleNamespace(output=first.output, initial_output=initial_output,
                                    abi_sizes=first.lease.sizes, graphs=isolated_graphs,
                                    mark_launched=mark_launched, mark_completed=mark_completed)
            try:
                rows = compare_fetch_only(torch, owner, first.rotating, first.forward, config)
                atomic_json(directory / 'diagnose' / 'fetch_only.json', dict(status='PASS', rows=rows))
            finally:
                stream.synchronize()
                for isolated in isolated_graphs:
                    isolated.reset()
        all_spans = [v for timing in rounds.values() for v in timing['samples_us']]
        event_mean = sum(t['mean_us'] for t in rounds.values()) / len(rounds)
        overall = timing_summary(all_spans)
        overall.update(profiler_mean_us=overall['mean_us'], mean_us=event_mean,
                       event_mean_us=event_mean, measured_graphs=len(rounds),
                       graph_calls=len(states), timing_protocol='one_profiled_graph',
                       stage_mean_us={stage: sum(t['stage_mean_us'].get(stage, 0.) for t in rounds.values())/len(rounds)
                                      for stage in dict.fromkeys(s for t in rounds.values() for s in t['stage_mean_us'])},
                       inter_kernel_gap_mean_us=sum(t['inter_kernel_gap_mean_us'] for t in rounds.values())/len(rounds),
                       overlap_mean_us=sum(t['overlap_mean_us'] for t in rounds.values())/len(rounds),
                       outside_invocations_mean_us=sum(t['outside_invocations_mean_us'] for t in rounds.values())/len(rounds))
        return dict(rounds=rounds, all_rounds=overall, correctness=checks.status,
                    failures=checks.failures, sequence_graph_sum=dict(
                        mean_us=sum(t['mean_us'] for t in rounds.values()), samples=1))
    except BaseException as error:
        if pending:
            NativeFetchSession.quarantine('batched_replay_completion_uncertain', error)
        raise
    finally:
        if not pending:
            if fused:
                for state in states:
                    state.forward.set_diagnostics(False)
            graph.reset()


def run_path(config, directory, sequence, pages, *, fused, round_check=None, shared_pool=None):
    import acl
    import kvaf
    import torch
    import torch_npu
    from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library
    from kvip.runtime.build import build_libraries, kvaf_build_paths
    from kvip.runtime.device import query_device_resources

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
    borrowed = shared_pool.borrow(torch, device, g['batch'], g['history'], workers) if shared_pool else None
    runtime = borrowed.runtime if borrowed else kvaf.NpuRuntime(
        str(build / 'bin/kvaf_execute_batch_get_kernels.o'), width, 128, **kwargs)
    state = graph = table = None
    batch_states = []
    pending = False
    def replay_stage(label):
        progress(directory.parent, f"REPLAY {'fused' if fused else 'serial'}: {label}")
    replay_stage('transport initialization')
    try:
        block_bytes = 128 * width
        slot_bytes = ((block_bytes + 4095) // 4096 * 4096
                      if config.transport == 'ubmem' else block_bytes)
        if borrowed:
            table, view = borrowed.table, borrowed.device_table
            pages.cpu_payload = borrowed.cpu_payload
        else:
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
            replay_stage('DDR Put completed; access table and binding')
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
                transport=config.transport, device=device, aics=aics, windows=config.intermediate_fetches,
                prefetch_tokens=config.prefetch_tokens, soc=soc,
                prefetch_rank_start=getattr(config, "prefetch_rank_start", 1024))
        else:
            state = NativeReplayState(torch, torch_npu, runtime, expanded, initial, pages, device=device)
        if config.mode in ('bench', 'diagnose') and g.get('sparse_mode') == 0 and not fused:
            from kvip.cache.lru import SlotLRU
            selected = state.native_li(torch_npu, state.fixture).cpu().numpy().reshape(g['batch'], -1)
            chains = []
            for ids in selected:
                policy = SlotLRU(); policy.resize(g['resident'], g['resident'])
                policy.commit(ids[ids < g['resident']])
                chain = []; slot = int(policy.head[0])
                while slot >= 0:
                    chain.append(slot); slot = int(policy.next[0, slot])
                chains.append(chain)
            initial.captured_lru = np.asarray(chains, dtype=np.int32)
            state.reset()
        if fused:
            state.failure_directory = directory
        replay_stage('binding and initial inputs ready')
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
        replay_stage('first eager forward and device completion')
        state.forward()
        stream.synchronize()
        replay_stage('first eager completed; correctness check')
        checks.check('eager', lambda: validate(initial))
        if config.mode in ('bench', 'diagnose'):
            batch_states.append(state)
            for _ in range(1, config.iterations):
                if fused:
                    replica = FusedReplayState(
                        torch, runtime, view, library, initial, pages, logical_addresses,
                        transport=config.transport, device=device, aics=aics, windows=config.intermediate_fetches,
                        prefetch_tokens=config.prefetch_tokens, soc=soc,
                        prefetch_rank_start=getattr(config, 'prefetch_rank_start', 1024), lease=state.lease)
                    replica.failure_directory = directory
                else:
                    replica = NativeReplayState(torch, torch_npu, runtime, expanded, initial, pages, device=device)
                batch_states.append(replica)
            replay_stage(f'one graph with {len(batch_states)} independent inputs; event + profiler')
            result = measure_replay_batch(torch, batch_states, sequence, config, directory, fused=fused)
            if checks.failures:
                result['correctness'] = 'FAIL'
                result['failures'] = checks.failures + result['failures']
            result.update(transport=config.transport, path='fused' if fused else 'serial',
                          runtime_workers=shared_pool.workers if borrowed else workers, geometry=g,
                          input_paths=[str(ref.manifest_path) for refs in sequence.rounds for ref in refs])
            atomic_json(directory / 'path_result.json', result)
            return result
        state.reset()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        scope = state.lease.capture_scope() if fused else nullcontext()
        pending = True
        replay_stage('graph capture')
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
        replay_stage('graph warmup and measurement')
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
            from diagnostics import collect_replay

            # Full-graph timings above remain saved if profiling later fails.
            result['diagnostic'] = collect_replay(
                torch, state, TrackedGraph(), sequence, validate, config,
                directory / 'diagnose')
            if result['diagnostic']['correctness'] == 'FAIL':
                result['correctness'] = 'FAIL'
                result['failures'].extend(result['diagnostic']['failures'])
            atomic_json(directory / 'path_result.json', result)
        for failure in result.get('failures', []):
            print('REPLAY CHECK FAILURE: ' + json.dumps(failure), flush=True)
        return result
    except BaseException as error:
        if pending:
            NativeFetchSession.quarantine('real_replay_completion_uncertain', error)
        raise
    finally:
        if not pending:
            if graph is not None:
                graph.reset()
            for replica in reversed(batch_states[1:]):
                replica.close()
            if state is not None:
                state.close()
            if borrowed:
                shared_pool.release()
            else:
                if table is not None:
                    runtime.release_access_table(table)
                if runtime.initialized:
                    runtime.shutdown()


def run_synthetic_batch(config, directory, *, shared_pool=None):
    """Keep the compact fixture; benchmark it through independent state slots."""
    from workloads import prepare_compact_fixture, ReplayRound
    from kvip.cache.lru import SlotLRU
    directory = Path(directory)
    fixture = directory / 'fixture'
    key = ('graph_batch', config.history, config.resident, config.miss_percent,
           config.index_heads, config.seed)
    pool = shared_pool or BenchPool(config)
    owns_pool = shared_pool is None
    if key in pool.fixtures:
        fixture = pool.fixtures[key]
    else:
        prepare_compact_fixture(fixture, config.history, config.resident, config.miss_percent,
                                config.index_heads, config.seed)
        pool.fixtures[key] = fixture
    h, r, heads = config.history, config.resident, config.index_heads
    # The previous synthetic runner treats physical fixture rows as logical
    # tokens. Preserve that identity and the exact Q/Index/scale bit patterns.
    order = np.load(fixture / 'physical_to_logical.npy')
    scores = np.load(fixture / 'logical_scores.npy')[:, order]
    keys = np.fromfile(fixture / 'arg1.bin', np.uint8).reshape(h, 128)
    scales = np.fromfile(fixture / 'arg4.bin', np.float32)
    query = np.fromfile(fixture / 'arg0.bin', np.uint8).reshape(4, heads, 128)
    weights = (np.fromfile(fixture / 'arg2.bin', np.float32).view(np.uint32) >> 16).astype(np.uint16).reshape(4, heads)
    qscale = np.fromfile(fixture / 'arg3.bin', np.float32).reshape(4, heads)
    selected = np.argpartition(scores, -TOPK, axis=1)[:, -TOPK:].reshape(-1)
    policy = SlotLRU(); policy.resize(r, r); policy.commit(selected[selected < r].astype(np.int32))
    chain = []; slot = int(policy.head[0])
    while slot >= 0:
        chain.append(slot); slot = int(policy.next[0, slot])
    failed = False
    try:
        for batch in config.batches:
            def rows(value):
                return np.broadcast_to(value, (batch, *value.shape))
            g = dict(batch=batch, history=h, resident=r, heads=heads, token_bytes=TOKEN_BYTES,
                     output_stride=r, sparse_mode=0)
            inputs = ReplayRound(1, 1, tuple((0, str(i), i+1) for i in range(batch)), g,
                rows(keys), rows(scales), rows(query), rows(weights), rows(qscale),
                np.full((batch, 4), h, np.int32), np.zeros((batch, 2), np.int32),
                np.ones((batch, h), np.uint8), tuple(tuple(None for _ in range(h//128)) for _ in range(batch)),
                rows(np.arange(h)), np.ones((batch, r), np.int32), rows(np.asarray(chain, np.int32)),
                (str(fixture),), rows(scores), np.ones((batch, 4, h), bool))
            class Sequence:
                rounds = ((SimpleNamespace(source_round=1, manifest_path=fixture/'fixture.json'),),)
                def __iter__(self):
                    yield inputs
            sequence = Sequence()
            pages = ReplayPagePlan(sequence)
            # Shared DDR already holds payload(request, token), as before.
            from workloads import payload
            pages.token_payload = lambda row, ids: payload(row, np.asarray(ids))
            folder = directory / f'case-b{batch}'
            results = {}
            for name in ('serial', 'fused'):
                results[name] = run_path(config, folder / name, sequence, pages,
                                         fused=name == 'fused', shared_pool=pool)
                report = dict(batch=batch, miss_percent=config.miss_percent, input='synthetic',
                              correctness='NOT_RUN', execution='INCOMPLETE',
                              timing={'native_chain': results['serial']['all_rounds']})
                if 'fused' in results:
                    report.update(execution='COMPLETED', correctness='PASS' if all(
                        result['correctness']=='PASS' for result in results.values()) else 'FAIL')
                    report['timing']['prototype_with_update'] = results['fused']['all_rounds']
                atomic_json(folder / 'report.json', report)
            failed |= report['correctness'] != 'PASS'
        return int(failed)
    finally:
        if owns_pool:
            pool.close()


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
    from correctness import require_stateful_round, stateful_inputs
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


class StreamingForward(LIFetchForward):
    def __init__(self, rotating, update, torch, prefetch_limit, *, fused_final=True, q_workers=2):
        if torch is not rotating.torch or prefetch_limit != rotating.prefetch_limit or not fused_final:
            raise ValueError("Diagnostics must use the exact public LIFetch binding and budget")
        super().__init__(rotating, update, q_workers=q_workers)

    def calibrate_counter_mhz(self):
        """Approximate cycle frequency against a long independent graph event."""
        if not self.q_workers:
            return None
        torch = self.torch
        output = torch.empty(4, dtype=torch.int64, device=self.rotating.device)
        function = self.rotating.library.kvip_counter_calibration
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        function.restype = ctypes.c_uint32
        stream = torch.npu.current_stream(self.rotating.device)
        capture = torch.npu.Stream(device=self.rotating.device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture):
            if function(output.data_ptr(), capture.npu_stream):
                raise RuntimeError('counter calibration launch failed')
        capture.synchronize()
        samples = []
        for _ in range(3):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            values = output.cpu().numpy().astype(np.uint64)
            elapsed_us = start.elapsed_time(end) * 1000
            cycles = int(values[1]) - int(values[0])
            if elapsed_us <= 0 or cycles < 20_000_000:
                raise RuntimeError('counter calibration returned invalid timing')
            samples.append(cycles / elapsed_us)
        graph.reset()
        return float(np.median(samples))

    def set_deferred_fetch(self, enabled):
        self.set_options(deferred_fetch=enabled)

    def set_diagnostics(self, enabled):
        self.set_options(diagnostics=enabled)

    def diagnostic_cycles(self):
        raw = self.times.cpu().numpy().astype(np.uint64)
        rows = raw[:self.batch * 64].reshape(self.batch, 64)
        li = raw[self.batch * 64:self.batch * 128].reshape(self.batch, 4, 16)
        fetch = raw[self.batch * 128:self.batch * (256 if self.q_workers else 224)].reshape(self.batch, 128 if self.q_workers else 96) if self.fused_final else None
        flags = self.flags.cpu().numpy()
        posts = (self.rotating.final_posts.cpu().numpy() if hasattr(self, "rotating") else None)
        result = []
        for b in range(self.batch):
            p, u = rows[b, :32], rows[b, 32:]
            queries = []
            for q in range(4):
                t = li[b, q]
                if not all(int(v) for v in t[:8]) or not int(t[10]) or not int(t[11]):
                    raise AssertionError(f'missing LI diagnostic timestamps: batch={b} query={q}')
                queries.append(dict(mapping_wait=int(t[1]-t[0]), barrier=int(t[2]-t[1]),
                                    load_first=int(t[3]-t[2]), classify_first=int(t[4]-t[3]),
                                    wait_second=int(t[5]-t[4]), classify_second=int(t[6]-t[5]),
                                    bitmap_merge=int(t[11]-t[10]), publish=int(t[7]-t[11]),
                                    overlap=bool(t[8]), topk_only=bool(t[9])))
            producer = []
            for q in range(4):
                t = p[10+5*q:15+5*q]
                if not all(int(v) for v in t):
                    raise AssertionError(f'missing resolver diagnostic timestamps: batch={b} query={q}')
                producer.append(dict(resolve=int(t[1]-t[0]), output=int(t[2]-t[1]),
                                     pack=int(t[3]-t[2]), publish=int(t[4]-t[3])))
            update = [int(u[13+2*q]-u[12+2*q]) for q in range(5)]
            waits = [int(u[12]-u[10])] + [int(u[12+2*q]-u[11+2*q]) for q in range(1, 5)]
            fetch_detail = None
            if fetch is not None and self.q_workers:
                f = fetch[b]
                fetch_detail = dict(queries=[dict(wait=int(f[q*24+1]-f[q*24]),
                    prepare=int(f[q*24+2]-f[q*24+1]), safe_submit_or_sync=int(f[q*24+4]-f[q*24+3]),
                    prefetch_join=int(f[q*24+6]-f[q*24+5]) if f[q*24+5] else 0,
                    conflict_get=int(f[q*24+8]-f[q*24+7]), safe_post=int(f[q*24+12]),
                    safe_cq_wait=int(f[q*24+13]), conflict_post=int(f[q*24+14]),
                    conflict_cq_wait=int(f[q*24+15]), safe_groups=int(f[q*24+16]),
                    conflict_groups=int(f[q*24+17]), safe_mode=int(f[q*24+20]), total=int(f[q*24+9]-f[q*24]),
                    group_prepare=int(f[q*24+18]), group_pack=int(f[q*24+19]))
                    for q in range(4)], finalize=dict(prefetch_join=int(f[100]-f[96]), prefetch_polls=int(f[101]),
                    prefetch_cq_poll_cycles=int(f[102]), prefetch_last_cq_observed=int(f[103]), wait=int(f[97]-f[100]),
                    residency=int(f[98]-f[97]), status=int(f[99]-f[98])))
            elif fetch is not None:
                f = fetch[b]
                fetch_detail = dict(queries=[dict(wait=int(f[q*10+1]-f[q*10]),
                    prepare=int(f[q*10+2]-f[q*10+1]), safe_get=int(f[q*10+4]-f[q*10+3]),
                    prefetch_join=int(f[q*10+6]-f[q*10+5]) if f[q*10+5] else 0,
                    conflict_get=int(f[q*10+8]-f[q*10+7]),
                    safe_post=int(f[64+q*6]), safe_cq_wait=int(f[65+q*6]),
                    conflict_post=int(f[66+q*6]), conflict_cq_wait=int(f[67+q*6]),
                    safe_groups=int(f[68+q*6]), conflict_groups=int(f[69+q*6]),
                    total=int(f[q*10+9]-f[q*10])) for q in range(4)],
                    final_prefetch_join=int(f[50]-f[49]), wait_producer_update=int(f[51]-f[50]),
                    collect=int(f[52]-f[51]), residency=int(f[53]-f[52]),
                    write_status=int(f[54]-f[53]), span=int(f[54]-f[48]))
            if fetch_detail is not None and hasattr(self.rotating, "ubmem_status"):
                fetch_detail["transport"] = "ubmem"
                fetch_detail["ubmem_queries"] = [dict(
                    query_ready_wait=int(fetch[b,q*24+1]-fetch[b,q*24]),
                    dependency_classify=int(fetch[b,q*24+2]-fetch[b,q*24+1]),
                    safe_get=int(fetch[b,q*24+4]-fetch[b,q*24+3]),
                    dependency_wait=int(fetch[b,q*24+6]-fetch[b,q*24+5]),
                    conflict_get=int(fetch[b,q*24+8]-fetch[b,q*24+7]),
                    safe_entries=int(fetch[b,q*24+21]), conflict_entries=int(fetch[b,q*24+15]),
                    dependency_mask=int(fetch[b,q*24+19]),
                    safe_calls=int(fetch[b,q*24+16]), conflict_calls=int(fetch[b,q*24+17]),
                    completion_and_index_join=int(fetch[b,q*24+9]-fetch[b,q*24+8]))
                    for q in range(4)]
                fetch_detail["note"] = "UBmem safe/conflict Gets include KVAF MTE pipeline drain; no CQ"
            counts = [int(flags[b, phase, 1]) for phase in range(5)]
            final_counts = ([dict(total=int(post[0]), safe=int(post[5]), conflict=int(post[0]-post[5]))
                             for post in posts[b]] if posts is not None else None)
            result.append(dict(batch=b, li=queries, producer=producer, update=update, fetch=fetch_detail,
                               final_counts=final_counts, prefetch_changes=counts[0], q_new_misses=counts[1:],
                               update_wait=waits,
                               input_and_bitmap=int(p[2]-p[1]), initialize=int(p[3]-p[2]),
                               metadata=int(u[23]-u[22]), consumer_start=int(u[10]),
                               consumer_finish=int(u[24]), producer_finish=int(p[8])))
        if self.q_workers and hasattr(self, "rotating"):
            events = self.rotating.events.cpu().numpy().reshape(-1, 128).view('<u8')
            ubmem = hasattr(self.rotating, "ubmem_status")
            services = []
            for event_index, event in enumerate(events):
                if not int(event[0]) or int(event[1]) < int(event[0]):
                    continue  # Profiling-disabled events are not measured zero-work events.
                words = event.view('<u4')
                services.append(dict(event=event_index, core=int(words[4]), epoch=int(words[5]),
                    lane=event_index % self.rotating.service_lanes,
                    begin_cycles=int(event[0]), end_cycles=int(event[1]),
                    predecessor_wait_cycles=(int(event[13])-int(event[4])
                        if not ubmem and int(event[4]) and int(event[13]) >= int(event[4]) else None),
                    incremental_cycles=(int(event[14])-int(event[13])
                        if not ubmem and int(event[13]) and int(event[14]) >= int(event[13]) else None),
                    scan_cycles=int(event[5]), candidate_cycles=int(event[6]), post_cycles=int(event[7]),
                    bitmap_bytes=int(event[8]), candidate_count=int(event[9]), posted_count=int(event[10])))
            result[0]['prefetch_services'] = services
            first = events[:, 0 if ubmem else 11]
            last = events[:, 1 if ubmem else 12]
            service_scope = ubmem or not np.any(first)
            if service_scope:
                first, last = events[:, 0], events[:, 1]
            first_nonzero = first[first != 0]
            last_nonzero = last[last != 0]
            rows = self.rotating.rows.cpu().numpy().view(np.uint32).reshape(self.batch, -1)
            prefetch_count = int(rows[:, 9].astype(np.uint64).sum())
            finish = [int(fetch[b, 103] or fetch[b, 100]) for b in range(self.batch) if int(rows[b, 9]) and int(fetch[b, 103] or fetch[b, 100])]
            if prefetch_count and len(first_nonzero) and finish:
                start = int(first_nonzero.min())
                end = max(finish)
                result[0]['prefetch_scope'] = dict(
                    requests=prefetch_count, bytes=prefetch_count * self.rotating.runtime_shape.tokenBytes,
                    first_doorbell=start, last_doorbell=int(last_nonzero.max()),
                    last_cq_observed=end, interval_cycles=max(0, end - start),
                    note=('Prefetch service entry to software-observed completion; submission timestamps unavailable, includes control gaps' if service_scope else
                          'first doorbell to software-observed completion; join return used when CQ was already drained'))
        if self.q_workers:
            result[0]['core_timeline_raw'] = dict(
                batch=self.batch, workers_per_query=self.q_workers, resolver_cores=self.resolver_cores,
                token_bytes=self.rotating.runtime_shape.tokenBytes,
                clock_mhz=getattr(self, 'counter_clock_mhz', None),
                cores=raw[self.batch*256:].reshape(self.resolver_cores*self.batch+4*self.q_workers, 32).tolist(),
                producer_update=raw[:self.batch*64].reshape(self.batch, 64).tolist(),
                fetch=fetch.tolist(), flags=flags.tolist())
        return result


if __name__ == "__main__":
    sys.exit(run_config(sys.argv[1]))
