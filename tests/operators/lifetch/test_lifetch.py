#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run graph-mode LIFetch correctness and performance for one transport."""
import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / 'tests/operators/lifetch/support'
sys.path.insert(0, str(SUPPORT))
from compare_transports import execute  # noqa: E402
from runner import progress, supervise  # noqa: E402


def select_supported_pipeline(parser, args):
    """Keep one validated streaming implementation, not historical experiments."""
    if args.early_fetch or args.avoid_prefetch_slots or args.legacy_fetch_layout or args.parallel_fetch_finalize:
        parser.error("Historical early/safe/layout/finalize experiments are not included in this integration branch")
    args.optimized_resolve = True
    args.streaming = args.streaming_fetch = args.fused_final = True
    args.prepare_update_metadata = not args.fetch_only_compare
    if args.q_workers is None:
        args.q_workers = 2


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-run", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--fetch-only-compare", action="store_true", help="compare the actual Q Fetch worker with native KVAF Get")
    p.add_argument("--early-fetch", action="store_true")
    p.add_argument("--streaming", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--streaming-fetch", action="store_true", help="submit each Q during streaming resolver")
    p.add_argument("--legacy-fetch-layout", action="store_true")
    p.add_argument("--parallel-fetch-finalize", action="store_true")
    p.add_argument("--prepare-update-metadata", action="store_true")
    p.add_argument("--optimized-resolve", action="store_true")
    p.add_argument("--fused-final", action="store_true", help="finish fetch status in streaming kernel")
    p.add_argument("--q-workers", type=int, choices=(1, 2, 4), default=None,
                   help="AIV workers per query across all requests")
    p.add_argument("--final-workers", type=int, choices=(1, 2, 4), default=None)
    p.add_argument("--avoid-prefetch-slots", action="store_true")
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--kvaf-server", required=True)
    p.add_argument("--kvaf-root", type=Path)
    p.add_argument("--soc", default=None)
    p.add_argument("--cann", default="/usr/local/Ascend/cann")
    p.add_argument("--index-heads", type=int, choices=(32, 64), default=32)
    p.add_argument("--history", type=int, default=131072)
    p.add_argument("--resident", type=int, default=16384)
    p.add_argument("--intermediate-fetches", type=int, default=8)
    p.add_argument("--prefetch-tokens", type=int, default=512)
    p.add_argument("--runtime-shapes", action="store_true", help="one binary, four distinct runtime shapes")
    p.add_argument("--resolver-compare", action="store_true")
    p.add_argument("--trace-rounds", type=int, default=3)
    p.add_argument("--miss-percent", type=float, default=10)
    p.add_argument("--batches", default="16")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--comparison-output", type=Path)
    p.add_argument("--comparison-sfa", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "tests/results/lifetch")
    a = p.parse_args(argv)
    select_supported_pipeline(p, a)
    if a.prepare_update_metadata and (a.q_workers is None or a.fetch_only_compare):
        p.error("--prepare-update-metadata requires Q workers and full pipeline")
    if a.parallel_fetch_finalize and (a.q_workers is None or a.fetch_only_compare):
        p.error("--parallel-fetch-finalize requires Q workers and full pipeline")
    if a.fetch_only_compare and a.q_workers != 2:
        p.error("--fetch-only-compare requires --q-workers 2 --fused-final --streaming-fetch --streaming")
    if a.q_workers is not None and not a.fused_final:
        p.error("--q-workers requires --fused-final --streaming-fetch --streaming")
    if a.fused_final and not a.streaming_fetch:
        p.error("--fused-final requires --streaming-fetch")
    if a.streaming_fetch and not a.streaming:
        p.error("--streaming-fetch requires --streaming")
    if a.streaming and (a.early_fetch or a.avoid_prefetch_slots or a.resolver_compare):
        p.error("Only the integrated streaming pipeline is supported")
    if a.kvaf_root is None:
        state = ROOT / "build/dependencies.json"
        a.kvaf_root = Path(json.loads(state.read_text())["kvaf"]) if state.exists() else ROOT / "third_party/KVAF"
    try:
        batches = [int(b) for b in a.batches.split(",")]
    except ValueError:
        p.error("batches must be comma-separated integers")
    if not batches or any(not 1 <= b <= 16 for b in batches):
        p.error("LIFetch supports BS1..16; larger batches are not supported")
    if a.fetch_only_compare and any(b < 2 for b in batches):
        p.error("--fetch-only-compare requires batch >= 2 to use both workers per query")
    if not 0 <= a.miss_percent <= 100 or a.history <= a.resident + 4:
        p.error("invalid history/resident/miss geometry")
    if a.iterations < 1 or a.warmup < 0 or a.trace_rounds < 0:
        p.error("invalid timing iterations")
    if not 1 <= a.intermediate_fetches <= 32:
        p.error("windows must be 1..32")
    if not 0 <= a.prefetch_tokens <= 2048 or a.prefetch_tokens % a.intermediate_fetches:
        p.error("prefetch tokens must be 0..2048 and divisible by windows")
    return a


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


def run_urma(argv=None):
    args = arguments(argv)
    if args.worker_run is not None:
        code = run_worker(args, args.worker_run)
        (args.worker_run / "worker-result.json").write_text(json.dumps({"exit_code": code}))
        return code
    run = args.output.resolve() / (time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    run.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-u",
        "-X",
        "faulthandler",
        str(Path(__file__).resolve()),
        "--internal-urma",
        *(sys.argv[1:] if argv is None else argv),
        "--worker-run",
        str(run),
    ]
    return supervise(command, run, timeout_seconds=args.timeout_seconds)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--internal-urma"]:
        return run_urma(argv[1:])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', required=True, choices=('urma', 'ubmem'))
    p.add_argument('--fetch-only-compare', action='store_true')
    p.add_argument('--early-fetch', action='store_true')
    p.add_argument('--streaming', action='store_true')
    p.add_argument('--streaming-fetch', action='store_true')
    p.add_argument('--legacy-fetch-layout', action='store_true')
    p.add_argument('--parallel-fetch-finalize', action='store_true')
    p.add_argument('--prepare-update-metadata', action='store_true')
    p.add_argument('--optimized-resolve', action='store_true')
    p.add_argument('--ubmem-fetch-mode', choices=('serial', 'overlap'), default='overlap')
    p.add_argument('--fused-final', action='store_true')
    p.add_argument('--q-workers', type=int, choices=(1, 2, 4))
    p.add_argument('--final-workers', type=int, choices=(1, 2, 4))
    p.add_argument('--avoid-prefetch-slots', action='store_true')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--kvaf-server', required=True)
    p.add_argument('--kvaf-root', type=Path)
    p.add_argument('--soc')
    p.add_argument('--cann', default='/usr/local/Ascend/cann')
    p.add_argument('--batches', default='16', help='comma-separated request counts')
    p.add_argument('--history', type=int, default=131072)
    p.add_argument('--resident', type=int, default=16384)
    p.add_argument('--index-heads', type=int, choices=(32, 64), default=32)
    p.add_argument('--intermediate-fetches', type=int, default=8)
    p.add_argument('--prefetch-tokens', type=int, default=512)
    p.add_argument('--miss-percent', type=float, default=10)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--trace-rounds', type=int, default=3, help='separate device profiling replays')
    p.add_argument('--runtime-shapes', action='store_true')
    p.add_argument('--resolver-compare', action='store_true', help='URMA resolver comparison')
    p.add_argument('--comparison-output', type=Path)
    p.add_argument('--comparison-sfa', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'tests/results/lifetch')
    p.add_argument('--timeout-seconds', type=int, default=300)
    a = p.parse_args(argv)
    select_supported_pipeline(p, a)
    try:
        batches = [int(x) for x in a.batches.split(',')]
    except ValueError:
        p.error('batches must be comma-separated integers')
    if not batches or any(not 1 <= x <= 16 for x in batches):
        p.error('LIFetch supports BS1..16; larger batches are not supported')
    if a.fetch_only_compare and any(x < 2 for x in batches):
        p.error('--fetch-only-compare requires batch >= 2 to use both workers per query')
    if (a.history <= a.resident + 4 or not 0 <= a.miss_percent <= 100 or a.warmup < 0
            or a.iterations < 1 or a.trace_rounds < 0 or not 1 <= a.intermediate_fetches <= 32
            or not 0 <= a.prefetch_tokens <= 2048 or a.prefetch_tokens % a.intermediate_fetches
            or a.timeout_seconds < 20):
        p.error('invalid geometry, prefetch budget, timing or timeout')
    if (a.comparison_output or a.comparison_sfa) and (len(batches) != 1 or a.runtime_shapes):
        p.error('comparison snapshots require one batch and no runtime-shapes')
    state = ROOT / 'build/dependencies.json'
    kvaf_root = (a.kvaf_root or (Path(json.loads(state.read_text())['kvaf']) if state.exists()
                                 else ROOT / 'third_party/KVAF')).resolve()
    # The selected dependency must also control imports in the worker.
    # scripts/env.sh may refer to a different checkout used for URMA.
    selected_python = os.pathsep.join((str(ROOT / 'python'), str(kvaf_root / 'build-npu/python')))
    inherited_python = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = selected_python + (os.pathsep + inherited_python if inherited_python else '')
    common = ['--device', str(a.device), '--kvaf-server', a.kvaf_server, '--history', str(a.history),
              '--resident', str(a.resident), '--index-heads', str(a.index_heads), '--miss-percent',
              str(a.miss_percent), '--prefetch-tokens', str(a.prefetch_tokens), '--warmup', str(a.warmup),
              '--iterations', str(a.iterations)]
    if a.fetch_only_compare and (a.transport != 'urma' or a.q_workers != 2 or not a.fused_final or not a.streaming_fetch or not a.streaming):
        p.error('--fetch-only-compare requires URMA and --streaming --streaming-fetch --fused-final --q-workers 2')
    if a.parallel_fetch_finalize and a.transport != 'urma':
        p.error('--parallel-fetch-finalize is an URMA experiment')
    if a.prepare_update_metadata and (a.q_workers is None or a.fetch_only_compare):
        p.error('--prepare-update-metadata requires Q workers and full pipeline')
    if a.transport == 'urma':
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--internal-urma', *common,
                   '--kvaf-root', str(kvaf_root), '--cann', a.cann, '--batches', a.batches,
                   '--intermediate-fetches', str(a.intermediate_fetches), '--trace-rounds', str(a.trace_rounds),
                   '--output', str(a.output), '--timeout-seconds', str(a.timeout_seconds)]
        if a.optimized_resolve:
            if not (a.streaming and a.streaming_fetch and a.fused_final and a.q_workers and a.prepare_update_metadata):
                p.error('--optimized-resolve requires streaming, streaming-fetch, fused-final, q-workers and prepare-update-metadata')
            command.append('--optimized-resolve')
        if a.prepare_update_metadata:
            command.append('--prepare-update-metadata')
        if a.parallel_fetch_finalize:
            if a.q_workers is None or a.fetch_only_compare:
                p.error('--parallel-fetch-finalize requires Q workers and full pipeline')
            command.append('--parallel-fetch-finalize')
        if a.legacy_fetch_layout:
            command.append('--legacy-fetch-layout')
        if a.fetch_only_compare:
            command.append('--fetch-only-compare')
        if a.final_workers is not None:
            command += ['--final-workers', str(a.final_workers)]
        if a.early_fetch:
            command.append('--early-fetch')
        if a.streaming:
            command.append('--streaming')
        if a.streaming_fetch:
            command.append('--streaming-fetch')
        if a.fused_final:
            command.append('--fused-final')
        if a.q_workers is not None:
            command += ['--q-workers', str(a.q_workers)]
        if a.avoid_prefetch_slots:
            command.append('--avoid-prefetch-slots')
        if a.soc:
            command += ['--soc', a.soc]
        if a.runtime_shapes:
            command.append('--runtime-shapes')
        if a.resolver_compare:
            command.append('--resolver-compare')
        if a.comparison_output:
            command += ['--comparison-output', str(a.comparison_output)]
        if a.comparison_sfa:
            command += ['--comparison-sfa', str(a.comparison_sfa)]
        return subprocess.call(command)
    if a.final_workers is not None:
        p.error('--final-workers is supported only by URMA')
    if not a.optimized_resolve and (a.early_fetch or a.avoid_prefetch_slots or a.streaming or a.streaming_fetch or a.fused_final or a.q_workers is not None):
        p.error('--early-fetch and --avoid-prefetch-slots are supported only by URMA')
    if a.intermediate_fetches != 8:
        p.error('UBmem fixture currently supports exactly 8 windows')
    if not a.soc:
        p.error('--soc is required for UBmem, e.g. Ascend950PR_9599')
    if a.resolver_compare:
        p.error('resolver comparison is currently supported only by URMA')
    run = a.output.resolve() / (time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}')
    run.mkdir(parents=True)
    log, summary = run / 'worker.log', run / 'summary.log'
    print(f'Summary: {summary}', flush=True)
    command = [sys.executable, '-u', str(SUPPORT / 'ubmem_smoke.py'), *common,
               '--build-dir', str(kvaf_root / 'build-npu'), '--soc', a.soc,
               '--batches', *map(str, batches), '--token-bytes', '656',
               '--trace-rounds', str(a.trace_rounds), '--trace-output', str(run / 'trace')]
    if a.runtime_shapes:
        command.append('--runtime-shapes')
    if a.comparison_output:
        command += ['--comparison-output', str(a.comparison_output)]
    if a.comparison_sfa:
        command += ['--comparison-sfa', str(a.comparison_sfa)]
    if a.optimized_resolve:
        command += ['--optimized-resolve', '--q-workers', str(a.q_workers), '--ubmem-fetch-mode', a.ubmem_fetch_mode]
    print('UBmem stage traces use separate profiled replays outside graph timing.', flush=True)
    code = execute(command, log, a.timeout_seconds, 'ubmem')
    lines = log.read_text(errors='replace').splitlines()
    selected = [line for line in lines if line.startswith(('UBMEM_LIFETCH_PASS', 'UBMEM_SFA_PASS',
                'UBMEM_GRAPH_EVENT_US', 'UBMEM_FETCH_COUNTS', 'UBMEM_WINDOW_TRIGGERS', 'UBMEM_RUNTIME_BINARY',
                'UBMEM_TRACE', 'UBMEM_FETCH_MODE', 'UBMEM_LI_SEQUENCE_WINDOWS'))]
    expected = len(batches) + int(a.runtime_shapes and (a.history, a.resident) != (33024, 8192))
    ok = (code == 0 and sum(line.startswith('UBMEM_LIFETCH_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_SFA_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_GRAPH_EVENT_US') for line in selected) == expected)
    trace_ok = sum(line.startswith('UBMEM_TRACE round=') for line in selected) == expected * a.trace_rounds
    result = [f'{"PASS" if ok else "FAIL"}: transport=ubmem exit={code}; '
              f'correctness/timing={"PASS" if ok else "FAIL"}; '
              f'trace={"COMPLETE" if trace_ok else "INCOMPLETE"}', *selected]
    if not ok:
        result += ['--- worker.log tail ---', *lines[-25:]]
    summary.write_text('\n'.join(result) + '\n')
    print('\n'.join(result), flush=True)
    print(f'Full logs: {run}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())


def test_host_pipeline_order_and_first_error(tmp_path):
    import subprocess

    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    begin = source.index('extern "C" uint32_t kvip_lifetch_streaming_forward(')
    entry = source[begin:source.index('\nextern "C" uint32_t aclrtlaunch_kvip_transport_snapshot', begin)]
    stub = r"""
#include <cstdint>
#include <cassert>
#include "runtime_shape.h"
namespace pipeline_ab { constexpr uint32_t kResolveCores=3; }
int fail=0, count=0;
void* stream_expected=(void*)3;
uint32_t step(int n,void* a,void* stream) { assert(a==(void*)1);assert(stream==stream_expected);assert(++count==n);return fail==n?99:0; }
uint32_t v9_init(void*a,void*s,const rotating::RuntimeShape*){return step(1,a,s);}
uint32_t v9_li(void*a,void*s,const rotating::RuntimeShape*){return step(2,a,s);}
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t cores,void*s,void*a){assert(cores==56);return step(3,a,s);}
"""
    main = r"""
int main(){rotating::RuntimeShape shape{};shape.batch=16;shape.finalWorkers=4;shape.aicCount=32;
for(fail=0;fail<=3;++fail){count=0;assert(kvip_lifetch_streaming_forward((void*)1,(void*)2,(void*)1,stream_expected,&shape,2)==(fail?99:0));assert(count==(fail?fail:3));}
count=0;shape.batch=17;assert(kvip_lifetch_streaming_forward((void*)1,(void*)2,(void*)1,stream_expected,&shape,2)!=0);assert(count==0);return 0;}
"""
    path = tmp_path / "launch.cpp"
    path.write_text(stub + entry + main)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
            str(path),
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)


def test_default_fixture_preserves_requested_misses(tmp_path):
    import numpy as np
    from real_fetch_inputs import prepare_full_chain_fixture

    root = tmp_path / "fixture"
    prepare_full_chain_fixture(root, 131072, 16384, 10, index_heads=32)
    order = np.load(root / "physical_to_logical.npy")
    scores = np.load(root / "logical_scores.npy")[:, order]
    selected = []
    for row in scores:
        threshold = np.partition(row, -2048)[-2048]
        tokens = np.flatnonzero(row >= threshold)
        assert len(tokens) == 2048
        assert np.count_nonzero(tokens >= 16384) == 205
        selected.extend(tokens[tokens >= 16384])
    assert len(set(selected)) == 410


def test_runtime_tiling_matches_original_geometry(tmp_path):
    import ctypes
    import subprocess
    from kvip.operators.lifetch.tiling import RuntimeShape
    from kvip.operators.lifetch.geometry import li_geometry, final_aivs, trigger_tokens
    import pytest

    source = tmp_path / "shape.cpp"
    # Compile the actual host tiling and device-shared geometry helpers.
    (tmp_path / "batch_get_abi.h").write_text("")
    source.write_text("""
#include "context.h"
#include <cstddef>
static_assert(sizeof(rotating::Context)==328,"context ABI");
static_assert(offsetof(rotating::Context, thresholdBounds)==320,"threshold pointer ABI");
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* s){return rotating::PrepareShape(s);}
extern "C" uint32_t trigger(rotating::RuntimeShape* s,uint32_t e){return rotating::Trigger(s->history,s->resident,s->rounds,e);}
extern "C" uint32_t core(rotating::RuntimeShape* s,uint32_t e,uint32_t g){return rotating::ServiceCore(s->batch,s->splitQ,s->serviceAics,e,g);}
extern "C" uint32_t queue(rotating::RuntimeShape* s,uint32_t b,uint32_t w){return rotating::FinalQueue(s->batch,s->finalWorkers,b,w);}
""")
    library = tmp_path / "shape.so"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(source),
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch/include"),
            "-I" + str(tmp_path),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    for aics in (16, 24, 32):
        for batch in range(1, aics + 1):
            for history, resident in ((65536, 8192), (131072, 16384), (196608, 32768)):
                for windows in (1, 4, 8, 16, 32):
                    for heads in (32, 64):
                        shape = RuntimeShape.prepare(
                            lib,
                            batch=batch,
                            history=history,
                            resident=resident,
                            index_heads=heads,
                            aic_count=aics,
                            windows=windows,
                        )
                        geo = li_geometry(batch, "auto", windows, aics)
                        assert shape.liCores == geo["rotating_aic"]
                        assert shape.finalWorkers == final_aivs(batch, "auto", aics)
                        assert shape.serviceLanes == geo["service_aivs_per_window"]
                        assert shape.rounds == geo["intermediate_fetches"]
                        pointer = ctypes.byref(shape)
                        assert [128 * lib.trigger(pointer, e) for e in range(shape.rounds)] == trigger_tokens(
                            batch, history, resident, "auto", windows, aics
                        )
                        assert [
                            [lib.core(pointer, e, g) for g in range(shape.serviceAics)] for e in range(shape.rounds)
                        ] == geo["service_groups"]
                        queues = [lib.queue(pointer, b, w) for b in range(batch) for w in range(shape.finalWorkers)]
                        assert len(set(queues)) == len(queues)
                        assert set(queues).isdisjoint(range(0, 2 * batch, 2))
    for changes in ({"history": 1048576}, {"resident": 12345}, {"batch": 33}, {"index_heads": 16}, {"windows": 0}):
        inputs = dict(batch=16, history=131072, resident=16384, index_heads=32, aic_count=24, windows=8)
        with pytest.raises(ValueError):
            RuntimeShape.prepare(lib, **(inputs | changes))


def test_build_has_no_workload_specialization(tmp_path):
    import json
    from unittest.mock import patch
    from types import SimpleNamespace
    from kvip.runtime.build import build_libraries

    def build(command, **kwargs):
        if command[:2] == ["cmake", "--build"]:
            folder = Path(command[2])
            for name in ("libv9_bridge.so",):
                (folder / name).write_bytes(b"universal")
        return SimpleNamespace(returncode=0)

    with patch("kvip.runtime.build.subprocess.run", side_effect=build) as commands:
        args = dict(
            cache_dir=tmp_path / "cache", cann=tmp_path / "cann", headers=tmp_path / "headers", soc="Ascend950PR_9599"
        )
        first = build_libraries(**args)
        second = build_libraries(**args)
    assert first == second and commands.call_count == 2
    options = json.loads((first[0].parent / "complete.json").read_text())
    assert not any(
        x in options
        for x in ("SIM_BATCH", "SIM_HISTORY", "SIM_RESIDENT", "INDEX_HEADS", "INTERMEDIATE_FETCHES", "AIC_COUNT")
    )


def test_build_failure_reports_first_diagnostic(tmp_path):
    from kvip.runtime.build import build_libraries
    from unittest.mock import patch
    from types import SimpleNamespace
    import pytest

    def fail(command, **kwargs):
        kwargs["stdout"].write("CMake Error: unsupported target\n" + "detail\n" * 80)
        return SimpleNamespace(returncode=1)

    with patch("kvip.runtime.build.subprocess.run", side_effect=fail):
        with pytest.raises(RuntimeError, match="configure failed") as error:
            build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="Ascend950PR_9599")
    assert "CMake Error: unsupported target" in str(error.value)
    assert len(str(error.value).splitlines()) <= 40
    assert not list(tmp_path.glob("*/complete.json"))


def test_install_and_cmake_default_to_concrete_a5_target():
    import ast

    tree = ast.parse((ROOT / "scripts/build.py").read_text())
    soc_option = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--soc"
    )
    default = next(ast.literal_eval(k.value) for k in soc_option.keywords if k.arg == "default")
    assert default == "Ascend950PR_9599"
    cmake = (ROOT / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt").read_text()
    assert f"set(SOC_VERSION {default} CACHE" in cmake












def test_completion_diagnostics_distinguish_transport_states():
    import numpy as np
    from prefetch_audit import completion_failures

    rows = np.zeros((2, 256), dtype=np.uint32)
    posts = np.zeros((2, 4, 32), dtype=np.uint32)
    statuses = np.zeros((2, 32), dtype=np.uint32)
    cq = np.zeros((4, 8), dtype=np.uint64)
    cq[:, 0] = 0x4B5641464247444F
    assert not completion_failures(rows, posts, statuses, cq, cq)
    posts[1, 2, 3] = 1
    result = completion_failures(rows, posts, statuses, cq, cq)
    assert len(result) == 1 and result[0]["request"] == 1
    assert result[0]["posts_count_pending_error_deferred_descriptor"][2][3] == 1
    posts.fill(0)
    failed = cq.copy()
    failed[0, 2] = 7
    result = completion_failures(rows, posts, statuses, cq, failed)
    assert len(result) == 1 and result[0]["final_cq"][0][2] == 7






def test_load_bridge_without_legacy_update_library(tmp_path):
    import subprocess
    from kvip.runtime.loader import load_lifetch_library
    source = tmp_path / "bridge.cpp"
    source.write_text('extern "C" int value(){return 42;}')
    library = tmp_path / "libv9_bridge.so"
    subprocess.run(["c++", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    assert load_lifetch_library(library).value() == 42
