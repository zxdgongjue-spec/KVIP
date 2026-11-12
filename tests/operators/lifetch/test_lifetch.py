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
from actual_k_admission import parse_actual_key_lengths  # noqa: E402


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker-run", type=Path, help=argparse.SUPPRESS)
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
    p.add_argument("--active-k-map-clear", action="store_true",
                   help="P0-A1 opt-in: CMake-build shortened v9_init mapping reset with safety guards")
    p.add_argument("--native-union-fit", action="store_true",
                   help="test-only: choose overlapping Q4 Hot TopK and admit only a union fitting Resident")
    p.add_argument("--actual-key-lengths", type=str,
                   help="P0-A2: CSV of per-request actual K within fixed History")
    p.add_argument("--resolver-compare", action="store_true")
    p.add_argument("--trace-rounds", type=int, default=3)
    p.add_argument("--miss-percent", type=float, default=10)
    p.add_argument("--batches", default="16")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--comparison-output", type=Path)
    p.add_argument("--comparison-sfa", type=Path)
    p.add_argument("--li-only-varlen", action="store_true",
                   help="run only custom v9 LI with two mixed-length forwards; requires the URMA test fixture")
    p.add_argument("--epoch-audit-profile", action="store_true",
                   help="enable test-only V9 Event candidate/selected counters for nonzero-prefetch LI varlen")
    p.add_argument("--prefetch-payload-audit", action="store_true",
                   help="P1-C0: reconcile prefetch Row/CQ/mapping and DDR-to-HBM bytes in LI-only fixture")
    p.add_argument("--cold-prefetch-probe", action="store_true",
                   help="P1-C1: 1 strong cold TopK winner per query in first cold tile; cold-only prefetch")
    p.add_argument("--output", type=Path, default=ROOT / "tests/results/lifetch")
    a = p.parse_args(argv)
    if a.kvaf_root is None:
        state = ROOT / "build/dependencies.json"
        a.kvaf_root = Path(json.loads(state.read_text())["kvaf"]) if state.exists() else ROOT / "third_party/KVAF"
    try:
        batches = [int(b) for b in a.batches.split(",")]
    except ValueError:
        p.error("batches must be comma-separated integers")
    if not batches or any(not 1 <= b <= 32 for b in batches):
        p.error("batch must be 1..32")
    if not 0 <= a.miss_percent <= 100 or a.history <= a.resident + 4:
        p.error("invalid history/resident/miss geometry")
    if a.iterations < 1 or a.warmup < 0 or a.trace_rounds < 0:
        p.error("invalid timing iterations")
    if not 1 <= a.intermediate_fetches <= 32:
        p.error("windows must be 1..32")
    if not 0 <= a.prefetch_tokens <= 2048 or a.prefetch_tokens % a.intermediate_fetches:
        p.error("prefetch tokens must be 0..2048 and divisible by windows")
    if a.epoch_audit_profile and (not a.li_only_varlen or a.prefetch_tokens != 64):
        p.error("--epoch-audit-profile requires --li-only-varlen and --prefetch-tokens 64")
    if a.prefetch_payload_audit and not a.epoch_audit_profile:
        p.error("--prefetch-payload-audit requires --epoch-audit-profile")
    if a.cold_prefetch_probe and (
        not a.prefetch_payload_audit or round(2048 * a.miss_percent / 100) != 1
    ):
        p.error("--cold-prefetch-probe requires --prefetch-payload-audit and one final cold miss per query")
    if a.actual_key_lengths:
        if a.li_only_varlen or a.runtime_shapes or len(batches) != 1:
            p.error("--actual-key-lengths requires one full-chain fixed-History batch")
        try:
            parse_actual_key_lengths(a.actual_key_lengths, batches[0], a.history, a.resident)
        except ValueError as exc:
            p.error(str(exc))
    if a.native_union_fit and a.li_only_varlen:
        p.error("--native-union-fit is for the full-chain Native MTP baseline, not LI-only")
    if a.li_only_varlen and (
        batches != [4] or a.history != 8192 or a.resident != 2048 or a.index_heads != 32
        or (round(2048 * a.miss_percent / 100) != 1 if a.cold_prefetch_probe
            else a.miss_percent != 0)
        or a.prefetch_tokens not in (0, 64) or a.runtime_shapes
        or a.resolver_compare or a.comparison_output or a.comparison_sfa
    ):
        p.error(
            "li-only-varlen requires batch=4, history=8192, resident=2048, "
            "index-heads=32, miss-percent=0 (or one cold miss/query in cold probe), "
            "prefetch-tokens=0 or 64, and no runtime/resolver comparisons"
        )
    return a


def run_worker(args, run):
    progress(run, "checking native registration and KVAF dependency")
    if not args.li_only_varlen:
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
    return supervise(command, run)



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--internal-urma"]:
        return run_urma(argv[1:])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport', required=True, choices=('urma', 'ubmem'))
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
    p.add_argument("--active-k-map-clear", action="store_true",
                   help="P0-A1 opt-in: compile guarded active-K mapping clear")
    p.add_argument("--native-union-fit", action="store_true",
                   help="test-only: admit a Q4 TopK union that fits Resident capacity")
    p.add_argument("--actual-key-lengths", type=str,
                   help="P0-A2: CSV of per-request actual K within fixed History")
    p.add_argument('--resolver-compare', action='store_true', help='URMA resolver comparison')
    p.add_argument('--comparison-output', type=Path)
    p.add_argument('--comparison-sfa', type=Path)
    p.add_argument('--li-only-varlen', action='store_true',
                   help='run only custom v9 LI with two mixed-length forwards')
    p.add_argument('--epoch-audit-profile', action='store_true',
                   help='enable test-only candidate/selected counters for nonzero-prefetch varlen')
    p.add_argument('--prefetch-payload-audit', action='store_true',
                   help='P1-C0: reconcile Row/CQ/mapping and compare prefetch HBM bytes with DDR')
    p.add_argument('--cold-prefetch-probe', action='store_true',
                   help='P1-C1: one cold TopK winner/query in first cold tile, cold-only prefetch')
    p.add_argument('--output', type=Path, default=ROOT / 'tests/results/lifetch')
    p.add_argument('--timeout-seconds', type=int, default=1800)
    a = p.parse_args(argv)
    try:
        batches = [int(x) for x in a.batches.split(',')]
    except ValueError:
        p.error('batches must be comma-separated integers')
    if not batches or any(not 1 <= x <= 32 for x in batches):
        p.error('batches must be in 1..32')
    if (a.history <= a.resident + 4 or not 0 <= a.miss_percent <= 100 or a.warmup < 0
            or a.iterations < 1 or a.trace_rounds < 0 or not 1 <= a.intermediate_fetches <= 32
            or not 0 <= a.prefetch_tokens <= 2048 or a.prefetch_tokens % a.intermediate_fetches
            or a.timeout_seconds < 20):
        p.error('invalid geometry, prefetch budget, timing or timeout')
    if (a.comparison_output or a.comparison_sfa) and (len(batches) != 1 or a.runtime_shapes):
        p.error('comparison snapshots require one batch and no runtime-shapes')
    if a.epoch_audit_profile and (not a.li_only_varlen or a.prefetch_tokens != 64):
        p.error('--epoch-audit-profile requires --li-only-varlen and --prefetch-tokens 64')
    if a.prefetch_payload_audit and not a.epoch_audit_profile:
        p.error('--prefetch-payload-audit requires --epoch-audit-profile')
    if a.cold_prefetch_probe and (
        not a.prefetch_payload_audit or round(2048 * a.miss_percent / 100) != 1
    ):
        p.error('--cold-prefetch-probe requires --prefetch-payload-audit and one final cold miss per query')
    if a.actual_key_lengths:
        if a.li_only_varlen or a.runtime_shapes or len(batches) != 1:
            p.error("--actual-key-lengths requires one full-chain fixed-History batch")
        try:
            parse_actual_key_lengths(a.actual_key_lengths, batches[0], a.history, a.resident)
        except ValueError as exc:
            p.error(str(exc))
    if a.native_union_fit and a.li_only_varlen:
        p.error("--native-union-fit is for the full-chain Native MTP baseline, not LI-only")
    if a.li_only_varlen and (
        a.transport != 'urma' or batches != [4] or a.history != 8192 or a.resident != 2048
        or a.index_heads != 32
        or (round(2048 * a.miss_percent / 100) != 1 if a.cold_prefetch_probe
            else a.miss_percent != 0)
        or a.prefetch_tokens not in (0, 64)
        or a.runtime_shapes or a.resolver_compare or a.comparison_output or a.comparison_sfa
    ):
        p.error(
            'li-only-varlen requires URMA, batch=4, history=8192, resident=2048, '
            'index-heads=32, miss-percent=0 (or one cold miss/query in cold probe), '
            'prefetch-tokens=0 or 64, and no runtime/resolver comparisons'
        )
    state = ROOT / 'build/dependencies.json'
    kvaf_root = (a.kvaf_root or (Path(json.loads(state.read_text())['kvaf']) if state.exists()
                                 else ROOT / 'third_party/KVAF')).resolve()
    common = ['--device', str(a.device), '--kvaf-server', a.kvaf_server, '--history', str(a.history),
              '--resident', str(a.resident), '--index-heads', str(a.index_heads), '--miss-percent',
              str(a.miss_percent), '--prefetch-tokens', str(a.prefetch_tokens), '--warmup', str(a.warmup),
              '--iterations', str(a.iterations)]
    if a.li_only_varlen:
        common.append('--li-only-varlen')
    if a.native_union_fit:
        common.append('--native-union-fit')
    if a.active_k_map_clear:
        common.append("--active-k-map-clear")
    if a.epoch_audit_profile:
        common.append('--epoch-audit-profile')
    if a.prefetch_payload_audit:
        common.append('--prefetch-payload-audit')
    if a.cold_prefetch_probe:
        common.append('--cold-prefetch-probe')
    if a.actual_key_lengths:
        if a.transport != 'urma' or a.runtime_shapes or a.li_only_varlen or len(batches) != 1:
            p.error('--actual-key-lengths requires full-chain URMA with one fixed-History batch')
        try:
            parse_actual_key_lengths(a.actual_key_lengths, batches[0], a.history, a.resident)
        except ValueError as exc:
            p.error(str(exc))
        common.extend(['--actual-key-lengths', a.actual_key_lengths])
    if a.native_union_fit and a.transport != 'urma':
        p.error('--native-union-fit requires URMA')
    if a.transport == 'urma':
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--internal-urma', *common,
                   '--kvaf-root', str(kvaf_root), '--cann', a.cann, '--batches', a.batches,
                   '--intermediate-fetches', str(a.intermediate_fetches), '--trace-rounds', str(a.trace_rounds),
                   '--output', str(a.output)]
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
    print('UBmem stage traces use separate profiled replays outside graph timing.', flush=True)
    code = execute(command, log, a.timeout_seconds, 'ubmem')
    lines = log.read_text(errors='replace').splitlines()
    selected = [line for line in lines if line.startswith(('UBMEM_LIFETCH_PASS', 'UBMEM_SFA_PASS',
                'UBMEM_GRAPH_EVENT_US', 'UBMEM_FETCH_COUNTS', 'UBMEM_WINDOW_TRIGGERS', 'UBMEM_RUNTIME_BINARY',
                'UBMEM_TRACE'))]
    expected = len(batches) + int(a.runtime_shapes and (a.history, a.resident) != (33024, 8192))
    ok = (code == 0 and sum(line.startswith('UBMEM_LIFETCH_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_SFA_PASS') for line in selected) == expected
          and sum(line.startswith('UBMEM_GRAPH_EVENT_US') for line in selected) == expected
          and sum(line.startswith('UBMEM_TRACE round=') for line in selected) == expected * a.trace_rounds)
    result = [f'{"PASS" if ok else "FAIL"}: transport=ubmem exit={code}', *selected]
    if not ok:
        result += ['--- worker.log tail ---', *lines[-25:]]
    summary.write_text('\n'.join(result) + '\n')
    print('\n'.join(result), flush=True)
    print(f'Full logs: {run}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())


def test_public_launch_same_stream_and_addresses():
    from types import SimpleNamespace
    import kvip
    import pytest

    class Function:
        result = 0

        def __call__(self, *args):
            self.args = args
            return self.result

    function = Function()
    rotating = SimpleNamespace(
        device=5,
        shape_pointer=400,
        bindings=SimpleNamespace(data_ptr=lambda: 100),
        library=SimpleNamespace(kvip_lifetch_forward=function),
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=300))),
    )
    update = SimpleNamespace(device=5, bindings=SimpleNamespace(data_ptr=lambda: 200, numel=lambda: 28))
    forward = kvip.LIFetch(rotating, update)
    forward()
    assert function.args == (100, 200, 300, 400)
    function.result = 42
    with pytest.raises(RuntimeError, match="42"):
        forward()
    update.device = 6
    with pytest.raises(ValueError, match="same device"):
        kvip.LIFetch(rotating, update)


def test_host_pipeline_order_and_first_error(tmp_path):
    import subprocess

    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_host/launch.cpp").read_text()
    entry = source[source.index('extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,') :]
    stub = r"""
#include <cstdint>
#include <cassert>
#include "runtime_shape.h"
int fail=0, count=0;
void* stream_expected=(void*)3;
uint32_t step(int n,void* a,void* stream) { assert(a==(void*)1);assert(stream==stream_expected);assert(++count==n);return fail==n?99:0; }
uint32_t v9_init(void*a,void*s,const rotating::RuntimeShape*){return step(1,a,s);}
uint32_t v9_li(void*a,void*s,const rotating::RuntimeShape*){return step(2,a,s);}
uint32_t v9_resolve(void*a,void*s,const rotating::RuntimeShape*){return step(3,a,s);}
extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t batch,void*a,void*s){assert(batch==16);assert(a==(void*)2);return step(4,(void*)1,s);}
extern "C" uint32_t aclrtlaunch_v9_fetch_update_kernel(uint32_t batch,void*s,void*a,void*update){assert(batch==16);assert(update==(void*)2);return step(5,a,s);}
"""
    main = r"""
int main(){rotating::RuntimeShape shape{};shape.batch=16;for(fail=0;fail<=5;++fail){count=0;assert(kvip_lifetch_forward((void*)1,(void*)2,stream_expected,&shape)==(fail?99:0));assert(count==(fail?fail:5));}return 0;}
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
static_assert(sizeof(rotating::Context)==320,"context ABI");
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
            for name in ("libv9_bridge.so", "libkvop_cache_update.so"):
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


def test_bridge_keeps_sibling_rpath_when_toolchain_disables_build_rpath(tmp_path):
    import re
    import subprocess

    cmake = (ROOT / "csrc/operators/lightning_indexer_with_fetch/CMakeLists.txt").read_text()
    properties = re.search(r"set_target_properties\(v9_bridge PROPERTIES.*?\)", cmake, re.S).group()
    (tmp_path / "empty.cpp").write_text("int placeholder() { return 0; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(rpath_check LANGUAGES CXX)\n"
        "set(CMAKE_SKIP_BUILD_RPATH TRUE)\nadd_library(v9_bridge SHARED empty.cpp)\n"
        + properties
        + '\nfile(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/properties.txt" CONTENT '
        '"$<TARGET_PROPERTY:v9_bridge,SKIP_BUILD_RPATH>\\n'
        "$<TARGET_PROPERTY:v9_bridge,BUILD_WITH_INSTALL_RPATH>\\n"
        '$<TARGET_PROPERTY:v9_bridge,INSTALL_RPATH>")\n'
    )
    subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")], check=True, capture_output=True)
    assert (tmp_path / "build/properties.txt").read_text().splitlines() == ["FALSE", "TRUE", "$ORIGIN"]


def test_load_bridge_without_rpath(tmp_path):
    import subprocess
    import sys
    from kvip.runtime.loader import load_lifetch_library

    dependency = tmp_path / "libkvop_cache_update.so"
    bridge = tmp_path / "libv9_bridge.so"
    (tmp_path / "update.cpp").write_text('extern "C" int kvip_test_update() { return 42; }')
    (tmp_path / "bridge.cpp").write_text(
        'extern "C" int kvip_test_update(); extern "C" int kvip_test_forward() { return kvip_test_update(); }'
    )
    identity = (
        ["-Wl,-install_name,@rpath/libkvop_cache_update.so"]
        if sys.platform == "darwin"
        else ["-Wl,-soname,libkvop_cache_update.so"]
    )
    subprocess.run(
        ["c++", "-shared", "-fPIC", str(tmp_path / "update.cpp"), *identity, "-o", str(dependency)], check=True
    )
    subprocess.run(
        ["c++", "-shared", "-fPIC", str(tmp_path / "bridge.cpp"), str(dependency), "-o", str(bridge)], check=True
    )
    # Fresh process proves the fixture really fails without the dependency preload.
    failed = subprocess.run(
        [sys.executable, "-c", "import ctypes,sys;ctypes.CDLL(sys.argv[1])", str(bridge)], capture_output=True
    )
    assert failed.returncode != 0
    library = load_lifetch_library(bridge)
    assert library.kvip_test_forward() == 42
    assert library._kvip_update_dependency is not None


def test_hot_resolver_actual_function_on_empty_and_populated_cache(tmp_path):
    _check_hot_resolver(tmp_path, "urma")


def test_ubmem_hot_resolver_actual_function_on_empty_and_populated_cache(tmp_path):
    _check_hot_resolver(tmp_path, "ubmem")


def _check_hot_resolver(tmp_path, transport):
    """Execute production SIMT logic serially; CANN simulator covers barriers."""
    import subprocess

    operator = "lightning_indexer_with_fetch" + ("_ubmem" if transport == "ubmem" else "")
    kernel = ROOT / "csrc/operators" / operator / "op_kernel"
    metadata = kernel / ("controller/cache_metadata.h" if transport == "ubmem" else "prefetch.h")

    def function(path, name):
        text = path.read_text()
        begin = text.index(name + "(")
        begin = text.rfind("\n", 0, begin) + 1
        brace = text.index("{", begin)
        end, depth = brace + 1, 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        return text[begin:end]

    common = ROOT / "csrc/operators/common/lifetch"
    functions = "\n".join(
        function(common / "cache_metadata.h", name)
        for name in ("OwnerRow", "OutputStride", "StoreFallbackSource")
    )
    functions += "\n" + function(common / "cache_metadata.h", "ResolvePhysicalSlot")
    functions += "\n" + function(metadata, "InitializeTailMappings")
    functions += "\n" + function(ROOT / "csrc/operators/common/lifetch/resolver_candidates.h", "SparseHash")
    functions += "\ntemplate<unsigned Threads>\n" + function(
        ROOT / "csrc/operators/common/lifetch/resident_selection.h", "ResolveResidentPlan"
    )
    resolve = function(ROOT / "csrc/operators/common/lifetch/resolver.h", "ReplicatedResolve")
    shared = (ROOT / "csrc/operators/common/lifetch/resolver_plan.h").read_text()
    resolve = resolve.replace('#include "resolver_plan.h"', shared)
    functions += "\ntemplate<unsigned Threads>\n" + resolve
    source = r"""
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cassert>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {unsigned x=0;} threadIdx;
constexpr unsigned warpSize=1;
namespace AscendC {namespace Simt {unsigned GetThreadIdx(){return 0;} unsigned GetThreadNum(){return 1;}}}
#include "hot_slot_selection.h"
#include "context.h"
using namespace rotating;
constexpr unsigned kLruSlots=2048,kChangeCapacity=10240;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
unsigned asc_atomic_cas(unsigned* p,unsigned a,unsigned b){auto old=*p;if(old==a)*p=b;return old;}
void asc_atomic_or(unsigned* p,unsigned v){*p|=v;}
void asc_atomic_min(unsigned* p,unsigned v){*p=std::min(*p,v);}
void asc_atomic_max(unsigned* p,unsigned v){*p=std::max(*p,v);}
void asc_atomic_add(unsigned* p,unsigned v){*p+=v;}
unsigned asc_shfl_up(unsigned v,unsigned){return v;}
void asc_syncthreads(){} void asc_threadfence(){}
FUNCTIONS
int main(){
 std::mt19937 rng(71);
 for(unsigned trial=0;trial<27;++trial){
  const unsigned h=32768,r=trial%2?16384:8192,workers=4,used=trial%3?512:0;
  std::vector<unsigned> candidates(r);std::iota(candidates.begin(),candidates.end(),0);
  std::shuffle(candidates.begin(),candidates.end(),rng);candidates.resize(2048);
  std::vector<int32_t> reserved(r,-1);for(unsigned i=0;i<2048;++i)reserved[candidates[i]]=i;
  std::vector<unsigned> map(h,UINT32_MAX),changes(kChangeCapacity,UINT32_MAX),changeSlots(kChangeCapacity);
  std::copy(candidates.begin(),candidates.end(),changeSlots.begin());
  for(unsigned i=0;i<used;++i){map[r+i]=r+i;changes[i]=r+i;}
  std::vector<unsigned> selected(8192),base(h),hotValid(r,trial%4!=0);
  const bool abc=trial%2;unsigned tailBounds[2]={h-384,h};
  std::iota(base.begin(),base.end(),0);std::shuffle(base.begin(),base.end(),rng);
  std::copy(base.begin(),base.begin()+8192,selected.begin());
  if(trial%3==0)for(unsigned i=4096;i<8192;++i)selected[i]=selected[i-4096];
  if(trial%5==0)for(unsigned i=6144;i<8192;++i)selected[i]=UINT32_MAX;
  if(abc){for(unsigned i=0;i<384;++i)map[h-384+i]=r+8192+i;
   for(unsigned i=0;i<8;++i)selected[i]=h-384+i;}
  if(trial==24)for(unsigned i=0;i<8192;++i)selected[i]=r+i; // Maximum Q4 union, no prefetch.
  if(trial==25)for(unsigned i=0;i<8192;++i)selected[i]=r+i%used; // All prefetch hits.
  if(trial==26){std::vector<unsigned> hot;for(unsigned i=0;i<r;++i)if(reserved[i]<0)hot.push_back(i);
   for(unsigned i=0;i<8192;++i)selected[i]=hot[i%hot.size()];} // All untouched hot hits.
  std::vector<uint64_t> addresses(h),remote(8192);std::iota(addresses.begin(),addresses.end(),100000);
  std::vector<int32_t> blocks(8192),offsets(8192),destinations(8192);
  Context ctx{};unsigned changeCount=2048,total=0;
  ctx.lruSlots=(uint64_t)candidates.data();ctx.reservedHot=(uint64_t)reserved.data();
  ctx.changes=(uint64_t)changes.data();ctx.changeSlots=(uint64_t)changeSlots.data();ctx.changeCount=(uint64_t)&changeCount;
  ctx.fallbackAddresses=(uint64_t)remote.data();ctx.outputStride=r+(abc?384:0);
  if(abc)ctx.tailBounds=(uint64_t)tailBounds;
  ctx.mapping=(uint64_t)map.data();changeCount=0;
  for(unsigned lane=0;lane<256;++lane){threadIdx.x=lane;InitializeTailMappings(h,r,&ctx,0);}
  threadIdx.x=0;assert(changeCount==2048);
  std::vector<unsigned> output(8192),locations(8192),positions(8192),keys(8192),ranks(8192),slots(8192),cfg(2560);
  std::vector<uint16_t> ids(8192);unsigned cold=0;
  for(unsigned w=0;w<workers;++w){
   cold=0;locations=selected;
   for(unsigned i=0;i<8192;++i){unsigned t=selected[i];
    if(t!=UINT32_MAX && (t>=r || !hotValid[t] || reserved[t]>=0)){positions[cold++]=i;locations[i]=r;}}
   ReplicatedResolve<1>(h,r,workers,656,selected.data(),locations.data(),positions.data(),cold,
    keys.data(),ranks.data(),ids.data(),slots.data(),cfg.data(),map.data(),addresses.data(),
    blocks.data(),offsets.data(),destinations.data(),&total,w,used,0,&ctx);
   assert(cfg[1]==0);
   assert(cfg[5]<=cfg[3]);
   for(unsigned i=0;i<cfg[3];++i){
    int ordinal=reserved[destinations[cfg[2]+i]];
    bool conflict=ordinal>=0 && unsigned(ordinal)<used;
    assert(conflict==(i>=cfg[5]));
   }
   std::copy(locations.begin()+w*2048,locations.begin()+(w+1)*2048,output.begin()+w*2048);
  }
  std::vector<unsigned> ownership(r),seen(r);std::iota(ownership.begin(),ownership.end(),0);
  for(unsigned i=0;i<kChangeCapacity;++i)if(changes[i]!=UINT32_MAX){assert(changeSlots[i]<r);assert(!seen[changeSlots[i]]++);ownership[changeSlots[i]]=changes[i];}
  for(unsigned i=0;i<8192;++i){if(selected[i]==UINT32_MAX){assert(output[i]==UINT32_MAX);continue;}
   if(abc && selected[i]>=h-384){assert(output[i]==r+selected[i]%384);continue;}
   assert(output[i]<r);assert(ownership[output[i]]==selected[i]);}
  for(unsigned i=0;i<total;++i){assert(unsigned(destinations[i])<r);assert(remote[i]==addresses[ownership[destinations[i]]]);}
  assert(changeCount<=kChangeCapacity);
 }
}
""".replace("FUNCTIONS", functions)
    definitions = []
    if transport == "ubmem":
        definitions = [
            "-D" + value
            for value in (
                "SIM_BATCH=1",
                "SIM_HISTORY=32768",
                "SIM_RESIDENT=16384",
                "TOKEN_BYTES=656",
                "V9_UBPIPE_DEPTH=8",
                "V9_UBPIPE_ROWS=4",
                "V9_DEDICATED_SERVICE=0",
                "V9_SPLIT_Q=1",
                "V9_AIC_COUNT=32",
                "V9_INTERMEDIATE_FETCHES=2",
                "V9_FINAL_REPLICATED=1",
            )
        ]
    (tmp_path / "batch_get_abi.h").write_text("")
    cpp, exe = tmp_path / "resolver.cpp", tmp_path / "resolver"
    cpp.write_text(source)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            *definitions,
            str(cpp),
            "-I" + str(tmp_path),
            "-I" + str(kernel.parent / "include"),
            "-I" + str(kernel.parents[1] / "lidu/op_kernel"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True, timeout=30)


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


def test_final_fetch_uses_native_explicit_range_without_direct_submission():
    from pathlib import Path
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch/op_kernel"
    resolver = (root / "resolver.h").read_text()
    lifecycle = (root / "lifecycle.cpp").read_text()
    assert "PublishFinalWorker" not in resolver
    assert "CommitPostScalarStage" not in resolver
    phase = (root / "final_fetch_phase.h").read_text()
    phase_loop = (root.parents[1] / "common/lifetch/final_phase_loop.h").read_text()
    phase_loop = phase_loop.replace("#include LIFETCH_FINAL_PHASE", phase)
    lifecycle = lifecycle.replace('#include "../../common/lifetch/final_phase_loop.h"', phase_loop)
    finish = lifecycle[lifecycle.index("__aicore__ inline void FinishFetch") :]
    assert "if(phase==1)" in finish
    assert "(!phase || !row->pending)" in finish
    assert "kvaf_device::KvafBatchGetDevice(workerArgs,0,begin,count)" in finish
    assert "DrainQueue(" not in finish
    # Execute the production descriptor publisher: it must never mark an SQ post
    # pending, even for nonempty or oversized ranges; native KVAF schedules them.
    publisher = (root.parents[1] / "common/lifetch/final_post.h").read_text()
    start = publisher.index("__aicore__ inline void PrepareFinalFetch")
    function = publisher[start : publisher.index("\n}\n", start) + 2]
    function = function.replace("LIFETCH_CLEAN_CACHE_LINES", "kvaf_device::CleanCacheLines")
    program = (
        """#include <cstdint>
#include <cassert>
#define __aicore__
#define __gm__
struct FinalPost { uint32_t count,pending,error,deferred,descriptor,nonconflicting; uint64_t head; };
struct Context { FinalPost posts[8]; };
FinalPost* FinalPostAt(uint32_t,Context* c,uint32_t i) { return c->posts+i; }
namespace kvaf_device { void CleanCacheLines(uint8_t*,unsigned long) {} }
"""
        + function
        + """
int main() { Context c{};
 for(uint32_t n: {0U,608U,8192U}) {
  PrepareFinalFetch(4,0,&c,1,2,n,123,0,n/2);
  auto p=c.posts[6]; assert(p.count==n && p.descriptor==123 && p.pending==0 && p.deferred==(n!=0) && p.nonconflicting==n/2);
 }
 PrepareFinalFetch(4,0,&c,1,2,608,123,4,0); assert(c.posts[6].error==4 && !c.posts[6].deferred);
}
"""
    )
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "test.cpp"
        source.write_text("#include <initializer_list>\n" + program)
        subprocess.run(["c++", "-std=c++17", str(source), "-o", folder + "/test"], check=True)
        subprocess.run([folder + "/test"], check=True)


def test_native_final_phase_order_runs_production_loop(tmp_path):
    """Exercise the actual Fetch orchestration with a recording native SDK."""
    import subprocess

    common = ROOT / "csrc/operators/common/lifetch/final_phase_loop.h"
    phase = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/final_fetch_phase.h"
    loop = common.read_text().replace("#include LIFETCH_FINAL_PHASE", phase.read_text())
    loop = loop.replace("LIFETCH_FINAL_JOIN", "CompletePrefetch(ctx,b); error|=row->error")
    code = r"""
#include <cstdint>
#include <vector>
#include <cassert>
#define __gm__
#define V9_SIMULATION 0
#define KVAF_AIV_BATCH_GET_STATUS_MAGIC 99
struct Post { unsigned count=0,pending=0,error=0,deferred=0,descriptor=0,nonconflicting=0; } posts[4];
struct KvafBatchGetQueueState { uint64_t sq_producer_index=0,cq_consumer_index=0; } queues[4];
struct KvafBatchGetCoreStatus { uint64_t magic=99,result=0,completed_batch_count=1,processed_request_count=0; } statuses[4];
struct Args { uint64_t queue_state,core_status,indices,offsets,output_indices,miss_count,address_table,valid_table; unsigned request_capacity,table_capacity; };
struct Context { uint64_t sourceBlocks=1,sourceOffsets=2,destinations=3,count=4,fallbackAddresses=5,fallbackValid=6; } context;
struct Row { unsigned pending=1,error=0; } rowValue;
std::vector<int> events; bool joined=false,failJoin=false;
Post* FinalPostAt(unsigned,Context*,unsigned w){return posts+w;}
unsigned FinalQueue(unsigned,unsigned,unsigned,unsigned w){return w;}
Args GetArgs(Context*,unsigned w){Args a{};a.queue_state=(uint64_t)(queues+w);a.core_status=(uint64_t)(statuses+w);return a;}
void CompletePrefetch(Context*,unsigned){events.push_back(-1);joined=true;rowValue.pending=failJoin;rowValue.error=failJoin?6:0;}
namespace kvaf_device {
constexpr unsigned kInvalidArgs=1;
void CleanCacheLines(uint8_t*,unsigned long){}
void KvafBatchGetDevice(Args a,unsigned,unsigned begin,unsigned count){
 unsigned w=begin/100;auto& p=posts[w];bool suffix=begin!=p.descriptor;
 if(p.nonconflicting==0)suffix=true;
 assert(joined==suffix);assert(a.indices==1 && a.offsets==2 && a.output_indices==3 && a.address_table==5);
 events.push_back(int(begin));auto s=(KvafBatchGetCoreStatus*)a.core_status;s->result=0;s->processed_request_count=count;
}
}
void run(unsigned safe,bool failure=false){
 events.clear();joined=false;failJoin=failure;rowValue={1,0};
 for(unsigned w=0;w<4;++w)posts[w]={8,0,0,1,w*100,safe};
 unsigned rt_batch=1,rt_finalWorkers=4,rt_finalPostOffset=0,kSlots=8192,b=0;
 Context* ctx=&context;Row* row=&rowValue;
 uint64_t error=0,lastHead=0,lastCq=0;unsigned total=0,completed=0,deferred=0;
LOOP
 assert(total==32);unsigned index=0;
 if(safe)for(unsigned w=0;w<4;++w)assert(events[index++]==int(w*100));
 assert(events[index++]==-1);
 if(safe<8 && !failure)for(unsigned w=0;w<4;++w)assert(events[index++]==int(w*100+safe));
 assert(index==events.size());assert(bool(error)==failure);
 if(!failure)for(auto p:posts)assert(!p.deferred);
}
int main(){run(8);run(3);run(0);run(3,true);}
""".replace("LOOP", loop)
    source, exe = tmp_path / "phases.cpp", tmp_path / "phases"
    source.write_text(code)
    subprocess.run(["c++", "-std=c++17", str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
