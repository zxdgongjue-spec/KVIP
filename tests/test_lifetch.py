# SPDX-License-Identifier: Apache-2.0
"""LIFetch + Index update: native A/B, SFA/KV checks, resolver isolation, graph timing."""

import argparse
import contextlib
import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests/support")]
from runner import progress, supervise  # noqa: E402 — standalone test bootstrap


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
    p.add_argument("--runtime-shapes", action="store_true", help="one binary, four distinct runtime shapes")
    p.add_argument("--resolver-compare", action="store_true")
    p.add_argument("--trace-rounds", type=int, default=3)
    p.add_argument("--miss-percent", type=float, default=10)
    p.add_argument("--batches", default="16")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=4)
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


def main(argv=None):
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
        *(sys.argv[1:] if argv is None else argv),
        "--worker-run",
        str(run),
    ]
    return supervise(command, run)


if __name__ == "__main__":
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
    update = SimpleNamespace(device=5, bindings=SimpleNamespace(data_ptr=lambda: 200, numel=lambda: 26))
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

    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/launch.cpp").read_text()
    entry = source[source.index('extern "C" uint32_t kvop_direct_update_outputs(uint32_t,') :]
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
uint32_t v9_fetch(void*a,void*s,const rotating::RuntimeShape*){return step(4,a,s);}
extern "C" uint32_t kvop_direct_update_outputs(uint32_t batch,void*a,void*s){assert(batch==16);assert(a==(void*)2);return step(5,(void*)1,s);}
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
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch"),
            str(path),
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)


def test_operator_sources_match_migration_manifest():
    from kvip.integrity import verify_sources

    assert verify_sources() > 0


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
    from kvip._lifetch.runtime import RuntimeShape
    from kvip._lifetch.geometry import li_geometry, final_aivs, trigger_tokens
    import pytest

    source = tmp_path / "shape.cpp"
    # Compile the actual host tiling and device-shared geometry helpers.
    (tmp_path / "batch_get_abi.h").write_text("")
    source.write_text("""
#include "context.h"
static_assert(sizeof(rotating::Context)==304,"context ABI");
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
            "-I" + str(ROOT / "csrc/operators/lightning_indexer_with_fetch"),
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
    from kvip.fused_build import build_libraries

    def build(command, **kwargs):
        if command[:2] == ["cmake", "--build"]:
            folder = Path(command[2])
            for name in ("libv9_bridge.so", "libkvop_cache_update.so"):
                (folder / name).write_bytes(b"universal")
        return SimpleNamespace(returncode=0)

    with patch("kvip.fused_build.subprocess.run", side_effect=build) as commands:
        args = dict(cache_dir=tmp_path / "cache", cann=tmp_path / "cann", headers=tmp_path / "headers", soc="Ascend950")
        first = build_libraries(**args)
        second = build_libraries(**args)
    assert first == second and commands.call_count == 2
    options = json.loads((first[0].parent / "complete.json").read_text())
    assert not any(
        x in options
        for x in ("SIM_BATCH", "SIM_HISTORY", "SIM_RESIDENT", "INDEX_HEADS", "INTERMEDIATE_FETCHES", "AIC_COUNT")
    )


def test_build_failure_reports_first_diagnostic(tmp_path):
    from kvip.fused_build import build_libraries
    from unittest.mock import patch
    from types import SimpleNamespace
    import pytest

    def fail(command, **kwargs):
        kwargs["stdout"].write("CMake Error: unsupported target\n" + "detail\n" * 80)
        return SimpleNamespace(returncode=1)

    with patch("kvip.fused_build.subprocess.run", side_effect=fail):
        with pytest.raises(RuntimeError, match="configure failed") as error:
            build_libraries(cache_dir=tmp_path, cann=tmp_path, headers=tmp_path, soc="Ascend950")
    assert "CMake Error: unsupported target" in str(error.value)
    assert len(str(error.value).splitlines()) <= 40
    assert not list(tmp_path.glob("*/complete.json"))
