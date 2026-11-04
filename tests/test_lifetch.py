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
        bindings=SimpleNamespace(data_ptr=lambda: 100),
        library=SimpleNamespace(kvip_lifetch_forward=function),
        torch=SimpleNamespace(npu=SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=300))),
    )
    update = SimpleNamespace(device=5, bindings=SimpleNamespace(data_ptr=lambda: 200, numel=lambda: 24))
    forward = kvip.LIFetch(rotating, update)
    forward()
    assert function.args == (100, 200, 300)
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
#define SIM_BATCH 16
int fail=0, count=0;
void* stream_expected=(void*)3;
uint32_t step(int n,void* a,void* stream) { assert(a==(void*)1);assert(stream==stream_expected);assert(++count==n);return fail==n?99:0; }
uint32_t v9_init(void*a,void*s){return step(1,a,s);}
uint32_t v9_li(void*a,void*s){return step(2,a,s);}
uint32_t v9_resolve(void*a,void*s){return step(3,a,s);}
uint32_t v9_fetch(void*a,void*s){return step(4,a,s);}
extern "C" uint32_t kvop_direct_update_outputs(uint32_t batch,void*a,void*s){assert(batch==16);assert(a==(void*)2);return step(5,(void*)1,s);}
"""
    main = r"""
int main(){for(fail=0;fail<=5;++fail){count=0;assert(kvip_lifetch_forward((void*)1,(void*)2,stream_expected)==(fail?99:0));assert(count==(fail?fail:5));}return 0;}
"""
    path = tmp_path / "launch.cpp"
    path.write_text(stub + entry + main)
    subprocess.run(["c++", "-std=c++17", str(path), "-o", str(tmp_path / "test")], check=True)
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
