# SPDX-License-Identifier: Apache-2.0
"""Host compilation and workload regression checks retained from the old entry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


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
    from workloads import prepare_full_chain_fixture

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
    from diagnostics import completion_failures

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
