# SPDX-License-Identifier: Apache-2.0
"""Shared speculative limits and cache invalidation for hot-slot algorithms."""

import inspect
import random
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvip.operators.lifetch.binding import LIFetchBinding
from kvip.operators.lifetch.geometry import prefetch_budget
from kvip.operators.lifetch_ubmem import specialization
from kvip.operators.lifetch_ubmem.binding import UbmemBinding


def test_binding_defaults_and_budget_boundaries():
    for binding in (LIFetchBinding, UbmemBinding):
        params = inspect.signature(binding).parameters
        assert prefetch_budget(params["prefetch_limit"].default, params["window_limit"].default, 8) == (512, 64)
    assert prefetch_budget(0, None, 8) == (0, 1)
    assert prefetch_budget(2048, None, 8) == (2048, 256)
    for total, window, rounds in ((2049, 64, 8), (-1, 64, 8), (512, 481, 8), (512, 0, 8), (512, None, 0)):
        with pytest.raises(ValueError):
            prefetch_budget(total, window, rounds)


def test_device_budget_saturates_without_unsigned_wrap(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "budget.cpp"
    source.write_text(
        '#include <iostream>\n#define __aicore__\n#include "prefetch_budget.h"\n'
        "int main() { unsigned s,r,sl,pl,tl,w; while(std::cin>>s>>r>>sl>>pl>>tl>>w) "
        'std::cout<<rotating::PrefetchAllowance(s,r,sl,pl,tl,w)<<"\\n"; }\n'
    )
    binary = tmp_path / "budget"
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-I" + str(root / "csrc/operators/common/lifetch"), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    cases = [(0, 0, 8192, 0, 2048, 1), (8192, 0, 8192, 512, 2048, 64), (0, 512, 8192, 512, 2048, 64)]
    rng = random.Random(17)
    cases.extend(tuple(rng.randrange(10000) for _ in range(6)) for _ in range(1000))
    data = "".join(" ".join(map(str, case)) + "\n" for case in cases)
    result = subprocess.run([str(binary)], input=data, text=True, capture_output=True, check=True)
    expected = [min(w, max(0, sl - s), max(0, pl - r), max(0, tl - r)) for s, r, sl, pl, tl, w in cases]
    assert list(map(int, result.stdout.split())) == expected


def test_shared_hot_selector_invalidates_specialization_cache(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    module = repo / "python/kvip/operators/lifetch_ubmem/specialization.py"
    module.parent.mkdir(parents=True)
    module.touch()
    source = repo / "csrc/operators/lightning_indexer_with_fetch_ubmem"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("# fixture\n")
    selector = source.parent / "lidu/op_kernel/hot_slot_selection.h"
    selector.parent.mkdir(parents=True)
    selector.write_text("// first algorithm\n")
    monkeypatch.setattr(specialization, "__file__", str(module))
    calls = []

    def compile_fixture(command, **kwargs):
        calls.append(command)
        if command[1] == "--build":
            (Path(command[2]) / "libkvip_lifetch_ubmem.so").write_bytes(b"fixture")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(specialization.subprocess, "run", compile_fixture)
    options = dict(
        cache_dir=tmp_path / "cache",
        cann=tmp_path / "cann",
        shape=specialization.Specialization(2, 33024, 8192, 64, 656, 32, 8, "Ascend950DT_9582"),
    )
    old = specialization.build_specialization(**options)
    assert specialization.build_specialization(**options) == old
    assert len(calls) == 2
    alternate = specialization.Specialization(16, 131072, 16384, 32, 2624, 32, 8, "Ascend950DT_9582")
    assert specialization.build_specialization(**(options | {"shape": alternate})) == old
    assert len(calls) == 2  # no recompilation for a new runtime geometry or row width
    selector.write_text("// changed hot-slot algorithm\n")
    new = specialization.build_specialization(**options)
    assert new != old and len(calls) == 4


def test_service_chunk_assignment_covers_every_request_once(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3]
    header = root / "csrc/operators/lightning_indexer_with_fetch_ubmem/op_kernel/controller"
    source = tmp_path / "chunks.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n#include <algorithm>\n"
        "#define SIM_BATCH 32\nnamespace rotating { constexpr uint32_t kServiceLanes=2; }\n"
        '#include "post_geometry.h"\n'
        "int main() { for(uint32_t b=1;b<=32;++b) for(uint32_t lanes=2;lanes<=64;lanes+=2) {"
        "uint32_t groups=(b+lanes-1)/lanes,chunk=std::min(groups,8U),seen[32]={};"
        "auto geometry=rotating::MakePostGeometry(b,lanes);assert(geometry.groups==chunk);"
        "assert(geometry.threads*geometry.groups==512 || geometry.groups==3 || geometry.groups>=5);"
        "assert(geometry.groups*geometry.bank<=480 && geometry.bank%8==0);"
        "assert(geometry.groups*1024<=8448);"
        "for(uint32_t base=0;base<groups;base+=chunk) for(uint32_t lane=0;lane<lanes;++lane){"
        "auto first=rotating::ServiceChunkFirst(lane,base,lanes);"
        "auto count=rotating::ServiceChunkCount(b,first,lanes,chunk);"
        "assert(count<=8);for(uint32_t i=0;i<count;++i){auto row=first+i*lanes;assert(row<b);++seen[row];}}"
        "for(uint32_t i=0;i<b;++i)assert(seen[i]==1); } }\n"
    )
    binary = tmp_path / "chunks"
    subprocess.run([compiler, "-std=c++17", str(source), "-I" + str(header), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("batch,windows,history,resident", [
    (1, 1, 33024, 8192),
    (1, 8, 33024, 8192),
    (8, 8, 131072, 16384),
    (16, 8, 131072, 16384),
    (32, 32, 131072, 16384),
])
def test_ubmem_window_triggers_match_urma(tmp_path, batch, windows, history, resident):
    from kvip.operators.lifetch.geometry import trigger_tokens

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "triggers.cpp"
    source.write_text(
        '#include "context.h"\n#include <cstdio>\n'
        'int main() { for(unsigned e=0;e<ROUNDS;++e) '
        'std::printf("%u ",128*rotating::RuntimeTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,e)); '
        'for(unsigned t=0;t<=SIM_HISTORY/128+1;++t) { bool found=false; '
        'for(unsigned e=0;e<ROUNDS;++e)found |= rotating::RuntimeTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,e)==t; '
        'if(found != rotating::IsTrigger(SIM_HISTORY,SIM_RESIDENT,ROUNDS,t))return 1; } }\n'
    )
    definitions = dict(
        SIM_BATCH=batch, SIM_HISTORY=history, SIM_RESIDENT=resident, TOKEN_BYTES=656,
        ROUNDS=min(windows,batch*(2 if 2*batch<=32 else 1)),
        V9_UBPIPE_DEPTH=8, V9_UBPIPE_ROWS=4, V9_DEDICATED_SERVICE=0,
        V9_AIC_COUNT=32, V9_SPLIT_Q=int(2 * batch <= 32),
        V9_INTERMEDIATE_FETCHES=windows, V9_FINAL_REPLICATED=1,
    )
    binary = tmp_path / "triggers"
    subprocess.run(
        [compiler, "-std=c++17", str(source), *[f"-D{k}={v}" for k, v in definitions.items()],
         "-I" + str(root / "csrc/operators/lightning_indexer_with_fetch_ubmem/include"), "-o", str(binary)],
        check=True, capture_output=True,
    )
    actual = subprocess.check_output([str(binary)], text=True)
    assert list(map(int, actual.split())) == trigger_tokens(batch, history, resident, intermediate_fetches=windows)


def test_runtime_trigger_and_epoch_assignment_use_urma_schedule(tmp_path):
    from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ unavailable")
    root = Path(__file__).resolve().parents[3]
    source = tmp_path / "runtime_schedule.cpp"
    source.write_text(
        '#include "context.h"\n#include <cstdio>\n'
        'int main(){ unsigned b,h,r,s,n,lanes; '
        'while(std::scanf("%u%u%u%u%u%u",&b,&h,&r,&s,&n,&lanes)==6){'
        'for(unsigned e=0;e<n;++e)std::printf("%u ",128*rotating::RuntimeTrigger(h,r,n,e));'
        'std::printf("\\n"); for(unsigned t=0;t<=h/128;++t){bool found=false;'
        'for(unsigned e=0;e<n;++e)found |= rotating::RuntimeTrigger(h,r,n,e)==t;'
        'if(found!=rotating::IsTrigger(h,r,n,t))return 2;}'
        'for(unsigned c=0;c<b*(1+s);++c){unsigned rank=s?(c%2)*b+c/2:c,e=rank/lanes;'
        'if(e<n && rotating::EpochAt(b,h,r,s,n,lanes,rotating::RuntimeTrigger(h,r,n,e),c)!=int(e))return 3;}'
        '} }\n'
    )
    defs = dict(SIM_BATCH=1, SIM_HISTORY=33024, SIM_RESIDENT=8192, TOKEN_BYTES=656,
                V9_UBPIPE_DEPTH=8, V9_UBPIPE_ROWS=4, V9_DEDICATED_SERVICE=0,
                V9_AIC_COUNT=32, V9_SPLIT_Q=1, V9_INTERMEDIATE_FETCHES=2, V9_FINAL_REPLICATED=1)
    binary = tmp_path / "runtime_schedule"
    subprocess.run([compiler, "-std=c++17", str(source), *[f"-D{k}={v}" for k, v in defs.items()],
                    "-I" + str(root / "csrc/operators/lightning_indexer_with_fetch_ubmem/include"),
                    "-o", str(binary)], check=True, capture_output=True)
    cases = [(1, 33024, 8192, 1), (8, 131072, 16384, 8), (16, 131072, 16384, 8),
             (32, 131072, 16384, 32)]
    rows, expected = [], []
    for batch, history, resident, windows in cases:
        geo = li_geometry(batch, intermediate_fetches=windows)
        rows.append(f"{batch} {history} {resident} {geo['cores_per_batch']-1} "
                    f"{geo['intermediate_fetches']} {geo['service_aics_per_window']}\n")
        expected.append(trigger_tokens(batch, history, resident, intermediate_fetches=windows))
    result = subprocess.run([str(binary)], input="".join(rows), text=True, capture_output=True, check=True)
    assert [list(map(int, row.split())) for row in result.stdout.splitlines()] == expected
