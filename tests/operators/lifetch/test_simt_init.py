# SPDX-License-Identifier: Apache-2.0
"""Execute both production init bodies with delayed DMA and guarded storage."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _function(source, name):
    begin = source.rfind("\n", 0, source.index(name + "(")) + 1
    brace = source.index("{", begin)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.mark.parametrize("transport,tagged", (("urma", 0), ("ubmem", 0), ("ubmem", 1)))
@pytest.mark.parametrize("batch,rounds", ((1, 0), (16, 8)))
def test_simt_init_matches_all_baseline_records_and_reduces_dma(tmp_path, transport, tagged, batch, rounds):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for the init model")
    operator = ROOT / "csrc/operators" / ("lightning_indexer_with_fetch" + ("_ubmem" if transport == "ubmem" else ""))
    lifecycle = (operator / "op_kernel" / ("lifecycle.cpp" if transport == "urma" else "init.h")).read_text()
    tail = (operator / "op_kernel" / ("resolver/direct_slots.h" if transport == "ubmem" else "prefetch.h")).read_text()
    context = (operator / "include/context.h").read_text()
    row_start = context.index("struct alignas(128) Row {")
    row_end = context.index("\n};", row_start) + 3

    # Reuse the existing delayed-MTE and SIMT host intrinsics. The model runs
    # real bulk-init/VF bodies; only the prevalidated UBmem lease values are
    # supplied by a host stub, since lease validation is unchanged here.
    model = (ROOT / "tests/operators/lifetch/reserved_hot_model.cpp").read_text().split("int main() {")[0]
    model = model.replace(
        "inline void asc_stcg(uint32_t* p,uint32_t value)", "template<class T> inline void asc_stcg(T* p,T value)"
    )
    model = model.replace(
        "inline uint32_t currentPart=0", "inline uint32_t copies=0,loads=0,zeroFills=0;\ninline uint32_t currentPart=0"
    )
    model = model.replace(
        "negativeFills+=(value==UINT32_MAX);", "negativeFills+=(value==UINT32_MAX);zeroFills+=(value==0);"
    )
    model = model.replace(
        "assert(count%8==0 && pendingMte2.empty());", "++copies;assert(count%8==0 && pendingMte2.empty());"
    )
    model = model.replace(
        "assert(count%8==0 && pendingMte3.empty());", "++loads;assert(count%8==0 && pendingMte3.empty());"
    )
    model = model.replace("constexpr uint32_t kRounds=8", f"constexpr uint32_t kMaxRounds=32,kRounds={rounds}")
    model = model.replace("struct Row {uint64_t mappingState;uint32_t words[254];};", context[row_start:row_end])
    model = model.replace("uint64_t locations;", "uint64_t locations;uint32_t prefetchLimit=2048;")
    model = model.replace(
        '#include "reserved_hot_lifecycle.h"',
        '#include "simt_init.h"\n'
        "struct InitRowValues {uint64_t addresses,output;uint32_t error;};\n"
        "inline uint32_t modelError=0;\n"
        "inline InitRowValues GetInitRowValues(Context*,uint32_t b) {"
        "return {0x123456780000ULL+b*131328ULL*8,0x887766550000ULL+b*16384ULL*2624,modelError};}\n"
        '#include "reserved_hot_lifecycle.h"',
    )
    functions = [
        _function(tail, name) for name in ("OutputStride", "InitializeTailMappingsBody", "InitializeTailMappings")
    ]
    names = (
        ("FillRoundWords", "InitializeRoundSimt", "InitializeRound")
        if transport == "urma"
        else (
            "InitializeRoundSimt",
            "FillWords",
            "InitializeControl",
            "CopyInitialLruSlots",
            "InitializeRound",
            "InitializeFetchCore",
        )
    )
    functions.extend(_function(lifecycle, name) for name in names)
    (tmp_path / "reserved_hot_lifecycle.h").write_text("\n".join(functions))
    source = tmp_path / "model.cpp"
    source.write_text(model + (ROOT / "tests/operators/lifetch/simt_init_model.cpp").read_text())
    results = []
    for enabled in (0, 1):
        binary = tmp_path / f"simt-init-{enabled}"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                f"-DV9_SIMT_INIT={enabled}",
                "-DV9_COMPACT_RESERVED=1",
                f"-DSIM_BATCH={batch}",
                "-DSIM_HISTORY=131328",
                "-DSIM_RESIDENT=16384",
                f"-DV9_UBMEM_TAGGED_MAP={tagged}",
                f"-DMODEL_URMA={int(transport == 'urma')}",
                f"-I{ROOT / 'csrc/operators/common/lifetch'}",
                f"-I{ROOT / 'csrc/operators/lightning_indexer_with_fetch_ubmem/include'}",
                f"-I{tmp_path}",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
        results.append([int(value) for value in subprocess.check_output([str(binary)], text=True).split()])
    before, after = results
    assert before[0] == after[0]  # Byte-identical rows, control, maps and input copies.
    assert after[1] < before[1]  # Fewer actual MTE3 calls, not an estimated latency.
    assert before[2] > 0 and after[2] == 0  # LRU no longer has an MTE2/MTE3 chain.
    assert before[3] > 0 and after[3] == 0  # Zero Duplicate and its joins are gone.
    assert before[4] == after[4]  # Exactly one VF per packed row in either path.
