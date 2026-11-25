"""Exercise the real terminal output publisher across all 32-bit route words."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_active_and_padding_routes_past_query_64(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for the native output router")
    source = tmp_path / "routes.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
#include "terminal_runtime_abi.h"

enum class HardEvent { MTE3_V, V_MTE3 };
constexpr int EVENT_ID0 = 0;
template<HardEvent> void SetFlag(int) {}
template<HardEvent> void WaitFlag(int) {}
template<class T> struct LocalTensor {
    T* p;
    T GetValue(size_t i) const { return p[i]; }
};
template<class T> struct GlobalTensor {
    T* p;
    void SetGlobalBuffer(T* value) { p = value; }
    T* operator[](size_t i) const { return p+i; }
};
void DataCopy(uint32_t* dst, LocalTensor<uint32_t> src, size_t count) {
    std::copy_n(src.p, count, dst);
}
void Duplicate(LocalTensor<uint32_t> dst, uint32_t value, size_t count) {
    std::fill_n(dst.p, count, value);
}
#include "terminal_output_routes.h"

template<bool large> void check(uint32_t batch) {
    constexpr uint32_t k = 2048, sentinel = 407;
    const uint32_t queries = batch*4;
    std::vector<uint32_t> out(queries*k, sentinel), slots(queries*k, sentinel);
    uint64_t update[24]{};
    update[22] = reinterpret_cast<uint64_t>(out.data());
    update[23] = reinterpret_cast<uint64_t>(slots.data());
    // Process in reverse so the destination normally precedes its source.
    // Each destination selects a different Q, with padding in every word.
    for (uint32_t remaining = queries; remaining; --remaining) {
        const uint32_t query = remaining-1;
        uint32_t activeAndPad[2*terminal_runtime::Routes<large>::words]{};
        for (uint32_t row = 0; row < queries; ++row) {
            const bool padding = row%11 == 0;
            if (!padding && (query == (row+41)%queries))
                activeAndPad[row/32] |= uint32_t(1) << (row%32);
            if (padding && query%4 == 0 && row/4 == query/4)
                activeAndPad[terminal_runtime::Routes<large>::words+row/32] |= uint32_t(1) << (row%32);
        }
        std::vector<uint32_t> ids(k, query), positions(k, query+500);
        PublishTerminalOutputs<large>(update,batch,query,{ids.data()},{positions.data()},{activeAndPad},0);
    }
    for (uint32_t row=0; row<queries; ++row) {
        const uint32_t want = (row+41)%queries;
        for (uint32_t i=0; i<k; ++i) {
            assert(out[row*k+i] == (row%11 ? want : UINT32_MAX));
            assert(slots[row*k+i] == (row%11 ? want+500 : UINT32_MAX));
        }
    }
}
int main() { check<false>(8); check<false>(16); for(auto b: {20,24,28,32}) check<true>(b); }
""")
    terminal = (Path(__file__).resolve().parents[3] / "csrc/operators/lightning_indexer_with_fetch"
                / "op_kernel/terminal")
    exe = tmp_path / "routes"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(terminal), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
