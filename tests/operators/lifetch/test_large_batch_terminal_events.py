"""Exercise the production event retirement code independently of the transport driver."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_each_service_record_is_retired_by_its_request(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required for the terminal event test")
    source = tmp_path / "events.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __gm__
#define __simt_callee__
namespace pipeline_ab { struct Args { uintptr_t completedEvents; }; }
namespace rotating {
struct Event { uint32_t words[32]; };
struct Shape { uint32_t rounds, serviceLanes; };
struct Context { uintptr_t events; Shape shape; };
}
uint32_t asc_ldcg(const uint32_t* p) { return *p; }
void asc_stcg(uint32_t* p, uint32_t value) { *p = value; }
#include "terminal_events.h"

int main() {
    std::mt19937 rng(421);
    for (uint32_t batch : {20, 24, 28, 32}) for (uint32_t rounds : {2, 3, 4, 7, 32}) {
        std::vector<uint32_t> live(rounds * batch * 32), saved(live.size()), order(batch * 4);
        std::iota(live.begin(), live.end(), 1);
        const auto initial = live;
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        rotating::Context ctx{uintptr_t(live.data()), {rounds, batch}};
        pipeline_ab::Args target{};
        for (const auto query : order) for (uint32_t t = 0; t < 512; ++t)
            SaveRequestEvents(&target, &ctx, query/4, query%4, t);
        assert(live == initial);
        target.completedEvents = uintptr_t(saved.data());
        std::vector<bool> finished(batch*4);
        for (const auto query : order) {
            for (uint32_t t = 0; t < 512; ++t)
                SaveRequestEvents(&target, &ctx, query/4, query%4, t);
            finished[query] = true;
            for (uint32_t word = 0; word < live.size(); ++word) {
                const uint32_t event = word/32, owner = (event%batch)*4 + (event/batch)%4;
                assert(live[word] == (finished[owner] ? 0 : initial[word]));
                assert(saved[word] == (finished[owner] ? initial[word] : 0));
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    operators = Path(__file__).resolve().parents[3] / "csrc/operators"
    include = operators / "lightning_indexer_with_fetch/op_kernel/terminal"
    binary = tmp_path / "events"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(include), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
