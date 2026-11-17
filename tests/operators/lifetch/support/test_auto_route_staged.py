# SPDX-License-Identifier: Apache-2.0
"""Compare staged routing with the original scalar contract, including fallback."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[4]
INCLUDE = ROOT / "csrc/operators/lightning_indexer_with_fetch/include"


@pytest.mark.parametrize("simt", [0, 1])
def test_staged_owner_binding_refreshes_scalar_cache(tmp_path, simt):
    """MTE2/asc_ldcg routing must not bind an arena through a stale scalar row."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the routing contract")
    lifecycle = (INCLUDE.parent / "op_kernel/lifecycle.cpp").read_text()
    start = lifecycle.index("#if KVIP_AUTO_ROUTE_SIMT ||")
    end = lifecycle.index("        uint64_t outputStride=", start)
    binding = lifecycle[start:end]
    selection = re.search(r"^\s*parallelRoute=([^;\n]+);", lifecycle, re.M).group(1)
    source = tmp_path / "owner_binding.cpp"
    binary = tmp_path / "owner_binding"
    source.write_text(r"""
#include <cassert>
#include <cstdint>
#include <cstddef>
#define __gm__
struct Context { uint64_t rowOwners; bool publish; };
uint32_t ownerCache[16], dmaOwners[16];
unsigned cleans;
namespace kvaf_device {
void CleanCacheLines(uint8_t* pointer, size_t size) {
    assert(size == sizeof(uint32_t));
    auto index = reinterpret_cast<uint32_t*>(pointer) - ownerCache;
    assert(index >= 0 && index < 16);
    ownerCache[index] = dmaOwners[index]; ++cleans;
}
}
namespace kvip_auto_route {
bool CanPublishRows(Context* ctx) { return ctx->publish; }
}
uint32_t Bind(Context* ctx, uint32_t b, bool stageRoute, bool parallelRoute) {
""" + binding + r"""
    return owner;
}
int main() {
    Context state{uint64_t(ownerCache),true};
    Context* ctx=&state;
    for (int staged=0;staged<2;++staged) {
        bool stageRoute=staged;
        bool parallelRoute=""" + selection + r""";
        assert(!(stageRoute && parallelRoute));
        for(uint32_t b=0;b<16;++b) {
            ownerCache[b]=b+1;dmaOwners[b]=16-b;cleans=0;
            const bool fresh=stageRoute || (KVIP_AUTO_ROUTE_SIMT && parallelRoute);
            assert(Bind(ctx,b,stageRoute,parallelRoute)==(fresh?dmaOwners[b]:b+1));
            assert(cleans==unsigned(fresh));
        }
    }
    ctx->rowOwners=0;cleans=0;
    assert(Bind(ctx,7,true,false)==7 && cleans==0);
}
""")
    subprocess.run([compiler, "-std=c++17", "-O2",
                    f"-DKVIP_AUTO_ROUTE_SIMT={simt}",
                    "-DKVIP_LOW_MISS_POST_INDEX=1", "-DKVIP_AUTO_ROUTE_ROW_PUBLISH=1",
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_staged_route_matches_original_contract(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the routing contract")
    source = tmp_path / "staged_route.cpp"
    binary = tmp_path / "staged_route"
    source.write_text(SOURCE)
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(INCLUDE),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


SOURCE = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <random>
#include "auto_route.h"
#define __aicore__
#define __gm__
#define __ubuf__
#include "auto_route_staged.h"

using namespace kvip_auto_route;
static unsigned comparisons=0,rowComparisons=0;
struct Context {
    uint64_t autoRoute, admissionControls, rowOwners, queryEnds;
    struct { uint32_t batch = 16, history = 131328, tokenBytes = 656; } shape;
};
struct Fixture {
    std::array<Request, 16> requests{};
    std::array<Feedback, 129> feedback{};
    std::array<uint32_t, 16> owners{};
    std::array<uint32_t, 64> ends{};
    alignas(128) std::array<Pending, 16> fullPending{}, shortPending{};
    alignas(128) std::array<int32_t, 64> fullControls{}, shortControls{};
    Config config{};
    bool noOwners=false,noEnds=false;uint32_t batch=16;
    Fixture() {
        config.version = 1; config.owners = 17;
        config.lowBudget = 256; config.highBudget = 384; config.maxStepGap = 8;
        config.learning = 1; config.fullTotal = 512; config.fullWindow = 128;
        config.fullSteps = 8; config.fullCarry = 1;
        ends.fill(131328);
        for (uint32_t b = 0; b < 16; ++b) {
            const uint32_t owner = b == 0 ? 2 : b == 1 ? 1 : b + 1;
            owners[b] = owner;
            requests[b] = {owner, 10 + owner, 100 + owner, 4};
            auto& f = feedback[owner];
            f.generation = requests[b].generation; f.step = requests[b].step - 1;
            f.necessary = 60; f.queries = 4; f.scoreWork = uint64_t(131328) * 4;
            f.tokenBytes = 656; f.mode = 0; f.valid = kValid;
            std::fill(std::begin(f.reserved), std::end(f.reserved), uint64_t(0xdeadbeef));
        }
    }
    bool compare() {
        ++comparisons;
        config.requests = uint64_t(requests.data());
        config.feedback = uint64_t(feedback.data()); config.pending = uint64_t(fullPending.data());
        Context ctx{uint64_t(&config), uint64_t(fullControls.data()), noOwners?0:uint64_t(owners.data()), noEnds?0:uint64_t(ends.data())};
        ctx.shape.batch=batch;
        fullPending.fill({}); shortPending.fill({}); fullControls.fill(-7); shortControls.fill(-7);
        bool full = false;
        bool expectedResult=Evaluate<true>(&ctx, full);
        config.pending = uint64_t(shortPending.data());
        ctx.admissionControls = uint64_t(shortControls.data());
        RouteTables table{};table.config=config;
        std::copy(requests.begin(),requests.end(),table.requests);
        std::copy(owners.begin(),owners.end(),table.owners);
        std::copy(ends.begin(),ends.end(),table.ends);
        if(config.owners<=64) for(uint32_t i=0;i<config.owners;++i)
            std::memcpy(&table.feedback[i],&feedback[i],sizeof(FeedbackHead));
        for (uint32_t reverse = std::max(1U,batch); reverse; --reverse) {
            ++rowComparisons;bool row = !full;bool got;
            if(CanStage(&ctx,reverse-1))got=EvaluateStaged(&ctx,&table,row,reverse-1);
            else got=Evaluate<true,true>(&ctx,row,reverse-1);
            assert(got==expectedResult);assert(row == full);
        }
        assert(!std::memcmp(fullPending.data(), shortPending.data(), sizeof(fullPending)));
        assert(fullControls == shortControls);
        return full;
    }
};
int main() {
    static_assert(offsetof(Feedback, reserved) == 64 && sizeof(Feedback) == 128);
    Fixture f; assert(f.compare());
    std::reverse(f.requests.begin(), f.requests.end());
    std::reverse(f.owners.begin(), f.owners.end()); assert(f.compare());
    f.requests[0].generation += 1; f.requests[0].step = 1; assert(!f.compare());
    f = Fixture(); f.requests[0].step = f.feedback[f.requests[0].owner].step; assert(!f.compare());
    f = Fixture(); f.requests[0].generation -= 1; assert(!f.compare());
    f = Fixture(); f.requests[1].owner = f.requests[0].owner; f.owners[1] = f.owners[0]; assert(!f.compare());
    f = Fixture(); f.requests[1].queries = 0; std::fill(f.ends.begin() + 4, f.ends.begin() + 8, 0); assert(f.compare());
    f = Fixture(); f.owners[0] = 0; assert(!f.compare());
    f = Fixture(); f.feedback[f.owners[0]].valid = 0; assert(!f.compare());
    f = Fixture(); f.feedback[f.owners[0]].necessary = 8193; assert(!f.compare());
    f = Fixture(); f.feedback[f.owners[0]].queries = 0; assert(!f.compare());
    f = Fixture(); f.feedback[f.owners[0]].mode = 2; assert(!f.compare());
    f = Fixture(); f.feedback[f.owners[0]].tokenBytes = 2624; assert(!f.compare());
    f = Fixture(); f.config.learning = 0; assert(!f.compare());
    f = Fixture(); for (auto& sample : f.feedback) sample.necessary = 4096; assert(!f.compare());
    f=Fixture(); f.ends[0]=131329;assert(!f.compare());
    f=Fixture();for(auto& r:f.requests)r.queries=0;f.ends.fill(0);assert(!f.compare());
    f=Fixture();f.config.owners=129;f.requests[0].owner=128;f.owners[0]=128;
    f.feedback[128]=f.feedback[2];assert(f.compare());
    f=Fixture();f.config.fullWindow=481;assert(!f.compare());
    f=Fixture();f.noOwners=true;assert(!f.compare());
    f=Fixture();f.noEnds=true;assert(f.compare());
    f=Fixture();f.batch=0;assert(!f.compare());
    for(uint32_t b:{1U,7U,8U}){f=Fixture();f.batch=b;assert(f.compare());}
    f=Fixture();f.config.version=9;assert(!f.compare());
    f=Fixture();f.config.lowBudget=0;assert(!f.compare());
    f=Fixture();f.config.fullSteps=65536;assert(!f.compare());
    f=Fixture();f.config.owners=0;assert(!f.compare());
    std::mt19937 rng(117);
    for (uint32_t iteration = 0; iteration < 5000; ++iteration) {
        f = Fixture();
        for (uint32_t b = 0; b < 16; ++b) {
            auto& request = f.requests[b]; auto& previous = f.feedback[request.owner];
            request.queries = 1 + rng() % 4;
            previous.queries = 1 + rng() % 4;
            previous.necessary = rng() % 8193; previous.mode = rng() % 2;
            for (uint32_t q = 0; q < 4; ++q) f.ends[b * 4 + q] = q < request.queries ? 130000 + rng() % 1329 : 0;
            if ((iteration + b) % 31 == 0) previous.valid = 0;
            if ((iteration + b) % 37 == 0) request.generation += 1;
            if ((iteration + b) % 41 == 0) request.step = previous.step;
        }
        f.compare();
    }
    std::printf("INIT_STAGED_CPU_PASS cases=%u row_comparisons=%u all_pending_controls_exact=pass identity_reorder_generation_invalid_padding_large_owner_fallback=pass\n",comparisons,rowComparisons);
}

"""
