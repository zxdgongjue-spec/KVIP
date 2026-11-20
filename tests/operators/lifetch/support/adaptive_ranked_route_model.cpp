// SPDX-License-Identifier: Apache-2.0
// Exercise the device policy and commit functions against host memory.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include "runtime_shape.h"

#define __aicore__
#define __gm__
#define KVIP_OUTPUT_WORKER 1
namespace kvaf_device {
inline void CleanCacheLines(uint8_t*, uint32_t) {}
}
namespace rotating {
struct Context { uint64_t rowOwners, queryEnds, reservedHot, fallbackAddresses, sourceBlocks; };
}
#include "adaptive_route_probe.h"

int main() {
    using namespace pipeline_ab;
    using namespace adaptive_route;
    constexpr unsigned batch = 16;
    std::array<std::array<uint64_t, kRequestWords>, batch> requests{};
    std::array<std::array<uint64_t, kFeedbackWords>, batch+1> feedback{};
    std::array<std::array<uint32_t, 32>, batch> demands{};
    std::array<uint32_t, batch> owners{};
    std::array<uint32_t, 4*batch> ends{};
    uint32_t mode = High;
    for (unsigned b = 0; b < batch; ++b) {
        owners[b] = b+1;
        requests[b] = {owners[b], 80+b, 1, 4};
    }
    ends.fill(130048);
    rotating::RuntimeShape shape{};
    shape.batch=batch; shape.history=131328; shape.resident=16384;
    shape.indexHeads=32; shape.splitQ=1; shape.tokenBytes=656; shape.aicCount=32; shape.liCores=32;
    rotating::Context ctx{reinterpret_cast<uint64_t>(owners.data()),
        reinterpret_cast<uint64_t>(ends.data()), 1, 1, 1};
    Args args{};
    args.context=reinterpret_cast<uint64_t>(&ctx);
    args.routeRequests=reinterpret_cast<uint64_t>(requests.data());
    args.routeFeedback=reinterpret_cast<uint64_t>(feedback.data());
    args.routeResult=reinterpret_cast<uint64_t>(&mode);
    args.routeDemand=reinterpret_cast<uint64_t>(demands.data());
    args.routeOwners=feedback.size(); args.routePolicy=Zero;
    args.completedFlags=1; args.prefetchCount=0x10000000U;
    auto choose = [&] { return Choose(&args,&ctx,shape); };
    auto advance = [&] { for (auto& r : requests) ++r[2]; };
    auto finish = [&](uint32_t total, int failed=-1) {
        for (unsigned b = 0; b < batch; ++b)
            Commit(&args,b,int(b)==failed,total/batch+(b<total%batch));
        advance();
    };
    assert(choose()==High);
    finish(955); assert(choose()==Zero);
    std::swap(requests.front(),requests.back()); std::swap(owners.front(),owners.back());
    assert(choose()==Zero);
    for (unsigned total : {3152U,4717U,7142U,12516U}) {
        finish(total); assert(choose()==High);
        finish(955); assert(choose()==Zero);
    }
    // Exact boundary and a different current workload with the same past.
    const unsigned budget=uint64_t(130048)*4*batch*5/(32*shape.tokenBytes);
    finish(budget); assert(choose()==Zero);
    finish(budget+1); assert(choose()==High);
    finish(955); ends.fill(65536); assert(choose()==Zero);
    ends.fill(32768); assert(choose()==High); ends.fill(130048);
    auto& request=requests[3];
    auto& record=feedback[owners[3]];
    ++request[1]; assert(choose()==High);
    finish(955); assert(choose()==Zero);
    auto prev=record;
    finish(7500,3); assert(record==prev && choose()==High);
    finish(955); assert(choose()==Zero);
    const auto step=request[2]; request[2]-=2;
    prev=record; Commit(&args,3,0,900); assert(record==prev && choose()==High);
    request[2]=step; request[3]=3; assert(choose()==High); request[3]=4;
    --request[1]; Commit(&args,3,0,900); assert(record==prev && choose()==High); ++request[1];
    std::swap(owners[0],owners[3]); assert(choose()==High); std::swap(owners[0],owners[3]);
    ends[8]=0; assert(choose()==High); ends[8]=130048;
    assert(choose()==Zero);
    // Every active warp lane, including the last, must veto a bad row.
    for (unsigned b=0; b<batch; ++b) {
        const auto selected = feedback[owners[b]][2];
        feedback[owners[b]][2] = uint64_t{1} << 32;
        assert(choose()==High);
        feedback[owners[b]][2] = selected;
        for (unsigned q=0; q<4; ++q) {
            ends[4*b+q] = shape.history+1;
            assert(choose()==High);
            ends[4*b+q] = 130048;
        }
    }
    shape.aicCount=33; assert(choose()==High); shape.aicCount=32;
    shape.liCores=16; assert(choose()==High); shape.liCores=32;
    mode=choose(); assert(AdaptiveLow(&args));
    args.routePolicy=Early; assert(choose()==Early);
    args.routePolicy=9; assert(choose()==High);
    args.routeFeedback=0; assert(choose()==High); Commit(&args,3,0,900);
    std::cout<<"adaptive device policy: cold start, boundary, traffic, reorder, "
               "generation, stale steps and error recovery PASS\n";
}
