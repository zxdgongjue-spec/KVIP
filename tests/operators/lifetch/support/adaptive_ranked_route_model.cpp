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
    // A certified HIGH must agree with the complete policy for every
    // 16-owner pack. In particular the seventeenth owner may be live.
    static_assert(sizeof(Args) == 256 && offsetof(Args,routeBound) == 240);
    uint32_t rng = 0x83c43dU;
    auto next = [&] { return rng = rng*1664525U + 1013904223U; };
    for (uint32_t count : {16U,17U}) {
        assert(BoundGeometry(count));
        args.routeOwners = count;
        for (unsigned trial=0; trial<500; ++trial) {
            uint32_t sum = 0, maximum = 0;
            for (uint32_t owner=0; owner<count; ++owner) {
                uint64_t reads = next()%512;
                if (trial%17==0 && owner==0) reads = UINT64_MAX;
                feedback[owner] = {trial%3 && owner==0 ? 0U : 13U, 30U, reads, 4U};
                const uint32_t cost = OwnerCost(feedback[owner][0],reads);
                sum += cost;
                maximum = std::max(maximum,cost);
            }
            const uint32_t bound = MinimumBatchReads(count,sum,maximum);
            for (unsigned skip=0; skip<(count==16 ? 1U : 17U); ++skip) {
                for (unsigned b=0; b<16; ++b) {
                    const unsigned owner=(b+skip+1)%count;
                    owners[b]=owner;
                    requests[b]={owner,13U,31U,4U};
                    for (unsigned q=0;q<4;++q) ends[b*4+q] = shape.history-next()%2048;
                }
                if (ForcesHigh(bound,shape)) assert(choose()==High);
            }
        }
    }
    assert(!BoundGeometry(15) && !BoundGeometry(18));
    for (unsigned b=0;b<batch;++b) {
        owners[b]=b+1; requests[b]={b+1,13,31,4}; feedback[b+1]={13,30,1,4};
    }
    ends.fill(130048);
    const uint32_t maximumBudget=uint64_t(shape.history)*64*5/(32*shape.tokenBytes);
    assert(!ForcesHigh(maximumBudget,shape) && ForcesHigh(maximumBudget+1,shape));
    shape.aicCount=33; assert(choose()==High); shape.aicCount=32;
    shape.liCores=16; assert(choose()==High); shape.liCores=32;
    mode=choose(); assert(AdaptiveLow(&args));
    args.routePolicy=Early; assert(choose()==Early);
    args.routePolicy=9; assert(choose()==High);
    args.routeFeedback=0; assert(choose()==High); Commit(&args,3,0,900);
    std::cout<<"adaptive device policy: cold start, boundary, traffic, reorder, "
               "generation, stale steps and error recovery PASS\n";
}
