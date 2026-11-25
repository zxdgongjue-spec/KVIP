// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifndef KVIP_STAGE_TRACE
#define KVIP_STAGE_TRACE 0
#endif

// MR12 binary recorder. The opt-in caller stores the pointer in the
// production streaming workspace's zero-filled reserved tail (byte 192).
namespace kvip_trace {
constexpr uint64_t kMagic = 0x4b56495054524331ULL;
constexpr uint32_t kVersion = 1, kLanes = 96;
constexpr uint32_t kWorkspaceOffset = 192;
struct alignas(128) Header {
    uint64_t magic;
    uint32_t version, lanes, capacity, eventBytes;
    uint64_t reserved[13];
};
struct alignas(128) Lane {
    uint32_t count, overflow;
    uint64_t reserved[15];
};
struct Event {
    uint64_t begin, end, coreId;
    uint32_t kernel, phase, block, subblock, iteration, detail, complete, reserved;
    uint64_t padding;
};
static_assert(sizeof(Header)==128 && sizeof(Lane)==128 && sizeof(Event)==64,"trace ABI");
enum Kernel : uint32_t { Hot = 20, Threshold, Cold, Terminal, Final };
enum Phase : uint32_t {
    KernelBody = 1, Setup = 2, LiMatmul = 21, LiScore = 22, LiHotTopK = 23,
    LiWindow = 25, LiFinalTopK = 26, LiCubeWait = 27,
    ServiceBody = 150, ServicePredecessor = 151, ServiceSelect = 152,
    PrefetchPrepare = 153, PrefetchSubmit = 154, WindowQueryTopK = 155,
    TerminalTopK = 160, TerminalResolve = 161, TerminalCapacity = 162,
    TerminalOutput = 163, TerminalWqe = 164, TerminalDoorbell = 165,
    TerminalIndex = 166, TerminalGetCq = 167, TerminalPrefetchCq = 168,
    TerminalFinish = 169,
    AicCreditWait = 180, AicKeyMte2 = 181, AicQueryMte2 = 182,
    AicL0Mte1 = 183, AicCubeIssue = 184, AicFixpIssue = 185, AicFixpWait = 186,
    AivQueryMte2 = 188, AivKeyScaleMte2 = 189, AivCubeWait = 190,
    AivScoreVf = 191, AivScoreMte3 = 192, AivBankWait = 193,
    ThresholdInputMte2 = 196, ThresholdTopKVf = 197, ThresholdOutputMte3 = 198,
    WindowInputMte2 = 200, WindowTopKVf = 201, WindowOutputMte3 = 202,
    ServiceUnionVf = 203, PrefetchBuildVf = 204, PrefetchSqMte3 = 205, ServiceInputMte2 = 206,
    TerminalInputMte2 = 210, TerminalTopKVf = 211, TerminalTopKMte3 = 212,
    TerminalResolveVf = 213, TerminalOutputMte3 = 214, TerminalSqMte3 = 215,
    TerminalIndexVf = 216, TerminalWqeVf = 217,
    // Detail stores the request, first local Q, and the leader's partition index.
    ColdPartitionWait = 218,
    ColdT2BankWait = 219, ColdT2ScoreMte2 = 220,
    ColdT2PredicateVf = 221, ColdT2PublishMte3 = 222,
    PrefetchDoorbell = 223, DedicatedDriverWait = 224, DedicatedReadySnapshot = 225
};
}

#if KVIP_STAGE_TRACE && defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "c_api/sys_var/sys_var.h"
namespace kvip_trace {
__aicore__ inline uint64_t WorkspaceAddress(GM_ADDR workspace) {
    if (!workspace) return 0;
    return *reinterpret_cast<__gm__ uint64_t*>(workspace + kWorkspaceOffset);
}

// Detail holds the logical request row, local Q, and absolute 128-score tile.
constexpr uint32_t kNoTile = (1U << 20) - 1;
constexpr uint32_t kAllQueries = 15;
__aicore__ inline uint32_t PipeDetail(uint32_t row, uint32_t query = kAllQueries, uint32_t tile = kNoTile) {
    return (row << 24) | (query << 20) | tile;
}
struct ScanSample {
    uint64_t base = 0;
    uint32_t kernel = 0;
    uint32_t epoch = 0;
    uint32_t detail = 0;
};
__aicore__ inline uint32_t ScanPhase() {
    if ASCEND_IS_AIV { return LiScore; }
    return LiMatmul;
}

__aicore__ inline void Flush(uint64_t address) {
    AscendC::GlobalTensor<uint8_t> gm;
    // dav_3510 SINGLE_CACHE_LINE is 64 bytes. Rounding a 64-byte event to
    // 128 would flush its neighbour and lose every odd event in the CA model.
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(address & ~uint64_t(63)));
    AscendC::DataCacheCleanAndInvalid<uint8_t,AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(gm);
}
// One scalar writer per logical AIC/AIV lane; no atomics, cross-core waits or
// new pipeline barriers. Regions are cache-line separated. Scope duration is
// source execution (including existing waits), not a hardware pipe busy time.
class Scope {
    __gm__ Event* event_=nullptr;
    __gm__ Lane* lane_=nullptr;
public:
    __aicore__ inline Scope(uint64_t base, uint32_t kernel, uint32_t phase,
                           uint32_t iteration=0, uint32_t detail=0) {
        if (!base) return;
        auto h=reinterpret_cast<__gm__ Header*>(base);
        if(h->magic!=kMagic || h->version!=kVersion || h->eventBytes!=sizeof(Event) ||
           !h->capacity || h->capacity>65536)return;
        uint32_t block=AscendC::GetBlockIdx(),lane;
        if ASCEND_IS_AIV { lane=block; } else { lane=64+block; }
        if(lane>=h->lanes || lane>=kLanes)return;
        lane_=reinterpret_cast<__gm__ Lane*>(base+sizeof(Header)+
            uint64_t(lane)*(sizeof(Lane)+uint64_t(h->capacity)*sizeof(Event)));
        uint32_t slot=lane_->count;
        if(slot>=h->capacity){lane_->overflow=1;Flush(reinterpret_cast<uint64_t>(lane_));return;}
        lane_->count=slot+1;
        event_=reinterpret_cast<__gm__ Event*>(lane_+1)+slot;
        event_->kernel=kernel;event_->phase=phase;event_->block=block;
        event_->subblock=AscendC::GetSubBlockIdx();
        event_->coreId=asc_get_core_id();
        event_->iteration=iteration;event_->detail=detail;
        event_->complete=0;event_->end=0;
        event_->begin=AscendC::GetSystemCycle();
    }
    __aicore__ inline void End() {
        if(!event_)return;
        event_->end=AscendC::GetSystemCycle();event_->complete=1;
        Flush(reinterpret_cast<uint64_t>(event_));
        Flush(reinterpret_cast<uint64_t>(lane_));event_=nullptr;
    }
    __aicore__ inline ~Scope(){End();}
};
}
#define KVIP_TRACE_WORKSPACE(w) kvip_trace::WorkspaceAddress(w)
#define KVIP_TRACE_RECORD(n,b,k,p,i,d) kvip_trace::Scope n(b,k,p,i,d)
#define KVIP_TRACE_BEGIN(n,b,k,p,i,d) KVIP_TRACE_RECORD(n,b,kvip_trace::k,kvip_trace::p,i,d)
#define KVIP_TRACE_END(n) n.End()
#define KVIP_PIPE_PARAM , const kvip_trace::ScanSample& pipeSample
#define KVIP_PIPE_PARAM_DEFAULT KVIP_PIPE_PARAM = {}
#define KVIP_PIPE_ARG , pipeSample
#define KVIP_PIPE_SAMPLE(n,s,p) KVIP_TRACE_RECORD(n,(s).base,(s).kernel,kvip_trace::p,(s).epoch,(s).detail)
#define KVIP_RANK_SAMPLE(n,s,w,t) KVIP_TRACE_RECORD(n,(s).base,(s).kernel, \
    (s).kernel==kvip_trace::Terminal ? kvip_trace::t : kvip_trace::w,(s).epoch,(s).detail)
#else
#define KVIP_TRACE_WORKSPACE(w) uint64_t(0)
#define KVIP_TRACE_RECORD(n,b,k,p,i,d)
#define KVIP_TRACE_BEGIN(n,b,k,p,i,d)
#define KVIP_TRACE_END(n)
#define KVIP_PIPE_PARAM
#define KVIP_PIPE_PARAM_DEFAULT
#define KVIP_PIPE_ARG
#define KVIP_PIPE_SAMPLE(n,s,p)
#define KVIP_RANK_SAMPLE(n,s,w,t)
#endif
