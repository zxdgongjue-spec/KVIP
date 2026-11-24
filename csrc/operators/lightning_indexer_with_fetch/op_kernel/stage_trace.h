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
    TerminalFinish = 169
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
#else
#define KVIP_TRACE_WORKSPACE(w) uint64_t(0)
#define KVIP_TRACE_RECORD(n,b,k,p,i,d)
#define KVIP_TRACE_BEGIN(n,b,k,p,i,d)
#define KVIP_TRACE_END(n)
#endif
