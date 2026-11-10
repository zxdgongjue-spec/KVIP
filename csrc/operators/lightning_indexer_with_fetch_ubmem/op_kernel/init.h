// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "context.h"
#include "copy/device_utils.h"
#include "resolver/direct_slots.h"
#include "init_partition.h"
#include "../../common/lifetch/simt_init.h"
namespace rotating { namespace init {
using namespace AscendC;
__aicore__ inline __gm__ Context* GetContext(GM_ADDR bindings) {
    return reinterpret_cast<__gm__ Context*>(reinterpret_cast<__gm__ uint64_t*>(bindings)[14]);
}
struct InitRowValues {uint64_t addresses,output;uint32_t error;};
__aicore__ inline InitRowValues GetInitRowValues(__gm__ Context* ctx,uint32_t b) {
    auto region=UbmemRegionAt(ctx);
    uint64_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
    uint64_t stride=ctx->outputStride?ctx->outputStride:SIM_RESIDENT;
    uint64_t rowBytes=stride*kTokenBytes;
    uint32_t error=(!ctx->ubmemStatus || !UbmemRegionValid(region) || !ctx->tokenAddresses ||
        !ctx->lruSlots || !ctx->reservedHot || !ctx->changeSlots || !ctx->changeCount ||
        ctx->addressStride<SIM_HISTORY || stride<SIM_RESIDENT || (stride-SIM_RESIDENT)%128 || (ctx->tailBounds && stride<SIM_RESIDENT+384) ||
        rowBytes%32 || ctx->prefetchLimit>kLruSlots || !ctx->windowLimit || ctx->windowLimit>480 ||
        (V9_UBMEM_TAGGED_MAP && stride>=kMappingMaxOutputStride))?kUbmemInvalidRegion:0;
    if(!error && (rowBytes>region->outputBytes || owner>region->outputBytes/rowBytes-1))error=kUbmemInvalidRegion;
    return {ctx->tokenAddresses+owner*ctx->addressStride*8,
            error?0:region->outputBase+owner*rowBytes,error};
}
#if V9_SIMT_INIT
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeRoundSimt(__gm__ Context* ctx,uint32_t b,uint64_t reservedRow,
    uint32_t mappingTag,uint64_t mappingState,uint64_t addresses,uint64_t output,uint32_t error) {
    uint64_t status=ctx->ubmemStatus?ctx->ubmemStatus+uint64_t(b)*kUbmemStatusWords*8:0;
    InitializeSimtControl<Row,Progress,Event,FinalPost>(ctx,b,SIM_BATCH,kRounds,kServiceLanes,
        kFinalWorkers,kFinalPostOffset,status,kUbmemStatusWords*2,
        mappingState,0,addresses,output,0,kLruSlots,error);
    InitializeTailMappingsBody(SIM_HISTORY,SIM_RESIDENT,ctx,b,reservedRow,mappingTag);
}
#endif
#if V9_RESERVED_UB
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeReservedLocal(__gm__ Context* ctx,uint32_t b,uint64_t reservedRow,
    __ubuf__ uint32_t* inverse,const __ubuf__ uint32_t* candidates,uint32_t mappingTag) {
    if(reservedRow)ScatterReservedHotLocal(inverse,candidates,SIM_RESIDENT,kLruSlots);
    InitializeTailMappingsBody(SIM_HISTORY,SIM_RESIDENT,ctx,b,0,mappingTag);
}
#endif
__aicore__ inline void FillWords(uint64_t address,uint32_t words,LocalTensor<uint32_t> local) {
    GlobalTensor<uint32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    for(uint32_t offset=0;offset<words;offset+=4096) {
        uint32_t count=words-offset;if(count>4096)count=4096;
        DataCopy(output[offset],local,count);
    }
}
__aicore__ inline void InitializeControl(__gm__ Context* ctx,uint32_t b,LocalTensor<uint32_t> local) {
    FillWords(ctx->rows+b*sizeof(Row),sizeof(Row)/4,local);
    FillWords(ctx->progress+b*4*sizeof(Progress),4*sizeof(Progress)/4,local);
    FillWords(ctx->owners+b*128,32,local);
    FillWords(ctx->owners+SIM_BATCH*128+b*kFinalWorkers*kFinalStateWords*4,kFinalWorkers*kFinalStateWords,local);
    FillWords(ctx->owners+kFinalPostOffset+b*kFinalWorkers*sizeof(FinalPost),kFinalWorkers*sizeof(FinalPost)/4,local);
    FillWords(ctx->rowStatus+b*kRowStatusWords*4,kRowStatusWords,local);
    if(ctx->ubmemStatus)FillWords(ctx->ubmemStatus+b*kUbmemStatusWords*8,kUbmemStatusWords*2,local);
    for(uint32_t e=b;e<(kRounds?kRounds:1)*kServiceLanes;e+=SIM_BATCH)
        FillWords(ctx->events+e*sizeof(Event),sizeof(Event)/4,local);
    if(!b) {
        *reinterpret_cast<__gm__ uint32_t*>(ctx->count)=0;
        ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->count),4);
    }
}
__aicore__ inline void CopyInitialLruSlots(__gm__ Context* ctx,uint32_t b,LocalTensor<uint32_t> local,bool stage=false) {
    if(ctx->changeSlots || stage) {
        GlobalTensor<uint32_t> source,target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots);
        DataCopy(local,source,kLruSlots);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        if(ctx->changeSlots) {
            target.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+b*kChangeCapacity);
            DataCopy(target,local,kLruSlots);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
}
__aicore__ inline void InitializeRound(__gm__ Context* ctx,uint32_t b) {
    MappingGeneration generation{0,0,true};
#if V9_UBMEM_TAGGED_MAP
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    generation=NextMappingGeneration(row->mappingState);
    // Scalar loading the old epoch refills this cache line. Discard it before
    // the DMA zero-fill, so publishing the new epoch cannot write old Row
    // counters back over those zeros.
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
#endif
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,(4096+V9_RESERVED_UB*16384)*4);
    auto local=scratch.Get<uint32_t>();
#if !V9_SIMT_INIT
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    InitializeControl(ctx,b,local);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
#endif
    uint64_t reservedRow=0;
#if V9_RESERVED_UB
    reservedRow=ReservedHotRow(ctx->lruSlots,ctx->reservedHot,ctx->queryEnds,b,SIM_RESIDENT);
#endif
    Duplicate(local,UINT32_MAX,4096+(V9_RESERVED_UB && reservedRow?16384:0));
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(generation.clear)FillWords(ctx->mapping+uint64_t(b)*SIM_HISTORY*4,SIM_HISTORY,local);
    FillWords(ctx->selected+uint64_t(b)*kSlots*4,kSlots,local);
    // Only prefetch candidate holes need a round-wide sentinel fill. Resolver
    // publishes every extended rank below changeCount before Index consumes it.
    if(ctx->changes)FillWords(ctx->changes+b*kChangeCapacity*4,kLruSlots,local);
#if V9_COMPACT_RESERVED && !V9_RESERVED_UB
    reservedRow=ClearReservedHot(ctx->lruSlots,ctx->reservedHot,ctx->queryEnds,b,SIM_RESIDENT,local);
#endif
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if V9_SIMT_INIT
    const auto values=GetInitRowValues(ctx,b);
    // Old epoch cache lines were discarded above. The VF is the only Row
    // writer; no later scalar writeback can restore stale epoch/counter data.
    Simt::VF_CALL<InitializeRoundSimt>(Simt::Dim3{256,1,1},ctx,b,reservedRow,generation.tag,
        generation.state,values.addresses,values.output,values.error);
#else
#if V9_UBMEM_TAGGED_MAP
    // The Row zero-fill is complete. Persist the epoch for later windows;
    // the init entry publishes this cache line after filling row metadata.
    row->mappingState=generation.state;
#endif
    CopyInitialLruSlots(ctx,b,local,V9_RESERVED_UB && reservedRow);
#if V9_RESERVED_UB
    // The existing candidate copy owns the first 8 KiB. The inverse occupies
    // a disjoint 64 KiB and remains live until its single MTE3 has completed.
    if(reservedRow) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    auto inverse=local[4096];
    Simt::VF_CALL<InitializeReservedLocal>(Simt::Dim3{256,1,1},ctx,b,reservedRow,
        reinterpret_cast<__ubuf__ uint32_t*>(inverse.GetPhyAddr()),
        reinterpret_cast<const __ubuf__ uint32_t*>(local.GetPhyAddr()),generation.tag);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(reservedRow) {
        GlobalTensor<uint32_t> output;output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(reservedRow));
        DataCopy(output,inverse,SIM_RESIDENT);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#else
    Simt::VF_CALL<InitializeTailMappings>(Simt::Dim3{256,1,1},SIM_HISTORY,SIM_RESIDENT,ctx,b,reservedRow,generation.tag);
#endif
#endif
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    pipe.Destroy();
}
#if V9_UBMEM_PARALLEL_INIT
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailShard(__gm__ Context* ctx,uint32_t b,uint32_t part,
    uint32_t first,uint32_t last,uint64_t reservedRow) {
#if V9_COMPACT_RESERVED
    if(!part && reservedRow)ScatterReservedHot(reinterpret_cast<__gm__ uint32_t*>(reservedRow),
        reinterpret_cast<const __gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*kLruSlots,SIM_RESIDENT,kLruSlots);
#endif
    if(!part && !threadIdx.x && ctx->changeCount)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots);
    if(ctx->tailBounds) {
        const uint32_t tailTokens=OutputStride(SIM_RESIDENT,ctx)-SIM_RESIDENT;
        if(tailTokens) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
            // Keep the original bounded ABC initialization, including when
            // a malformed end exceeds the available tail output capacity.
            uint32_t begin=asc_ldcg(bounds),end=asc_ldcg(bounds+1);
            uint64_t capacityEnd=uint64_t(begin)+tailTokens;
            if(uint64_t(end)>capacityEnd)end=uint32_t(capacityEnd);
            auto range=InitTailIntersection({first,last},begin,end,SIM_HISTORY);
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*SIM_HISTORY;
            for(uint32_t i=threadIdx.x;i<range.end-range.begin;i+=256) {
                uint32_t token=range.begin+i;
                asc_stcg(map+token,SIM_RESIDENT+kSlots+token%tailTokens);
            }
        }
    }
    asc_threadfence();
}
__aicore__ inline void InitializeRoundParallel(__gm__ Context* ctx,uint32_t b,uint32_t part) {
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,4096*4);
    auto local=scratch.Get<uint32_t>();
    Duplicate(local,uint32_t(0),4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(!part) {
        // Only one writer initializes each control record. Mapping writers
        // need no global barrier: the next kernel waits for every init AIV.
        InitializeControl(ctx,b,local);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    Duplicate(local,UINT32_MAX,4096);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    auto map=PartitionInitWords<SIM_HISTORY>(part,kInitWorkersPerBatch);
    auto selected=PartitionInitWords<kSlots>(part,kInitWorkersPerBatch);
    auto changes=PartitionInitWords<kLruSlots>(part,kInitWorkersPerBatch);
    FillWords(ctx->mapping+(uint64_t(b)*SIM_HISTORY+map.begin)*4,map.end-map.begin,local);
    FillWords(ctx->selected+(uint64_t(b)*kSlots+selected.begin)*4,selected.end-selected.begin,local);
    if(ctx->changes)FillWords(ctx->changes+(uint64_t(b)*kChangeCapacity+changes.begin)*4,changes.end-changes.begin,local);
    uint64_t reservedRow=0;
#if V9_COMPACT_RESERVED
    if(!part)reservedRow=ClearReservedHot(ctx->lruSlots,ctx->reservedHot,ctx->queryEnds,b,SIM_RESIDENT,local);
#endif
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if(!part)CopyInitialLruSlots(ctx,b,local);
    // Each shard installs ABC only after its own sentinel DMA has completed;
    // no shard can clear or scalar-write another shard's cache line.
    Simt::VF_CALL<InitializeTailShard>(Simt::Dim3{256,1,1},ctx,b,part,map.begin,map.end,reservedRow);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    pipe.Destroy();
}
#endif

__aicore__ inline void InitializeFetchCore(__gm__ Context* ctx,uint32_t core) {
#if V9_UBMEM_PARALLEL_INIT
    if(core>=kInitCores)return;
    uint32_t b=core%SIM_BATCH,part=core/SIM_BATCH;
    InitializeRoundParallel(ctx,b,part);
    if(part)return;
#else
    uint32_t b=core;if(b>=SIM_BATCH)return;
    InitializeRound(ctx,b);
#endif
#if !V9_SIMT_INIT
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    const auto values=GetInitRowValues(ctx,b);
#if !V9_UBMEM_TAGGED_MAP
    row->mappingState=0;
#endif
    row->reserved1=0;row->reserved2=0;
    row->addresses=values.addresses;row->output=values.output;
    row->limit=kSlots;row->prefetchLimit=ctx->prefetchLimit;row->copyLimit=kLruSlots;row->error=values.error;
    ubmem_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
#endif
}
#if V9_UBMEM_FUSED_INIT
__aicore__ inline void InitializeBeforeLi(__gm__ Context* ctx,uint32_t core) {
    // Non-owner AIVs return only from the row helper, never from this barrier.
    // The init TPipe is destroyed and all GM writes are complete first. No
    // LI UB allocation or cube/vector credit exists on any AIV before here.
    InitializeFetchCore(ctx,core);
    // Ascend950's hardware AIV-only collective needs no reusable GM flags.
    // Every launched AIV participates, including inactive/padding producers.
    SyncAll<true>();
}
#endif

}}
#endif
