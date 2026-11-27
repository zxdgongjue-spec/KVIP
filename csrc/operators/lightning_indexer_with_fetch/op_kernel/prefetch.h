#pragma once
#include "stage_trace.h"
#if V9_UBMEM
#include "../../lightning_indexer_with_fetch_ubmem/op_kernel/controller/prefetch.h"
namespace kvaf_device { using ubmem_device::CleanCacheLines; constexpr uint32_t kInvalidArgs=0x20000, kCqTimeout=0x40000; }
#else
#pragma once

// Imported from rotation_device.h

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "context.h"
#include "indexer/ranked_candidates.h"
#include "../../common/lifetch/prefetch_budget.h"

// Imported from direct_slots.h

#include "simt_api/device_functions.h"
// Prefetch ordinals map to LRU hot destinations. Final repair uses hot slots;
// virtual ABC tags remain disjoint from prefetch ordinals.
namespace rotating {
#include "../../common/lifetch/cache_metadata.h"
// ABC destinations form a ring immediately after the resident output.
__simt_callee__ inline uint32_t DirectTailSlot(uint32_t token,uint32_t resident,
    uint32_t begin,uint32_t end,uint32_t capacity) {
    if(token>=begin && token<end && token-begin<capacity)return resident+token%capacity;
    return UINT32_MAX;
}
__simt_callee__ inline uint32_t DirectTailSlot(__gm__ Context* ctx,uint32_t batch,
    uint32_t resident,uint32_t token) {
    if(!ctx->tailBounds)return UINT32_MAX;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+batch*2;
    return DirectTailSlot(token,resident,asc_ldcg(bounds),asc_ldcg(bounds+1),
                          OutputStride(resident,ctx)-resident);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailMappings(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t beginOffset,uint32_t endOffset,bool materialize) {
    // Use the existing SIMT launch: scalar cache-line writeback would race
    // adjacent request counts initialized by other cores.
    if(!threadIdx.x && beginOffset == 0 && ctx->changeCount) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots);
        asc_threadfence();
    }
    if(!threadIdx.x && beginOffset==0 && ctx->mappingJournal){
        auto journal=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal)+b;
        auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds);
        uint32_t first=materialize && bounds?asc_ldcg(bounds+2*b):0;
        uint32_t last=materialize && bounds?asc_ldcg(bounds+2*b+1):0;
        uint32_t capacity=OutputStride(rt_resident,ctx)-rt_resident;
        if(last>rt_history)last=rt_history;
        if(last<first)last=first;
        if(last-first>capacity)last=first+capacity;
        asc_stcg(&journal->count,0U);
        asc_stcg(&journal->tailBegin,first);
        asc_stcg(&journal->tailEnd,last);
        asc_threadfence();
    }
    if(!materialize || !ctx->tailBounds)return;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
    uint32_t begin=asc_ldcg(bounds),end=asc_ldcg(bounds+1);
    const uint32_t tailTokens=OutputStride(rt_resident,ctx)-rt_resident;
    for(uint32_t i=threadIdx.x;i<tailTokens;i+=256) {
        uint32_t token=begin+i;
        if(token<end && token<rt_history && token>=beginOffset && token<endOffset)
            asc_stcg(map+token,rt_resident+kSlots+token%tailTokens);
    }
    asc_threadfence();
}

// Each request's prefetch queue appends contiguous ordinals in service order.
__simt_callee__ inline void TrackMapping(__gm__ Context* ctx,uint32_t b,uint32_t ordinal,
    uint32_t token,bool last) {
    if (!ctx->mappingJournal || ordinal>=kLruSlots) return;
    auto journal=reinterpret_cast<__gm__ MappingJournal*>(ctx->mappingJournal)+b;
    asc_stcg(journal->tokens+ordinal,token);
    if(last) asc_stcg(&journal->count,ordinal+1);
}

// Keep exported change records paired for callers of Index update. Only
// ordinals actually admitted by this round have a corresponding token.
__simt_callee__ inline void RecordPrefetchSlot(uint32_t resident,__gm__ Context* ctx,
    uint32_t batch,uint32_t slot,uint32_t token,uint32_t physical) {
    RecordChange(resident,ctx,batch,slot,token);
    const uint32_t ordinal=slot-resident;
    if(ctx->changeSlots && ordinal<kLruSlots)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeSlots)+batch*kChangeCapacity+ordinal,
                 physical);
}

// Resolver workers consume the pre-round validity. Publish only after all
// resolver workers finish; consumers of KV still wait for Fetch completion.



}

namespace rotating {
using namespace AscendC;
// v9 GatherCandidateMap's indexed-load loop, adapted to packed uint16 scores.
// One thread owns both halves of each UB word; no half-word write sharing.
#include "../../common/lifetch/victim_ops.h"

__simd_vf__ inline void PrepareProgress(__ubuf__ uint32_t* table, uint32_t baseTile) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> sequence;
    Arange(sequence, int32_t(0));
    auto& index = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
    RegTensor<uint32_t> value;
    ShiftRights(value, index, int16_t(4), all);
    Adds(value, value, uint32_t(baseTile + 1), all);
    Muls(value, value, uint32_t(128), all);
    for (uint16_t offset = 0; offset < 64 * 16; offset += 64) {
        StoreAlign(table + offset, value, all);
        Adds(value, value, uint32_t(512), all);
    }
}

}

// Imported from window_fetch.h


// Imported from fetch_profiling.h

#define V9_FETCH_PROFILE(...) if constexpr (captureIo) { __VA_ARGS__ }

#include "kvaf_batch_get_device.h"
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
#include <stddef.h>
#endif

// Imported from final_submit.h

// WQE fields/owner/completion encoding are inherited from the Codex SIMD demo.
// Only the address gather and destination slots differ: long histories do not
// fit in UB, and final repair may safely recycle unreferenced speculative slots.
namespace rotating {
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
constexpr uint32_t kUrmaRegisteredBaseWord =
    offsetof(KvafBatchGetDirectUrmaEntity, registered_base) / sizeof(uint64_t);
constexpr uint32_t kUrmaCanonicalBaseWord =
    offsetof(KvafBatchGetDirectUrmaEntity, remote_host_ddr_address) / sizeof(uint64_t);
static_assert(kUrmaRegisteredBaseWord < 24 && kUrmaCanonicalBaseWord < 24,
              "URMA bases must be in the staged entity words");
#endif

__simd_vf__ inline void FinalBuild(uint32_t rt_tokenBytes,__ubuf__ uint32_t* remoteLow, __ubuf__ uint32_t* remoteHigh, __ubuf__ uint32_t* slots, __ubuf__ uint32_t* words,
    __ubuf__ uint32_t* config, uint32_t begin, uint32_t count, uint32_t total) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> sequenceSigned;
    auto& sequence = reinterpret_cast<RegTensor<uint32_t>&>(sequenceSigned);
    RegTensor<uint32_t> index, offsets, remoteLo, remoteHi, localLo, localHi;
    RegTensor<uint32_t> baseLo, baseHi, delta, one, zero, carry, value, pi, depth, owner, control;
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(baseLo, config + 14);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(baseHi, config + 15);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(pi, config + 16);
    LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(depth, config + 17);
    Duplicate(one, uint32_t(1)); Duplicate(zero, uint32_t(0));
    uint32_t remaining = count;
    for (uint16_t j = 0; j < count; j += 64) {
        auto mask = UpdateMask<uint32_t>(remaining);
        Arange(sequenceSigned, int32_t(j));
        Adds(index, sequence, begin, mask);
        RegTensor<uint32_t> slot;
        LoadAlign(slot, slots + begin + j);
        LoadAlign(remoteLo,remoteLow+begin+j);LoadAlign(remoteHi,remoteHigh+begin+j);
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
        RegTensor<uint32_t> remoteDeltaLo, remoteDeltaHi, translatedLo;
        LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(remoteDeltaLo, config + 10);
        LoadAlign<uint32_t, LoadDist::DIST_BRC_B32>(remoteDeltaHi, config + 11);
        Add(translatedLo, remoteLo, remoteDeltaLo, mask);
        MaskReg remoteOverflow;
        Compare<uint32_t, CMPMODE::LT>(remoteOverflow, translatedLo, remoteLo, mask);
        Select(carry, one, zero, remoteOverflow);
        Add(remoteHi, remoteHi, remoteDeltaHi, mask);
        Add(remoteHi, remoteHi, carry, mask);
        Add(remoteLo, translatedLo, zero, mask);
#endif
        Muls(delta, slot, uint32_t(rt_tokenBytes), mask);
        Add(localLo, baseLo, delta, mask);
        MaskReg overflow;
        Compare<uint32_t, CMPMODE::LT>(overflow, localLo, baseLo, mask);
        Select(carry, one, zero, overflow); Add(localHi, baseHi, carry, mask);
        Add(value, pi, index, mask); And(value, value, depth, mask);
        MaskReg firstLap, completion;
        Compare<uint32_t, CMPMODE::EQ>(firstLap, value, zero, mask);
        Select(owner, one, zero, firstLap); ShiftLefts(owner, owner, int16_t(31), mask);
        CompareScalar<uint32_t, CMPMODE::EQ>(completion, index, total - 1, mask);
        Duplicate(control, uint32_t(0x30150000)); Duplicate(value, uint32_t(0x30360000));
        Select(control, value, control, completion); Or(control, control, owner, mask);
        // Transpose 16 field vectors into 64 complete WQEs in registers.
        // Contiguous stores avoid the stride-16 Scatter bank conflicts.
        RegTensor<uint32_t> f1,f2,f3,f4,f5,f6,f7,f8,f9,length,localToken,low,high;
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f1,config+1);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f2,config+2);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f3,config+3);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f4,config+4);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f5,config+5);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f6,config+6);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f7,config+7);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f8,config+8);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(f9,config+9);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(length,config+12);
        LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(localToken,config+13);
        // Interleave field groups of 8.
        Interleave(low,high,control,f8);control=low;f8=high;
        Interleave(low,high,f1,f9);f1=low;f9=high;
        Interleave(low,high,f2,remoteLo);f2=low;remoteLo=high;
        Interleave(low,high,f3,remoteHi);f3=low;remoteHi=high;
        Interleave(low,high,f4,length);f4=low;length=high;
        Interleave(low,high,f5,localToken);f5=low;localToken=high;
        Interleave(low,high,f6,localLo);f6=low;localLo=high;
        Interleave(low,high,f7,localHi);f7=low;localHi=high;
        // Interleave field groups of 4.
        Interleave(low,high,control,f4);control=low;f4=high;
        Interleave(low,high,f1,f5);f1=low;f5=high;
        Interleave(low,high,f2,f6);f2=low;f6=high;
        Interleave(low,high,f3,f7);f3=low;f7=high;
        Interleave(low,high,f8,length);f8=low;length=high;
        Interleave(low,high,f9,localToken);f9=low;localToken=high;
        Interleave(low,high,remoteLo,localLo);remoteLo=low;localLo=high;
        Interleave(low,high,remoteHi,localHi);remoteHi=low;localHi=high;
        // Interleave field groups of 2.
        Interleave(low,high,control,f2);control=low;f2=high;
        Interleave(low,high,f1,f3);f1=low;f3=high;
        Interleave(low,high,f4,f6);f4=low;f6=high;
        Interleave(low,high,f5,f7);f5=low;f7=high;
        Interleave(low,high,f8,remoteLo);f8=low;remoteLo=high;
        Interleave(low,high,f9,remoteHi);f9=low;remoteHi=high;
        Interleave(low,high,length,localLo);length=low;localLo=high;
        Interleave(low,high,localToken,localHi);localToken=low;localHi=high;
        // Interleave field groups of 1.
        Interleave(low,high,control,f1);control=low;f1=high;
        Interleave(low,high,f2,f3);f2=low;f3=high;
        Interleave(low,high,f4,f5);f4=low;f5=high;
        Interleave(low,high,f6,f7);f6=low;f7=high;
        Interleave(low,high,f8,f9);f8=low;f9=high;
        Interleave(low,high,remoteLo,remoteHi);remoteLo=low;remoteHi=high;
        Interleave(low,high,length,localToken);length=low;localToken=high;
        Interleave(low,high,localLo,localHi);localLo=low;localHi=high;
        StoreAlign(words+j*16+0,control,all);
        StoreAlign(words+j*16+64,f1,all);
        StoreAlign(words+j*16+128,f2,all);
        StoreAlign(words+j*16+192,f3,all);
        StoreAlign(words+j*16+256,f4,all);
        StoreAlign(words+j*16+320,f5,all);
        StoreAlign(words+j*16+384,f6,all);
        StoreAlign(words+j*16+448,f7,all);
        StoreAlign(words+j*16+512,f8,all);
        StoreAlign(words+j*16+576,f9,all);
        StoreAlign(words+j*16+640,remoteLo,all);
        StoreAlign(words+j*16+704,remoteHi,all);
        StoreAlign(words+j*16+768,length,all);
        StoreAlign(words+j*16+832,localToken,all);
        StoreAlign(words+j*16+896,localLo,all);
        StoreAlign(words+j*16+960,localHi,all);
    }
}

__aicore__ inline bool SubmitWqesAt(uint32_t rt_tokenBytes,uint64_t entityAddress,uint64_t queueAddress,uint64_t output,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize,bool final,__gm__ Row* row KVIP_PIPE_PARAM_DEFAULT) {
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(queueAddress);
    // Intermediate service exclusively owns this queue; CQ draining starts
    // after LI completes. Keep its state on the DMA path, like the batch row.
    auto queueLocal=cfg[320];auto queue64=queueLocal.ReinterpretCast<uint64_t>();
    GlobalTensor<uint32_t> queueGm;queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queueAddress));
    uint64_t head;
    if(final) {
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
        head=queue->sq_producer_index;
    } else {
        DataCopy(queueLocal,queueGm,16);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        head=queue64.GetValue(1);
    }
    if(final) {row->finalRequests=count;row->finalHead=head;}
    if(!count)return false;
    // The entity is immutable while leased. Fetch its six cache lines with
    // one DMA instead of dependent scalar GM loads for individual fields.
    static_assert(sizeof(KvafBatchGetDirectUrmaEntity)==192,"entity ABI changed");
    auto meta=cfg[256];auto meta64=meta.ReinterpretCast<uint64_t>();
    GlobalTensor<uint32_t> entityGm;entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(entityAddress));
    DataCopy(meta,entityGm,48);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    uint32_t sqDepth=meta.GetValue(19);uint64_t sqAddress=meta64.GetValue(7),doorbell=meta64.GetValue(8);
    if(final && (head>UINT64_MAX-count || count>=sqDepth))return false;
    cfg.SetValue(1,6U<<8);cfg.SetValue(2,(meta.GetValue(20)&0xffffffU)|0x1000000U);
    cfg.SetValue(3,meta.GetValue(6)&0xfffffU);
    for(uint32_t j=0;j<4;++j)cfg.SetValue(4+j,meta.GetValue(22+j));
    cfg.SetValue(8,meta.GetValue(7));cfg.SetValue(9,0);
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
    const uint64_t remoteDelta=meta64.GetValue(kUrmaRegisteredBaseWord)-meta64.GetValue(kUrmaCanonicalBaseWord);
    cfg.ReinterpretCast<uint64_t>().SetValue(5,remoteDelta);
#endif
    cfg.SetValue(12,rt_tokenBytes);cfg.SetValue(13,meta.GetValue(12));
    cfg.SetValue(14,uint32_t(output));cfg.SetValue(15,uint32_t(output>>32));
    cfg.SetValue(16,uint32_t(head));cfg.SetValue(17,sqDepth);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sqAddress));
    // selected and locations are both dead: their contiguous 64KiB holds
    // 1024 WQEs, halving the dispatch/DMA fences for the usual final batch.
    for(uint32_t begin=0;begin<count;begin+=bankSize) {
        uint32_t n=count-begin<bankSize?count-begin:bankSize;
        KVIP_PIPE_SAMPLE(buildScope, pipeSample, PrefetchBuildVf);
        FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),begin,n,count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        KVIP_TRACE_END(buildScope);
        KVIP_PIPE_SAMPLE(sqScope, pipeSample, PrefetchSqMte3);
        uint32_t start=uint32_t(head+begin)&(sqDepth-1),first=n<sqDepth-start?n:sqDepth-start;
        DataCopy(sq[start*16],words,first*16);
        if(first!=n)DataCopy(sq,words[first*16],(n-first)*16);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if(!final) {
#if !V9_SIMULATION
        queue64.SetValue(0,(uint64_t(1)<<32)|kvaf_device::kQueueMagic);
#endif
        queue64.SetValue(1,head+count);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(queueGm,queueLocal,16);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        KVIP_PIPE_SAMPLE(doorbellScope, pipeSample, PrefetchDoorbell);
#if V9_SIMULATION
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+count;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        st_dev(uint32_t(head+count),reinterpret_cast<__gm__ uint32_t*>(doorbell),0);
#endif
    } else {
#if V9_SIMULATION
        queue->sq_producer_index=head+count;
        *reinterpret_cast<__gm__ uint64_t*>(doorbell)=head+count;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(doorbell),64);
#else
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        kvaf_device::PendingPost post{doorbell,queueAddress,head+count,0U};
        kvaf_device::CommitPostScalarStage(&post);
#endif
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
    }
    return true;
}
#if KVIP_PARALLEL_TERMINAL
__aicore__ inline bool WriteReservedWqes(uint32_t rt_tokenBytes,uint64_t entityAddress,uint64_t queueAddress,uint64_t output,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize,bool final,__gm__ Row* row,uint32_t offset,uint32_t total,__gm__ uint32_t* boundTemplate,uint64_t originalHead KVIP_PIPE_PARAM_DEFAULT) {
    if(!count)return false;
    auto cfg64=cfg.ReinterpretCast<uint64_t>();uint64_t head=originalHead+offset;
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
    auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(entityAddress);
    cfg64.SetValue(5,entity->registered_base-entity->remote_host_ddr_address);
#endif
    uint32_t sqDepth=cfg.GetValue(17);uint64_t sqAddress=cfg64.GetValue(12);
    if(final && (head>UINT64_MAX-count || count>=sqDepth))return false;
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sqAddress));
    // selected and locations are both dead: their contiguous 64KiB holds
    // 1024 WQEs, halving the dispatch/DMA fences for the usual final batch.
    if(count<=bankSize){
        KVIP_PIPE_SAMPLE(buildScope, pipeSample, TerminalWqeVf);
        FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,count,total-offset);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        KVIP_TRACE_END(buildScope);
        KVIP_PIPE_SAMPLE(sqScope, pipeSample, TerminalSqMte3);
        uint32_t start=uint32_t(head)&(sqDepth-1),first=count<sqDepth-start?count:sqDepth-start;
        DataCopy(sq[start*16],words,first*16);
        if(first!=count)DataCopy(sq,words[first*16],(count-first)*16);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }else{
    for(uint32_t begin=0;begin<count;begin+=bankSize) {
        uint32_t n=count-begin<bankSize?count-begin:bankSize;
        KVIP_PIPE_SAMPLE(buildScope, pipeSample, TerminalWqeVf);
        FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),begin,n,total-offset);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        KVIP_TRACE_END(buildScope);
        KVIP_PIPE_SAMPLE(sqScope, pipeSample, TerminalSqMte3);
        uint32_t start=uint32_t(head+begin)&(sqDepth-1),first=n<sqDepth-start?n:sqDepth-start;
        DataCopy(sq[start*16],words,first*16);
        if(first!=n)DataCopy(sq,words[first*16],(n-first)*16);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    return true;
}
#endif
__aicore__ inline bool SubmitWqes(uint32_t rt_tokenBytes,__gm__ Row* row,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize=1024,bool final=false) {
    return SubmitWqesAt(rt_tokenBytes,row->entity,row->queue,row->output,slots,words,remoteLow,remoteHigh,cfg,count,bankSize,final,row);
}
__aicore__ inline void SubmitFinal(uint32_t rt_tokenBytes,__gm__ Row* row,LocalTensor<uint32_t> slots,LocalTensor<uint32_t> words,
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count) {
    if(SubmitWqes(rt_tokenBytes,row,slots,words,remoteLow,remoteHigh,cfg,count,1024,true))row->finalPending=1;
}
}


// Imported from batch_post.h

namespace rotating {
// Four 256-ID input rows must fit before the compact output in scratch.
constexpr uint32_t kWindowOutputOffset = 23808 / sizeof(uint32_t);
constexpr uint32_t kDirectWindowWords = kWindowOutputOffset - 4 * 256;
constexpr uint32_t kWindowSetWords = 4096;
static_assert(kWindowSetWords <= kDirectWindowWords);

// One SIMT task covers every batch assigned to this AIV. Per-batch queues
// remain independent. locations is dead until final resolve and stages IDs.

template<bool BuildSimt>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void BatchBuild(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,bool sparseWindow,__gm__ Context* ctx,uint32_t epoch,uint32_t lane,uint32_t begin,
    __ubuf__ uint64_t* remote,__ubuf__ uint32_t* meta,__ubuf__ uint32_t* inputs,__ubuf__ uint32_t* wqes) {
    const uint32_t g=threadIdx.x/rt_postThreads,t=threadIdx.x%rt_postThreads,b=lane+rt_serviceLanes*g;
    const bool active=b<rt_batch;
    uint32_t count=0,n=0,slotBase=0,depth=0;uint64_t head=0,output=0;
    __gm__ Row* row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    __gm__ uint32_t* rowWords=reinterpret_cast<__gm__ uint32_t*>(row);
    if(active) {
        auto input=inputs+g*80;auto input64=reinterpret_cast<__ubuf__ uint64_t*>(input);
        count=input[64];
        if(count) {
            auto entity=input;
            output=input64[35];
            head=input64[25];depth=entity[19];
            auto addresses=reinterpret_cast<__gm__ uint64_t*>(input64[34]);
            auto jobs=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
            slotBase=input[65];
            n=count>begin?count-begin:0;if(n>rt_postBank)n=rt_postBank;
            for(uint32_t i=t;i<n;i+=rt_postThreads) {
                uint32_t token=asc_ldcg(jobs+begin+i);
                uint32_t slot=slotBase+begin+i;
                uint32_t physical=PhysicalSlot(rt_resident,ctx,b,slot);
                uint64_t address=asc_ldcg(addresses+token);
#ifdef KVIP_KVAF_URMA_REGISTERED_BASE
                address+=input64[kUrmaRegisteredBaseWord]-input64[kUrmaCanonicalBaseWord];
#endif
                if constexpr(BuildSimt)remote[g*rt_postBank+i]=address;
                else {
                    auto fields=reinterpret_cast<__ubuf__ uint32_t*>(remote);
                    fields[g*rt_postBank+i]=uint32_t(address);
                    fields[512+g*rt_postBank+i]=uint32_t(address>>32);
                    fields[1024+g*rt_postBank+i]=physical;
                }
                TrackMapping(ctx,b,slot-rt_resident,token,i+1==n);
                asc_stcg(map+token,slot);
                if (sparseWindow) {
                    auto covered = reinterpret_cast<__gm__ uint32_t*>(input64[37]);
                    asc_atomic_or(covered + token / 32, 1U << (token % 32));
                }

                RecordPrefetchSlot(rt_resident,ctx,b,slot,token,physical);
            }
            if(t<16) {
                uint32_t value=0;
                if(t==1)value=6U<<8;
                if(t==2)value=(entity[20]&0xffffffU)|0x1000000U;
                if(t==3)value=entity[6]&0xfffffU;
                if(t>=4 && t<8)value=entity[22+t-4];
                if(t==8)value=entity[7];
                if(t==12)value=rt_tokenBytes;
                if(t==13)value=entity[12];
                if(t==14)value=uint32_t(output);
                if(t==15)value=uint32_t(output>>32);
                meta[g*32+t]=value;
            }
            if(!t) {meta[g*32+16]=uint32_t(head+begin);meta[g*32+17]=depth;}
        }
    }
    asc_syncthreads();
    if(BuildSimt && active && count) {
        for(uint32_t i=t;i<n*16;i+=rt_postThreads) {
            uint32_t token=i/16,field=i%16,value=meta[g*32+field];
            uint32_t slot=slotBase+begin+token;
            uint64_t address=remote[g*rt_postBank+token],local=output+uint64_t(PhysicalSlot(rt_resident,ctx,b,slot))*rt_tokenBytes;
            if(field==0)value=((begin+token+1==count)?0x30360000U:0x30150000U)|(((head+begin+token)&depth)?0U:0x80000000U);
            if(field==10)value=uint32_t(address);
            if(field==11)value=uint32_t(address>>32);
            if(field==14)value=uint32_t(local);
            if(field==15)value=uint32_t(local>>32);
            wqes[g*rt_postBank*16+i]=value;
        }
    }
    asc_threadfence();asc_syncthreads();
    if(active && count && begin+n==count && n && !t) {
        asc_stcg(rowWords+211,inputs[g*80+66]+1);
    }
    asc_threadfence();
}
}

#ifndef V9_BATCH_POST
#define V9_BATCH_POST 2
#endif
namespace rotating {
// The native predicate has 256 bits for 128 BF16 lanes. No per-token address
// or map DMA is performed while scanning these masks.
// Reserve the last 8KiB of the 32000-byte Q4 scratch for candidate IDs
// when indicesOut holds the complete lane's batch rows.
constexpr uint32_t kCandidateCapacity=2048;
// V7's direct transport pattern: address lookup/slot assignment only for the
// compact miss list. FinalBuild/SubmitWqes are shared with final completion.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PrepareCandidateWqes(uint32_t rt_resident,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* lo,
    __ubuf__ uint32_t* hi,__ubuf__ uint32_t* slots,__gm__ uint32_t* map,
    __gm__ uint64_t* addresses,uint32_t count,uint32_t slotBase,__gm__ Context* ctx,uint32_t batch,__gm__ uint32_t* admitted) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
#if KVIP_PARALLEL_TERMINAL
        uint32_t token=ids[i],slot=slotBase+i;/* Addresses prefetched with compact IDs. */
#else
        uint32_t token=ids[i],slot=slotBase+i;uint64_t remote=asc_ldcg(addresses+token);
#endif
        uint32_t physical=PhysicalSlot(rt_resident,ctx,batch,slot);
#if KVIP_PARALLEL_TERMINAL
        slots[i]=physical;asc_stcg(map+token,physical);
#else
        lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);slots[i]=physical;asc_stcg(map+token,slot);
#endif
        TrackMapping(ctx,batch,slot-rt_resident,token,i+1==count);
#if KVIP_PARALLEL_TERMINAL
        // Final window: no later window reads the covered bitmap.
#else
        asc_atomic_or(admitted+token/32,1U<<(token%32));
#endif
        RecordPrefetchSlot(rt_resident,ctx,batch,slot,token,physical);
    }
    asc_threadfence();
}
// Hash collisions probe the next slot and compare the full token ID. If the
// table is full, at least kCandidateCapacity distinct IDs were already saved.
static_assert(kWindowSetWords >= kCandidateCapacity && (kWindowSetWords & (kWindowSetWords - 1)) == 0);
__simt_callee__ inline bool InsertWindowToken(__ubuf__ uint32_t* keys, uint32_t token) {
    uint32_t slot = (token * 0x9e3779b9U) >> 20;
    for (uint32_t i = 0; i < kWindowSetWords; ++i) {
        uint32_t previous = asc_atomic_cas(keys + slot, UINT32_MAX, token);
        if (previous == UINT32_MAX || previous == token) return previous == UINT32_MAX;
        slot = (slot + 1) & (kWindowSetWords - 1);
    }
    return false;
}
// Expand only newly inserted compact IDs. Each warp reserves output once.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void UnionColdTopk(__ubuf__ uint32_t* input,__ubuf__ uint32_t* bits,
    __ubuf__ uint32_t* ids,__ubuf__ uint32_t* cfg,uint32_t stride,
    uint32_t n0,uint32_t n1,uint32_t n2,uint32_t n3,uint32_t limit,uint32_t words,bool last,
    __gm__ uint32_t* mapping,__gm__ Context* ctx,uint32_t batch,uint32_t resident) {
    uint32_t tid=threadIdx.x,lane=tid%32;
    for(uint32_t i=tid;i<4*stride;i+=256) {
        uint32_t q=i/stride,j=i%stride,n=q==0?n0:q==1?n1:q==2?n2:n3;
        bool fresh=false;uint32_t token=0;
        if(j<n) {
            token=input[i];
            // The predecessor service is joined before this window. Its map
            // is authoritative even when the exported eligibility is stale.
            // Never create a second Index exchange for an admitted source.
            if(token/32<words && DirectTailSlot(ctx,batch,resident,token)==UINT32_MAX && asc_ldcg(mapping+token)==UINT32_MAX){
                if (words > kDirectWindowWords) {
                    fresh = InsertWindowToken(bits, token);
                } else {
                    uint32_t bit=1U<<(token%32);
                    fresh=(asc_atomic_or(bits+token/32,bit)&bit)==0;
                }
            }
        }
        uint32_t mask=asc_ballot(fresh),base=0;
        if(!lane && mask)base=asc_atomic_add(cfg+40,uint32_t(__popc(mask)));
        base=asc_shfl(base,0);
        uint32_t rank=base+__popc(mask&((1U<<lane)-1));
        if(fresh && rank<kCandidateCapacity)ids[rank]=token;
    }
    asc_syncthreads();
    // Earlier chunks still need all bits for dedup. Prune only after the last.
    if(words <= kDirectWindowWords && last && limit && cfg[40]>limit) {
        for(uint32_t i=tid;i<words;i+=256)bits[i]=0;
        asc_syncthreads();
        for(uint32_t i=tid;i<limit;i+=256){uint32_t token=ids[i];asc_atomic_or(bits+token/32,1U<<(token%32));}
        asc_syncthreads();
    }
}

template<bool captureIo, bool predecessorChecked=false>
__aicore__ inline void FastService(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer,uint32_t epoch,uint32_t core,uint32_t lane,
    LocalTensor<uint32_t> scratch,LocalTensor<uint16_t> flags,LocalTensor<uint32_t> ids,
    uint32_t sequenceHalfWindow KVIP_PIPE_PARAM_DEFAULT) {
    // Only IDs in the service lane's scoring range can be nominated. Size the
    // local set by that range; mappings and global covered rows use the full stride.
    const uint32_t tokenLimit = sequenceHalfWindow
        ? SequenceWindowEnd(rt_history, rt_resident, sequenceHalfWindow, rt_rounds - 1)
        : rt_history;
    const uint32_t windowWords = tokenLimit / 32;
    const bool sparseWindow = windowWords > kDirectWindowWords;
    uint64_t scanTicks=0,candidateTicks=0,postTicks=0,bitmapBytes=0,totalCandidates=0;
    uint64_t scanBegin=0,candidateBegin=0,postBegin=0,begin=0;
    uint64_t firstDoorbell=0,lastDoorbell=0;
    auto ctx=reinterpret_cast<__gm__ Context*>(pointer);auto events=reinterpret_cast<__gm__ Event*>(ctx->events);
    auto event=events+epoch*rt_serviceLanes+lane;event->core=core;event->epoch=epoch;V9_FETCH_PROFILE(if constexpr (!predecessorChecked) event->waitBegin=GetSystemCycle();)
    if constexpr (!predecessorChecked) if(epoch) {
        KVIP_TRACE_BEGIN(waitScope, pipeSample.base, Cold, ServicePredecessor, epoch, lane);
        auto previous=reinterpret_cast<__gm__ volatile Event*>(events)+(epoch-1)*rt_serviceLanes+lane;
        do { kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+(epoch-1)*rt_serviceLanes+lane),64); }
        while(!previous->complete);
    }
    auto work=flags.ReinterpretCast<uint32_t>();
    V9_FETCH_PROFILE(event->begin=GetSystemCycle();)
    uint64_t postedCount=0; // Business state: controls batched WQE submission.

    auto cfg=work[4096];
    auto rowStorage=ids;
    if(rt_batchedRows)ids=scratch[kWindowOutputOffset];
    GlobalTensor<uint32_t> allRows;allRows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows));
    if(rt_batchedRows) {
        if(lane<rt_batch) {
            KVIP_TRACE_BEGIN(rowRead, pipeSample.base, Cold, ServiceInputMte2, epoch, lane);
            DataCopyExtParams copy;copy.blockCount=(rt_batch-lane+rt_serviceLanes-1)/rt_serviceLanes;
            copy.blockLen=sizeof(Row);copy.srcStride=(rt_serviceLanes-1)*sizeof(Row);copy.dstStride=0;
            DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
            DataCopyPad(rowStorage,allRows[lane*256],copy,pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
        // Transfer state as a block. Avoid scalar GM read/modify/write and
        // repeated cache maintenance for every counter/cache line.
        auto state=rt_batchedRows?rowStorage[(b/rt_serviceLanes)*256]:work[2048];auto state64=state.ReinterpretCast<uint64_t>();
        GlobalTensor<uint32_t> rowGm;
        rowGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows)+b*256);
        if(!rt_batchedRows) {
            KVIP_TRACE_BEGIN(rowRead, pipeSample.base, Cold, ServiceInputMte2, epoch, b);
            DataCopy(state,rowGm,256);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
auto postInput=cfg[256+(b/rt_serviceLanes)*80];
        if (rt_batchedRows) postInput.SetValue(64,0);
        uint32_t slots=state.GetValue(8),requests=state.GetValue(9),pending=state.GetValue(211);
        uint32_t sequence=state.GetValue(246)+1;
        state.SetValue(246,sequence);state.SetValue(214+epoch,sequence);

        const uint32_t lower=ctx->lruSlots?0:rt_resident;
        uint32_t candidateCount=0;
        uint32_t available=requests<ctx->prefetchLimit?ctx->prefetchLimit-requests:0;
        if(available>ctx->windowLimit)available=ctx->windowLimit;
        V9_FETCH_PROFILE(scanBegin=GetSystemCycle();)
        KVIP_TRACE_BEGIN(selectScope, pipeSample.base, Cold, ServiceSelect, epoch, b);
        // Counts occupy spare incremental metadata, exported alongside TopK.
        GlobalTensor<uint32_t> metadata,compact;
        metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+b*128);
        compact.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
        KVIP_TRACE_BEGIN(headerRead, pipeSample.base, Cold, ServiceInputMte2, epoch, b);
        DataCopy(cfg[128],metadata,128);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        KVIP_TRACE_END(headerRead);
        uint32_t counts[4],maxCount=0;
        for(uint32_t q=0;q<4;++q){counts[q]=cfg.GetValue(128+q*32+6);if(counts[q]>maxCount)maxCount=counts[q];}
        cfg.SetValue(40,0);
        const uint32_t setWords = sparseWindow ? kWindowSetWords : windowWords;
        Duplicate(scratch, sparseWindow ? UINT32_MAX : 0U, setWords);
        auto input=scratch[setWords];
        // Four 256-ID rows fit before the unchanged service output at 23808.
        // Dense tails use the same bounded loop; all reads finish before jobs
        // overwrite ctx->locations in the original downstream submission path.
        for(uint32_t start=0;start<maxCount;start+=256) {
            uint32_t width=maxCount-start;if(width>256)width=256;width=(width+7)/8*8;
            KVIP_TRACE_BEGIN(idsRead, pipeSample.base, Cold, ServiceInputMte2, epoch, b);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
            DataCopyExtParams copy;copy.blockCount=4;copy.blockLen=width*4;copy.srcStride=(2048-width)*4;copy.dstStride=0;
            DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
            DataCopyPad(input,compact[start],copy,pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
            KVIP_TRACE_END(idsRead);
            KVIP_TRACE_BEGIN(unionScope, pipeSample.base, Cold, ServiceUnionVf, epoch, b);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            uint32_t n[4];for(uint32_t q=0;q<4;++q)n[q]=counts[q]>start?((counts[q]-start)<width?counts[q]-start:width):0;
            Simt::VF_CALL<UnionColdTopk>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(input.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),width,n[0],n[1],n[2],n[3],available<kCandidateCapacity?available:kCandidateCapacity,windowWords,start+256>=maxCount,
                reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+uint64_t(b)*rt_history,ctx,b,rt_resident);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        KVIP_TRACE_END(selectScope);
        candidateCount=cfg.GetValue(40);if(candidateCount>kCandidateCapacity)candidateCount=kCandidateCapacity;


        V9_FETCH_PROFILE(scanTicks+=GetSystemCycle()-scanBegin;totalCandidates+=candidateCount;)
        V9_FETCH_PROFILE(candidateBegin=GetSystemCycle();postBegin=0;)
        uint32_t count=0,missing=0;
        if(candidateCount) {
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
            // ExportWindowCold already removed local hits and prior admissions.
            // AdvanceWindow joined the predecessor; this request has no other writer.
            missing=candidateCount;count=missing<available?missing:available;
            V9_FETCH_PROFILE(candidateTicks+=GetSystemCycle()-candidateBegin;)
            if(count) {
                const auto& args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)[b*2];
                const uint64_t entityAddress=args.channel_entity,queueAddress=args.queue_state;
                uint32_t owner=ctx->rowOwners?reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b]:b;
                uint64_t stride=ctx->outputStride?ctx->outputStride:rt_resident;
                uint64_t addresses=ctx->rowAddressBase?RequestAddressTable(ctx,b,owner,rt_history):state64.GetValue(2);
                uint64_t output=args.output+uint64_t(owner)*stride*rt_tokenBytes;
                V9_FETCH_PROFILE(postBegin=GetSystemCycle();)
if (rt_batchedRows) {
                KVIP_TRACE_BEGIN(prepareScope, pipeSample.base, Cold, PrefetchPrepare, epoch, b);
                auto post64=postInput.ReinterpretCast<uint64_t>();
                postInput.SetValue(64,count);postInput.SetValue(65,rt_resident+slots);
                postInput.SetValue(66,pending);post64.SetValue(34,addresses);
                post64.SetValue(35,output);post64.SetValue(36,queueAddress);
                post64.SetValue(37,kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b));
                GlobalTensor<uint32_t> entityGm,queueGm;
                entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(entityAddress));
                queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queueAddress));
                DataCopy(postInput,entityGm,48);DataCopy(postInput[48],queueGm,16);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                // Long windows publish from the capped job list in BatchBuild.
                // Dropped nominations must remain eligible in the next window.
                if (!sparseWindow) {
                    GlobalTensor<int32_t> admitted;
                    admitted.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(post64.GetValue(37)));
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID5);WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
                    SetAtomicAdd<int32_t>();DataCopy(admitted,scratch.ReinterpretCast<int32_t>(),windowWords);SetAtomicNone();
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID5);WaitFlag<HardEvent::MTE3_S>(EVENT_ID5);
                }
                GlobalTensor<uint32_t> jobs;
                jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(jobs,ids,((count+7)/8)*8);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                slots+=count;requests+=count;
} else {
                {
                    KVIP_TRACE_BEGIN(prepareScope, pipeSample.base, Cold, PrefetchPrepare, epoch, b);
                    Simt::VF_CALL<PrepareCandidateWqes>(Simt::Dim3{256,1,1},rt_resident,
                        reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(work[512].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[1024].GetPhyAddr()),
                        map,reinterpret_cast<__gm__ uint64_t*>(addresses),count,rt_resident+slots,ctx,b,reinterpret_cast<__gm__ uint32_t*>(kvip_prefetch_rank::CoveredAddress(ctx->incremental,rt_batch,rt_history,b)));
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    KVIP_TRACE_END(prepareScope);
                    KVIP_TRACE_BEGIN(postScope, pipeSample.base, Cold, PrefetchSubmit, epoch, b);
                    if(captureIo && !firstDoorbell)firstDoorbell=GetSystemCycle();
                    SubmitWqesAt(rt_tokenBytes,entityAddress,queueAddress,output,work[1024],scratch,work,work[512],cfg,count,256,false,nullptr KVIP_PIPE_ARG);
                    ++pending;
                    if(captureIo)lastDoorbell=GetSystemCycle();
                    slots+=count;requests+=count;
                }
}
            }
        }
        V9_FETCH_PROFILE(if(postBegin)postTicks+=GetSystemCycle()-postBegin;)
        postedCount+=count;
        state.SetValue(16+epoch,count);state.SetValue(48+epoch,missing-count);
        state.SetValue(10,state.GetValue(10)+missing-count);
        state.SetValue(8,slots);state.SetValue(9,requests);state.SetValue(211,pending);
        if(!rt_batchedRows) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(rowGm,state,256);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    if(rt_batchedRows) {
        if(lane<rt_batch) {
            DataCopyExtParams copy;copy.blockCount=(rt_batch-lane+rt_serviceLanes-1)/rt_serviceLanes;
            copy.blockLen=sizeof(Row);copy.srcStride=0;copy.dstStride=(rt_serviceLanes-1)*sizeof(Row);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(allRows[lane*256],rowStorage,copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
if (rt_batchedRows) {
    if(postedCount) {
        KVIP_TRACE_BEGIN(postScope, pipeSample.base, Cold, PrefetchSubmit, epoch, lane);
        V9_FETCH_PROFILE(begin=GetSystemCycle();)
        uint32_t maxCount=0;
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            uint32_t n=cfg.GetValue(256+(b/rt_serviceLanes)*80+64);if(n>maxCount)maxCount=n;
        }
        for(uint32_t start=0;start<maxCount;start+=rt_postBank) {
            KVIP_TRACE_BEGIN(batchBuildScope, pipeSample.base, Cold, PrefetchBuildVf, epoch, lane);
            Simt::VF_CALL<BatchBuild<V9_BATCH_POST==1>>(Simt::Dim3{512,1,1},rt_batch,rt_history,rt_resident,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,sparseWindow,ctx,epoch,lane,start,
                reinterpret_cast<__ubuf__ uint64_t*>(work[512].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg[256].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            KVIP_TRACE_END(batchBuildScope);
            for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
                uint32_t g=b/rt_serviceLanes;auto input=cfg[256+g*80];auto in64=input.ReinterpretCast<uint64_t>();
                uint32_t count=input.GetValue(64);if(count<=start)continue;
                uint32_t n=count-start;if(n>rt_postBank)n=rt_postBank;
                uint32_t depth=input.GetValue(19),offset=uint32_t(in64.GetValue(25)+start)&(depth-1);
                uint32_t first=n<depth-offset?n:depth-offset;
#if V9_BATCH_POST == 2
                KVIP_TRACE_BEGIN(wqeBuildScope, pipeSample.base, Cold, PrefetchBuildVf, epoch, b);
                FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(work[512+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1024+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1536+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(scratch[g*rt_postBank*16].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[g*32].GetPhyAddr()),0,n,count-start);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                KVIP_TRACE_END(wqeBuildScope);
#endif
                KVIP_TRACE_BEGIN(sqScope, pipeSample.base, Cold, PrefetchSqMte3, epoch, b);
                GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(7)));
                DataCopy(sq[offset*16],scratch[g*rt_postBank*16],first*16);
                if(first<n)DataCopy(sq,scratch[(g*rt_postBank+first)*16],(n-first)*16);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            auto input=cfg[256+(b/rt_serviceLanes)*80];auto in64=input.ReinterpretCast<uint64_t>();
            uint32_t count=input.GetValue(64);if(!count)continue;
#if !V9_SIMULATION
            in64.SetValue(24,(uint64_t(1)<<32)|kvaf_device::kQueueMagic);
#endif
            in64.SetValue(25,in64.GetValue(25)+count);
            GlobalTensor<uint32_t> queue;queue.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(36)));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(queue,input[48],16);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            auto input=cfg[256+(b/rt_serviceLanes)*80];if(!input.GetValue(64))continue;
            auto in64=input.ReinterpretCast<uint64_t>();uint64_t db=in64.GetValue(8),head=in64.GetValue(25);
            KVIP_TRACE_BEGIN(doorbellScope, pipeSample.base, Cold, PrefetchDoorbell, epoch, b);
            if(captureIo && !firstDoorbell)firstDoorbell=GetSystemCycle();
#if V9_SIMULATION
            *reinterpret_cast<__gm__ uint64_t*>(db)=head;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(db),64);
#else
            st_dev(uint32_t(head),reinterpret_cast<__gm__ uint32_t*>(db),0);
#endif
            if(captureIo)lastDoorbell=GetSystemCycle();
        }
        V9_FETCH_PROFILE(postTicks+=GetSystemCycle()-begin;)
    }
}
    V9_FETCH_PROFILE(event->scanTicks=scanTicks;event->candidateTicks=candidateTicks;event->postTicks=postTicks;)
    V9_FETCH_PROFILE(event->bitmapBytes=bitmapBytes;event->candidateCount=totalCandidates;event->postedCount=postedCount;)
    if(captureIo){auto stamp=reinterpret_cast<__gm__ uint64_t*>(event->pad);stamp[0]=firstDoorbell;stamp[1]=lastDoorbell;}
    V9_FETCH_PROFILE(event->end=GetSystemCycle();)event->complete=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),captureIo?sizeof(Event):64);
}
}

namespace rotating {
__aicore__ inline void Service(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_splitQ,uint32_t rt_rounds,uint32_t rt_serviceAics,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer, uint32_t epoch, uint32_t core,
    uint32_t lane, LocalTensor<uint32_t> scratch, LocalTensor<uint16_t> flags,
    LocalTensor<uint32_t> indices, uint32_t sequenceHalfWindow = 0,bool captureIo = false
#if KVIP_STAGE_TRACE
    , uint64_t traceBase = 0
#endif
    ) {
#if KVIP_STAGE_TRACE
    const kvip_trace::ScanSample pipeSample{traceBase, kvip_trace::Cold, epoch,
        ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane)};
#endif
    // RotateFetch calls AdvanceWindow first, which joins the same lane's
    // predecessor before touching incremental state. A second CQ-independent
    // poll here adds no ordering and delays the resumed score producer.
    if(captureIo) FastService<true,true>(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,pointer, epoch, core, ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane), scratch, flags, indices, sequenceHalfWindow KVIP_PIPE_ARG);
    else FastService<false,true>(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,pointer, epoch, core, ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane), scratch, flags, indices, sequenceHalfWindow KVIP_PIPE_ARG);
}
}

#endif
