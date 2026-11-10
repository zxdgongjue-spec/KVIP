#pragma once

// Imported from rotation_device.h

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "context.h"
#include "../../common/lifetch/prefetch_budget.h"

// Imported from direct_slots.h

#include "simt_api/device_functions.h"
#include "../../common/lifetch/reserved_hot.h"
// Prefetch ordinals map to LRU hot destinations. Final repair uses hot slots;
// virtual ABC tags remain disjoint from prefetch ordinals.
namespace rotating {
constexpr uint32_t kLruSlots = 2048;
constexpr uint32_t kChangeCapacity = kLruSlots + kSlots;
__simt_callee__ inline uint32_t OwnerRow(__gm__ Context* ctx,uint32_t b) {
    return ctx->rowOwners ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)+b) : b;
}
__simt_callee__ inline uint32_t OutputStride(uint32_t rt_resident,__gm__ Context* ctx) {
    return ctx->outputStride ? uint32_t(ctx->outputStride) : rt_resident;
}
__simt_callee__ inline void InitializeTailMappingsBody(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,
    uint64_t reservedRow=0) {
#if V9_COMPACT_RESERVED
    if(reservedRow)ScatterReservedHot(reinterpret_cast<__gm__ uint32_t*>(reservedRow),
        reinterpret_cast<const __gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*kLruSlots,rt_resident,kLruSlots);
#endif
    // Use the existing SIMT launch: scalar cache-line writeback would race
    // adjacent request counts initialized by other cores.
    if(!threadIdx.x && ctx->changeCount) {
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changeCount)+b,kLruSlots);
        asc_threadfence();
    }
    if(!ctx->tailBounds)return;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
    uint32_t begin=asc_ldcg(bounds),end=asc_ldcg(bounds+1);
    const uint32_t tailTokens=OutputStride(rt_resident,ctx)-rt_resident;
    for(uint32_t i=threadIdx.x;i<tailTokens;i+=256) {
        uint32_t token=begin+i;
        if(token<end && token<rt_history)
            asc_stcg(map+token,rt_resident+kSlots+token%tailTokens);
    }
    asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailMappings(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,
    uint64_t reservedRow=0) {
    InitializeTailMappingsBody(rt_history,rt_resident,ctx,b,reservedRow);
}
__simt_callee__ inline bool IsTailToken(__gm__ Context* ctx,uint32_t b,uint32_t token) {
    if(!ctx->tailBounds)return false;
    auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+b*2;
    return token>=asc_ldcg(bounds) && token<asc_ldcg(bounds+1);
}
__simt_callee__ inline uint32_t QueryEnd(uint32_t rt_history,__gm__ Context* ctx,uint32_t query) {
    return ctx->queryEnds ? asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+query) : rt_history;
}
__simt_callee__ inline void StoreFallbackSource(uint32_t rt_history,uint32_t rt_tokenBytes,__gm__ Context* ctx,uint32_t batch,uint32_t dst,
    uint32_t token,uint64_t remote,__gm__ int32_t* blocks,__gm__ int32_t* offsets) {
    if(ctx->fallbackAddresses) {
        asc_stcg(reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses)+dst,remote);
        asc_stcg(blocks+dst,int32_t(dst));asc_stcg(offsets+dst,int32_t(0));
    } else {
        asc_stcg(blocks+dst,int32_t(batch*(rt_history/128)+token/128));
        asc_stcg(offsets+dst,int32_t((token%128)*rt_tokenBytes));
    }
}
__simt_callee__ inline bool HotValid(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t token) {
    return !ctx->hotValid || asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*rt_resident+token)!=0;
}
__simt_callee__ inline uint32_t PhysicalSlot(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot) {
    if(ctx->lruSlots && uint32_t(slot-rt_resident)<kLruSlots)
        return asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots+slot-rt_resident);
    return slot;
}
__simt_callee__ inline void RecordChange(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t slot,uint32_t token) {
    if(ctx->changes && uint32_t(slot-rt_resident)<kLruSlots)
        asc_stcg(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+b*kChangeCapacity+slot-rt_resident,token);

}
// Resolver workers consume the pre-round validity. Publish only after all
// resolver workers finish; consumers of KV still wait for Fetch completion.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PublishDestinationResidency(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,uint32_t fetched) {
    if(!ctx->hotValid)return;
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*rt_resident;
    // The original resolver already emits exactly the final Fetch destinations.
    // Hits need no publication; do not rescan all four TopK rows.
    auto destinations=reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+b*kSlots;
    uint32_t ownerBase=OwnerRow(ctx,b)*OutputStride(rt_resident,ctx);
    for(uint32_t i=threadIdx.x;i<fetched;i+=256) {
        uint32_t slot=uint32_t(asc_ldcg(destinations+i))-ownerBase;
        if(slot<rt_resident)asc_stcg(valid+slot,uint32_t(1));
    }
    if(ctx->changes && ctx->lruSlots) {
        auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
        auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
        for(uint32_t i=threadIdx.x;i<kLruSlots;i+=256)
            if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),uint32_t(1));
    }
    asc_threadfence();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicate(uint32_t rt_resident,__gm__ Context* ctx,uint32_t b,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate) {
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+b*rt_resident;
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident &&
            (IsTailToken(ctx,b,token) || (!HotValid(rt_resident,ctx,b,token)) || (ctx->reservedHot && asc_ldcg(reserved+token)>=0)))?rt_resident:token;
    }
}
// Bulk metadata is staged once in UB by the resolver, not fetched per TopK item.
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void AdaptColdPredicateLocal(uint32_t rt_resident,__ubuf__ uint32_t* selected,__ubuf__ uint32_t* predicate,
    __ubuf__ uint32_t* valid,__ubuf__ int32_t* reserved,bool hasValid,bool hasReserved,
    uint32_t tailBegin,uint32_t tailEnd) {
    for(uint32_t i=threadIdx.x;i<kSlots;i+=512) {
        uint32_t token=selected[i];
        predicate[i]=(token<rt_resident && ((token>=tailBegin && token<tailEnd) ||
            (hasValid && !valid[token]) || (hasReserved && reserved[token]>=0)))?rt_resident:token;
    }
}
__simt_callee__ inline uint32_t ResolvePhysicalSlot(uint32_t rt_resident,__gm__ uint32_t* lru,uint32_t slot) {
    if(lru && uint32_t(slot-rt_resident)<kLruSlots)return asc_ldcg(lru+slot-rt_resident);
    // Missing-rank markers have their high bit set and are not ABC slots.
    return slot>=rt_resident+kSlots && slot<0x80000000U ? slot-kSlots : slot;
}

}

namespace rotating {
using namespace AscendC;
// v9 GatherCandidateMap's indexed-load loop, adapted to packed uint16 scores.
// One thread owns both halves of each UB word; no half-word write sharing.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherVictimScores(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* output,
    const __gm__ uint32_t* scores) {
    auto input=const_cast<__gm__ uint32_t*>(scores);
    for(uint32_t i=threadIdx.x;i<1024;i+=256) {
        uint32_t a=ids[2*i],b=ids[2*i+1];
        uint32_t low=(asc_ldcg(input+a/2)>>((a%2)*16))&0xffffU;
        uint32_t high=(asc_ldcg(input+b/2)>>((b%2)*16))&0xffffU;
        output[i]=low|(high<<16);
    }
}

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
__simd_callee__ inline void Visible(MicroAPI::RegTensor<uint16_t>& value,
    MicroAPI::RegTensor<uint16_t>& index, __ubuf__ uint16_t* input,
    __ubuf__ uint32_t* control, uint32_t q, MicroAPI::MaskReg& all) {
    using namespace MicroAPI;
    RegTensor<uint16_t> low, high, zero;
    MaskReg lower, upper, valid;
    LoadAlign(value, input);
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(low, reinterpret_cast<__ubuf__ uint16_t*>(control + 4 + q));
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(high, reinterpret_cast<__ubuf__ uint16_t*>(control + q));
    Compare<uint16_t, CMPMODE::GE>(lower, index, low, all);
    Compare<uint16_t, CMPMODE::LT>(upper, index, high, all);
    MaskAnd(valid, lower, upper, all);
    Duplicate(zero, uint16_t(0));
    Select(value, value, zero, valid);
}
}

// Imported from window_fetch.h


// Imported from fetch_profiling.h

#ifndef V9_INTERMEDIATE_PROFILING
#define V9_INTERMEDIATE_PROFILING 1
#endif
#if V9_INTERMEDIATE_PROFILING
#define V9_FETCH_PROFILE(...) __VA_ARGS__
#else
#define V9_FETCH_PROFILE(...)
#endif

#include "kvaf_batch_get_device.h"

// Imported from final_submit.h

// WQE fields/owner/completion encoding are inherited from the Codex SIMD demo.
// Only the address gather and destination slots differ: long histories do not
// fit in UB, and final repair may safely recycle unreferenced speculative slots.
namespace rotating {
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
    LocalTensor<uint32_t> remoteLow,LocalTensor<uint32_t> remoteHigh,LocalTensor<uint32_t> cfg,uint32_t count,uint32_t bankSize,bool final,__gm__ Row* row,uint32_t pending=0) {
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
    if(count>=sqDepth || (!final && pending+1>=meta.GetValue(31)))return false;
    cfg.SetValue(1,6U<<8);cfg.SetValue(2,(meta.GetValue(20)&0xffffffU)|0x1000000U);
    cfg.SetValue(3,meta.GetValue(6)&0xfffffU);
    for(uint32_t j=0;j<4;++j)cfg.SetValue(4+j,meta.GetValue(22+j));
    cfg.SetValue(8,meta.GetValue(7));cfg.SetValue(9,0);
    cfg.SetValue(12,rt_tokenBytes);cfg.SetValue(13,meta.GetValue(12));
    cfg.SetValue(14,uint32_t(output));cfg.SetValue(15,uint32_t(output>>32));
    cfg.SetValue(16,uint32_t(head));cfg.SetValue(17,sqDepth);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sqAddress));
    // selected and locations are both dead: their contiguous 64KiB holds
    // 1024 WQEs, halving the dispatch/DMA fences for the usual final batch.
    for(uint32_t begin=0;begin<count;begin+=bankSize) {
        uint32_t n=count-begin<bankSize?count-begin:bankSize;
        FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(remoteLow.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(remoteHigh.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(slots.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(words.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),begin,n,count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t start=uint32_t(head+begin)&(sqDepth-1),first=n<sqDepth-start?n:sqDepth-start;
        DataCopy(sq[start*16],words,first*16);
        if(first!=n)DataCopy(sq,words[first*16],(n-first)*16);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    if(!final) {
        queue64.SetValue(1,head+count);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(queueGm,queueLocal,16);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
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
// One SIMT task covers every batch assigned to this AIV. Per-batch queues
// remain independent. locations is dead until final resolve and stages IDs.

template<bool BuildSimt>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void BatchBuild(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,__gm__ Context* ctx,uint32_t epoch,uint32_t lane,uint32_t begin,
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
            uint32_t pending=input[66],cqDepth=entity[31];
            if(count>=depth || pending+1>=cqDepth) {
                if(!t){asc_stcg(rowWords+11,uint32_t(kvaf_device::kInvalidArgs));input[64]=0;}count=0;
            } else {
                auto addresses=reinterpret_cast<__gm__ uint64_t*>(input64[34]);
                auto jobs=reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192;
                auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
                slotBase=input[65];
                n=count>begin?count-begin:0;if(n>rt_postBank)n=rt_postBank;
                for(uint32_t i=t;i<n;i+=rt_postThreads) {
                    uint32_t token=asc_ldcg(jobs+begin+i);
                    uint32_t slot=slotBase+begin+i;
                    uint64_t address=asc_ldcg(addresses+token);
                    if constexpr(BuildSimt)remote[g*rt_postBank+i]=address;
                    else {
                        auto fields=reinterpret_cast<__ubuf__ uint32_t*>(remote);
                        fields[g*rt_postBank+i]=uint32_t(address);
                        fields[512+g*rt_postBank+i]=uint32_t(address>>32);
                        fields[1024+g*rt_postBank+i]=PhysicalSlot(rt_resident,ctx,b,slot);
                    }
                    asc_stcg(map+token,slot);
                    RecordChange(rt_resident,ctx,b,slot,token);
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
#ifndef V9_PARTIAL_SCAN_SIMD
#define V9_PARTIAL_SCAN_SIMD 1
#endif
// Clip each query before union so publication skew never admits future bits.
__simd_callee__ inline void ClipWords(MicroAPI::RegTensor<uint32_t>& value,
    MicroAPI::RegTensor<uint32_t>& index,__ubuf__ uint32_t* cfg,uint32_t q,
    MicroAPI::MaskReg& all) {
    using namespace MicroAPI;
    RegTensor<uint32_t> low,high,lowMask,highMask,mask,ones,zero;
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(low,cfg+80+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(high,cfg+81+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(lowMask,cfg+82+q*4);
    LoadAlign<uint32_t,LoadDist::DIST_BRC_B32>(highMask,cfg+83+q*4);
    Duplicate(ones,uint32_t(0x55555555U));Duplicate(zero,uint32_t(0));
    MaskReg lower,upper,valid,edge;
    Compare<uint32_t,CMPMODE::GE>(lower,index,low,all);
    Compare<uint32_t,CMPMODE::LE>(upper,index,high,all);
    MaskAnd(valid,lower,upper,all);Select(value,value,zero,valid);
    Compare<uint32_t,CMPMODE::EQ>(edge,index,low,all);Select(mask,lowMask,ones,edge);And(value,value,mask,all);
    Compare<uint32_t,CMPMODE::EQ>(edge,index,high,all);Select(mask,highMask,ones,edge);And(value,value,mask,all);
}
template<bool Partial=false>
__simd_vf__ inline void UnionWords(__ubuf__ uint32_t* masks,__ubuf__ uint32_t* indices,__ubuf__ uint32_t* values,
    __ubuf__ uint32_t* cfg,uint32_t words,uint32_t wordBase) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto one=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> a,b,c,d,packed,counts,accum,total,bitMask;
    RegTensor<int32_t> indexSigned;Arange(indexSigned,int32_t(wordBase));
    auto& index=reinterpret_cast<RegTensor<uint32_t>&>(indexSigned);
    Duplicate(accum,uint32_t(0));Duplicate(bitMask,uint32_t(0x55555555U));uint32_t remaining=words;
    UnalignRegForStore indexStore;ClearSpr<SpecialPurposeReg::AR>();
    for(uint16_t i=0;i<words;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);
        LoadAlign(a,masks+i);LoadAlign(b,masks+words+i);LoadAlign(c,masks+2*words+i);LoadAlign(d,masks+3*words+i);
        if constexpr(Partial){
            ClipWords(a,index,cfg,0,all);ClipWords(b,index,cfg,1,all);
            ClipWords(c,index,cfg,2,all);ClipWords(d,index,cfg,3,all);
        }
        Or(a,a,b,all);Or(c,c,d,all);Or(a,a,c,all);
        And(a,a,bitMask,all);
        MaskReg nonzero;CompareScalar<uint32_t,CMPMODE::NE>(nonzero,a,uint32_t(0),valid);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(counts,uint32_t(1),nonzero);Add(accum,accum,counts,all);
        StoreAlign(masks+i,a,valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,index,nonzero);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(indices,packed,indexStore);
        Adds(index,index,uint32_t(64),all);
    }
    StoreUnAlignPost(indices,indexStore);
    ReduceSum(total,accum,all);StoreAlign(cfg+72,total,one);
    // The second pass reloads q0 after the OR stores above. A scalar V_S
    // after this VF cannot order its internal vector store/load pipelines.
    LocalMemBar<MemType::VEC_STORE,MemType::VEC_LOAD>();
    ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore valueStore;remaining=words;
    for(uint16_t i=0;i<words;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);LoadAlign(a,masks+i);
        MaskReg nonzero;CompareScalar<uint32_t,CMPMODE::NE>(nonzero,a,uint32_t(0),valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,a,nonzero);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(values,packed,valueStore);
    }
    StoreUnAlignPost(values,valueStore);
}
// Each SIMT thread owns consecutive bitmap words. A two-level prefix sum
// assigns stable, token-ordered output positions without per-candidate atomics.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ExpandWords(__ubuf__ uint32_t* input,__ubuf__ uint32_t* wordIndices,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* cfg,uint32_t base,uint32_t span,uint32_t oldCount,uint32_t words) {
    const uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    const uint32_t perThread=(words+255)/256;
    uint32_t count=0;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t word=tid*perThread+i;
        if(word<words)count+=uint32_t(__popc(input[word]));
    }
    uint32_t inclusive=count;
    for(uint32_t delta=1;delta<warpSize;delta*=2) {
        uint32_t prior=asc_shfl_up(inclusive,delta);
        if(lane>=delta)inclusive+=prior;
    }
    if(lane==warpSize-1)cfg[64+warp]=inclusive;
    asc_syncthreads();
    if(!tid) {
        uint32_t total=oldCount;
        for(uint32_t w=0;w<256/warpSize;++w) {
            uint32_t n=cfg[64+w];cfg[64+w]=total;total+=n;
        }
        cfg[40]=total<kCandidateCapacity?total:kCandidateCapacity;cfg[48]=span;
    }
    asc_syncthreads();
    uint32_t rank=cfg[64+warp]+inclusive-count,stopValue=span;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t word=tid*perThread+i;
        uint32_t bits=word<words?input[word]:0;
        while(bits) {
            uint32_t offset=wordIndices[word]*16+uint32_t(__ffs(int32_t(bits))-1)/2;
            if(rank<kCandidateCapacity)ids[rank]=base+offset;
            uint32_t select=0U-uint32_t(rank==kCandidateCapacity);
            stopValue=(stopValue&~select)|(offset&select);
            ++rank;bits&=bits-1;
        }
    }
    if(stopValue<span)asc_atomic_min(cfg+48,stopValue);
}
// Explicit two-bit predicate decoding for unequal query publication bounds.
__simt_callee__ inline uint32_t ReadyWord(__ubuf__ uint32_t* input,__ubuf__ uint32_t* cfg,uint32_t word,uint32_t words) {
    uint32_t bits=0,start=word*16;
    for(uint32_t q=0;q<4;++q) {
        uint32_t low=cfg[28+q],high=cfg[24+q];
        if(start>=high || start+16<=low)continue;
        uint32_t lo=low>start?low-start:0,hi=high-start<16?high-start:16;
        uint32_t mask=0x55555555U<<(lo*2);
        if(hi<16)mask&=(1U<<(hi*2))-1;
        bits|=input[q*words+word]&mask;
    }
    return bits;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ScanPartial(__ubuf__ uint32_t* input,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* cfg,
    uint32_t base,uint32_t span,uint32_t oldCount) {
    uint32_t tid=threadIdx.x,lane=tid%warpSize,warp=tid/warpSize;
    uint32_t words=span/16,perThread=(words+255)/256,count=0;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t w=tid*perThread+i;
        if(w<words)count+=uint32_t(__popc(ReadyWord(input,cfg,w,words)));
    }
    uint32_t inclusive=count;
    for(uint32_t d=1;d<warpSize;d*=2) {
        uint32_t previous=asc_shfl_up(inclusive,d);if(lane>=d)inclusive+=previous;
    }
    if(lane==warpSize-1)cfg[64+warp]=inclusive;
    asc_syncthreads();
    if(!tid) {
        uint32_t total=oldCount;
        for(uint32_t w=0;w<256/warpSize;++w) {uint32_t n=cfg[64+w];cfg[64+w]=total;total+=n;}
        cfg[40]=total<kCandidateCapacity?total:kCandidateCapacity;cfg[48]=span;
    }
    asc_syncthreads();
    uint32_t rank=cfg[64+warp]+inclusive-count,stop=span;
    for(uint32_t i=0;i<perThread;++i) {
        uint32_t w=tid*perThread+i,bits=w<words?ReadyWord(input,cfg,w,words):0;
        while(bits) {
            uint32_t offset=w*16+uint32_t(__ffs(int32_t(bits))-1)/2;
            if(rank<kCandidateCapacity)ids[rank]=base+offset;
            uint32_t select=0U-uint32_t(rank==kCandidateCapacity);
            stop=(stop&~select)|(offset&select);++rank;bits&=bits-1;
        }
    }
    if(stop<span)asc_atomic_min(cfg+48,stop);
}
// LIDU pattern: gather only candidate map entries, then SIMD squeeze misses.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherCandidateMap(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __gm__ uint32_t* map,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=256)slots[i]=asc_ldcg(map+ids[i]);
}
__simd_vf__ inline void CompactMisses(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,uint32_t count,uint32_t available) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> token,old,rank,index,written,flags,total,zeroIndex,broadcast,packed;
    Duplicate(written,uint32_t(0));Duplicate(zeroIndex,uint32_t(0));
    uint32_t remaining=count;auto output=ids;ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
    for(uint16_t i=0;i<count;i+=64) {
        auto valid=UpdateMask<uint32_t>(remaining);LoadAlign(token,ids+i);LoadAlign(old,slots+i);
        MaskReg miss,capacity;CompareScalar<uint32_t,CMPMODE::EQ>(miss,old,UINT32_MAX,valid);
        Unsqueeze(rank,miss);Add(index,rank,written,all);
        Duplicate<uint32_t,MaskMergeMode::ZEROING>(flags,uint32_t(1),miss);
        ReduceSum(total,flags,all);Gather(broadcast,total,zeroIndex);Add(written,written,broadcast,all);
        CompareScalar<uint32_t,CMPMODE::LT>(capacity,index,available,valid);MaskAnd(miss,miss,capacity,valid);
        Squeeze<uint32_t,MicroAPI::GatherMaskMode::STORE_REG>(packed,token,miss);
        StoreUnAlign<uint32_t,PostLiteral::POST_MODE_UPDATE>(output,packed,store);
    }
    StoreUnAlignPost(output,store);StoreAlign(cfg+56,written,lane);
}
// V7's direct transport pattern: address lookup/slot assignment only for the
// compact miss list. FinalBuild/SubmitWqes are shared with final completion.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PrepareCandidateWqes(uint32_t rt_resident,__ubuf__ uint32_t* ids,__ubuf__ uint32_t* lo,
    __ubuf__ uint32_t* hi,__ubuf__ uint32_t* slots,__gm__ uint32_t* map,
    __gm__ uint64_t* addresses,uint32_t count,uint32_t slotBase,__gm__ Context* ctx,uint32_t batch) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i],slot=slotBase+i;uint64_t remote=asc_ldcg(addresses+token);
        lo[i]=uint32_t(remote);hi[i]=uint32_t(remote>>32);slots[i]=PhysicalSlot(rt_resident,ctx,batch,slot);asc_stcg(map+token,slot);
        RecordChange(rt_resident,ctx,batch,slot,token);
    }
    asc_threadfence();
}
// Same indexed-load pattern as v9 GatherCandidateMap; cold IDs pass through.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void RestoreVictimTokens(__ubuf__ uint32_t* ids,__gm__ uint32_t* victims,uint32_t count) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        uint32_t token=ids[i];
        if(token<kLruSlots)ids[i]=asc_ldcg(victims+token);
    }
}
__aicore__ inline void FastService(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer,uint32_t epoch,uint32_t core,uint32_t lane,
    LocalTensor<uint32_t> scratch,LocalTensor<uint16_t> flags,LocalTensor<uint32_t> ids) {
    auto ctx=reinterpret_cast<__gm__ Context*>(pointer);auto events=reinterpret_cast<__gm__ Event*>(ctx->events);
    auto event=events+epoch*rt_serviceLanes+lane;event->core=core;event->epoch=epoch;V9_FETCH_PROFILE(event->waitBegin=GetSystemCycle();)
    if(epoch) {
        auto previous=reinterpret_cast<__gm__ volatile Event*>(events)+(epoch-1)*rt_serviceLanes+lane;
        do { kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+(epoch-1)*rt_serviceLanes+lane),64); }
        while(!previous->complete);
    }
    V9_FETCH_PROFILE(event->begin=GetSystemCycle();)
    uint64_t postedCount=0; // Business state: controls batched WQE submission.
    V9_FETCH_PROFILE(uint64_t scanTicks=0,candidateTicks=0,postTicks=0,bitmapBytes=0,totalCandidates=0;)
    auto work=flags.ReinterpretCast<uint32_t>();auto cfg=work[4096];
    auto rowStorage=ids;
    if(rt_batchedRows)ids=scratch[23808/4];
    // Freeze every ready snapshot before doing any batch work. Cap to the
    // scheduled boundary so slow service cannot chase newly produced history.
    GlobalTensor<uint32_t> progressGm;
    progressGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    if(lane<rt_batch) {
        DataCopyExtParams gather;gather.blockCount=(rt_batch-lane+rt_serviceLanes-1)/rt_serviceLanes;
        gather.blockLen=4*sizeof(Progress);gather.srcStride=(rt_serviceLanes-1)*4*sizeof(Progress);gather.dstStride=0;
        DataCopyPadExtParams<uint32_t> padding{false,0,0,0};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(work,progressGm[lane*128],gather,padding);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes)for(uint32_t q=0;q<4;++q) {
            uint32_t end=work.GetValue((b/rt_serviceLanes)*128+q*32),cap=Trigger(rt_history,rt_resident,rt_rounds,epoch)*128;
            if(ctx->queryEnds) {
                auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds);
                uint32_t valid=bounds[b*4+q];
                if(ctx->lruSlots && valid<rt_resident)valid=rt_resident;
                if(cap>valid)cap=valid;
            }
            cfg.SetValue(128+b*4+q,end<cap?end:cap);
        }
    }
    GlobalTensor<uint32_t> allRows;allRows.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rows));
    if(rt_batchedRows) {
        if(lane<rt_batch) {
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
            DataCopy(state,rowGm,256);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
auto postInput=cfg[256+(b/rt_serviceLanes)*80];
        if (rt_batchedRows) postInput.SetValue(64,0);
        if(state.GetValue(11))continue;
        uint32_t slots=state.GetValue(8),requests=state.GetValue(9),pending=state.GetValue(211);
        uint32_t sequence=state.GetValue(246)+1;
        state.SetValue(246,sequence);state.SetValue(214+epoch,sequence);
        const uint32_t available=PrefetchAllowance(slots,requests,state.GetValue(208),
            state.GetValue(209),state.GetValue(210),ctx->windowLimit);
        const uint64_t entityAddress=state64.GetValue(0),queueAddress=state64.GetValue(1);
        const uint64_t addresses=state64.GetValue(2),output=state64.GetValue(3);
        // Optional transport policy: reserve speculative traffic for cold
        // history. Final TopK repair still handles selected evicted hot slots.
        const uint32_t lower=V9_COLD_ONLY_PREFETCH?rt_resident:(ctx->lruSlots?0:rt_resident);
        uint32_t end[4],first=rt_history,last=lower;
        for(uint32_t q=0;q<4;++q) {
            uint32_t saved=cfg.GetValue(128+b*4+q);
            end[q]=saved>state.GetValue(12+q)?saved:state.GetValue(12+q);
            uint32_t lo=state.GetValue(12+q)>lower?state.GetValue(12+q):lower;
            if(lo<first)first=lo;if(end[q]>last)last=end[q];
        }
        V9_FETCH_PROFILE(uint64_t scanBegin=GetSystemCycle();)
        // Exhausted budgets cannot recover within this invocation: counters
        // only increase and limits are fixed. Retain the unscanned cursor;
        // final TopK repair is independent of this speculative scan position.
        // Suppression statistics count examined candidates only, not skipped bits.
        uint32_t candidateCount=0,stop=available?last:0,wordCount=0;
        bool aggregate=first%128==0 && last%128==0;
        for(uint32_t q=0;q<4;++q) {
            uint32_t consumed=state.GetValue(12+q);if(consumed<lower)consumed=lower;
            if(consumed!=first || end[q]!=last)aggregate=false;
        }
        GlobalTensor<uint8_t> bitmap;
        bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(ctx->hits)+b*rt_history);
        for(uint32_t base=(first/128)*128;available && first<last && base<last;) {
            // Compact victim coordinates occupy [0,2048); the remaining
            // hot bitmap is not written or read. Cold coordinates are unchanged.
            if(ctx->lruSlots && base>=kLruSlots && base<rt_resident) {
                base=rt_resident;continue;
            }
            uint32_t span=((last-base+127)/128)*128;
            if(span>rt_scanTokens)span=rt_scanTokens;
            if(ctx->lruSlots && base<kLruSlots && span>kLruSlots-base)span=kLruSlots-base;
            if(aggregate) {
                uint32_t room=((1792-wordCount)/8)*128;
                if(!room) {stop=base;break;}
                if(span>room)span=room;
            }
            // ready ends are 128-aligned; a capacity cutoff can leave a
            // half-tile cursor, which is masked on the next service.
            bool full=true;
            for(uint32_t q=0;q<4;++q) {
                uint32_t lo=state.GetValue(12+q)>base?state.GetValue(12+q)-base:0;
                if(base<lower && lo<lower-base)lo=lower-base;
                uint32_t hi=end[q]>base?end[q]-base:0;
                cfg.SetValue(28+q,lo<span?lo:span);cfg.SetValue(24+q,hi<span?hi:span);
                if(lo || hi<span)full=false;
#if V9_PARTIAL_SCAN_SIMD
                uint32_t clippedLo=lo<span?lo:span,clippedHi=hi<span?hi:span;
                cfg.SetValue(80+q*4,clippedLo/16);cfg.SetValue(81+q*4,clippedHi/16);
                cfg.SetValue(82+q*4,0x55555555U<<((clippedLo%16)*2));
                cfg.SetValue(83+q*4,((1U<<((clippedHi%16)*2))-1U)&0x55555555U);
#endif
            }
            DataCopyExtParams copy;copy.blockCount=4;copy.blockLen=span/4;
            copy.srcStride=(rt_history-span)/4;copy.dstStride=0;
            V9_FETCH_PROFILE(bitmapBytes+=span;)
            DataCopyPadExtParams<uint8_t> padding{false,0,0,0};
            DataCopyPad(scratch.ReinterpretCast<uint8_t>(),bitmap[base/4],copy,padding);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if(full || V9_PARTIAL_SCAN_SIMD) {
                uint32_t prior=aggregate?wordCount:0;
                if(full)UnionWords<false>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                else UnionWords<true>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint32_t words=cfg.GetValue(72);wordCount+=words;
                if(!aggregate && words)Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount,words);
                else {cfg.SetValue(40,candidateCount);cfg.SetValue(48,span);}
            } else Simt::VF_CALL<ScanPartial>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);uint32_t scanned=cfg.GetValue(48);
            // V_S above joins the scan; forward that dependency to MTE2.
            // Do not consume LI's persistent V_MTE2 ping-pong credits.
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            if(scanned<span) { stop=base+scanned;break; }
            base+=span;
        }
        if(aggregate && wordCount) {
            Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,stop,0,wordCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);stop=cfg.GetValue(48);
        }
        if(ctx->lruSlots && first<kLruSlots && candidateCount) {
            Simt::VF_CALL<RestoreVictimTokens>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        V9_FETCH_PROFILE(scanTicks+=GetSystemCycle()-scanBegin;totalCandidates+=candidateCount;)
        V9_FETCH_PROFILE(uint64_t candidateBegin=GetSystemCycle(),postBegin=0;)
        uint32_t count=0,missing=0;
        if(candidateCount) {
            auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
            if(!slots && !ctx->tailBounds) {
                // No slot has ever been assigned: the OR bitmap already
                // deduplicated tokens, so every candidate is a new miss.
                missing=candidateCount;count=missing<available?missing:available;
            } else {
                Simt::VF_CALL<GatherCandidateMap>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidateCount);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),candidateCount,available);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                missing=cfg.GetValue(56);count=missing<available?missing:available;
            }
            V9_FETCH_PROFILE(candidateTicks+=GetSystemCycle()-candidateBegin;)
            if(count) {
                V9_FETCH_PROFILE(postBegin=GetSystemCycle();)
if (rt_batchedRows) {
                auto post64=postInput.ReinterpretCast<uint64_t>();
                postInput.SetValue(64,count);postInput.SetValue(65,rt_resident+slots);
                postInput.SetValue(66,pending);post64.SetValue(34,addresses);
                post64.SetValue(35,output);post64.SetValue(36,queueAddress);
                GlobalTensor<uint32_t> entityGm,queueGm;
                entityGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(entityAddress));
                queueGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queueAddress));
                DataCopy(postInput,entityGm,48);DataCopy(postInput[48],queueGm,16);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                GlobalTensor<uint32_t> jobs;
                jobs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->locations)+b*8192);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(jobs,ids,((count+7)/8)*8);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                slots+=count;requests+=count;
} else {
                {
                    Simt::VF_CALL<PrepareCandidateWqes>(Simt::Dim3{256,1,1},rt_resident,
                        reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(work[512].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[1024].GetPhyAddr()),
                        map,reinterpret_cast<__gm__ uint64_t*>(addresses),count,rt_resident+slots,ctx,b);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    if(SubmitWqesAt(rt_tokenBytes,entityAddress,queueAddress,output,work[1024],scratch,work,work[512],cfg,count,256,false,nullptr,pending))++pending;
                    else state.SetValue(11,kvaf_device::kInvalidArgs);
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
        for(uint32_t q=0;q<4;++q) {
            uint32_t processed=end[q]<stop?end[q]:stop;
            if(processed<state.GetValue(12+q))processed=state.GetValue(12+q);
            state.SetValue(80+epoch*4+q,processed);state.SetValue(12+q,processed);
        }
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
        V9_FETCH_PROFILE(uint64_t begin=GetSystemCycle();)
        uint32_t maxCount=0;
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            uint32_t n=cfg.GetValue(256+(b/rt_serviceLanes)*80+64);if(n>maxCount)maxCount=n;
        }
        for(uint32_t start=0;start<maxCount;start+=rt_postBank) {
            Simt::VF_CALL<BatchBuild<V9_BATCH_POST==1>>(Simt::Dim3{512,1,1},rt_batch,rt_history,rt_resident,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,ctx,epoch,lane,start,
                reinterpret_cast<__ubuf__ uint64_t*>(work[512].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg[256].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
                uint32_t g=b/rt_serviceLanes;auto input=cfg[256+g*80];auto in64=input.ReinterpretCast<uint64_t>();
                uint32_t count=input.GetValue(64);if(count<=start)continue;
                uint32_t n=count-start;if(n>rt_postBank)n=rt_postBank;
                uint32_t depth=input.GetValue(19),offset=uint32_t(in64.GetValue(25)+start)&(depth-1);
                uint32_t first=n<depth-offset?n:depth-offset;
#if V9_BATCH_POST == 2
                FinalBuild(rt_tokenBytes,reinterpret_cast<__ubuf__ uint32_t*>(work[512+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1024+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[1536+g*rt_postBank].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(scratch[g*rt_postBank*16].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[g*32].GetPhyAddr()),0,n,count-start);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
#endif
                GlobalTensor<uint32_t> sq;sq.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(7)));
                DataCopy(sq[offset*16],scratch[g*rt_postBank*16],first*16);
                if(first<n)DataCopy(sq,scratch[(g*rt_postBank+first)*16],(n-first)*16);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            auto input=cfg[256+(b/rt_serviceLanes)*80];auto in64=input.ReinterpretCast<uint64_t>();
            uint32_t count=input.GetValue(64);if(!count)continue;
            in64.SetValue(25,in64.GetValue(25)+count);
            GlobalTensor<uint32_t> queue;queue.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(in64.GetValue(36)));
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopy(queue,input[48],16);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            auto input=cfg[256+(b/rt_serviceLanes)*80];if(!input.GetValue(64))continue;
            auto in64=input.ReinterpretCast<uint64_t>();uint64_t db=in64.GetValue(8),head=in64.GetValue(25);
#if V9_SIMULATION
            *reinterpret_cast<__gm__ uint64_t*>(db)=head;
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(db),64);
#else
            st_dev(uint32_t(head),reinterpret_cast<__gm__ uint32_t*>(db),0);
#endif
        }
        V9_FETCH_PROFILE(postTicks+=GetSystemCycle()-begin;)
    }
}
    V9_FETCH_PROFILE(event->scanTicks=scanTicks;event->candidateTicks=candidateTicks;event->postTicks=postTicks;)
    V9_FETCH_PROFILE(event->bitmapBytes=bitmapBytes;event->candidateCount=totalCandidates;event->postedCount=postedCount;)
    V9_FETCH_PROFILE(event->end=GetSystemCycle();)event->complete=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),V9_INTERMEDIATE_PROFILING?sizeof(Event):64);
}
}

namespace rotating {
__aicore__ inline void Service(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_splitQ,uint32_t rt_rounds,uint32_t rt_serviceAics,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer, uint32_t epoch, uint32_t core,
    uint32_t lane, LocalTensor<uint32_t> scratch, LocalTensor<uint16_t> flags,
    LocalTensor<uint32_t> indices) {
    FastService(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,pointer, epoch, core, ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane), scratch, flags, indices);
}
}
