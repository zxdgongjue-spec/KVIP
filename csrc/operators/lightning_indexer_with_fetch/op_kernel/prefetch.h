#pragma once
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
#include "cold_scan.h"
#include "rolling_threshold.h"
#if KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
#include "indexer/li_types.h"
#include "indexer/recent_threshold.h"
#include "indexer/topk_common.h"
#endif
#include "../../common/lifetch/prefetch_budget.h"

// Imported from direct_slots.h

#include "simt_api/device_functions.h"
// Prefetch ordinals map to LRU hot destinations. Final repair uses hot slots;
// virtual ABC tags remain disjoint from prefetch ordinals.
namespace rotating {
#include "../../common/lifetch/cache_metadata.h"
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void InitializeTailMappings(uint32_t rt_history,uint32_t rt_resident,__gm__ Context* ctx,uint32_t b) {
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

#define V9_FETCH_PROFILE(...) if constexpr (captureIo) { __VA_ARGS__ }

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

template<bool BuildSimt, bool traceWqe>
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
                    if constexpr(traceWqe) {
                        if(rt_batch==16 && ctx->shape.rounds==9 && ctx->windowLimit==64) {
                            auto audit=reinterpret_cast<__gm__ uint32_t*>(ctx->events+
                                uint64_t(9)*rt_serviceLanes*sizeof(Event));
                            asc_stcg(audit+(b*9+epoch)*64+begin+i,token);
                        }
                    }
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
__simt_callee__ inline uint32_t DecodeScanToken(uint32_t token, uint32_t begin, uint32_t end) {
#if KVIP_REVERSE_COLD
    if (token >= begin && token < end)
        return begin + end - 2048 - (token & ~2047U) + (token & 2047U);
#endif
    return token;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ExpandWords(__ubuf__ uint32_t* input,__ubuf__ uint32_t* wordIndices,__ubuf__ uint32_t* ids,
    __ubuf__ uint32_t* cfg,uint32_t base,uint32_t span,uint32_t oldCount,uint32_t words) {
#if KVIP_REVERSE_COLD
    // Load once per SIMT thread; do not reread control words for each ID.
    const uint32_t reverseBegin=cfg[60],reverseEnd=cfg[61];
#else
    constexpr uint32_t reverseBegin=0,reverseEnd=0;
#endif
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
            if(rank<kCandidateCapacity)ids[rank]=DecodeScanToken(base+offset,reverseBegin,reverseEnd);
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
#if KVIP_REVERSE_COLD
    // Load once per SIMT thread; do not reread control words for each ID.
    const uint32_t reverseBegin=cfg[60],reverseEnd=cfg[61];
#else
    constexpr uint32_t reverseBegin=0,reverseEnd=0;
#endif
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
            if(rank<kCandidateCapacity)ids[rank]=DecodeScanToken(base+offset,reverseBegin,reverseEnd);
            uint32_t select=0U-uint32_t(rank==kCandidateCapacity);
            stop=(stop&~select)|(offset&select);++rank;bits&=bits-1;
        }
    }
    if(stop<span)asc_atomic_min(cfg+48,stop);
}
// LIDU pattern: gather only candidate map entries, then SIMD squeeze misses.
// VF_CALL invokes a function pointer; every argument must be explicit.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void GatherCandidateMap(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __gm__ uint32_t* map,uint32_t count, __gm__ uint32_t* scores,
    __ubuf__ uint32_t* cfg, uint32_t history) {
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        const uint32_t token=ids[i]; uint32_t slot=asc_ldcg(map+token);
#if KVIP_WINDOW_REFILTER
        if (scores && slot==UINT32_MAX) {
            // The candidate list is already de-duplicated and restored to
            // physical coordinates. Reuse its existing map-gather launch.
            bool keep=false;
            const uint32_t scan=DecodeScanToken(token,cfg[60],cfg[61]);
#if KVIP_ROLLING_FAST_PATH && KVIP_REFILTER_SAFE
            // Any unknown, causal-valid Q already forces conservative keep.
            // Check that decision before issuing any random score loads.
            if(scan>=cfg[108])
                for(uint32_t q=0;q<4 && !keep;++q)
                    keep = token<cfg[104+q] && scan>=cfg[100+q];
#endif
            for(uint32_t q=0;q<4 && !keep;++q) {
#if KVIP_REFILTER_SAFE
                if(token>=cfg[104+q])continue;
#if !KVIP_ROLLING_FAST_PATH
                // Q4 union: unknown is not below Kth. Reuse this loop,
                // keep the ID and never read that query's unpublished GM.
                if(scan>=cfg[100+q]) {keep=true;break;}
#endif
#else
                if(scan>=cfg[100+q] || token>=cfg[104+q])continue;
#endif
                uint32_t pair=asc_ldcg(scores+(uint64_t(q)*history+token)/2);
                uint32_t score=(pair>>((token&1)*16))&0xffffU;
                keep=score>=cfg[96+q]; // Keep boundary ties; final TopK decides IDs.
            }
            if(!keep)slot=0; // UB-only rejection, never mutate the cache map.
        }
#endif
        slots[i]=slot;
    }
}
#if KVIP_RANKED_PORT
// Residency is stable through the LI kernel. Tag a private copy of seed IDs
// once; subsequent merges preserve old IDs and add only cold-range positions.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void TagRankedSeed(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* tagged,
    uint32_t count,__gm__ Context* ctx,uint32_t batch,uint32_t resident) {
    const uint32_t owner=OwnerRow(ctx,batch);
    auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid);
    auto reserved=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot);
    uint32_t tailBegin=0,tailEnd=0;
    if(ctx->tailBounds) {
        auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->tailBounds)+batch*2;
        tailBegin=asc_ldcg(bounds);tailEnd=asc_ldcg(bounds+1);
    }
    for(uint32_t i=threadIdx.x;i<count;i+=256) {
        const uint32_t token=ids[i];
        const bool safeHot=token<resident && !(token>=tailBegin && token<tailEnd) &&
            (!valid || asc_ldcg(valid+uint64_t(owner)*resident+token)) &&
            (!reserved || asc_ldcg(reserved+uint64_t(batch)*resident+token)<0);
        tagged[i]=safeHot ? (token | 0x80000000U) : token;
    }
}
#endif
__simd_vf__ inline void CompactMisses(__ubuf__ uint32_t* ids,__ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* cfg,uint32_t count,uint32_t available,
    __ubuf__ uint32_t* destination) {
    using namespace MicroAPI;
    auto all=CreateMask<uint32_t,MaskPattern::ALL>();auto lane=CreateMask<uint32_t,MaskPattern::VL1>();
    RegTensor<uint32_t> token,old,rank,index,written,flags,total,zeroIndex,broadcast,packed;
    Duplicate(written,uint32_t(0));Duplicate(zeroIndex,uint32_t(0));
    uint32_t remaining=count;auto output=destination;ClearSpr<SpecialPurposeReg::AR>();UnalignRegForStore store;
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
#if KVIP_RANKED_PORT || KVIP_INCREMENTAL_TOPK || KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
// Reserve only admitted IDs, before the next Q list is examined. This gives
// Q4 deduplication with the existing map and no per-window hash table.
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ReserveIncremental(__ubuf__ uint32_t* ids,__gm__ uint32_t* map,
    uint32_t count,uint32_t firstSlot) {
    for(uint32_t i=threadIdx.x;i<count;i+=256)asc_stcg(map+ids[i],firstSlot+i);
    asc_threadfence();
}
#endif
#include "fetch_rank.h"
// Same indexed-load pattern as v9 GatherCandidateMap; cold IDs pass through.

#if KVIP_FETCH_PACKETS
template<bool captureIo>
__aicore__ inline void FlushFetchBatch(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,
    uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,
    __gm__ Context* ctx,uint32_t epoch,uint32_t lane,LocalTensor<uint32_t> cfg,
    LocalTensor<uint32_t> work,LocalTensor<uint32_t> scratch,
    uint64_t& firstDoorbell,uint64_t& secondDoorbell,uint64_t& lastDoorbell) {
        uint32_t maxCount=0;
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            uint32_t n=cfg.GetValue(256+(b/rt_serviceLanes)*80+64);if(n>maxCount)maxCount=n;
        }
        for(uint32_t start=0;start<maxCount;start+=rt_postBank) {
            Simt::VF_CALL<BatchBuild<V9_BATCH_POST==1,captureIo>>(Simt::Dim3{512,1,1},rt_batch,rt_history,rt_resident,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,ctx,epoch,lane,start,
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
            if constexpr(captureIo) {
                lastDoorbell=GetSystemCycle();
                if(!firstDoorbell)firstDoorbell=lastDoorbell;
                else if(!secondDoorbell)secondDoorbell=lastDoorbell;
            }
        }
}

#endif
template<bool captureIo>
__aicore__ inline void FastService(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_rounds,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer,uint32_t epoch,uint32_t core,uint32_t lane,
    LocalTensor<uint32_t> scratch,LocalTensor<uint16_t> flags,LocalTensor<uint32_t> ids,
    uint32_t sequenceHalfWindow, GM_ADDR rollingScores = nullptr, const uint32_t* rankedCounts = nullptr) {
    uint64_t scanTicks=0,candidateTicks=0,postTicks=0,bitmapBytes=0,totalCandidates=0;
    uint64_t scanBegin=0,candidateBegin=0,postBegin=0,begin=0;
    uint64_t firstDoorbell=0,secondDoorbell=0,lastDoorbell=0;
    auto ctx=reinterpret_cast<__gm__ Context*>(pointer);auto events=reinterpret_cast<__gm__ Event*>(ctx->events);
    auto event=events+epoch*rt_serviceLanes+lane;event->core=core;event->epoch=epoch;V9_FETCH_PROFILE(if(!KVIP_RANKED_PORT)event->waitBegin=GetSystemCycle();)
    uint64_t postWaitStart=0;
#if KVIP_RANKED_PORT
    if constexpr(captureIo)postWaitStart=GetSystemCycle();
#endif
    if(epoch) {
        auto previous=reinterpret_cast<__gm__ volatile Event*>(events)+(epoch-1)*rt_serviceLanes+lane;
        do { kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(events+(epoch-1)*rt_serviceLanes+lane),64); }
        while(!previous->complete);
    }
#if KVIP_RANKED_PORT
    if constexpr(captureIo)bitmapBytes=GetSystemCycle()-postWaitStart; // rankedPostWait, same event word
#endif
    GlobalTensor<uint32_t> progressGm;
    progressGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->progress));
    auto work=flags.ReinterpretCast<uint32_t>();
    const uint32_t windowEnd=sequenceHalfWindow?SequenceWindowEnd(rt_history,rt_resident,sequenceHalfWindow,epoch):
        Trigger(rt_history,rt_resident,rt_rounds,epoch)*128;
    // Each shard publishes an independent ready boundary. Snapshot whatever
    // has completed; the final resolver repairs selected tokens that were not
    // prefetched. Waiting for another shard here can stall its producer.
    V9_FETCH_PROFILE(event->begin=GetSystemCycle();)
    uint64_t postedCount=0; // Business state: controls batched WQE submission.

    auto cfg=work[4096];
    auto rowStorage=ids;
    if(rt_batchedRows)ids=scratch[23808/4];
#if !KVIP_RANKED_PORT
    // Dynamic TopK already froze and consumed the per-query progress. Its
    // Fetch branch reads the committed rank, so a second progress DMA is dead.
    // Freeze every ready snapshot before doing any batch work. Cap to the
    // scheduled boundary so slow service cannot chase newly produced history.
    if(lane<rt_batch) {
        DataCopyExtParams gather;gather.blockCount=(rt_batch-lane+rt_serviceLanes-1)/rt_serviceLanes;
        gather.blockLen=4*sizeof(Progress);gather.srcStride=(rt_serviceLanes-1)*4*sizeof(Progress);gather.dstStride=0;
        DataCopyPadExtParams<uint32_t> padding{false,0,0,0};
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(work,progressGm[lane*128],gather,padding);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes)for(uint32_t q=0;q<4;++q) {
            uint32_t progress=(b/rt_serviceLanes)*128+q*32;
            uint32_t end=work.GetValue(progress),cap=windowEnd;
#if KVIP_INCREMENTAL_TOPK
            cfg.SetValue(1024+b*4+q,work.GetValue(progress+kvip_incremental::kReadyWord));
#endif
#if KVIP_WINDOW_REFILTER
            cfg.SetValue(1024+b*4+q,work.GetValue(progress+kvip_rolling::kProgressThreshold)&0xffffU);
#endif
            if(ctx->queryEnds) {
                auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds);
                uint32_t valid=bounds[b*4+q];
                if(ctx->lruSlots && valid<rt_resident)valid=rt_resident;
                if(cap>valid)cap=valid;
            }
            if(sequenceHalfWindow) {
                // A separate hot phase has already published the hot bitmap.
                if(end<rt_resident)end=rt_resident;
                uint32_t high=work.GetValue(progress+16);
                // Split-Q geometry limits the batch to 16, leaving this band
                // disjoint from both the low snapshot and batched post input.
                cfg.SetValue(192+b*4+q,high<cap?high:cap);
            }
            cfg.SetValue(128+b*4+q,end<cap?end:cap);
        }
    }
#endif
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
#if KVIP_FETCH_PACKETS
    // Immediate posts may run before later rows have been visited. Explicitly
    // clear their BatchBuild counts so they cannot consume stale UB contents.
    for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes)
        cfg.SetValue(256+(b/rt_serviceLanes)*80+64,0);
#endif
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
        const uint64_t entityAddress=state64.GetValue(0),queueAddress=state64.GetValue(1);
        const uint64_t addresses=state64.GetValue(2),output=state64.GetValue(3);
        const uint32_t lower=ctx->lruSlots?0:rt_resident;
#if KVIP_REVERSE_COLD || KVIP_WINDOW_REFILTER
        uint32_t ends[4] = {rt_history,rt_history,rt_history,rt_history};
        if(ctx->queryEnds) {
            auto bounds=reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+b*4;
            for(uint32_t q=0;q<4;++q)ends[q]=bounds[q];
        }
        // Sequence-split LI retains its original ordering in this experiment.
        auto range=sequenceHalfWindow?kvip_cold_scan::Range{0,0}:
            kvip_cold_scan::Bounds(rt_resident,rt_history,ends);
        // Unused control words, disjoint from counters and warp-prefix scratch.
        cfg.SetValue(60,KVIP_REVERSE_COLD?range.begin:0);
        cfg.SetValue(61,KVIP_REVERSE_COLD?range.end:0);
#if KVIP_WINDOW_REFILTER
#if KVIP_ROLLING_FAST_PATH
        uint32_t allReady=UINT32_MAX;
#endif
        for(uint32_t q=0;q<4;++q) {
            cfg.SetValue(96+q,cfg.GetValue(1024+b*4+q));
            cfg.SetValue(100+q,cfg.GetValue(128+b*4+q));
            cfg.SetValue(104+q,ends[q]);
#if KVIP_ROLLING_FAST_PATH
            uint32_t ready=cfg.GetValue(128+b*4+q);
            allReady=allReady<ready?allReady:ready;
#endif
        }
#if KVIP_ROLLING_FAST_PATH
        // Existing free word immediately after Q4 causal ends. No new DMA.
        cfg.SetValue(108,allReady);
#endif
#endif
#endif
        auto map=reinterpret_cast<__gm__ uint32_t*>(ctx->mapping)+b*rt_history;
        uint32_t count=0,missing=0;
#if KVIP_RANKED_PORT || KVIP_REFILTER_REFILL || KVIP_COMPACT_OUT_OF_PLACE || KVIP_INCREMENTAL_TOPK || KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
        // Existing Q2/16K work tensor: cfg[1536:2016] is disjoint from
        // bitmap words, frozen Q4 progress, thresholds and batched post inputs.
        // Window allowance is validated <=480. No new UB/GM allocation.
        auto selectedIds=cfg[1536];
#else
        auto selectedIds=ids;
#endif
#if KVIP_RANKED_PORT
        const uint32_t allowance=PrefetchAllowance(slots,requests,state.GetValue(208),state.GetValue(209),state.GetValue(210),ctx->windowLimit);
        for(uint32_t q=0;q<4 && count<allowance;++q) {
            const uint32_t n = rankedCounts[(b / rt_serviceLanes) * 4 + q];
            if (!n) continue;
            const uint64_t row = (uint64_t(epoch) * rt_batch + b) * 4 + q;
            GlobalTensor<uint32_t> offers;
            offers.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->rankedOffers) + row * kTop);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopy(ids,offers,(n+7U)&~7U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Simt::VF_CALL<GatherCandidateMap>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,n,
                static_cast<__gm__ uint32_t*>(nullptr),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),rt_history);
            PipeBarrier<PIPE_V>();
            CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),n,allowance-count,
                reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint32_t found=cfg.GetValue(56),admitted=found<allowance-count?found:allowance-count;
            missing+=found;
            if(admitted) {
                Simt::VF_CALL<ReserveIncremental>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()),map,admitted,rt_resident+slots+count);
                PipeBarrier<PIPE_V>();
            }
            count+=admitted;
        }
#elif KVIP_FETCH_BITMAP || KVIP_FETCH_DENSE
        // consumed[0] is the count of committed rank versions for this row.
        uint32_t versions=state.GetValue(12);
#if KVIP_FETCH_PACKETS
        uint32_t target=0; // Per-query consumed cursors are independent.
#else
        uint32_t target=versions;
#endif
        for(;target<=epoch;++target) {
            uint32_t bound=Trigger(rt_history,rt_resident,rt_rounds,target)*128;
            bool complete=true;
            for(uint32_t q=0;q<4;++q) {
                uint32_t want=bound<ends[q]?bound:ends[q];
                if(cfg.GetValue(128+b*4+q)<want)complete=false;
            }
            if(!complete)break;
        }
        uint32_t allowance=PrefetchAllowance(slots,requests,state.GetValue(208),state.GetValue(209),
                                             state.GetValue(210),ctx->windowLimit);
#if KVIP_FETCH_PACKETS
        count=ServiceFetchDemand<captureIo>(ctx,scratch,cfg,selectedIds,rollingScores,map,b,rt_history,rt_resident,
                               rt_rounds,state,target,ends,range,allowance,slots,missing);
#else
        count=ServiceFetchRank<captureIo>(ctx,scratch,cfg,selectedIds,rollingScores,map,b,rt_history,rt_resident,
                               rt_rounds,versions,target,ends,range,allowance,slots,missing,false);
        state.SetValue(12,target);
#endif
        // The ordinary row flush already transfers this history. Actual posted
        // IDs are recorded in BatchBuild, just as for the forward/reverse controls.
        if constexpr (captureIo) state.SetValue(80+epoch*4, target);
#elif KVIP_INCREMENTAL_TOPK
        const uint32_t allowance=PrefetchAllowance(slots,requests,state.GetValue(208),
            state.GetValue(209),state.GetValue(210),ctx->windowLimit);
        V9_FETCH_PROFILE(candidateBegin=GetSystemCycle();)
        for(uint32_t q=0;q<4 && count<allowance;++q) {
            uint32_t ready=cfg.GetValue(1024+b*4+q);
            if(!ready)continue;
            uint32_t published=ready-1;if(published>epoch)published=epoch;
            GlobalTensor<uint32_t> list;
            list.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incrementalLists)+
                uint64_t((b*4+q)*rt_rounds+published)*kvip_incremental::kListWords);
            auto header=work[3072];
            DataCopy(header,list,32);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            uint32_t candidates=header.GetValue(0);
            if(candidates>kvip_incremental::kTop) {state.SetValue(11,kvaf_device::kInvalidArgs);break;}
            if(!candidates)continue;
#if KVIP_INCREMENTAL_SEGMENTS
            const uint32_t baseEpoch = header.GetValue(3);
            const uint32_t prefix = header.GetValue(4);
            const uint32_t suffixEpoch = header.GetValue(5);
            if (baseEpoch > published || suffixEpoch > published || prefix > candidates || prefix % 8) {
                state.SetValue(11, kvaf_device::kInvalidArgs);
                break;
            }
            GlobalTensor<uint32_t> payload;
            payload.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(ctx->incrementalLists)
                + uint64_t(b * 4 + q) * rt_rounds * kvip_incremental::kListWords + kvip_incremental::kHeader);
            uint32_t skip = 0;
#if KVIP_CONSUMED_PREFIX
            // The last snapshot row is unused when rounds < kMaxRounds. Its
            // four words are cleared by InitializeRound and carried in the
            // existing Row DMA. The preceding service event orders its reuse.
            constexpr uint32_t cursorWord = __builtin_offsetof(Row, snapshot) / sizeof(uint32_t)
                                          + 4 * (kMaxRounds - 1);
            const bool useCursor = rt_rounds < kMaxRounds;
            const uint32_t cursor = useCursor ? state.GetValue(cursorWord + q) : 0;
            if ((cursor >> 18) == baseEpoch + 1) {
                const uint32_t consumed = cursor & 0xfffU;
                const bool sameSuffix = ((cursor >> 12) & 63U) == suffixEpoch + 1;
                if (sameSuffix && consumed == candidates) continue;
                skip = (sameSuffix ? consumed : (consumed < prefix ? consumed : prefix)) & ~7U;
            }
#endif
            const uint32_t head = prefix > skip ? prefix - skip : 0;
            const uint32_t tail = skip > prefix ? skip - prefix : 0;
            if (head) DataCopy(ids, payload[baseEpoch * kvip_incremental::kListWords + skip], head);
            if (prefix + tail < candidates) {
                DataCopy(ids[head], payload[suffixEpoch * kvip_incremental::kListWords + tail],
                         (candidates - prefix - tail + 7) / 8 * 8);
            }
            candidates -= skip;
#else
            DataCopy(ids,list[kvip_incremental::kHeader],(candidates+7)/8*8);
#endif
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Simt::VF_CALL<GatherCandidateMap>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidates,
                static_cast<__gm__ uint32_t*>(nullptr),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),rt_history);
            PipeBarrier<PIPE_V>();
            CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),candidates,allowance-count,
                reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint32_t found=cfg.GetValue(56),admitted=found<allowance-count?found:allowance-count;
            V9_FETCH_PROFILE(totalCandidates+=candidates;)
#if KVIP_CONSUMED_PREFIX
            if (useCursor) {
                // An unsubmitted miss may never be skipped. If the cap bound,
                // the first admitted positions are safe even with arbitrary
                // hits interleaved; later positions will be inspected again.
                uint32_t proven = skip + (found == admitted ? candidates : admitted);
                state.SetValue(cursorWord + q, ((baseEpoch + 1) << 18) |
                               ((suffixEpoch + 1) << 12) | proven);
            }
#endif
            missing+=found;
            if(admitted) {
                Simt::VF_CALL<ReserveIncremental>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr()),map,
                    admitted,rt_resident+slots+count);
                PipeBarrier<PIPE_V>();
            }
            count+=admitted;
        }
        V9_FETCH_PROFILE(candidateTicks+=GetSystemCycle()-candidateBegin;)
        for(uint32_t q=0;q<4;++q) {
            state.SetValue(12+q,cfg.GetValue(128+b*4+q));
            state.SetValue(80+epoch*4+q,cfg.GetValue(128+b*4+q));
        }
#else
#if KVIP_REFILTER_REFILL
        const uint32_t allowance=PrefetchAllowance(slots,requests,state.GetValue(208),
            state.GetValue(209),state.GetValue(210),ctx->windowLimit);
        for(;;) {
#endif
#define KV_WINDOW_HISTORY rt_history
#define KV_WINDOW_RESIDENT rt_resident
#define KV_WINDOW_SCAN_TOKENS rt_scanTokens
#include "../../common/lifetch/sequence_candidates.h"
#undef KV_WINDOW_HISTORY
#undef KV_WINDOW_RESIDENT
#undef KV_WINDOW_SCAN_TOKENS
        V9_FETCH_PROFILE(scanTicks+=GetSystemCycle()-scanBegin;totalCandidates+=candidateCount;)
        V9_FETCH_PROFILE(candidateBegin=GetSystemCycle();postBegin=0;)
        if(candidateCount) {
#if KVIP_REFILTER_REFILL
            uint32_t available=allowance-count;
#else
            uint32_t available=PrefetchAllowance(slots,requests,state.GetValue(208),
                state.GetValue(209),state.GetValue(210),ctx->windowLimit);
#endif
            if(!KVIP_REFILTER_REFILL && !slots && !ctx->tailBounds && !rollingScores) {
                // No slot has ever been assigned: the OR bitmap already
                // deduplicated tokens, so every candidate is a new miss.
                missing=candidateCount;count=missing<available?missing:available;
                selectedIds=ids; // No compaction on this existing all-miss shortcut.
            } else {
                Simt::VF_CALL<GatherCandidateMap>(Simt::Dim3{256,1,1},
                    reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),map,candidateCount,
                    rollingScores ? reinterpret_cast<__gm__ uint32_t*>(rollingScores)+uint64_t(b)*4*rt_history/2 : nullptr,
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),rt_history);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
                CompactMisses(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),candidateCount,available,
#if KVIP_REFILTER_REFILL || KVIP_COMPACT_OUT_OF_PLACE
                    reinterpret_cast<__ubuf__ uint32_t*>(selectedIds[count].GetPhyAddr())
#else
                    reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr())
#endif
                );
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint32_t found=cfg.GetValue(56);
                missing+=found;count+=found<available?found:available;
            }
            V9_FETCH_PROFILE(candidateTicks+=GetSystemCycle()-candidateBegin;)
        }
        for(uint32_t q=0;q<4;++q) {
            uint32_t processed=end[0][q]<stop?end[0][q]:stop;
            if(processed<consumed[0][q])processed=consumed[0][q];
            state.SetValue(12+q,processed);
            if(sequenceHalfWindow) {
                processed=end[1][q]<stop?end[1][q]:stop;
                if(processed<consumed[1][q])processed=consumed[1][q];
            }
            state.SetValue(80+epoch*4+q,processed);
        }
#if KVIP_REFILTER_REFILL
            // Continue only through already-frozen progress. Never wait for
            // another Q, enlarge the budget, or resnapshot advancing producers.
            if(count>=allowance || stop>=last || stop<=first)break;
        }
#endif
#endif
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
                DataCopy(jobs,selectedIds,((count+7)/8)*8);
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
#if KVIP_FETCH_PACKETS
                // Publish this request now. A different request's rank work is
                // not a dependency of these WQEs. Preserve in-request batching.
                FlushFetchBatch<captureIo>(rt_batch,rt_history,rt_resident,rt_serviceLanes,rt_tokenBytes,
                    rt_postThreads,rt_postBank,ctx,epoch,lane,cfg,work,scratch,
                    firstDoorbell,secondDoorbell,lastDoorbell);
                // BatchBuild normally updates pending AFTER the row flush.
                // Immediate posting moves it before that flush, so retain the
                // increment in the UB row as well instead of overwriting it.
                if(postInput.GetValue(64))++pending;
                else state.SetValue(11,kvaf_device::kInvalidArgs);
                postInput.SetValue(64,0);
#endif
                slots+=count;requests+=count;
} else {
                {
                    Simt::VF_CALL<PrepareCandidateWqes>(Simt::Dim3{256,1,1},rt_resident,
                        reinterpret_cast<__ubuf__ uint32_t*>(selectedIds.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ uint32_t*>(work[512].GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[1024].GetPhyAddr()),
                        map,reinterpret_cast<__gm__ uint64_t*>(addresses),count,rt_resident+slots,ctx,b);
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    if(captureIo && !firstDoorbell)firstDoorbell=GetSystemCycle();
                    if(SubmitWqesAt(rt_tokenBytes,entityAddress,queueAddress,output,work[1024],scratch,work,work[512],cfg,count,256,false,nullptr,pending))++pending;
                    else state.SetValue(11,kvaf_device::kInvalidArgs);
                    if(captureIo)lastDoorbell=GetSystemCycle();
                    slots+=count;requests+=count;
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
#if !KVIP_FETCH_PACKETS
if (rt_batchedRows) {
    if(postedCount) {
        V9_FETCH_PROFILE(begin=GetSystemCycle();)
        uint32_t maxCount=0;
        for(uint32_t b=lane;b<rt_batch;b+=rt_serviceLanes) {
            uint32_t n=cfg.GetValue(256+(b/rt_serviceLanes)*80+64);if(n>maxCount)maxCount=n;
        }
        for(uint32_t start=0;start<maxCount;start+=rt_postBank) {
            Simt::VF_CALL<BatchBuild<V9_BATCH_POST==1,captureIo>>(Simt::Dim3{512,1,1},rt_batch,rt_history,rt_resident,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,ctx,epoch,lane,start,
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
            if constexpr(captureIo) {
                lastDoorbell=GetSystemCycle();
                if(!firstDoorbell)firstDoorbell=lastDoorbell;
                else if(!secondDoorbell)secondDoorbell=lastDoorbell;
            }
        }
        V9_FETCH_PROFILE(postTicks+=GetSystemCycle()-begin;)
    }
}
#endif
    V9_FETCH_PROFILE(event->scanTicks=scanTicks;event->candidateTicks=candidateTicks;event->postTicks=postTicks;)
    V9_FETCH_PROFILE(event->bitmapBytes=bitmapBytes;event->candidateCount=totalCandidates;event->postedCount=postedCount;)
    if(captureIo){auto stamp=reinterpret_cast<__gm__ uint64_t*>(event->pad);stamp[0]=firstDoorbell;stamp[1]=lastDoorbell;stamp[4]=secondDoorbell;}
    V9_FETCH_PROFILE(event->end=GetSystemCycle();)event->complete=1;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(event),captureIo?sizeof(Event):64);
}
}

namespace rotating {
__aicore__ inline void Service(uint32_t rt_batch,uint32_t rt_history,uint32_t rt_resident,uint32_t rt_splitQ,uint32_t rt_rounds,uint32_t rt_serviceAics,uint32_t rt_serviceLanes,uint32_t rt_tokenBytes,uint32_t rt_postThreads,uint32_t rt_postBank,uint32_t rt_batchedRows,uint32_t rt_scanTokens,GM_ADDR pointer, uint32_t epoch, uint32_t core,
    uint32_t lane, LocalTensor<uint32_t> scratch, LocalTensor<uint16_t> flags,
    LocalTensor<uint32_t> indices, uint32_t sequenceHalfWindow = 0,bool captureIo = false, GM_ADDR rollingScores = nullptr, const uint32_t* rankedCounts = nullptr) {
    if(captureIo) FastService<true>(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,pointer, epoch, core, ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane), scratch, flags, indices, sequenceHalfWindow, rollingScores, rankedCounts);
    else FastService<false>(rt_batch,rt_history,rt_resident,rt_rounds,rt_serviceLanes,rt_tokenBytes,rt_postThreads,rt_postBank,rt_batchedRows,rt_scanTokens,pointer, epoch, core, ServiceLane(rt_batch,rt_splitQ,rt_serviceAics,core,lane), scratch, flags, indices, sequenceHalfWindow, rollingScores, rankedCounts);
}
}

#endif
