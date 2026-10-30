#pragma once
#include "resident_plan_parallel.h"
#include "contract_device.h"
#include "contract_maintenance.h"
#include "index_plan_records.h"
#include "publication_plan.h"
namespace rotating {
constexpr uint32_t kContractControl = 49664;
constexpr uint32_t kContractScratch = (kContractControl + 2560) * sizeof(uint32_t);
static_assert(kContractScratch <= V9_USER_UB_BYTES, "Contract final scratch exceeds V9 UB");
static_assert(kIndexScratchWords * sizeof(uint32_t) <= kContractScratch, "Index scratch exceeds final UB");

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void ResolveContract(__gm__ Context* ctx, uint32_t b, uint32_t w,
    __ubuf__ uint32_t* selected, __ubuf__ uint32_t* loc,
    __ubuf__ uint32_t* tokens, __ubuf__ uint32_t* slots,
    __ubuf__ uint32_t* hash, __ubuf__ uint32_t* protectedSlots,
    __ubuf__ uint32_t* ctl) {
    uint32_t tid=threadIdx.x;
    auto c=contract_simt::ContractOf(ctx);
    auto row=contract_simt::ContractPtr<Row>(ctx->rows)+b;
    auto result=ctl+8,prebits=ctl+16,scan=ctl+544;
    uint32_t pool=contract_simt::ContractPool(c,b);
    for(uint32_t i=tid;i<512;i+=256)prebits[i]=0;
    asc_syncthreads();
    for(uint32_t i=tid;i<row->slots;i+=256) {
        uint32_t s=uint32_t(contract_simt::ContractPtr<int32_t>(c->reserved_slots)[b*2048+i]);
        asc_atomic_or(prebits+s/32,1U<<(s%32));
    }
    asc_syncthreads();
    BuildResidentPlanParallel(contract_simt::ContractBudget(c,b),contract_simt::ContractLength(c,b),
        contract_simt::ContractPtr<int32_t>(c->cursors)[pool],
        contract_simt::ContractPtr<uint32_t>(c->hot_bitmap)+b*((c->candidate_capacity+31)/32),
        contract_simt::ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity,
        contract_simt::ContractPtr<int32_t>(ctx->mapping)+uint64_t(b)*c->candidate_capacity,
        contract_simt::ContractPtr<int32_t>(c->victims)+b*2048,
        contract_simt::ContractPtr<uint32_t>(c->victim_bitmap)+b*512,
        selected,loc,tokens,slots,hash,protectedSlots,result,prebits,scan);
    uint32_t begin=result[kPlanSupplement]*w/kFinalWorkers,end=result[kPlanSupplement]*(w+1)/kFinalWorkers;
    if(!tid) {
        ctl[0]=result[kPlanSupplement]; ctl[1]=result[kPlanError]; ctl[4]=0;
        ctl[2]=begin; ctl[3]=end-begin;
    }
    asc_syncthreads();
    if(result[kPlanError]) {
        if(!tid) {
            ctl[0]=ctl[3]=0;
            if(!w) {
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->status)+b,int32_t(result[kPlanError]));
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->error_detail)+b*4,int32_t(result[kPlanError]));
                asc_atomic_cas(contract_simt::ContractPtr<int32_t>(c->sticky),0,int32_t(result[kPlanError]));
            }
        }
        return;
    }
    StageResidentDestination(c,b,w,selected,loc,scan);
    if(!w) {
        // Reject the direct record route before writing if its conservative
        // extent exceeds the existing S+C metadata workspace.
        bool emit=contract_simt::IndexOwned(c,b) &&
            2*(result[kPlanMisses]+row->slots)<=c->resident_capacity+c->candidate_capacity;
        auto state=contract_simt::ContractPtr<int32_t>(contract_simt::IndexBase(c,b));
        auto assigned=ctl+576; // 512 words, dead before WQE packing reuses UB.
        if(!tid) { ctl[5]=0;asc_stcg(state+27,0); }
        for(uint32_t i=tid;i<512;i+=256)assigned[i]=0;
        asc_syncthreads();
        uint32_t extras=0,extraMask=0;
        for(uint32_t i=tid;i<row->slots;i+=256) {
            int32_t s=contract_simt::ContractPtr<int32_t>(c->reserved_slots)[b*2048+i];
            bool keep=!PlanBit(protectedSlots,uint32_t(s));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->extra_prefetch_slots)+b*2048+i,keep?s:-1);
            extras+=keep;if(keep)extraMask|=1U<<(i/256);
        }
        for(uint32_t i=tid;i<result[kPlanMisses];i+=256) {
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->planned_tokens)+b*kSlots+i,int32_t(tokens[i]));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->planned_slots)+b*kSlots+i,int32_t(slots[i]));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->changed_slots)+b*c->resident_capacity+i,int32_t(slots[i]));
            asc_atomic_or(assigned+slots[i]/32,1U<<(slots[i]%32));
            if(emit) {
                if(!StoreIndexPlanRecord(c,b,i,slots[i],tokens[i]))asc_atomic_or(ctl+5,1U);
            }
        }
        asc_syncthreads();
        uint32_t unplanned=0,unplannedMask=0;
        for(uint32_t i=tid;i<row->slots;i+=256) {
            uint32_t s=uint32_t(contract_simt::ContractPtr<int32_t>(c->reserved_slots)[b*2048+i]);
            if(!PlanBit(assigned,s)){++unplanned;unplannedMask|=1U<<(i/256);}
        }
        // Both counts <= 2048; combine the two stable compactions in one scan.
        auto sum=ContractPrefixUB(extras+(unplanned<<16),scan);
        {
            uint32_t rank=result[kPlanMisses]+(sum.begin>>16);
            for(uint32_t i=tid;i<row->slots;i+=256)if(unplannedMask&(1U<<(i/256))) {
                uint32_t slot=uint32_t(contract_simt::ContractPtr<int32_t>(c->reserved_slots)[b*2048+i]);
                uint32_t token=(extraMask&(1U<<(i/256)))?
                    uint32_t(contract_simt::ContractPtr<int32_t>(c->reserved_tokens)[b*2048+i]):UINT32_MAX;
                asc_stcg(contract_simt::ContractPtr<int32_t>(c->changed_slots)+b*c->resident_capacity+rank,int32_t(slot));
                if(emit && !StoreIndexPlanRecord(c,b,rank,slot,token))asc_atomic_or(ctl+5,1U);
                ++rank;
            }
        }
        asc_threadfence();asc_syncthreads();
        if(!tid) {
            auto meta=contract_simt::ContractPtr<int32_t>(c->publish_meta)+b*4;
            asc_stcg(meta+1,int32_t(result[kPlanMisses]+(sum.total>>16)));
            asc_stcg(meta+2,int32_t(result[kPlanMisses]-result[kPlanSupplement]));
            asc_threadfence();asc_stcg(meta+3,1);
            if(emit) {
                asc_stcg(state+28,int32_t(result[kPlanMisses]+(sum.total>>16)));
                asc_stcg(state+29,int32_t(result[kPlanMisses]));
                asc_stcg(state+30,int32_t(row->slots));
                asc_threadfence();asc_stcg(state+27,ctl[5]?0:1);
            }
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->extra_prefetch_counts)+b,int32_t(sum.total&0xffffU));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->plan_count)+b,int32_t(result[kPlanMisses]));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->logical_misses)+b,int32_t(result[kPlanMisses]));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->proposed_cursor)+b,int32_t(result[kPlanCursor]));
            asc_stcg(contract_simt::ContractPtr<int32_t>(c->error_detail)+b*4+2,1);
            asc_atomic_add(contract_simt::ContractPtr<uint32_t>(ctx->count),result[kPlanSupplement]);
        }
    }
    if(!result[kPlanSupplement]) {
        asc_threadfence();asc_syncthreads();return;
    }
    uint32_t n=result[kPlanMisses],chunk=(n+255)/256,lo=tid*chunk,hi=lo+chunk,count=0,needMask=0;
    if(hi>n)hi=n;
    for(uint32_t i=lo;i<hi;++i)
        if(contract_simt::ContractPtr<int32_t>(ctx->mapping)[uint64_t(b)*c->candidate_capacity+tokens[i]]<0) {
            needMask|=1U<<(i-lo);++count;
        }
    auto prefix=ContractPrefixUB(count,scan); uint32_t rank=prefix.begin;
    for(uint32_t i=lo;i<hi;++i) {
        uint32_t token=tokens[i],slot=slots[i];
        if(!(needMask&(1U<<(i-lo))))continue;
        uint32_t current=rank++; if(current<begin || current>=end)continue;
        uint32_t own=current-begin,dst=b*kSlots+current;
        uint64_t address=contract_simt::ContractRemote(c,b,token);
        hash[own]=uint32_t(address); hash[8192+own]=uint32_t(address>>32);
        selected[own]=contract_simt::ContractPhysical(c,b,slot);
        if(row->pending && PlanBit(prebits,slot))asc_atomic_or(ctl+4,1U);
        asc_atomic_or(contract_simt::ContractPtr<uint32_t>(c->touched)+b*512+slot/32,1U<<(slot%32));
        asc_stcg(contract_simt::ContractPtr<int32_t>(ctx->sourceBlocks)+dst,int32_t(contract_simt::ContractDirectory(c,b,token)));
        asc_stcg(contract_simt::ContractPtr<int32_t>(ctx->sourceOffsets)+dst,int32_t(token%128*kTokenBytes));
        asc_stcg(contract_simt::ContractPtr<int32_t>(ctx->destinations)+dst,int32_t(selected[own]));
    }
    asc_syncthreads();
    for(uint32_t i=tid;i<end-begin;i+=256)tokens[i]=selected[i];
    asc_threadfence(); asc_syncthreads();
}

__aicore__ inline void ContractFinal(__gm__ Context* ctx) {
    uint64_t kernelStarted=GetSystemCycle();
    uint32_t core = GetBlockIdx(), b = core / kFinalWorkers, w = core % kFinalWorkers;
    if (b >= SIM_BATCH) return;
    auto c = ContractOf(ctx);
    auto row = ContractPtr<Row>(ctx->rows) + b;
    // Snapshot preflight state before resolving. A different request's final
    // capacity failure must not prevent this request's queues from being drained.
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->status) + b*4, 4);
    if (ContractPtr<int32_t>(c->status)[b] != kPending || !row->entity || row->error) {
        return; // Fetch drains queues and publishes failures.
    }
    TPipe pipe; TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, kContractScratch);
    auto local = scratch.Get<uint32_t>(), selected = local, loc = local[8192];
    auto tokens = local[16384], slots = local[24576], hash = local[32768];
    auto protectedSlots = local[49152], ctl = local[kContractControl];
    uint64_t started = GetSystemCycle();
    GlobalTensor<uint32_t> input;
    input.SetGlobalBuffer(ContractPtr<uint32_t>(c->topk) + b * c->queries * kTop);
    DataCopy(selected, input, c->queries * kTop);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Simt::VF_CALL<ResolveContract>(Simt::Dim3{256, 1, 1}, ctx, b, w,
        FinalPtr(selected), FinalPtr(loc), FinalPtr(tokens), FinalPtr(slots),
        FinalPtr(hash), FinalPtr(protectedSlots), FinalPtr(ctl));
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(!w) RecordContractTicks(ctx,b,11,GetSystemCycle()-started);
    uint32_t count = ctl.GetValue(0), error = ctl.GetValue(1);
    uint32_t begin = ctl.GetValue(2), own = ctl.GetValue(3);
    // ResolveContract already published the query partition from live UB.
    // No intermediate locations D2D copy or MTE3 round trip is necessary.
    // Planning rejection has no final transport operation. Keep FinalPost zero
    // so host completion can distinguish a drained capacity rejection from an
    // uncertain queue failure; fetch still drains any earlier prefetch.
    if (!error) PublishFinalWorker(ctx, row, b, w, own, count, b * kSlots + begin,
        0, local, hash, hash[8192], tokens, ctl, ctl.GetValue(4) != 0);
    if (!w) {
        auto status = ContractPtr<uint32_t>(ctx->rowStatus) + b * kRowStatusWords;
        status[0] = error; status[1] = count; status[2] = b * kSlots;
        row->finalBegin = started; row->finalEnd = GetSystemCycle();
        row->finalRequests = count; row->finalPending = count ? 1 : 0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status), 128);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row) + 960, 64);
    }
    if(!w && !error) {
        // Worker 0 owns the complete plan and has already posted its GETs.
        // Keep maintenance on this worker: a spare AIV count does not establish
        // a scheduling/progress guarantee for a cross-AIV polling protocol.
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        MaintainContractIndex(ctx,b,local);
    }
    pipe.Destroy();
    if(!w)RecordContractTicks(ctx,b,14,GetSystemCycle()-kernelStarted);
    RecordKernelSpan(ctx,96+b*4+w,kernelStarted,GetSystemCycle());
}
}
