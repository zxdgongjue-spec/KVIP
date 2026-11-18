// SPDX-License-Identifier: Apache-2.0
#pragma once

// Experimental HIGH tail. Resolve the original eight-window events, drain
// speculative reads before posting any final overwrite, and compose the same
// ordered per-row permutation as Consumer. Only then can all AIVs safely move
// immutable tuples in parallel, including cycles and repeated victims.
#include "high_index_workspace.h"
#include "high_index_partition.h"

namespace pipeline_ab {
namespace high_post_index {
static_assert(offsetof(Args, highIndex) == 152, "high Index workspace address ABI");
struct PrivatePlan {
    uint32_t pairs = 0, singles = 0, invalidations = 0, chunks = 0, error = 0;
    uint64_t activeCycles = 0;
};

// One owner composes each row's sparse/dense map preserving the original
// prefetch/Q0/Q1/Q2/Q3 phase and 512-event chunk boundaries. Dirty destinations
// include displaced rows. No key/scale/address/logical writes here.
__aicore__ inline void Plan(__gm__ Args* a, uint32_t b, __gm__ Edge* edges,
                           __gm__ uint32_t* invalidations, bool profile, PrivatePlan& plan) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    auto shape = LoadRuntimeShape(ctx);
    auto src = reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+uint64_t(b)*a->maxEvents;
    auto dst = reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*a->maxEvents;
    auto dense = reinterpret_cast<__gm__ uint32_t*>(a->denseShadow)+uint64_t(b)*shape.history*6;
    auto flags = reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,SCRATCH_WORDS*4);
    auto local = buf.Get<uint32_t>();
    auto mem = reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    uint64_t begin = profile ? GetSystemCycle() : 0;
    Simt::VF_CALL<InitShadow>(Simt::Dim3{THREADS,1,1},mem);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    bool denseMode = false;
    uint32_t seen = 0, finalCount = 0, error = 0;
    for (uint32_t phase = 0; phase < 5 && !error; ++phase) {
        auto flag = flags+(phase ? kRawQueryFlag+phase-1 : 0);
        if (profile) plan.activeCycles += GetSystemCycle() - begin;
        error = AwaitPublication(flag);
        if (profile) begin = GetSystemCycle();
        if (error) break;
        uint32_t n = flag->count, offset = phase ? kPrefetch+finalCount : 0;
        if (!flag->ready || flag->error || n > kTop || offset+n > a->maxEvents) { error=1; break; }
        if (!n) continue;
        if (!denseMode && uint64_t(seen+n)*2 >= MAP) {
            Simt::VF_CALL<InitDenseShadow>(Simt::Dim3{THREADS,1,1},dense,shape.history,mem);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            denseMode=true;
        }
        __gm__ uint64_t* absent = nullptr;
        __gm__ uint32_t* noScales = nullptr;
        if (denseMode)
            Simt::VF_CALL<ApplyDenseBatch<false,true,sizeof(Tuple)/sizeof(uint32_t)>>(Simt::Dim3{THREADS,1,1},src,dst,offset,n,
                absent,noScales,invalidations,shape.resident,dense,shape.history,mem,absent,absent,absent);
        else
            Simt::VF_CALL<ApplyBatch<false,true,sizeof(Tuple)/sizeof(uint32_t)>>(Simt::Dim3{THREADS,1,1},src,dst,offset,n,
                absent,noScales,invalidations,shape.resident,mem,absent,absent,absent,shape.history);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error=local.GetValue(CONTROL); seen+=n;
        if (phase) finalCount+=n;
    }
    if (!error && seen) {
        Simt::VF_CALL<EmitEdges>(Simt::Dim3{THREADS,1,1},edges,dense,shape.history,denseMode,mem);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error=local.GetValue(CONTROL);
        if (!error) {
            plan.pairs=local.GetValue(CONTROL+1);
            plan.singles=local.GetValue(CONTROL+2);
            plan.invalidations=local.GetValue(CONTROL+6);
        }
    }
    plan.error=error;
    plan.chunks=local.GetValue(CONTROL+4);
    if (profile) plan.activeCycles += GetSystemCycle() - begin;
    pipe.Destroy();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void PackFinal(__gm__ Context* ctx, uint32_t b, uint32_t first, uint32_t end,
                      uint32_t c0,uint32_t c1,uint32_t c2,
                      __gm__ int32_t* out,uint64_t remoteBase,uint64_t remoteBytes,
                      uint32_t outputRows,uint64_t outputBytes,uint32_t tokenBytes,
                      __ubuf__ uint32_t* error) {
    if (!threadIdx.x) error[0]=0;
    asc_syncthreads();
    auto sources=reinterpret_cast<__gm__ int32_t*>(ctx->sourceBlocks);
    auto off=reinterpret_cast<__gm__ int32_t*>(ctx->sourceOffsets);
    auto dst=reinterpret_cast<__gm__ int32_t*>(ctx->destinations);
    auto valid=reinterpret_cast<__gm__ uint8_t*>(ctx->fallbackValid);
    auto addr=reinterpret_cast<__gm__ uint64_t*>(ctx->fallbackAddresses);
    for (uint32_t i=first+threadIdx.x;i<end;i+=256) {
        uint32_t q=i<c0 ? 0 : i<c0+c1 ? 1 : i<c0+c1+c2 ? 2 : 3;
        uint32_t prefix=q==0 ? 0 : q==1 ? c0 : q==2 ? c0+c1 : c0+c1+c2;
        uint32_t pos=b*kSlots+q*kTop+i-prefix;
        int32_t src=asc_ldcg(sources+pos), dest=asc_ldcg(dst+pos);
        uint64_t address=asc_ldcg(addr+pos);
        if (src!=int32_t(pos) || asc_ldcg(off+pos)!=0 || !asc_ldcg(valid+pos) ||
            address<remoteBase || remoteBytes<tokenBytes || address-remoteBase>remoteBytes-tokenBytes ||
            dest<0 || uint32_t(dest)>=outputRows || uint64_t(dest+uint64_t(1))*tokenBytes>outputBytes) {
            asc_atomic_or(error,1U); continue;
        }
        asc_stcg(out+i,src); asc_stcg(out+kSlots+i,0); asc_stcg(out+2*kSlots+i,dest);
    }
    asc_threadfence();
}

__aicore__ inline uint64_t RowErrors(__gm__ post_index_tail::Record* records,
                                    uint32_t b,uint32_t batch,uint32_t cores) {
    uint64_t error=0;
    for (uint32_t i=b;i<cores;i+=batch) {
        post_index_tail::Flush(records+i);
        error |= records[i].error;
    }
    return error;
}

#include "high_index_balance.h"

// Every block takes all joins, including empty plans and failed rows. The
// orders perform the same work; only the CQ wait/join moves around Index.
__aicore__ __attribute__((noinline)) inline void Run(__gm__ Args* a,uint32_t block) {
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    auto shape=LoadRuntimeShape(ctx);
    const uint32_t batch=shape.batch, cores=2*shape.aicCount;
    const uint32_t b=block%batch,lane=block/batch,lanes=(cores-1-b)/batch+1;
    const bool profile=(a->prefetchCount & 0x80000000U)!=0;
    const bool waitFirst=(a->prefetchCount & 0x02000000U)!=0;
    auto records=reinterpret_cast<__gm__ post_index_tail::Record*>(a->postIndex);
    auto r=records+block;
    auto ticks=reinterpret_cast<__gm__ uint64_t*>(a->times)+uint64_t(batch)*256+block*32;
    auto ws=reinterpret_cast<__gm__ uint32_t*>(a->workerStatus)+b*32;
    auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
    auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
    auto edges=reinterpret_cast<__gm__ Edge*>(a->highIndex+uint64_t(b)*kRowBytes);
    auto spill=reinterpret_cast<__gm__ Tuple*>(edges+kCapacity);
    // Tuple padding belongs to the private invalidation list, even in a spill.
    auto invalidations=&spill[0].padding;
    PrivatePlan plan;
    r->error=0; r->globalError=0; r->completed=0; r->count=0;
    r->packedRow=b; r->lane=lane; r->lanes=lanes;
    r->planDone=0; r->packDone=0; r->postBegin=0; r->postDone=0;
    r->indexDone=0; r->cqDone=0; r->start=profile ? GetSystemCycle() : 0;
    if (profile) { ticks[14]=0; ticks[15]=0; ticks[20]=0; ticks[21]=0; ticks[22]=0; }
    uint32_t useful=0;
    if (!lane) useful=Producer(a,b);
    else if (lane==1) OutputWorker<false>(a,b);
    else if (lane==2) {
        if (profile) ticks[14]=GetSystemCycle();
        Plan(a,b,edges,invalidations,profile,plan);
        if (profile) {
            ticks[15]=GetSystemCycle(); ticks[20]=2*plan.pairs+plan.singles;
            ticks[21]=plan.activeCycles; ticks[22]=plan.pairs;
        }
    }
    if (!lane) CompletePrefetch(ctx,b,true);
    SyncAll(); // original-address readers and all speculative CQs are finished
    if (profile) r->planDone=GetSystemCycle();

    auto posts=FinalPostAt(shape.finalPostOffset,ctx,4*b);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(posts),4*sizeof(FinalPost));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    uint64_t error=row->error | info[0] | plan.error;
    uint32_t count[4]={},total=0;
    for (uint32_t q=0;q<4;++q) {
        if (posts[q].count>kTop || posts[q].descriptor!=b*kSlots+q*kTop) error|=1;
        count[q]=posts[q].count<=kTop ? posts[q].count : 0;
        total+=count[q]; error|=posts[q].error;
    }
    if (total!=info[1]) error|=1;
    uint32_t begin=total*lane/lanes,end=total*(lane+1)/lanes,own=end-begin;
    auto packed=reinterpret_cast<__gm__ int32_t*>(a->packedFetch)+uint64_t(b)*3*kSlots;
    auto args=GetArgs(ctx,block);
    // Reuse the drained physical queue, never its first-phase completion
    // record: the service exposes those prefetch records through the lease.
    auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
    args.core_status=a->highIndex+uint64_t(batch)*kRowBytes+uint64_t(block)*update[15];
    args.block_count=1; args.jettys_per_core=1;
    args.indices=uint64_t(packed+begin); args.offsets=uint64_t(packed+kSlots+begin);
    args.output_indices=uint64_t(packed+2*kSlots+begin); args.request_capacity=own;
    args.address_table=ctx->fallbackAddresses; args.valid_table=ctx->fallbackValid;
    args.table_capacity=batch*kSlots; args.tokens_per_block=1;
    bool blocking=false;
    if (!error) error=post_index_transport::Preflight(args,0,own,blocking);
    if (!error && own) {
        auto entity=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
        TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,64);
        auto local=buf.Get<uint32_t>();
        Simt::VF_CALL<PackFinal>(Simt::Dim3{256,1,1},ctx,b,begin,end,count[0],count[1],count[2],packed,
             entity->remote_host_ddr_address,entity->remote_host_ddr_size,args.output_rows,
             entity->local_hbm_size,args.token_bytes,reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error=local.GetValue(0); pipe.Destroy();
    }
    r->error=error; r->count=own; r->fallback=uint64_t(blocking)|(waitFirst?2U:0U);
    if (profile) r->packDone=GetSystemCycle();
    post_index_tail::Flush(r);
    SyncAll();
    // Publish the rejected row once before any of its queues can be posted.
    if (!lane) {
        ws[0]=RowErrors(records,b,batch,cores) ? 1U : 0U;
        ws[1]=0; ws[2]=0; ws[3]=0;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    }
    SyncAll();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    error|=ws[0]; r->globalError=ws[0];
    post_index_transport::State transport;
    if (profile) r->postBegin=GetSystemCycle();
    if (!error) error=blocking ? post_index_transport::GetBlocking(args,0,own,transport)
                               : post_index_transport::Post(args,0,own,transport);
    if (profile) r->postDone=GetSystemCycle();
    r->error=error; post_index_tail::Flush(r);
    SyncAll();
    if (profile) ticks[3]=GetSystemCycle();
    if (waitFirst) {
        error|=post_index_transport::Wait(transport);
        if (profile) ticks[7]=r->cqDone=GetSystemCycle();
        r->error=error; post_index_tail::Flush(r);
        SyncAll();
    }
    if (profile) ticks[4]=GetSystemCycle();
    // The public launch requires 3*batch + 4*workers <= cores. The third
    // lane holds this row's private plan across both transport orderings.
    if (lane==2) {
        uint64_t rejected=RowErrors(records,b,batch,cores);
        ws[0]=rejected ? 1U : plan.error;
        ws[1]=total; ws[2]=plan.chunks;
        ws[3]=plan.pairs; ws[4]=plan.singles; ws[5]=plan.invalidations;
        ws[6]=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    }
    SyncAll(); // all Post/Wait permissions and private row descriptors are visible
    if (profile) ticks[11]=GetSystemCycle();
    ApplyGlobalIndex(a,ctx,b,block,batch,cores,lane,lanes,update,ticks,profile);
    if (profile) r->indexDone=ticks[5];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    // Wait-first already drained its queue before Index. Publish its results
    // at the Index join; both orders have exactly one Index join and one CQ join.
    if (waitFirst) {
        r->error=error|ws[0]; r->completed=transport.completed; r->active=2;
        post_index_tail::Flush(r);
    }
    SyncAll();
    if (profile) ticks[6]=GetSystemCycle();
    if (!waitFirst) {
        error|=post_index_transport::Wait(transport);
        if (profile) ticks[7]=r->cqDone=GetSystemCycle();
        r->error=error|ws[0]; r->completed=transport.completed; r->active=2;
        post_index_tail::Flush(r);
        SyncAll();
    }
    // Common endpoint after the last collective in either order.
    if (profile) ticks[13]=GetSystemCycle();
    if (!lane) {
        uint32_t completed=0;
        uint64_t failed=RowErrors(records,b,batch,cores);
        for (uint32_t i=b;i<cores;i+=batch) completed+=uint32_t(records[i].completed);
        if (completed!=total) failed|=kvaf_device::kInvalidArgs;
        // FinalizeStreaming checks and publishes the existing transport ABI.
        for (uint32_t q=0;q<4;++q) {
            posts[q].pending=q==0?completed:0; posts[q].deferred=0;
            posts[q].error=uint32_t(failed);
        }
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(posts),4*sizeof(FinalPost));
        Simt::VF_CALL<PublishStreamingUpdateStatus>(Simt::Dim3{32,1,1},
            reinterpret_cast<__gm__ uint32_t*>(update[6]),b,failed ? 1U : 0U);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (!failed && (a->prefetchCount & 0x10000000U)) {
            Simt::VF_CALL<PublishStreamingResidency>(Simt::Dim3{256,1,1},shape.resident,
                shape.finalPostOffset,ctx,b);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        FinalizeStreaming<false,true>(a,b,useful);
    }
}
} // namespace high_post_index
} // namespace pipeline_ab
