// SPDX-License-Identifier: Apache-2.0
#pragma once

// Experimental HIGH tail. Resolve the original eight-window events, drain
// speculative reads before posting any final overwrite, and compose the same
// ordered per-row permutation as Consumer. Only then can all AIVs safely move
// immutable tuples in parallel, including cycles and repeated victims.
namespace pipeline_ab {
namespace high_post_index {
constexpr uint32_t kCapacity = 2 * (kPrefetch + kSlots);
struct Move {
    uint32_t src, dst;
    uint64_t key[16], address, logical;
    uint32_t scale, padding;
};
static_assert(sizeof(Move) == 160, "must match optional host scratch allocation");
static_assert(offsetof(Args, highIndex) == 152, "high Index workspace address ABI");

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS)
inline void EmitMoves(__gm__ Move* moves, __gm__ uint32_t* dense, uint32_t history,
                      bool denseMode, __ubuf__ uint32_t* work) {
    uint32_t tid = threadIdx.x, count = 0;
    const uint32_t size = denseMode ? history : MAP;
    for (uint32_t base = 0; base < size; base += THREADS) {
        uint32_t i = base + tid;
        uint32_t dst = i, src = i;
        if (i < size) {
            if (denseMode) src = asc_ldcg(dense + i);
            else { dst = work[KEYS_POS+i]; src = dst == EMPTY ? dst : work[VALUES_POS+i]; }
        }
        const uint32_t changed = dst != src;
        uint32_t rank = prefix(changed, work+CONTROL+32);
        uint32_t found = work[CONTROL+32+15];
        if (count + found > kCapacity || (changed && (dst >= history || src >= history)))
            asc_atomic_or(work+CONTROL, 1U);
        asc_syncthreads();
        if (work[CONTROL]) return;
        if (changed) {
            asc_stcg(&moves[count+rank].src, src);
            asc_stcg(&moves[count+rank].dst, dst);
        }
        count += found;
        asc_syncthreads();
    }
    if (!tid) work[CONTROL+1] = count;
    asc_threadfence();
}

// One owner composes each row's sparse/dense map preserving the original
// prefetch/Q0/Q1/Q2/Q3 phase and 512-event chunk boundaries. Dirty destinations
// include displaced rows. No key/scale/address/logical writes here.
__aicore__ inline uint32_t Plan(__gm__ Args* a, uint32_t b, __gm__ Move* moves,
                               uint32_t& moveCount, uint32_t& chunks) {
    auto ctx = reinterpret_cast<__gm__ Context*>(a->context);
    auto shape = LoadRuntimeShape(ctx);
    uint32_t owner = ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
    auto src = reinterpret_cast<__gm__ uint32_t*>(a->eventsSrc)+uint64_t(b)*a->maxEvents;
    auto dst = reinterpret_cast<__gm__ uint32_t*>(a->eventsDst)+uint64_t(b)*a->maxEvents;
    auto valid = reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*shape.resident;
    auto dense = reinterpret_cast<__gm__ uint32_t*>(a->denseShadow)+uint64_t(b)*shape.history*6;
    auto flags = reinterpret_cast<__gm__ BatchFlag*>(a->flags)+b*kFlagCount;
    TPipe pipe; TBuf<TPosition::VECCALC> buf; pipe.InitBuffer(buf,SCRATCH_WORDS*4);
    auto local = buf.Get<uint32_t>();
    auto mem = reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    Simt::VF_CALL<InitShadow>(Simt::Dim3{THREADS,1,1},mem);
    SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    bool denseMode = false;
    uint32_t seen = 0, finalCount = 0, error = 0;
    for (uint32_t phase = 0; phase < 5 && !error; ++phase) {
        auto flag = flags+(phase ? kRawQueryFlag+phase-1 : 0);
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(flag),sizeof(*flag));
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
            Simt::VF_CALL<ApplyDenseBatch<false,true>>(Simt::Dim3{THREADS,1,1},src,dst,offset,n,
                absent,noScales,valid,shape.resident,dense,shape.history,mem,absent,absent,absent);
        else
            Simt::VF_CALL<ApplyBatch<false,true>>(Simt::Dim3{THREADS,1,1},src,dst,offset,n,
                absent,noScales,valid,shape.resident,mem,absent,absent,absent,shape.history);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error=local.GetValue(CONTROL); seen+=n;
        if (phase) finalCount+=n;
    }
    if (!error && seen) {
        Simt::VF_CALL<EmitMoves>(Simt::Dim3{THREADS,1,1},moves,dense,shape.history,denseMode,mem);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
        error=local.GetValue(CONTROL);
        if (!error) moveCount=local.GetValue(CONTROL+1);
    }
    chunks=local.GetValue(CONTROL+4);
    pipe.Destroy();
    return error;
}

template<bool Scatter>
__simt_vf__ __aicore__ LAUNCH_BOUND(256)
inline void MoveTuples(__gm__ Move* moves, uint32_t first, uint32_t end,
                       __gm__ uint64_t* keys, __gm__ uint32_t* scales,
                       __gm__ uint64_t* addresses, __gm__ uint64_t* logical) {
    const uint32_t field=threadIdx.x%16;
    for (uint32_t i=first+threadIdx.x/16; i<end; i+=16) {
        auto m=moves+i;
        const uint32_t pos=Scatter ? asc_ldcg(&m->dst) : asc_ldcg(&m->src);
        if constexpr (Scatter) {
            asc_stcg(keys+uint64_t(pos)*16+field,asc_ldcg(m->key+field));
            if (!field) {
                asc_stcg(scales+pos,asc_ldcg(&m->scale));
                asc_stcg(addresses+pos,asc_ldcg(&m->address));
                asc_stcg(logical+pos,asc_ldcg(&m->logical));
            }
        } else {
            asc_stcg(m->key+field,asc_ldcg(keys+uint64_t(pos)*16+field));
            if (!field) {
                asc_stcg(&m->scale,asc_ldcg(scales+pos));
                asc_stcg(&m->address,asc_ldcg(addresses+pos));
                asc_stcg(&m->logical,asc_ldcg(logical+pos));
            }
        }
    }
    asc_threadfence();
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
    auto moves=reinterpret_cast<__gm__ Move*>(a->highIndex)+uint64_t(b)*kCapacity;
    r->error=0; r->globalError=0; r->completed=0; r->count=0;
    r->packedRow=b; r->lane=lane; r->lanes=lanes;
    r->planDone=0; r->packDone=0; r->postBegin=0; r->postDone=0;
    r->indexDone=0; r->cqDone=0; r->start=profile ? GetSystemCycle() : 0;
    uint32_t useful=0;
    if (!lane) useful=Producer(a,b);
    else if (lane==1) OutputWorker<false>(a,b);
    if (!lane) CompletePrefetch(ctx,b,true);
    SyncAll(); // original-address readers and all speculative CQs are finished
    if (profile) r->planDone=GetSystemCycle();

    auto posts=FinalPostAt(shape.finalPostOffset,ctx,4*b);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(posts),4*sizeof(FinalPost));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info),128);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),64);
    uint64_t error=row->error | info[0];
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
    args.core_status=a->highIndex+uint64_t(batch)*kCapacity*sizeof(Move)+uint64_t(block)*update[15];
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
    if (!lane) {
        uint32_t count=0,chunks=0;
        uint64_t rejected=RowErrors(records,b,batch,cores);
        ws[0]=rejected ? 1U : Plan(a,b,moves,count,chunks);
        ws[1]=total; ws[2]=chunks; ws[3]=count;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    }
    SyncAll(); // all rows' final maps have been composed
    if (profile) ticks[11]=GetSystemCycle();
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ws),128);
    uint32_t n=ws[0]?0:ws[3],first=n*lane/lanes,last=n*(lane+1)/lanes;
    uint32_t owner=ctx->rowOwners ? reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners)[b] : b;
    auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*shape.history*16;
    auto scales=reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*shape.history;
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*update[7];
    auto logical=reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*shape.history;
    if (last>first) {
        Simt::VF_CALL<MoveTuples<false>>(Simt::Dim3{256,1,1},moves,first,last,keys,scales,addresses,logical);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    SyncAll(); // no writer can clobber an unread source, even in long cycles
    if (profile) ticks[12]=GetSystemCycle();
    if (last>first) {
        Simt::VF_CALL<MoveTuples<true>>(Simt::Dim3{256,1,1},moves,first,last,keys,scales,addresses,logical);
        SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    if (profile) {
        ticks[2]=0x50494831U; ticks[5]=r->indexDone=GetSystemCycle();
        ticks[9]=last-first; ticks[10]=last-first;
    }
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
