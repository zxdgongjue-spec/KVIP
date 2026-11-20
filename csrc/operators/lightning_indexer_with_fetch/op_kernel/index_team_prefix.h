// SPDX-License-Identifier: Apache-2.0
#pragma once

// The fourth field is exactly the last 32 bytes of each Index key. Save
// chunk zero before Q4 finishes taking its address snapshot. Never write
// keys until the whole row plan AND that snapshot succeed.
constexpr uint32_t kStagedOwner = 9*kMaxMoves;
template<bool Gather>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StageFirstFourth(
        __gm__ uint32_t* edges,__gm__ uint64_t* keys,__ubuf__ uint32_t* work,
        uint32_t count,uint32_t block) {
    auto values=reinterpret_cast<__ubuf__ uint64_t*>(work);
    auto destinations=work+8*kMaxMoves;
    const uint32_t tid=threadIdx.x;
    if constexpr(Gather) {
        // Only take a fresh, untouched field. Leave a previously handed-off
        // suffix to its normal consumer. No losing reader races a writer.
        if(!tid)work[kStagedOwner]=asc_atomic_cas(edges+kClaimOffset,0U,block+1)==0;
        asc_syncthreads();
        if(!work[kStagedOwner])return;
        for(uint32_t x=tid;x<count*4;x+=256){
            uint32_t i=x/4, w=x%4;
            uint32_t src=asc_ldcg(edges+kIndexEdgesOffset+2*i);
            if(!w)destinations[i]=asc_ldcg(edges+kIndexEdgesOffset+2*i+1)&0x7fffffffU;
            values[x]=asc_ldcg(keys+uint64_t(src)*16+12+w);
        }
    } else {
        for(uint32_t x=tid;x<count*4;x+=256)
            asc_stcg(keys+uint64_t(destinations[x/4])*16+12+x%4,values[x]);
        asc_threadfence();
    }
}

// Process home using only registers while keeping another row's first
// chunk in UB. Every probe is non-blocking and the elected owner publishes
// exactly once. Empty or failed home plans go through the normal Copy.
__aicore__ inline bool TryReadyHome(__gm__ Args* a,uint32_t home,uint32_t block,uint32_t batch) {
    auto control=Control(a,home);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(control),sizeof(*control));
    if(!control->ready)return false;
    if(control->error || !control->count)return true;
    auto q4=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+home*kFlagCount+4;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(q4),kBatchFlagPayloadBytes);
    if(!q4->ready)return false;
    if(q4->error)return true;
    auto target=ProbeFourth(a,home,false);
    if(!target.remaining)return true;
    // A long home row must not monopolize the worker holding the peer's UB.
    // Its nominal Copy or any other helper can resume the ordered suffix.
    TailTicket ticket{};
    HelpOneChunk(a,block,batch,target,ticket);
    return true;
}

__aicore__ inline uint32_t PrefixSize(__gm__ Args* a,uint32_t row) {
    auto edges=Moves(a,row);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(edges),sizeof(*edges));
    uint32_t n=edges[0];
    return n<=kMaxMoves?n:0;
}

__aicore__ __attribute__((noinline)) inline bool CopySealedPrefix(__gm__ Args* a,
        uint32_t block,uint32_t batch) {
    const uint32_t home=block-3*batch;
    const uint32_t queues=4*((a->prefetchCount>>16)&0xffU);
    const uint32_t peer=home<2*queues?home-queues:home;
    auto homeFlag=Control(a,home);
    auto peerFlag=Control(a,peer);
    uint32_t row=home, count=0;
    bool profile=(a->prefetchCount&0x80000000U)!=0;
    auto times=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+32*block;
    if(profile){times[19]=0;times[24]=0;} // One staged row, even when the trace buffer is reused.
    uint64_t arrival=GetSystemCycle();
    // Observing the full plan wins ties. A late arrival can use the normal
    // register-only copy without allocating UB or claiming the field twice.
    while(true){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(homeFlag),sizeof(*homeFlag));
        if(peer!=home)kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(peerFlag),sizeof(*peerFlag));
        if(homeFlag->ready || (peer!=home && peerFlag->ready))return false;
        if(homeFlag->reserved[0])count=PrefixSize(a,home);
        if(peer!=home && peerFlag->reserved[0]){
            uint32_t n=PrefixSize(a,peer);
            if(n && n>=count){row=peer;count=n;}
        }
        if(count)break;
        if(GetSystemCycle()-arrival>5000000000ULL)return false;
    }
    auto control=Control(a,row);
    auto edges=Moves(a,row);
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    auto owners=reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners);
    uint32_t owner=owners?owners[row]:row;
    auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf,(kStagedOwner+8)*sizeof(uint32_t));
    auto local=buf.Get<uint32_t>();
    auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    uint64_t begin=profile?GetSystemCycle():0;
    Simt::VF_CALL<StageFirstFourth<true>>(Simt::Dim3{256,1,1},edges,keys,work,count,block);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(!local.GetValue(kStagedOwner)){pipe.Destroy();return false;}
    uint64_t readEnd=profile?GetSystemCycle():0;
    auto q4=reinterpret_cast<__gm__ BatchFlag*>(a->flags)+row*kFlagCount+4;
    uint32_t error=0;
    if(row==home){
        error=AwaitPublication(control);
        error|=AwaitPublication(q4);
    }else{
        // Aliased storage must not be changed by this helper while its peer
        // still holds a saved prefix. The caller services home after the peer.
        bool checkedHome=owners && owners[home]==owner;
        uint64_t wait=GetSystemCycle();
        while(true){
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(control),sizeof(*control));
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(q4),kBatchFlagPayloadBytes);
            if(control->ready && q4->ready){error=control->error|q4->error;break;}
            if(!checkedHome)checkedHome=TryReadyHome(a,home,block,batch);
            if(GetSystemCycle()-wait>5000000000ULL){error=8;break;}
        }
    }
    uint32_t chunks=control->ready?control->count:0;
    if(!error && (!chunks || chunks>=kIndexEdgesOffset))error=1;
    uint64_t committed=profile?GetSystemCycle():0;
    if(!error){
        Simt::VF_CALL<StageFirstFourth<false>>(Simt::Dim3{256,1,1},edges,keys,work,count,block);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        // A later chunk may read the just-written positions.
        if(chunks>1)CopyOwnedFourth(a,row,1,chunks);
    }
    if(profile){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(edges),kIndexEdgesOffset*4);
        auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+4*batch*32;
        TraceField(trace,batch,row,3,block,error?0:edges[chunks-1],arrival,begin,GetSystemCycle(),error);
        times[16]=begin;times[17]=readEnd;times[18]=committed;times[19]=count;times[20]=row+1;
    }
    Publish(control+4,0,error);
    pipe.Destroy();
    // A peer never replaces the nominal home claimant. Re-enter Copy even
    // when home was helped; the claim makes that second look harmless.
    if(row!=home)return false;
    HelpAfterLocalWork(a,block,batch);
    return true;
}
