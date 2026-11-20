// SPDX-License-Identifier: Apache-2.0
#pragma once

// The fourth field is exactly the last 32 bytes of each Index key. Save
// chunk zero before Q4 finishes taking its address snapshot. Never write
// keys until the whole row plan AND that snapshot succeed.
constexpr uint32_t kStagedOwner = 9*kMaxMoves;
template<bool Gather,bool Elect=Gather>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StageFirstFourth(
        __gm__ uint32_t* edges,__gm__ uint64_t* keys,__ubuf__ uint32_t* work,
        uint32_t first,uint32_t end,uint32_t block) {
    auto values=reinterpret_cast<__ubuf__ uint64_t*>(work);
    auto destinations=work+8*kMaxMoves;
    const uint32_t tid=threadIdx.x;
    if constexpr(Gather) {
        // Only take a fresh, untouched field. Leave a previously handed-off
        // suffix to its normal consumer. No losing reader races a writer.
        if constexpr(Elect){
            if(!tid)work[kStagedOwner]=asc_atomic_cas(edges+kClaimOffset,0U,block+1)==0;
            asc_syncthreads();
            if(!work[kStagedOwner])return;
        }
        for(uint32_t x=first*4+tid;x<end*4;x+=256){
            uint32_t i=x/4, w=x%4;
            uint32_t src=asc_ldcg(edges+kIndexEdgesOffset+2*i);
            if(!w)destinations[i]=asc_ldcg(edges+kIndexEdgesOffset+2*i+1)&0x7fffffffU;
            values[x]=asc_ldcg(keys+uint64_t(src)*16+12+w);
        }
    } else {
        for(uint32_t x=first*4+tid;x<end*4;x+=256)
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

// A spare AIV may have an immutable chunk for each of its two distinct
// requests. Only the fourth key field is owned; no key or metadata becomes
// visible until that request's complete plan and Q4 snapshot are successful.
constexpr uint32_t kBufferWords=kStagedOwner+8, kPeerReadSlice=256;
struct StagedRow {
    __gm__ BatchFlag* plan;
    __gm__ BatchFlag* snapshot;
    __gm__ uint32_t* edges;
    __gm__ uint64_t* keys;
    __ubuf__ uint32_t* work;
    uint32_t row, count, gathered=0, next=0, chunks=0;
    uint64_t begin=0, readEnd=0, scatter=0;
    bool done=false;
};

__aicore__ inline bool BothReady(const StagedRow& p) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p.plan),sizeof(*p.plan));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p.snapshot),kBatchFlagPayloadBytes);
    return p.plan->ready && p.snapshot->ready;
}

__aicore__ inline StagedRow BindStagedRow(__gm__ Args* a,uint32_t row,uint32_t count,__ubuf__ uint32_t* work) {
    auto ctx=reinterpret_cast<__gm__ Context*>(a->context);
    auto owners=reinterpret_cast<__gm__ uint32_t*>(ctx->rowOwners);
    uint32_t owner=owners?owners[row]:row;
    return {Control(a,row),reinterpret_cast<__gm__ BatchFlag*>(a->flags)+row*kFlagCount+4,
            Moves(a,row),reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*a->history*16,
            work,row,count};
}

template<bool Elect>
__aicore__ inline void ReadStage(StagedRow& p,uint32_t end,uint32_t block) {
    Simt::VF_CALL<StageFirstFourth<true,Elect>>(Simt::Dim3{256,1,1},p.edges,p.keys,p.work,p.gathered,end,block);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    p.gathered=end;
}

__aicore__ inline void ScatterStage(StagedRow& p,uint32_t block,bool profile) {
    p.scatter=profile?GetSystemCycle():0;
    Simt::VF_CALL<StageFirstFourth<false>>(Simt::Dim3{256,1,1},p.edges,p.keys,p.work,0U,p.count,block);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    p.next=1;
}

__aicore__ inline void FinishStage(__gm__ Args* a,uint32_t block,uint32_t batch,StagedRow& p,
        uint32_t error,uint64_t arrival,__gm__ uint64_t* stamp,bool profile) {
    if(profile){
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p.edges),kIndexEdgesOffset*4);
        auto trace=reinterpret_cast<__gm__ uint64_t*>(a->times)+batch*256+4*batch*32;
        TraceField(trace,batch,p.row,3,block,error?0:p.edges[p.chunks-1],arrival,p.begin,GetSystemCycle(),error);
        stamp[0]=p.begin;stamp[1]=p.readEnd;stamp[2]=error?GetSystemCycle():p.scatter;stamp[3]=p.count;stamp[4]=p.row+1;
    }
    Publish(p.plan+4,0,error);
    p.done=true;
}

__aicore__ inline uint32_t PlanError(StagedRow& p) {
    p.chunks=p.plan->count;
    return p.plan->error|p.snapshot->error|uint32_t(!p.chunks || p.chunks>=kIndexEdgesOffset);
}

struct SliceStats {
    uint32_t maxMoves=0,slices=0;
    uint64_t longest=0,total=0;
};

template<bool Elect>
__aicore__ inline void ReadPeerSlice(StagedRow& p,uint32_t block,bool profile,SliceStats& stats) {
    uint32_t n=p.count-p.gathered;
    if(n>kPeerReadSlice)n=kPeerReadSlice;
    uint64_t start=profile?GetSystemCycle():0;
    ReadStage<Elect>(p,p.gathered+n,block);
    if(profile){
        uint64_t end=GetSystemCycle();
        if(!stats.slices)p.begin=start;
        p.readEnd=end;
        if(n>stats.maxMoves)stats.maxMoves=n;
        if(end-start>stats.longest)stats.longest=end-start;
        stats.total+=end-start;++stats.slices;
    }
}

// Each row keeps its own claim and UB for the entire operation. At a ready
// boundary, commit the first chunks first. Limit other work before a recheck
// to one gather slice or one dependent suffix chunk.
__aicore__ __attribute__((noinline)) inline void FinishPair(__gm__ Args* a,uint32_t block,uint32_t batch,
        StagedRow* rows,SliceStats& stats,uint64_t arrival,__gm__ uint64_t* times,bool profile) {
    uint32_t turn=0;
    while(!rows[0].done || !rows[1].done){
        for(uint32_t i=0;i<2;++i){
            auto& p=rows[i];
            if(p.done || p.next)continue;
            uint32_t error=0;
            if(BothReady(p))error=PlanError(p);
            else if(GetSystemCycle()-arrival>5000000000ULL)error=8;
            else continue;
            if(!error && p.gathered!=p.count)continue;
            if(!error)ScatterStage(p,block,profile);
            if(error || p.next==p.chunks)FinishStage(a,block,batch,p,error,arrival,times+16+5*i,profile);
        }
        // A waiting second row can use spare issue slots for a bounded
        // read. A long ordered suffix must give the other row another turn.
        for(uint32_t j=0;j<2;++j){
            uint32_t i=(turn+j)%2;
            auto& p=rows[i];
            if(p.done)continue;
            if(i==1 && p.gathered<p.count){
                ReadPeerSlice<false>(p,block,profile,stats);
            }else if(p.next){
                // Once the partner has retired there is no wakeup to serve.
                // The existing single VF handles all remaining ordered chunks.
                const uint32_t end=rows[i^1].done?p.chunks:p.next+1;
                CopyOwnedFourth(a,p.row,p.next,end);
                p.next=end;
                if(end==p.chunks)FinishStage(a,block,batch,p,0,arrival,times+16+5*i,profile);
            }else continue;
            turn=i^1;break;
        }
    }
    if(profile){
        times[26]=stats.maxMoves;times[27]=stats.longest;times[28]=stats.slices;times[29]=stats.total;
    }
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
    if(profile){times[19]=0;times[24]=0;times[26]=times[27]=times[28]=times[29]=0;}
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
    const uint32_t other=row==home?peer:home;
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf,kBufferWords*(other==row?1:2)*sizeof(uint32_t));
    auto local=buf.Get<uint32_t>();
    auto work=reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr());
    StagedRow rows[2]{};
    auto& first=rows[0];
    first=BindStagedRow(a,row,count,work);
    first.begin=profile?GetSystemCycle():0;
    ReadStage<true>(first,count,block);
    if(!local.GetValue(kStagedOwner)){pipe.Destroy();return false;}
    first.readEnd=profile?GetSystemCycle():0;
    uint32_t error=0;
    bool homeChecked=false,triedOther=other==row;
    auto otherFlag=Control(a,other);
    while(!BothReady(first)){
        if(!triedOther){
            kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(otherFlag),sizeof(*otherFlag));
            if(otherFlag->reserved[0] && !(otherFlag->ready && otherFlag->error)){
                triedOther=true;
                auto n=PrefixSize(a,other);
                if(n && !BothReady(first)){
                    auto& second=rows[1];
                    second=BindStagedRow(a,other,n,work+kBufferWords);
                    // Distinct packed rows can name the same storage owner.
                    if(second.keys==first.keys){homeChecked=true;continue;}
                    SliceStats stats{};
                    ReadPeerSlice<true>(second,block,profile,stats);
                    if(local.GetValue(kBufferWords+kStagedOwner)){
                        FinishPair(a,block,batch,rows,stats,arrival,times,profile);
                        pipe.Destroy();
                        HelpAfterLocalWork(a,block,batch);
                        return true;
                    }
                }
            }else if(otherFlag->ready)triedOther=true;
        }
        if(row!=home && !homeChecked)homeChecked=TryReadyHome(a,home,block,batch);
        if(GetSystemCycle()-arrival>5000000000ULL){error=8;break;}
    }
    if(!error)error=PlanError(first);
    if(!error){
        ScatterStage(first,block,profile);
        if(first.chunks>1)CopyOwnedFourth(a,row,1,first.chunks);
    }
    FinishStage(a,block,batch,first,error,arrival,times+16,profile);
    pipe.Destroy();
    if(row!=home)return false;
    HelpAfterLocalWork(a,block,batch);
    return true;
}
