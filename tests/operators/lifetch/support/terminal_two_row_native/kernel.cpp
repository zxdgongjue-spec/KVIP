// SPDX-License-Identifier: Apache-2.0
#include "row_kernel.h"
#if defined(__CCE_AICORE__)
#include "device.h"
namespace joint_tail {
// Every value has a single writer, a separate cache line and one use per invocation.
struct alignas(128) Word {uint64_t value; uint64_t pad[15];};
struct Mailbox {Word at[4], eligible[4], waiting[4], prepared[4], gate[4], done[4], child, next;};
struct Workspace {Word direct; Mailbox rows[32];};
static_assert(sizeof(Workspace)==128*(1+32*26),"joint workspace ABI");
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void Reset(__gm__ Workspace* ws){
    auto words=reinterpret_cast<__gm__ Word*>(ws);
    for(uint32_t i=threadIdx.x;i<1+32*26;i+=32)asc_stcg(&words[i].value,uint64_t(0));
    asc_threadfence();
}
__aicore__ inline uint64_t Load(__gm__ Word* w) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(w),sizeof(*w));
    return *reinterpret_cast<__gm__ volatile uint64_t*>(&w->value);
}
__aicore__ inline void Put(__gm__ Word* w,uint64_t v) {
    *reinterpret_cast<__gm__ volatile uint64_t*>(&w->value)=v;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(w),sizeof(*w));
}
__aicore__ inline uint64_t Wait(__gm__ Word* w) {uint64_t v;do{v=Load(w);}while(!v);return v;}
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void Claim(__gm__ TerminalRowSchedule* sched,__gm__ Word* out) {
    if(!threadIdx.x){
        const uint32_t rank=asc_atomic_add(&sched->next,1U);
        const uint64_t row=rank<32?asc_ldcg(sched->order+rank):32;
        asc_stcg(&out->value,row+1);asc_threadfence();
    }
}
__aicore__ inline uint32_t Next(__gm__ TerminalRowSchedule* sched,__gm__ Word* dest,uint32_t q) {
    if(!q){Simt::VF_CALL<Claim>(Simt::Dim3{32,1,1},sched,dest);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);}
    return uint32_t(Wait(dest)-1);
}
__aicore__ inline void Work(GM_ADDR binding,__gm__ Workspace* ws,uint64_t guardTicks,
                            uint32_t team,uint32_t q){
    auto r=reinterpret_cast<__gm__ terminal_runtime::Binding*>(binding);
    auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(r->pipeline);
    auto c=reinterpret_cast<__gm__ rotating::Context*>(a->context);
    if(c->shape.batch<=16){if(team<c->shape.batch)RunTerminalRow<false>(binding,team,q);return;}
    if(c->shape.batch!=32 || !ws || !guardTicks) {RunTerminalTeams(binding,team,q);return;}
    auto sched=reinterpret_cast<__gm__ TerminalRowSchedule*>(r->reserved[1]);
    if(!team && !q){
        Simt::VF_CALL<Reset>(Simt::Dim3{32,1,1},ws);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        Put(&ws->direct,DirectRows(a,c)?1:2);
        Simt::VF_CALL<PrepareTerminalTeams>(Simt::Dim3{32,1,1},sched,c,32);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    while(!LoadScheduleWord(&sched->ready)){}
    uint32_t row=LoadScheduleWord(sched->order+team);
    const bool direct=Load(&ws->direct)==1;
    while(row<32){
        __gm__ Mailbox* m=&ws->rows[row];
        if(!direct){
            RunTerminalRow<true>(binding,row,q);
            Simt::VF_CALL<AssignTerminalSuccessor>(Simt::Dim3{32,1,1},sched,row,32);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint32_t next;do{next=LoadScheduleWord(sched->successor+row);}while(!next);
            row=next-1;continue;
        }
        Saved old{},child{};
        RunTerminalRow<true,true>(binding,row,q,&old,false);
        auto p=Bind(r,c,row,q);
        Put(m->eligible+q,old.own>1 && old.count>1 && !old.error &&
                           !old.topError && !old.packError && !old.indexError?1:2);
        Put(m->at+q,GetSystemCycle());
        bool longRow=true;uint64_t last=0;
        for(uint32_t k=0;k<4;++k){
            const auto at=Wait(m->at+k);if(at>last)last=at;
            if(Wait(m->eligible+k)!=1)longRow=false;
        }
        if(!longRow)old.credit=1;
        if(longRow) PostOne(old,p);
        // If the first CQE is already visible, make progress without waiting for the deadline.
        bool waited=false;
        if(longRow && LoadScheduleWord(&sched->next)<32){
            while(Peek(old,p)==Probe::Empty){
                const auto now=GetSystemCycle();
                if(now>=last && now-last>=guardTicks){waited=true;break;}
                bool busy=true;for(uint32_t k=0;k<4;++k)if(Load(m->waiting+k)==2)busy=false;
                if(!busy)break;
            }
        }
        Put(m->waiting+q,waited?1:2);
        bool park=longRow;
        for(uint32_t k=0;k<4;++k)if(Wait(m->waiting+k)!=1)park=false;
        uint32_t other=32;
        if(park)other=Next(sched,&m->child,q);
        const bool pair=other<32;
        if(pair)RunTerminalRow<true,true>(binding,other,q,&child,true);
        auto p2=Bind(r,c,pair?other:row,q);
        bool vote=false,admit=false,abort=false;
        if(pair){
            Put(m->prepared+q,child.topError||child.packError||child.indexError||child.error?2:1);
            for(uint32_t k=0;k<4;++k)if(Wait(m->prepared+k)==2)abort=true;
        }
        for(;;){
            PostOne(old,p);PollOne(old,p);
            if(pair){
                if(old.error && Load(m->done+q)!=2)Put(m->done+q,2);
                for(uint32_t k=0;k<4;++k)if(Load(m->done+k)==2)abort=true;
                if(!vote && (Gate(old)||old.error)){
                    Put(m->gate+q,old.error?2:1);vote=true;
                }
                if(!admit && !abort){
                    uint32_t ready=0;
                    for(uint32_t k=0;k<4;++k){const auto v=Load(m->gate+k);
                        ready+=v==1;if(v==2)abort=true;}
                    admit=ready==4;
                }
                if(abort)Poison(child,p2.Invalid());
                if(admit)PostOne(child,p2);
                PollOne(child,p2);
                EarlyOwnPrefetch(c,old);EarlyOwnPrefetch(c,child);
            }
            if(Settled(old) && (!pair || ((admit||abort)&&Settled(child))))break;
        }
        if(pair){
            Put(m->done+q,old.error||old.topError||old.packError||old.indexError?2:1);
            for(uint32_t k=0;k<4;++k)if(Wait(m->done+k)==2)Poison(child,p2.Invalid());
        }
        Retire(r,a,c,old,p);
        if(pair)Retire(r,a,c,child,p2);
        row=Next(sched,&m->next,q);
    }
    if(!q){
        Simt::VF_CALL<RetireTerminalTeam>(Simt::Dim3{32,1,1},sched);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}
}
#endif
extern "C" __global__ __aicore__ void v9_test_two_row_terminal(GM_ADDR binding,GM_ADDR ws,uint64_t guard){
#if defined(__CCE_AICORE__)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIV {joint_tail::Work(binding,reinterpret_cast<__gm__ joint_tail::Workspace*>(ws),guard,GetBlockIdx()/4,GetBlockIdx()%4);}
#endif
}
