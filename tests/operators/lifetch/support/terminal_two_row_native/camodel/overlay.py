"""Install specific synthetic inputs and CQ delivery inside the unchanged joint control loop."""
from pathlib import Path
import sys
source,out=map(Path,sys.argv[1:])
def install(name,changes):
    s=(source/name).read_text()
    for old,new in changes:
        if s.count(old)!=1:raise ValueError(f"{name}: missing or duplicate hook: {old}")
        s=s.replace(old,new)
    (out/name).write_text(s)

install("kernel.cpp",[
('#include "device.h"','#include "device.h"'),
('__aicore__ inline void Work(','''#include "probe_hooks.h"
__aicore__ inline void Work('''),
('Simt::VF_CALL<PrepareTerminalTeams>(Simt::Dim3{32,1,1},sched,c,32);',
 'Simt::VF_CALL<OneTeam>(Simt::Dim3{32,1,1},sched);'),
('RunTerminalRow<true,true>(binding,row,q,&old,false);','PrepareFixture(r,c,old,row,q,false);'),
('if(pair)RunTerminalRow<true,true>(binding,other,q,&child,true);',
 '''if(pair){
            auto rec=reinterpret_cast<__gm__ tiny::Trace*>(p.args->indices);
            rec->atPark=old.issued-old.seen;
            PrepareFixture(r,c,child,other,q,true);
        }'''),
('PostOne(old,p);PollOne(old,p);','PostOne(old,p);Supply(old,p);PollOne(old,p);'),
('PollOne(child,p2);','Supply(child,p2);PollOne(child,p2);'),
('row=NextAfterRetirement(sched,pair?other:row);',
 '''uint32_t retired=pair?other:row;
        auto& port=pair?p2:p;
        auto tr=reinterpret_cast<__gm__ tiny::Trace*>(port.args->indices);
        // Force Q3 for the non-park case, Q4 for the parked case to be the fourth arrival.
        const uint32_t slow=port.args->table_capacity==3?2:3;
        if(q==slow){
            while(LoadScheduleWord(sched->arrivals+retired)!=3){}
            tr->forcedFourth=1;
        }
        tr->expectedArrival=LoadScheduleWord(sched->arrivals+retired);
        tr->returnTick=GetSystemCycle();
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(tr),sizeof(*tr));
        row=NextAfterRetirement(sched,retired);'''),
('v9_test_two_row_terminal(', 'two_row_camodel(')
])
install("device.h",[
('#include "binding_guard.h"','#include "binding_guard.h"\n#include "fixture.h"'),
('return kvaf_device::kMaxCqPolls*::kFusedCqPollMultiplier;', 'return args->table_capacity==2 ? 8 : kvaf_device::kMaxCqPolls*::kFusedCqPollMultiplier;'),
('rotating::PublishTerminalPrefix(args,entity,head);','''auto tr=reinterpret_cast<__gm__ tiny::Trace*>(args->indices);
        if(args->request_capacity==tiny::second && !post->count){
            tr->firstChild=GetSystemCycle();
            auto gates=reinterpret_cast<__gm__ uint64_t*>(args->offsets);
            uint64_t mask=0;
            for(uint32_t k=0;k<4;++k){
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(gates+k*16),128);
                if(*reinterpret_cast<__gm__ volatile uint64_t*>(gates+k*16)==1)mask|=1U<<k;
            }
            tr->mask=mask;
        }
        rotating::PublishTerminalPrefix(args,entity,head);'''),
('Access().Commit(next);','''Access().Commit(next);
        if(args->request_capacity==tiny::first && next==tiny::cursor+3 && success)
            reinterpret_cast<__gm__ tiny::Trace*>(args->indices)->parentThird=GetSystemCycle();'''),
('if(s.query==1 && !s.prefetched)rotating::CompleteTerminalPrefetch(c,s.row);','''auto rec=reinterpret_cast<__gm__ tiny::Trace*>(p.args->indices);
    rec->earlyQ2=s.prefetched;
    if(s.query==1){
        auto pa=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(c->args)+2*s.row;
        auto e=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(pa->channel_entity);
        auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(pa->queue_state);
        rec->q2Start=queue->cq_consumer_index;
        if(s.row==tiny::second){
            // Second-row prefetch arrives only at retirement, after the early observation.
            for(uint32_t k=0;k<2;++k){uint64_t at=tiny::cursor+k;
                auto ptr=reinterpret_cast<__gm__ uint32_t*>(e->cq_va)+(at%8)*16;
                *ptr=uint32_t(1^((at/8)&1))<<2;
                kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ptr),64);}
        }
        if(!s.prefetched)rotating::CompleteTerminalPrefetch(c,s.row);
        rec->q2End=queue->cq_consumer_index;
    }'''),
('Simt::VF_CALL<FinishTerminalRow<true>>(Simt::Dim3{512,1,1},a,c,r,s.row,s.query,s.owner,s.total,s.first,\n                                         s.n,flags,s.generation,s.scan,work);',
 '''rec->issued=s.issued;rec->seen=s.seen;rec->success=s.success;rec->error=s.error;
    rec->pending=p.post->pending;++rec->retired;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(rec),sizeof(*rec));''')
])
