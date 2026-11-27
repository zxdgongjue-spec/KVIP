// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "binding_guard.h"
namespace joint_tail {
struct Port {
    __gm__ KvafExecuteBatchGetKernelArgs* args;
    __gm__ KvafBatchGetDirectUrmaEntity* entity;
    __gm__ KvafBatchGetQueueState* queue;
    __gm__ rotating::FinalPost* post;
    __aicore__ inline uint64_t Key(){return args->queue_state;}
    __aicore__ inline uint64_t Producer(){return queue->sq_producer_index;}
    __aicore__ inline uint32_t Depth() { return entity->cq_depth; }
    __aicore__ inline uint64_t Cursor() { return queue->cq_consumer_index; }
    __aicore__ inline ::CqAccess Access() {
        return {reinterpret_cast<__gm__ uint8_t*>(entity->cq_va),entity->cqe_size,
                reinterpret_cast<__gm__ uint32_t*>(entity->cq_db_va),queue};
    }
    __aicore__ inline uint32_t Read(uint32_t slot) { return Access().Read(slot); }
    __aicore__ inline uint32_t Limit() { return kvaf_device::kMaxCqPolls*::kFusedCqPollMultiplier; }
    __aicore__ inline uint64_t Timeout(){return kvaf_device::kCqTimeout;}
    __aicore__ inline uint64_t Invalid(){return kvaf_device::kInvalidArgs;}
    __aicore__ inline uint64_t Failure(){return kvaf_device::kCqError;}
    __aicore__ inline void Issue(uint64_t head){
        rotating::PublishTerminalPrefix(args,entity,head);post->count=uint32_t(head-post->head);++post->pending;
    }
    __aicore__ inline void Consume(uint64_t next,bool success){
        Access().Commit(next);--post->pending;if(success)++queue->completed_batch_count;
    }
};
__aicore__ inline Port Bind(__gm__ terminal_runtime::Binding* r,__gm__ rotating::Context* c,uint32_t row,uint32_t q) {
    const auto i=row*4+q;auto a=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(r->queueArgs)+i;
    return {a,reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(a->channel_entity),
            reinterpret_cast<__gm__ KvafBatchGetQueueState*>(a->queue_state),
            reinterpret_cast<__gm__ rotating::FinalPost*>(c->owners+c->shape.finalPostOffset)+i};
}
__aicore__ inline void EarlyOwnPrefetch(__gm__ rotating::Context* c,Saved& s){
    if(s.query!=1 || s.checkedPrefetch || !Settled(s))return;
    s.checkedPrefetch=true;
    auto row=reinterpret_cast<__gm__ rotating::Row*>(c->rows)+s.row;
    auto a=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(c->args)+2*s.row;
    auto e=reinterpret_cast<__gm__ KvafBatchGetDirectUrmaEntity*>(a->channel_entity);
    auto q=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(a->queue_state);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(&row->pending),sizeof(row->pending));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(q),sizeof(*q));
    ::CqAccess pre{reinterpret_cast<__gm__ uint8_t*>(e->cq_va),e->cqe_size,
                   reinterpret_cast<__gm__ uint32_t*>(e->cq_db_va),q};
    if(PrefetchVisible(pre,q->cq_consumer_index,e->cq_depth,row->pending)){
        rotating::CompleteTerminalPrefetch(c,s.row);s.prefetched=true;
    }
}
__aicore__ inline void Retire(__gm__ terminal_runtime::Binding* r,__gm__ pipeline_ab::Args* a,
    __gm__ rotating::Context* c,Saved& s,Port& p) {
    if(!StartRetirement(s))return;
    p.post->error=uint32_t(s.error);p.post->pending=s.issued-s.seen;
    if(s.error){p.queue->last_error=s.error;p.queue->poisoned=1;}
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(p.args->core_status);
    kvaf_device::WriteStatus(status,s.error,p.post->count,s.success,p.queue->sq_producer_index,
                             p.queue->cq_consumer_index,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p.queue),sizeof(*p.queue));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(p.post),sizeof(*p.post));
    auto ts=reinterpret_cast<__gm__ uint64_t*>(r->times)+terminal_runtime::QueryTraceOffset(s.row,s.query);
    const bool profile=(r->options&1U)!=0;
    if(profile)ts[24]=GetSystemCycle();
    if(s.query==1 && !s.prefetched)rotating::CompleteTerminalPrefetch(c,s.row);
    if(profile){ts[17]=GetSystemCycle();ts[9]=GetSystemCycle();}
    auto flags=reinterpret_cast<__gm__ uint32_t*>(r->flags)+terminal_runtime::RowFlagOffset(s.row);
    auto d=flags+(28+s.query)*32;
    auto rows=reinterpret_cast<__gm__ rotating::Row*>(c->rows);
    const auto pre=s.query==1?rows[s.row].error:0;
    d[0]=d[1]=s.topError;d[2]=s.packError;d[3]=s.indexError;d[4]=p.post->error;
    d[5]=pre;d[6]=s.total;d[7]=s.first;d[8]=s.n;d[9]=s.own;d[10]=p.post->pending;d[11]=s.generation;
    if(!s.indexError || !s.n)for(uint32_t i=12;i<=16;++i)d[i]=UINT32_MAX;
    d[0]=s.topError|s.packError|(s.indexError<<4)|uint32_t(s.error)|pre;
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(d),128);
    TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,128);
    auto work=reinterpret_cast<__ubuf__ uint32_t*>(scratch.Get<uint32_t>().GetPhyAddr());
    Simt::VF_CALL<FinishTerminalRow<true>>(Simt::Dim3{512,1,1},a,c,r,s.row,s.query,s.owner,s.total,s.first,
                                         s.n,flags,s.generation,s.scan,work);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if(profile){ts[10]=GetSystemCycle();kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ts),256);}
    s.phase=Phase::Done;pipe.Destroy();
}
}
