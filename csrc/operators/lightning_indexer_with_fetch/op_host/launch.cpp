#include "../../common/lifetch/independent_schedule.h"
#include "../op_kernel/pipeline_types.h"
#include "api.h"
#include "context.h"
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }
#define ENTRY(name, blocks) \
extern "C" uint32_t aclrtlaunch_v9_##name##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t v9_##name(void* bindings,void* stream,const rotating::RuntimeShape* shape) { \
    if (!bindings || !stream || !shape) return 100000; \
    return aclrtlaunch_v9_##name##_kernel(blocks,stream,bindings); }
ENTRY(init,shape->batch)

extern "C" uint32_t aclrtlaunch_v9_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_3(uint32_t,void*,void*);
extern "C" uint32_t v9_li(void* bindings, void* stream, const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0 aclrtlaunch_v9_li_kernel
#define LI_LAUNCH_1 aclrtlaunch_v9_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_v9_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_v9_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
}
extern "C" uint32_t v9_li_cores(const rotating::RuntimeShape* shape) { return shape->liCores; }
extern "C" uint32_t v9_batch(const rotating::RuntimeShape* shape) { return shape->batch; }
extern "C" uint32_t v9_history(const rotating::RuntimeShape* shape) { return shape->history; }
extern "C" uint32_t v9_resident(const rotating::RuntimeShape* shape) { return shape->resident; }
extern "C" uint32_t v9_row_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Row); }
extern "C" uint32_t v9_context_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Context); }
extern "C" uint32_t v9_rounds(const rotating::RuntimeShape* shape) { return shape->rounds; }
extern "C" uint32_t v9_simulation(const rotating::RuntimeShape* shape) { return 0; }
extern "C" uint32_t v9_abi_version(const rotating::RuntimeShape* shape) { return 11; }
extern "C" uint32_t v9_aic_count(const rotating::RuntimeShape* shape) { return shape->aicCount; }
extern "C" uint32_t v9_service_lanes(const rotating::RuntimeShape* shape) { return shape->serviceLanes; }
extern "C" uint32_t v9_final_workers(const rotating::RuntimeShape* shape) { return shape->finalWorkers; }
extern "C" uint32_t v9_final_owner_bytes(const rotating::RuntimeShape* shape) { return shape->finalOwnerBytes; }
extern "C" uint32_t v9_final_replicated(const rotating::RuntimeShape* shape) { return 1; }
extern "C" uint32_t v9_token_bytes(const rotating::RuntimeShape* shape) { return shape->tokenBytes; }

#include "api.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include "batch_get_abi.h"

// Native header is the source of truth for structure sizes and padding.
extern "C" uint32_t refactor_fetch_args_bytes() { return sizeof(KvafExecuteBatchGetKernelArgs); }
extern "C" uint32_t refactor_fetch_status_bytes() { return sizeof(KvafBatchGetCoreStatus); }
extern "C" uint32_t refactor_fetch_entity_bytes() { return sizeof(KvafBatchGetDirectUrmaEntity); }
extern "C" uint32_t refactor_fetch_queue_bytes() { return sizeof(KvafBatchGetQueueState); }

extern "C" uint32_t refactor_fetch_pack_args(void *output, uint32_t capacity,
                                             const uint64_t *fields, uint32_t count)
{
    if (!output || !fields || (count != 17 && count != 18) || capacity != sizeof(KvafExecuteBatchGetKernelArgs)) return 1;
    for (uint32_t i = 10; i < count; ++i)
        if (fields[i] > std::numeric_limits<uint32_t>::max()) return 2;
    KvafExecuteBatchGetKernelArgs args{};
    args.address_table = fields[0]; args.valid_table = fields[1];
    args.indices = fields[2]; args.offsets = fields[3]; args.miss_count = fields[4];
    args.output_indices = fields[5]; args.output = fields[6]; args.core_status = fields[7];
    args.channel_entity = fields[8]; args.queue_state = fields[9];
    args.table_capacity = fields[10]; args.request_capacity = fields[11];
    args.output_rows = fields[12]; args.token_bytes = fields[13];
    args.tokens_per_block = fields[14]; args.block_count = fields[15];
    args.jettys_per_core = fields[16];
    if (count == 18) args.batch_get_max_inflight_total = fields[17];
    std::memcpy(output, &args, sizeof(args));
    return 0;
}

extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update(uint32_t,void*,void*);
extern "C" uint32_t kvip_lifetch_streaming_forward_v2(void* bindings,void* update,void* workspace,void* stream,const rotating::RuntimeShape* shape,uint32_t workers,uint32_t aivBudget) {
    if(!bindings||!update||!workspace||!stream||!shape||shape->resident>131072||shape->batch<1||shape->batch>16||shape->finalWorkers!=4)return 100000;
    if(rotating::ValidateStreamingBudget(shape->batch,shape->aicCount,aivBudget,shape->liCores,workers))return 100000;
    uint32_t result=v9_init(bindings,stream,shape);
    if(!result)result=v9_li(bindings,stream,shape);
    if(!result)result=aclrtlaunch_v9_streaming_resolve_update(shape->batch*pipeline_ab::kResolveCores+4*workers,stream,workspace);
    return result;
}

extern "C" uint32_t aclrtlaunch_kvip_transport_snapshot_kernel(uint32_t,void*,void*,void*,uint32_t);
extern "C" uint32_t kvip_transport_snapshot(void* source, void* destination, uint32_t bytes, void* stream) {
    if (!source || !destination || !stream || !bytes || bytes % 32 ||
        reinterpret_cast<uintptr_t>(source) % 32 || bytes > 64U * 1024U * 1024U) return 100000;
    return aclrtlaunch_kvip_transport_snapshot_kernel(1,stream,source,destination,bytes);
}

extern "C" uint32_t aclrtlaunch_v9_counter_calibration(uint32_t,void*,void*);
extern "C" uint32_t kvip_counter_calibration(void* output,void* stream) {
    if(!output || !stream)return 100000;
    return aclrtlaunch_v9_counter_calibration(1,stream,output);
}

extern "C" uint32_t aclrtlaunch_v9_query_fetch_only(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t kvip_query_fetch_only(void* workspace,void* stream,uint32_t query) {
    const uint32_t selected=query & 255U;
    if(!workspace||!stream||selected>4||(query & ~511U))return 100000;
    return aclrtlaunch_v9_query_fetch_only(selected==4?8:2,stream,workspace,query);
}

extern "C" uint32_t aclrtlaunch_v9_native_fetch_only(uint32_t,void*,void*,void*,uint32_t);
extern "C" uint32_t kvip_native_fetch_only(void* workspace,void* counts,void* stream,uint32_t query) {
    if(!workspace||!counts||!stream||query>4)return 100000;
    return aclrtlaunch_v9_native_fetch_only(query==4?8:2,stream,workspace,counts,query);
}

extern "C" uint32_t kvip_streaming_resolver_cores(){return pipeline_ab::kResolveCores;}

extern "C" uint32_t kvip_streaming_flag_count(){return pipeline_ab::kFlagCount;}

// Compatibility entry for old callers. New bindings use the explicit V budget.
extern "C" uint32_t kvip_lifetch_streaming_forward(void* bindings,void* update,void* workspace,
 void* stream,const rotating::RuntimeShape* shape,uint32_t workers) {
 return kvip_lifetch_streaming_forward_v2(bindings,update,workspace,stream,shape,workers,shape?2*shape->aicCount:0);
}

#include "../../common/lifetch/staged_schedule.h"
extern "C" uint32_t aclrtlaunch_v9_staged_prepare(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_score(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_scatter(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_topk(uint32_t,void*,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_progress(uint32_t,void*,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_service(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_staged_clear_packed(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_streaming_owner_wave(uint32_t,void*,void*,uint32_t,uint32_t);
extern "C" uint32_t kvip_prepare_staged_shape(rotating::RuntimeShape* shape,uint32_t v,uint32_t w) {
    return kvip_staged::PrepareShape(shape,v,w);
}
// All task arrays are immutable host inputs, prepared before graph capture.
// Device wave bindings reference original FP8 input pages and private scratch.
extern "C" uint32_t kvip_lifetch_staged_forward(void* bindings,void* workspace,void* stream,
    const rotating::RuntimeShape* s,const kvip_staged::Plan* p) {
    if(!bindings || !workspace || !stream || !s || !p || !p->canonical || !p->hot || !p->cold ||
       !p->hotCount || !p->coldCount || s->finalWorkers!=4 || s->splitQ || s->serviceLanes!=2 ||
       s->serviceAics!=1 || s->batch<1 || s->batch>32 || s->rounds<2) return 100000;
    rotating::RuntimeShape prepared=*s;
    if(kvip_staged::PrepareShape(&prepared,p->aivBudget,p->fetchWorkers) || std::memcmp(&prepared,s,sizeof(*s)))return 100000;
    uint32_t capacity=kvip_staged::OwnerCapacity(p->aivBudget,p->fetchWorkers);
    if(!capacity || p->aivBudget<s->batch)return 100000;
    auto validate=[&](const kvip_staged::ScoreWave* ws,uint32_t n){
        for(uint32_t i=0;i<n;++i)if(!ws[i].bindings || !ws[i].scores || !ws[i].tasks || !ws[i].rows ||
            ws[i].rows>s->aicCount || 2*ws[i].rows>p->aivBudget || ws[i].stride%128 || !ws[i].stride ||
            (ws[i].queryGroup!=2 && ws[i].queryGroup!=4))return false;
        return true;
    };
    if(!validate(p->hot,p->hotCount) || !validate(p->cold,p->coldCount))return 100000;
    // Validate every epoch before mutating device state.
    uint32_t epochs=0,last=s->resident;
    for(uint32_t i=0;i<p->coldCount;++i)if(p->cold[i].readyEnd){
        uint32_t e=p->cold[i].readyEnd;
        if(e<=last || e>s->history || e%128)return 100000;
        last=e;++epochs;
    }
    if(last!=s->history || epochs!=s->rounds-1)return 100000;
    uint32_t result=v9_init(bindings,stream,s);
    auto scoreWave=[&](const kvip_staged::ScoreWave& w){
        uint32_t r=aclrtlaunch_v9_staged_prepare(w.rows,stream,bindings,reinterpret_cast<void*>(w.bindings),reinterpret_cast<void*>(w.tasks),w.rows,w.stride,w.queryGroup);
        if(!r)r=aclrtlaunch_v9_staged_score(w.rows,stream,reinterpret_cast<void*>(w.bindings),w.rows,w.stride,s->indexHeads,w.queryGroup);
        if(!r)r=aclrtlaunch_v9_staged_scatter(w.rows,stream,reinterpret_cast<void*>(w.scores),reinterpret_cast<void*>(p->canonical),
            reinterpret_cast<void*>(w.tasks),w.rows,w.stride,s->history,w.queryGroup);
        return r;
    };
    for(uint32_t i=0;!result && i<p->hotCount;++i)result=scoreWave(p->hot[i]);
    for(uint32_t q=0;!result && q<s->batch*4;q+=p->aivBudget){
        uint32_t n=s->batch*4-q;if(n>p->aivBudget)n=p->aivBudget;
        result=aclrtlaunch_v9_staged_topk(n,stream,bindings,reinterpret_cast<void*>(p->canonical),q,1);
    }
    if(!result)result=aclrtlaunch_v9_staged_progress(s->batch,stream,bindings,reinterpret_cast<void*>(p->canonical),s->resident,s->resident);
    if(!result)result=aclrtlaunch_v9_staged_service(s->serviceLanes,stream,bindings,0);
    uint32_t begin=s->resident,epoch=1;
    for(uint32_t i=0;!result && i<p->coldCount;++i){
        result=scoreWave(p->cold[i]);
        if(!result && p->cold[i].readyEnd){
            uint32_t end=p->cold[i].readyEnd;
            result=aclrtlaunch_v9_staged_progress(s->batch,stream,bindings,reinterpret_cast<void*>(p->canonical),begin,end);
            if(!result)result=aclrtlaunch_v9_staged_service(s->serviceLanes,stream,bindings,epoch++);
            begin=end;
        }
    }
    for(uint32_t q=0;!result && q<s->batch*4;q+=p->aivBudget){
        uint32_t n=s->batch*4-q;if(n>p->aivBudget)n=p->aivBudget;
        result=aclrtlaunch_v9_staged_topk(n,stream,bindings,reinterpret_cast<void*>(p->canonical),q,0);
    }
    for(uint32_t b=0;!result && b<s->batch;b+=capacity){
        uint32_t n=s->batch-b;if(n>capacity)n=capacity;
        result=aclrtlaunch_v9_staged_clear_packed(4*p->fetchWorkers,stream,workspace);
        if(!result)result=aclrtlaunch_v9_streaming_owner_wave(3*n+4*p->fetchWorkers,stream,workspace,b,n);
    }
    return result;
}
