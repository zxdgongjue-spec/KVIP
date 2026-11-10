#include "api.h"
#include "context.h"

#define ENTRY(api,kernel,blocks) \
extern "C" uint32_t aclrtlaunch_kvip_ubmem_##kernel##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t api(void* bindings,void* stream) { \
    if(!bindings || !stream)return 100000; \
    return aclrtlaunch_kvip_ubmem_##kernel##_kernel(blocks,stream,bindings); }
ENTRY(v9_init,init,rotating::kInitCores)
ENTRY(v9_li,li,rotating::kLaunchCores)
ENTRY(v9_resolve,resolve,rotating::kFinalCores)
ENTRY(v9_fetch,complete,SIM_BATCH)

extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_outputs_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_apply_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_kernel(uint32_t,void*,void*,void*);
extern "C" uint32_t kvip_ubmem_complete_plan(void* bindings,void* update,void* stream) {
    if(!bindings || !update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_complete_plan_kernel(SIM_BATCH,stream,bindings,update);
}
extern "C" uint32_t kvip_ubmem_update_plan(void* update,void* stream) {
    if(!update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_direct_update_owned_kernel(SIM_BATCH,stream,update);
}
extern "C" uint32_t kvip_ubmem_update_outputs(void* update,void* stream) {
    if(!update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_direct_update_outputs_kernel(rotating::kOutputCores,stream,update);
}
extern "C" uint32_t kvip_ubmem_update_apply(void* update,void* stream) {
    if(!update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_direct_apply_owned_kernel(SIM_BATCH*4,stream,update);
}
extern "C" uint32_t kvip_ubmem_update(void* update,void* stream) {
    if(!update || !stream)return 100000;
    uint32_t result=kvip_ubmem_update_plan(update,stream);
    if(!result)result=kvip_ubmem_update_outputs(update,stream);
    return result?result:kvip_ubmem_update_apply(update,stream);
}
extern "C" uint32_t kvip_ubmem_forward(void* bindings,void* update,void* stream) {
    if(!bindings || !update || !stream)return 100000;
    uint32_t result=v9_init(bindings,stream);
    if(!result)result=v9_li(bindings,stream);
    if(!result)result=v9_resolve(bindings,stream);
    if(!result)result=kvip_ubmem_complete_plan(bindings,update,stream);
    if(!result)result=kvip_ubmem_update_outputs(update,stream);
    if(!result)result=kvip_ubmem_update_apply(update,stream);
    return result;
}

#define VALUE(name,value) extern "C" uint32_t name() {return value;}
VALUE(kvip_ubmem_abi_version,2)
VALUE(kvip_ubmem_context_bytes,sizeof(rotating::Context))
VALUE(kvip_ubmem_region_bytes,sizeof(rotating::UbmemRegion))
VALUE(kvip_ubmem_status_bytes,rotating::kUbmemStatusWords*sizeof(uint64_t))
VALUE(kvip_ubmem_plan_words,8+2*20480)
VALUE(v9_abi_version,1)
VALUE(v9_context_bytes,sizeof(rotating::Context))
VALUE(v9_row_bytes,sizeof(rotating::Row))
VALUE(v9_batch,SIM_BATCH)
VALUE(v9_history,SIM_HISTORY)
VALUE(v9_resident,SIM_RESIDENT)
VALUE(v9_index_heads,INDEX_HEADS)
VALUE(v9_token_bytes,TOKEN_BYTES)
VALUE(v9_aic_count,V9_AIC_COUNT)
VALUE(v9_parallel_init,V9_UBMEM_PARALLEL_INIT)
VALUE(v9_init_cores,rotating::kInitCores)
VALUE(v9_balanced_index,V9_UBMEM_BALANCED_INDEX)
VALUE(v9_skip_inactive,V9_SKIP_INACTIVE)
VALUE(v9_compact_reserved,V9_COMPACT_RESERVED)
VALUE(v9_grouped_copy,V9_UBMEM_GROUPED_COPY)
VALUE(v9_li_cores,rotating::kLiCores)
VALUE(v9_launch_cores,rotating::kLaunchCores)
VALUE(v9_rounds,rotating::kRounds)
VALUE(v9_service_aics,rotating::kServiceAics)
VALUE(v9_service_lanes,rotating::kServiceLanes)
VALUE(v9_final_workers,rotating::kFinalWorkers)
VALUE(v9_final_owner_bytes,rotating::kFinalOwnerBytes)
VALUE(v9_final_post_offset,rotating::kFinalPostOffset)
VALUE(v9_final_replicated,1)
VALUE(v9_simulation,0)
VALUE(v9_transport_backend,1)
VALUE(v9_fused_lifecycle,0)
VALUE(v9_ubpipe_depth,V9_UBMEM_GROUPED_COPY?4:8)
VALUE(v9_ubpipe_rows,TOKEN_BYTES==656?4:(V9_UBMEM_GROUPED_COPY?2:1))
VALUE(v9_window_ubpipe_rows,TOKEN_BYTES==656?8:(V9_UBMEM_GROUPED_COPY?2:1))
VALUE(v9_window_ubpipe_depth,V9_UBMEM_GROUPED_COPY?4:8)
extern "C" uint32_t v9_window_trigger_tile(uint32_t epoch) {
    return epoch<rotating::kRounds?rotating::Trigger(epoch):0;
}
