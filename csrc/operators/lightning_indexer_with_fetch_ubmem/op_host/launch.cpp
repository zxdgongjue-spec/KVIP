#include "api.h"
#include "context.h"

#define ENTRY(api,kernel,blocks) \
extern "C" uint32_t aclrtlaunch_kvip_ubmem_##kernel##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t api(void* bindings,void* stream) { \
    if(!bindings || !stream)return 100000; \
    return aclrtlaunch_kvip_ubmem_##kernel##_kernel(blocks,stream,bindings); }
ENTRY(v9_init,init,rotating::kInitCores)
ENTRY(v9_li,li,rotating::kLaunchCores)
#if V9_UBMEM_FUSED_INIT
ENTRY(v9_init_li,init_li,rotating::kLaunchCores)
#endif
ENTRY(kvip_ubmem_resolve_plan,resolve_plan,SIM_BATCH)
ENTRY(kvip_ubmem_resolve_copy,resolve_copy,rotating::kFinalCores)
#if V9_UBMEM_SINGLE_PLAN
extern "C" uint32_t v9_resolve(void* bindings,void* stream) {
    uint32_t result=kvip_ubmem_resolve_plan(bindings,stream);
    return result?result:kvip_ubmem_resolve_copy(bindings,stream);
}
#else
ENTRY(v9_resolve,resolve,rotating::kFinalCores)
#endif
ENTRY(v9_fetch,complete,SIM_BATCH)

extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_outputs_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_apply_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_kernel(uint32_t,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_complete_plan_outputs_kernel(uint32_t,void*,void*,void*);
extern "C" uint32_t kvip_ubmem_complete_plan(void* bindings,void* update,void* stream) {
    if(!bindings || !update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_complete_plan_kernel(SIM_BATCH,stream,bindings,update);
}
extern "C" uint32_t kvip_ubmem_complete_plan_outputs(void* bindings,void* update,void* stream) {
    if(!bindings || !update || !stream)return 100000;
    return aclrtlaunch_kvip_ubmem_complete_plan_outputs_kernel(SIM_BATCH,stream,bindings,update);
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
#if V9_UBMEM_FUSED_INIT
    uint32_t result=v9_init_li(bindings,stream);
#else
    uint32_t result=v9_init(bindings,stream);
    if(!result)result=v9_li(bindings,stream);
#endif
    if(!result)result=v9_resolve(bindings,stream);
#if V9_UBMEM_MERGE_OUTPUTS
    if(!result)result=kvip_ubmem_complete_plan_outputs(bindings,update,stream);
#else
    if(!result)result=kvip_ubmem_complete_plan(bindings,update,stream);
    if(!result)result=kvip_ubmem_update_outputs(update,stream);
#endif
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
VALUE(v9_tagged_map,V9_UBMEM_TAGGED_MAP)
VALUE(v9_init_cores,rotating::kInitCores)
VALUE(v9_balanced_index,V9_UBMEM_BALANCED_INDEX)
VALUE(v9_merge_outputs,V9_UBMEM_MERGE_OUTPUTS)
VALUE(v9_single_plan,V9_UBMEM_SINGLE_PLAN)
VALUE(v9_staged_outputs,V9_STAGED_OUTPUTS)
VALUE(v9_rolling_threshold,V9_ROLLING_THRESHOLD)
VALUE(v9_cached_buckets,V9_CACHED_BUCKETS)
VALUE(v9_skip_inactive,V9_SKIP_INACTIVE)
VALUE(v9_compact_reserved,V9_COMPACT_RESERVED)
VALUE(v9_grouped_copy,V9_UBMEM_GROUPED_COPY)
VALUE(v9_final_copy_stages,V9_UBMEM_FINAL_COPY_STAGES)
VALUE(v9_final_copy_rows,V9_UBMEM_FINAL_COPY_ROWS)
VALUE(v9_ordered_pair,V9_UBMEM_ORDERED_PAIR)
VALUE(v9_cold_only_prefetch,V9_COLD_ONLY_PREFETCH)
VALUE(v9_threshold_only,V9_THRESHOLD_ONLY)
VALUE(v9_simt_init,V9_SIMT_INIT)
VALUE(v9_bitmap_index,V9_BITMAP_INDEX)
VALUE(v9_reserved_ub,V9_RESERVED_UB)
VALUE(v9_fused_init,V9_UBMEM_FUSED_INIT)
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
VALUE(v9_ubpipe_depth,V9_UBMEM_GROUPED_COPY?V9_UBMEM_FINAL_COPY_STAGES:8)
VALUE(v9_ubpipe_rows,TOKEN_BYTES==656?4:(V9_UBMEM_GROUPED_COPY?V9_UBMEM_FINAL_COPY_ROWS:1))
VALUE(v9_window_ubpipe_rows,TOKEN_BYTES==656?8:(V9_UBMEM_GROUPED_COPY?2:1))
VALUE(v9_window_ubpipe_depth,V9_UBMEM_GROUPED_COPY?4:8)
extern "C" uint32_t v9_window_trigger_tile(uint32_t epoch) {
    return epoch<rotating::kRounds?rotating::Trigger(epoch):0;
}
