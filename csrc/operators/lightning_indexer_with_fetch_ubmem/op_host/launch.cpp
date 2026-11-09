#include "api.h"
#include "context.h"

extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }

#define ENTRY(api,kernel,blocks) \
extern "C" uint32_t aclrtlaunch_kvip_ubmem_##kernel##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t api(void* bindings,void* stream,const rotating::RuntimeShape* shape) { \
    if(!bindings || !stream || !shape)return 100000; \
    return aclrtlaunch_kvip_ubmem_##kernel##_kernel(blocks,stream,bindings); }
ENTRY(v9_init,init,shape->batch)
ENTRY(v9_li,li,shape->liCores)
ENTRY(v9_resolve,resolve,shape->finalCores)
ENTRY(v9_fetch,complete,shape->batch)

extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_outputs_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_update_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_direct_apply_owned_kernel(uint32_t,void*,void*);
extern "C" uint32_t kvip_ubmem_update(void* update,void* stream,const rotating::RuntimeShape* shape) {
    if(!update || !stream || !shape)return 100000;
    uint32_t result=aclrtlaunch_kvip_ubmem_direct_update_owned_kernel(shape->batch,stream,update);
    if(!result)result=aclrtlaunch_kvip_ubmem_direct_update_outputs_kernel(shape->batch,stream,update);
    return result?result:aclrtlaunch_kvip_ubmem_direct_apply_owned_kernel(shape->batch*4,stream,update);
}
extern "C" uint32_t kvip_ubmem_forward(void* bindings,void* update,void* stream,const rotating::RuntimeShape* shape) {
    if(!bindings || !update || !stream || !shape)return 100000;
    uint32_t result=v9_init(bindings,stream,shape);
    if(!result)result=v9_li(bindings,stream,shape);
    if(!result)result=v9_resolve(bindings,stream,shape);
    if(!result)result=v9_fetch(bindings,stream,shape);
    if(!result)result=kvip_ubmem_update(update,stream,shape);
    return result;
}

#define VALUE(name,value) extern "C" uint32_t name() {return value;}
VALUE(kvip_ubmem_abi_version,3)
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
VALUE(v9_ubpipe_depth,8)
VALUE(v9_ubpipe_rows,TOKEN_BYTES==656?4:1)
VALUE(v9_window_ubpipe_rows,TOKEN_BYTES==656?8:1)
VALUE(v9_window_ubpipe_depth,8)
extern "C" uint32_t v9_window_trigger_tile(uint32_t epoch,const rotating::RuntimeShape* shape) {
    return shape && epoch<shape->rounds?rotating::RuntimeTrigger(shape->history,shape->resident,shape->rounds,epoch):0;
}
