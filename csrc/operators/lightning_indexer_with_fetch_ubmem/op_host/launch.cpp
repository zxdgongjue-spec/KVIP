#include "api.h"
#include "context.h"

extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }

#define ENTRY(api,kernel,blocks) \
extern "C" uint32_t aclrtlaunch_kvip_ubmem_656_##kernel##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t aclrtlaunch_kvip_ubmem_2624_##kernel##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t api(void* bindings,void* stream,const rotating::RuntimeShape* shape) { \
    if(!bindings || !stream || !shape)return 100000; \
    if(shape->tokenBytes==656)return aclrtlaunch_kvip_ubmem_656_##kernel##_kernel(blocks,stream,bindings); \
    if(shape->tokenBytes==2624)return aclrtlaunch_kvip_ubmem_2624_##kernel##_kernel(blocks,stream,bindings); \
    return 100000; }
ENTRY(v9_init,init,shape->batch)
#if !KVIP_STREAMING_UBMEM
ENTRY(v9_li,li,shape->liCores)
#else
extern "C" uint32_t aclrtlaunch_kvip_ubmem_656_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_656_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_656_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_656_li_kernel_3(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_2624_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_2624_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_2624_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_kvip_ubmem_2624_li_kernel_3(uint32_t,void*,void*);
extern "C" uint32_t v9_li(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
 if(!shape)return 100000;
 if(shape->tokenBytes==656) {
#define LI_LAUNCH_0 aclrtlaunch_kvip_ubmem_656_li_kernel
#define LI_LAUNCH_1 aclrtlaunch_kvip_ubmem_656_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_kvip_ubmem_656_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_kvip_ubmem_656_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
 }
 if(shape->tokenBytes==2624) {
#define LI_LAUNCH_0 aclrtlaunch_kvip_ubmem_2624_li_kernel
#define LI_LAUNCH_1 aclrtlaunch_kvip_ubmem_2624_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_kvip_ubmem_2624_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_kvip_ubmem_2624_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
 }
 return 100000;
}
#endif
ENTRY(v9_resolve,resolve,shape->finalCores)
ENTRY(v9_fetch_workers,fetch,shape->finalCores)
ENTRY(v9_finish_fetch,complete,shape->batch)
extern "C" uint32_t v9_fetch(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
    uint32_t result=v9_fetch_workers(bindings,stream,shape);
    return result?result:v9_finish_fetch(bindings,stream,shape);
}

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
extern "C" uint32_t v9_window_trigger_tile(uint32_t epoch,const rotating::RuntimeShape* shape) {
    return shape && epoch<shape->rounds?rotating::RuntimeTrigger(shape->history,shape->resident,shape->rounds,epoch):0;
}

#if KVIP_STREAMING_UBMEM
#include "../../lightning_indexer_with_fetch_streaming_qworkers/op_kernel/pipeline_types.h"
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update_656(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_streaming_resolve_update_2624(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_counter_calibration_656(uint32_t,void*,void*);
extern "C" uint32_t kvip_counter_calibration(void* output,void* stream){
 if(!output||!stream)return 100000;
 return aclrtlaunch_v9_counter_calibration_656(1,stream,output);
}
extern "C" uint32_t kvip_streaming_resolver_cores(){return pipeline_ab::kResolveCores;}
extern "C" uint32_t kvip_streaming_flag_count(){return pipeline_ab::kFlagCount;}
extern "C" uint32_t kvip_lifetch_streaming_forward(void* bindings,void* update,void* workspace,
 void* stream,const rotating::RuntimeShape* shape,uint32_t workers) {
 if(!bindings || !update || !workspace || !stream || !shape || shape->batch<1 || shape->batch>16 ||
    shape->history!=131072 || shape->resident!=16384 || shape->indexHeads!=32 || shape->finalWorkers!=4 ||
    (workers!=1 && workers!=2 && workers!=4))return 100000;
 uint32_t cores=shape->batch*pipeline_ab::kResolveCores+4*workers;
 if(cores>shape->aicCount*2)return 100000;
 uint32_t result=v9_init(bindings,stream,shape);
 if(!result)result=v9_li(bindings,stream,shape);
 if(result)return result;
 if(shape->tokenBytes==656)return aclrtlaunch_v9_streaming_resolve_update_656(cores,stream,workspace);
 if(shape->tokenBytes==2624)return aclrtlaunch_v9_streaming_resolve_update_2624(cores,stream,workspace);
 return 100000;
}
#endif
