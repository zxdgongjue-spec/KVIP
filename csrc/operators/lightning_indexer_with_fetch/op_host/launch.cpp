#include "../op_kernel/pipeline_types.h"
#include "api.h"
#include "context.h"
#include "fixed_score_groups.h"
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }
extern "C" uint32_t aclrtlaunch_v9_init_kernel(uint32_t,void*,void*,uint32_t,void*,
    uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
static uint32_t LaunchInit(void* bindings,void* stream,const rotating::RuntimeShape* shape,void* workspace,uint32_t fullReset) {
    return aclrtlaunch_v9_init_kernel(shape->batch*4,stream,bindings,fullReset,workspace,
        shape->batch, shape->history, shape->resident, shape->rounds,
        shape->serviceLanes, shape->finalWorkers, shape->finalPostOffset, shape->tokenBytes);
}
extern "C" uint32_t v9_init(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
    if (!bindings || !stream || !shape) return 100000;
    return LaunchInit(bindings,stream,shape,nullptr,1);
}

extern "C" uint32_t aclrtlaunch_v9_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_3(uint32_t,void*,void*);
static uint32_t LaunchLi(void* bindings, void* stream, const rotating::RuntimeShape* shape) {
#define LI_LAUNCH_0(blocks,stream,bindings) aclrtlaunch_v9_li_kernel(blocks,stream,bindings)
#define LI_LAUNCH_1 aclrtlaunch_v9_li_kernel_1
#define LI_LAUNCH_2 aclrtlaunch_v9_li_kernel_2
#define LI_LAUNCH_3 aclrtlaunch_v9_li_kernel_3
#include "../../common/lifetch/li_launch.h"
#undef LI_LAUNCH_0
#undef LI_LAUNCH_1
#undef LI_LAUNCH_2
#undef LI_LAUNCH_3
}
extern "C" uint32_t v9_li(void* bindings,void* stream,const rotating::RuntimeShape* shape) {
    // Large Q4 LI shares per-row readiness with its bound terminal workspace.
    if (shape && shape->batch > 16 && shape->finalWorkers == 4) return 100000;
    return LaunchLi(bindings,stream,shape);
}
extern "C" uint32_t v9_li_cores(const rotating::RuntimeShape* shape) { return shape->liCores; }
extern "C" uint32_t v9_batch(const rotating::RuntimeShape* shape) { return shape->batch; }
extern "C" uint32_t v9_history(const rotating::RuntimeShape* shape) { return shape->history; }
extern "C" uint32_t v9_resident(const rotating::RuntimeShape* shape) { return shape->resident; }
extern "C" uint32_t v9_row_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Row); }
extern "C" uint32_t v9_context_bytes(const rotating::RuntimeShape* shape) { return sizeof(rotating::Context); }
extern "C" uint32_t v9_rounds(const rotating::RuntimeShape* shape) { return shape->rounds; }
extern "C" uint32_t v9_simulation(const rotating::RuntimeShape* shape) { return 0; }
extern "C" uint32_t v9_abi_version(const rotating::RuntimeShape* shape) { return 12; }
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
extern "C" uint32_t kvip_lifetch_streaming_forward(void* bindings,void* update,void* workspace,void* stream,const rotating::RuntimeShape* shape,uint32_t workers) {
    if(!bindings||!update||!workspace||!stream||!shape||shape->resident>131072||shape->batch<1||shape->batch>16||shape->finalWorkers!=4)return 100000;
    if((workers!=1 && workers!=2 && workers!=4) || shape->batch*pipeline_ab::kResolveCores+4*workers>shape->aicCount*2)return 100000;
    uint32_t result=LaunchLi(bindings,stream,shape);
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

extern "C" uint32_t aclrtlaunch_v9_parallel_terminal(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_fixed_hot(uint32_t,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_fixed_cold(uint32_t,void*,void*,void*);
extern "C" uint32_t kvip_fixed_q4_score_groups(const rotating::RuntimeShape* shape) {
    return shape ? fixed_score::LaunchGroups(*shape) : 0;
}
extern "C" uint32_t aclrtlaunch_v9_li_large_hot(uint32_t,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_large_threshold_cold(uint32_t,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_parallel_terminal_large(uint32_t,void*,void*);
extern "C" uint32_t kvip_parallel_tail_supported(const rotating::RuntimeShape* shape) {
    if (shape && rotating::NativeLargeBatch(shape->batch, shape->aicCount))
        return shape->indexHeads == 32 && shape->finalWorkers == 4 &&
            shape->liCores == 32 && shape->serviceLanes == shape->batch && shape->rounds >= 2;
    return shape && shape->batch && shape->batch<=16 && shape->finalWorkers==4 &&
        shape->batch*4<=shape->aicCount*2 && rotating::UseSequenceSplit(*shape);
}
// This launch requires the extended terminal binding with ready lines. Old
// captured callers keep using kvip_parallel_tail_forward and its original ABI.
extern "C" uint32_t kvip_fixed_q4_small_forward(void* bindings, void* terminal,
    void* stream, const rotating::RuntimeShape* shape) {
    if (!bindings || !terminal || !stream || !shape) return 100000;
    uint32_t groups = fixed_score::LaunchGroups(*shape);
    if (!groups || !kvip_parallel_tail_supported(shape)) return 100000;
    uint32_t result = aclrtlaunch_v9_li_fixed_hot(groups, stream, bindings, terminal);
    if (!result) result = aclrtlaunch_v9_li_kernel_1(shape->liCores, stream, bindings);
    if (!result) result = aclrtlaunch_v9_li_fixed_cold(groups, stream, bindings, terminal);
    if (!result) result = aclrtlaunch_v9_parallel_terminal(shape->batch * 4, stream, terminal);
    return result;
}
extern "C" uint32_t kvip_parallel_tail_forward(void* bindings,void* terminal,
    void* stream,const rotating::RuntimeShape* shape) {
    if(!bindings || !terminal || !stream || !kvip_parallel_tail_supported(shape))return 100000;
    if (shape->batch > 16) {
        uint32_t result = aclrtlaunch_v9_li_large_hot(shape->liCores, stream, bindings, terminal);
        if (!result) result = aclrtlaunch_v9_li_large_threshold_cold(shape->liCores, stream, bindings, terminal);
        if (!result) result = aclrtlaunch_v9_parallel_terminal_large(64, stream, terminal);
        return result;
    }
    uint32_t result=aclrtlaunch_v9_li_kernel(shape->liCores,stream,bindings);
    if(!result)result=aclrtlaunch_v9_li_kernel_1(shape->liCores,stream,bindings);
    if(!result)result=aclrtlaunch_v9_li_kernel_2(shape->liCores,stream,bindings);
    if(!result)result=aclrtlaunch_v9_parallel_terminal(shape->batch*4,stream,terminal);
    return result;
}
