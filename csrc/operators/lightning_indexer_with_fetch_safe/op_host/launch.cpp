#include "api.h"
#include "context.h"
extern "C" uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape) { return rotating::PrepareShape(shape); }
#define ENTRY(name, blocks) \
extern "C" uint32_t aclrtlaunch_v9_##name##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t v9_##name(void* bindings,void* stream,const rotating::RuntimeShape* shape) { \
    if (!bindings || !stream || !shape) return 100000; \
    return aclrtlaunch_v9_##name##_kernel(blocks,stream,bindings); }
ENTRY(init,shape->batch)
ENTRY(wait,shape->batch)
ENTRY(resolve,shape->finalCores)
ENTRY(fetch,shape->batch)
ENTRY(probe,shape->serviceLanes)

extern "C" uint32_t aclrtlaunch_v9_li_kernel(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_1(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_2(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_3(uint32_t,void*,void*);
extern "C" uint32_t v9_li(void* bindings, void* stream, const rotating::RuntimeShape* shape) {
    if (!bindings || !stream || !shape) return 100000;
    uint32_t result = aclrtlaunch_v9_li_kernel(shape->liCores, stream, bindings);
    if (result || !rotating::UseSequenceSplit(*shape)) return result;
    result = aclrtlaunch_v9_li_kernel_1(shape->liCores, stream, bindings);
    if (!result) result = aclrtlaunch_v9_li_kernel_2(shape->liCores, stream, bindings);
    if (!result) result = aclrtlaunch_v9_li_kernel_3(shape->liCores, stream, bindings);
    return result;
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

#if V9_RESOLVER_COMPARE
ENTRY(resolve_original,shape->finalCores)
ENTRY(resolve_adapted,shape->finalCores)
#endif

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

extern "C" uint32_t kvop_direct_update_outputs_early(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_fetch_update_kernel(uint32_t,void*,void*,void*);
extern "C" uint32_t kvip_lifetch_forward(void* bindings, void* update, void* stream,const rotating::RuntimeShape* shape) {
    if (!bindings || !update || !stream || !shape) return 100000;
    uint32_t result = v9_init(bindings, stream, shape);
    if (!result) result = v9_li(bindings, stream, shape);
    if (!result) result = v9_resolve(bindings, stream, shape);
    if (!result) result = kvop_direct_update_outputs_early(shape->batch, update, stream);
    if (!result) result = aclrtlaunch_v9_fetch_update_kernel(shape->batch, stream, bindings, update);
    return result;
}
