#include "api.h"
#include <cstdint>
#include "context.h"
#define ENTRY(name, blocks) \
extern "C" uint32_t aclrtlaunch_v9_##name##_kernel(uint32_t,void*,void*); \
extern "C" uint32_t v9_##name(void* bindings,void* stream) { \
    if (!bindings || !stream) return 100000; \
    return aclrtlaunch_v9_##name##_kernel(blocks,stream,bindings); }
ENTRY(init,SIM_BATCH)
ENTRY(li,rotating::kLiCores)
ENTRY(wait,SIM_BATCH)
ENTRY(resolve,rotating::kFinalCores)
ENTRY(fetch,SIM_BATCH)
// Probe shares the production SIMD consumer but never rings a network doorbell.
ENTRY(probe,rotating::kServiceLanes)
extern "C" uint32_t v9_li_cores() { return rotating::kLiCores; }
extern "C" uint32_t v9_batch() { return SIM_BATCH; }
extern "C" uint32_t v9_history() { return SIM_HISTORY; }
extern "C" uint32_t v9_resident() { return SIM_RESIDENT; }
extern "C" uint32_t v9_row_bytes() { return sizeof(rotating::Row); }
extern "C" uint32_t v9_context_bytes() { return sizeof(rotating::Context); }
extern "C" uint32_t v9_rounds() { return rotating::kRounds; }

extern "C" uint32_t v9_simulation() { return V9_SIMULATION; }

extern "C" uint32_t v9_abi_version() { return 9; }

extern "C" uint32_t v9_aic_count() { return V9_AIC_COUNT; }

extern "C" uint32_t v9_service_lanes() { return rotating::kServiceLanes; }

extern "C" uint32_t v9_final_workers() { return rotating::kFinalWorkers; }
extern "C" uint32_t v9_final_owner_bytes() { return rotating::kFinalOwnerBytes; }

extern "C" uint32_t v9_final_replicated() { return rotating::kFinalReplicated; }

extern "C" uint32_t v9_token_bytes() { return rotating::kTokenBytes; }

extern "C" uint32_t aclrtlaunch_v9_get_kernel(uint32_t, void*, void*);
extern "C" uint32_t v9_get(void* arguments, void* stream, uint32_t cores) {
    if (!arguments || !stream || !cores || cores > KVAF_AIV_BATCH_GET_MAX_AIV_CORES) return 100000;
    return aclrtlaunch_v9_get_kernel(cores, stream, arguments);
}

#if V9_RESOLVER_COMPARE
ENTRY(resolve_original,rotating::kFinalCores)
ENTRY(resolve_adapted,rotating::kFinalCores)
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

extern "C" uint32_t kvop_direct_update_outputs(uint32_t,void*,void*);
extern "C" uint32_t kvip_lifetch_forward(void* bindings, void* update, void* stream) {
    if (!bindings || !update || !stream) return 100000;
    uint32_t result = v9_init(bindings, stream);
    if (!result) result = v9_li(bindings, stream);
    if (!result) result = v9_resolve(bindings, stream);
    if (!result) result = v9_fetch(bindings, stream);
    if (!result) result = kvop_direct_update_outputs(SIM_BATCH, update, stream);
    return result;
}
