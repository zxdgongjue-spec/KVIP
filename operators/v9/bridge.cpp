#include <cstdint>
#include "rotation_abi.h"
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
ENTRY(index_write,SIM_BATCH)
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

extern "C" uint32_t v9_abi_version() { return rotating::kContractVersion; }

extern "C" uint32_t v9_aic_count() { return V9_AIC_COUNT; }

extern "C" uint32_t v9_service_lanes() { return rotating::kServiceLanes; }

extern "C" uint32_t v9_final_workers() { return rotating::kFinalWorkers; }
extern "C" uint32_t v9_final_owner_bytes() { return rotating::kFinalOwnerBytes; }

extern "C" uint32_t v9_final_replicated() { return rotating::kFinalReplicated; }

extern "C" uint32_t v9_queries() { return SIM_QUERIES; }
extern "C" uint32_t v9_contract_mode() { return V9_CONTRACT_MODE; }
extern "C" uint32_t v9_contract_bytes() { return sizeof(rotating::Contract); }
extern "C" uint32_t v9_contract_offset(uint32_t field) {
#define OFFSET(name) uint32_t(offsetof(rotating::Contract, name)),
    static const uint32_t offsets[] = {
        V9_CONTRACT_POINTERS(OFFSET) V9_CONTRACT_GEOMETRY(OFFSET)
    };
#undef OFFSET
    return field < sizeof(offsets)/sizeof(offsets[0]) ? offsets[field] : UINT32_MAX;
}

extern "C" uint32_t v9_index_write_abi() { return 2; }
