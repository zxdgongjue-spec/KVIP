#pragma once
#include <cstdint>

// Stable host ABI; implementations live in launch.cpp.
extern "C" {
uint32_t kvip_lifetch_forward(void* bindings, void* update, void* stream);
uint32_t v9_init(void* bindings, void* stream);
uint32_t v9_li(void* bindings, void* stream);
uint32_t v9_wait(void* bindings, void* stream);
uint32_t v9_resolve(void* bindings, void* stream);
uint32_t v9_fetch(void* bindings, void* stream);
uint32_t v9_probe(void* bindings, void* stream);
uint32_t v9_li_cores();
uint32_t v9_batch();
uint32_t v9_history();
uint32_t v9_resident();
uint32_t v9_row_bytes();
uint32_t v9_context_bytes();
uint32_t v9_rounds();
uint32_t v9_simulation();
uint32_t v9_abi_version();
uint32_t v9_aic_count();
uint32_t v9_service_lanes();
uint32_t v9_final_workers();
uint32_t v9_final_owner_bytes();
uint32_t v9_final_replicated();
uint32_t refactor_fetch_args_bytes();
uint32_t refactor_fetch_status_bytes();
uint32_t refactor_fetch_entity_bytes();
uint32_t refactor_fetch_queue_bytes();
uint32_t refactor_fetch_pack_args(void* output, uint32_t capacity, const uint64_t* fields, uint32_t count);
}
