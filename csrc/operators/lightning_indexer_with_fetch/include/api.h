#pragma once
#include <cstdint>
#include "runtime_shape.h"

// Stable host ABI; implementations live in launch.cpp.
extern "C" {
uint32_t kvip_lifetch_forward(void* bindings, void* update, void* stream,const rotating::RuntimeShape* shape);
uint32_t kvip_prepare_shape(rotating::RuntimeShape* shape);
uint32_t v9_init(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_li(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_wait(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_resolve(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_fetch(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_probe(void* bindings, void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_li_cores(const rotating::RuntimeShape* shape);
uint32_t v9_batch(const rotating::RuntimeShape* shape);
uint32_t v9_history(const rotating::RuntimeShape* shape);
uint32_t v9_resident(const rotating::RuntimeShape* shape);
uint32_t v9_row_bytes(const rotating::RuntimeShape* shape);
uint32_t v9_context_bytes(const rotating::RuntimeShape* shape);
uint32_t v9_rounds(const rotating::RuntimeShape* shape);
uint32_t v9_simulation(const rotating::RuntimeShape* shape);
uint32_t v9_abi_version(const rotating::RuntimeShape* shape);
uint32_t v9_aic_count(const rotating::RuntimeShape* shape);
uint32_t v9_service_lanes(const rotating::RuntimeShape* shape);
uint32_t v9_final_workers(const rotating::RuntimeShape* shape);
uint32_t v9_final_owner_bytes(const rotating::RuntimeShape* shape);
uint32_t v9_final_replicated(const rotating::RuntimeShape* shape);
uint32_t refactor_fetch_args_bytes();
uint32_t refactor_fetch_status_bytes();
uint32_t refactor_fetch_entity_bytes();
uint32_t refactor_fetch_queue_bytes();
uint32_t refactor_fetch_pack_args(void* output, uint32_t capacity, const uint64_t* fields, uint32_t count);
}
