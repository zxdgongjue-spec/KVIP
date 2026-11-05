#pragma once
#include <stdint.h>

// All pointers refer to caller-owned, persistent NPU buffers. forward only
// enqueues work on stream. The caller holds its arena lease until stream work
// completes and every captured graph referring to these addresses is gone.
// bindings: 15 uint64 words, same LI tensor order as the KVIP LI/Fetch API.
// update: 26 uint64 words; UBmem replaces the transport fields at 12 and 13
// with ubmemStatus and UbmemRegion, respectively. No queue handle is accepted.
extern "C" {
uint32_t kvip_ubmem_forward(void* bindings,void* update,void* stream);
uint32_t kvip_ubmem_abi_version();
uint32_t kvip_ubmem_context_bytes();
uint32_t kvip_ubmem_region_bytes();
uint32_t kvip_ubmem_status_bytes();
uint32_t kvip_ubmem_plan_words();
uint32_t v9_init(void* bindings,void* stream);
uint32_t v9_li(void* bindings,void* stream);
uint32_t v9_resolve(void* bindings,void* stream);
uint32_t v9_fetch(void* bindings,void* stream);
uint32_t kvip_ubmem_update(void* update,void* stream);
}
