#pragma once
#include <stdint.h>
#include "../../lightning_indexer_with_fetch/include/runtime_shape.h"

// All pointers refer to caller-owned, persistent NPU buffers. forward only
// enqueues work on stream. The caller holds its arena lease until stream work
// completes and every captured graph referring to these addresses is gone.
// bindings: 15 uint64 words, same LI tensor order as the KVIP LI/Fetch API.
// update: 26 uint64 words; UBmem replaces the transport fields at 12 and 13
// with ubmemStatus and UbmemRegion, respectively. No queue handle is accepted.
extern "C" {
uint32_t kvip_lifetch_streaming_forward(void* bindings, void* update, void* workspace, void* stream, const rotating::RuntimeShape* shape, uint32_t workers);

uint32_t kvip_ubmem_abi_version();
uint32_t kvip_ubmem_context_bytes();
uint32_t kvip_ubmem_region_bytes();
uint32_t kvip_ubmem_status_bytes();
uint32_t v9_init(void* bindings,void* stream,const rotating::RuntimeShape* shape);
uint32_t v9_li(void* bindings,void* stream,const rotating::RuntimeShape* shape);
}

// Explicit independent AIV budget; legacy six-argument entry remains available.
extern "C" uint32_t kvip_lifetch_streaming_forward_v2(void*,void*,void*,void*,const rotating::RuntimeShape*,uint32_t,uint32_t);
