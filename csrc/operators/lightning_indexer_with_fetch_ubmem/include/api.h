#pragma once
#include <stdint.h>

// All pointers refer to caller-owned, persistent NPU buffers. forward only
// enqueues work on stream. The caller holds its arena lease until stream work
// completes and every captured graph referring to these addresses is gone.
// bindings: 15 uint64 words, same LI tensor order as the KVIP LI/Fetch API.
// update: 28 uint64 words; UBmem replaces the transport fields at 12 and 13
// with ubmemStatus and UbmemRegion, respectively. No queue handle is accepted.
extern "C" {
uint32_t kvip_ubmem_forward(void* bindings,void* update,void* stream);
uint32_t kvip_ubmem_abi_version();
uint32_t kvip_ubmem_context_bytes();
uint32_t kvip_ubmem_region_bytes();
uint32_t kvip_ubmem_status_bytes();
uint32_t kvip_ubmem_plan_words();
uint32_t v9_final_copy_stages();
uint32_t v9_final_copy_rows();
uint32_t v9_ordered_pair();
uint32_t v9_tagged_map();
uint32_t v9_merge_outputs();
uint32_t v9_cold_only_prefetch();
uint32_t v9_threshold_only();
// Optional eager-only diagnostics; zero Context pointer disables every write.
uint32_t v9_threshold_trace_abi();
uint32_t v9_simt_init();
uint32_t v9_bitmap_index();
uint32_t v9_reserved_ub();
uint32_t v9_fused_init();
uint32_t v9_init_li(void* bindings,void* stream);
uint32_t v9_single_plan();
uint32_t v9_staged_outputs();
uint32_t v9_rolling_threshold();
uint32_t v9_cached_buckets();
uint32_t v9_ubpipe_depth();
uint32_t v9_ubpipe_rows();
uint32_t v9_window_ubpipe_depth();
uint32_t v9_window_ubpipe_rows();
uint32_t v9_init(void* bindings,void* stream);
uint32_t v9_li(void* bindings,void* stream);
uint32_t v9_resolve(void* bindings,void* stream);
// Single-plan mode retains both launches inside v9_resolve. These entries
// expose the exact same boundaries for instrumented stage diagnostics.
uint32_t kvip_ubmem_resolve_plan(void* bindings,void* stream);
uint32_t kvip_ubmem_resolve_copy(void* bindings,void* stream);
uint32_t v9_fetch(void* bindings,void* stream);
// Completes the same packed batch before building its Index plan. Retain the
// outputs/apply launches after this kernel: unpack can cross packed batches.
uint32_t kvip_ubmem_complete_plan(void* bindings,void* update,void* stream);
// Same completion/plan checks, with outputs owned by their packed source row.
// This entry includes all TND publication; follow it only with Apply.
uint32_t kvip_ubmem_complete_plan_outputs(void* bindings,void* update,void* stream);
uint32_t kvip_ubmem_update(void* update,void* stream);
uint32_t kvip_ubmem_update_plan(void* update,void* stream);
uint32_t kvip_ubmem_update_outputs(void* update,void* stream);
uint32_t kvip_ubmem_update_apply(void* update,void* stream);
}
