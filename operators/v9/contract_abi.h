#pragma once
#include <stdint.h>
#include <stddef.h>

namespace rotating {
constexpr uint32_t kContractVersion = 13;
enum ContractStatus : int32_t {
    kOk = 0, kSkipped = 1, kPending = 2,
    kInvalid = -1, kCapacity = -2, kDirectory = -3, kCommunication = -4, kPoisoned = -5
};
// Keep field order in sync with contract.py; bridge exports every field offset.
#define V9_CONTRACT_POINTERS(X) \
    X(query) X(key) X(weights) X(query_scale) X(key_scale) \
    X(cumulative_query) X(candidate_lens) X(logical_lens) \
    X(index_table) X(resident_table) X(pool_rows) X(budgets) X(row_modes) \
    X(forward) X(reverse) X(cursors) X(victims) X(victim_bitmap) \
    X(address_table) X(valid_table) X(resident_output) X(profile) \
    X(topk) X(destination) X(miss_counts) X(resident_lengths) X(transfers) X(status) X(extra_prefetch_slots) X(extra_prefetch_counts) X(index_cache) X(index_dirty) \
    X(hot_bitmap) X(hot_tokens) X(hot_counts) X(scores) X(threshold_valid) \
    X(reserved_tokens) X(reserved_slots) X(planned_tokens) X(planned_slots) \
    X(plan_count) X(logical_misses) X(proposed_cursor) X(touched) X(sticky) X(error_detail) \
    X(eligible_victims) X(eligible_counts) X(hot_ready) X(hot_thresholds) X(cold_counts) X(next_reverse) X(mapping_epoch) X(staged_destination) X(changed_slots) X(publish_meta) X(kernel_timeline)
#define V9_CONTRACT_GEOMETRY(X) \
    X(batch) X(queries) X(candidate_capacity) X(resident_capacity) X(pool_capacity) \
    X(index_columns) X(resident_columns) X(index_blocks) X(resident_blocks) \
    X(layer_id) X(num_layers) X(max_requests) X(qcap) X(table_capacity) \
    X(output_tokens) X(score_stride)
struct Contract {
#define V9_PTR(name) uint64_t name;
    V9_CONTRACT_POINTERS(V9_PTR)
#undef V9_PTR
#define V9_DIM(name) uint32_t name;
    V9_CONTRACT_GEOMETRY(V9_DIM)
#undef V9_DIM
};
static_assert(sizeof(Contract) == 528, "Contract ABI 13 field layout changed");
}
