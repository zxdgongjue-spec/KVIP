#pragma once
#include "rotation_abi.h"
#include "kvaf_batch_get_device.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

namespace rotating {
// Scalar and SIMT have distinct function execution domains in CANN.
#define V9_CONTRACT_ACCESS __aicore__ inline
#include "contract_access.inc"
#undef V9_CONTRACT_ACCESS
namespace contract_simt {
#define V9_CONTRACT_ACCESS __simt_callee__ inline
#include "contract_access.inc"
#undef V9_CONTRACT_ACCESS
}
__aicore__ inline bool ContractDisabled(__gm__ Contract* c) {
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(c->sticky), 4);
    return ContractPtr<int32_t>(c->profile)[0] == 2 ||
           ContractPtr<int32_t>(c->row_modes)[0] == 0 ||
           ContractPtr<int32_t>(c->sticky)[0] != 0;
}
namespace contract_simt {
// Reservation state is private to this call; publication is fetch-owned.
__simt_callee__ inline uint32_t ReserveContract(__gm__ Context* ctx, uint32_t b,
    uint32_t token, uint32_t rank) {
    auto c = ContractOf(ctx);
    uint32_t slot = uint32_t(ContractPtr<int32_t>(c->eligible_victims)[b * 2048 + rank]);
    asc_stcg(ContractPtr<int32_t>(ctx->mapping) + uint64_t(b) * c->candidate_capacity + token, int32_t(slot));
    asc_stcg(ContractPtr<int32_t>(c->reserved_tokens) + b * 2048 + rank, int32_t(token));
    asc_stcg(ContractPtr<int32_t>(c->reserved_slots) + b * 2048 + rank, int32_t(slot));
    asc_atomic_or(ContractPtr<uint32_t>(c->touched) + b * 512 + slot / 32, 1U << (slot % 32));
    return ContractPhysical(c, b, slot);
}
}
__aicore__ inline uint32_t ContractRounds(__gm__ Contract* c) {
    uint32_t pages = (ColdExtent(c) + 127) / 128;
    return pages < kRounds ? pages : kRounds;
}
__aicore__ inline uint32_t ContractTrigger(__gm__ Contract* c, uint32_t epoch) {
    uint32_t rounds = ContractRounds(c);
    return rounds ? ((ColdExtent(c) + 127) / 128) * (epoch + 1) / rounds : 0;
}
__aicore__ inline int32_t ContractEpoch(__gm__ Contract* c, uint32_t pageEnd, uint32_t core) {
    uint32_t epoch = ServiceRank(core) / kServiceAics;
    return epoch < ContractRounds(c) && ContractTrigger(c, epoch) == pageEnd ? int32_t(epoch) : -1;
}
}
