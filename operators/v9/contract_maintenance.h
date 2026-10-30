#pragma once
#include "index_cache_device.h"
#include "kernel_timeline.h"
namespace rotating {
// Raw device ticks avoid an assumed clock frequency. These stores are outside
// hot loops; rowStatus has independent cache lines per request and only one writer at a
// time (init -> final worker 0, including maintenance -> fetch).
__aicore__ inline void RecordContractTicks(__gm__ Context* ctx, uint32_t b, uint32_t field, uint64_t ticks) {
  auto info = ContractPtr<uint64_t>(ctx->rowStatus) + b * (kRowStatusWords / 2);
  auto line=reinterpret_cast<__gm__ uint8_t*>(info)+((4+field)/16)*128;
  kvaf_device::CleanCacheLines(line,128);
  info[4 + field] = ticks;
  kvaf_device::CleanCacheLines(line,128);
}
__aicore__ inline void RecordIndexWork(__gm__ Context* ctx, uint32_t b, bool tail) {
  auto state = ContractPtr<int32_t>(IndexBase(ContractOf(ctx), b));
  auto info = ContractPtr<uint32_t>(ctx->rowStatus) + b * kRowStatusWords;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(state), 128);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
  info[28] = 0;  // Maintenance runs on final worker 0; no cross-AIV waiter.
  info[24 + (tail ? 2 : 0)] = uint32_t(state[7]);
  info[25 + (tail ? 2 : 0)] = uint32_t(state[17]);
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(info), 128);
}
__aicore__ inline void MaintainContractIndex(__gm__ Context* ctx, uint32_t b, LocalTensor<uint32_t> scratch) {
  uint64_t started = GetSystemCycle();
  Simt::VF_CALL<ReconcileIndexCache>(Simt::Dim3{256, 1, 1}, ctx, b, true, FinalPtr(scratch));
  SetFlag<HardEvent::V_S>(EVENT_ID0);
  WaitFlag<HardEvent::V_S>(EVENT_ID0);
  RecordContractTicks(ctx, b, 3, GetSystemCycle() - started);
  RecordIndexWork(ctx, b, true);
}
}  // namespace rotating
