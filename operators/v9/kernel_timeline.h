#pragma once
namespace rotating {
// Each lane owns an isolated cache line. No cross-core polling or subtraction.
__aicore__ inline void RecordKernelSpan(__gm__ Context* ctx,uint32_t slot,uint64_t start,uint64_t end) {
  auto c=ContractOf(ctx);
  auto dst=ContractPtr<uint64_t>(c->kernel_timeline)+slot*16;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(dst),128);
  dst[0]=start;dst[1]=end;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(dst),128);
}
}
