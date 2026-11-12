#pragma once
// SPDX-License-Identifier: Apache-2.0

// Finish the early update after Fetch has observed all transport completions.
// Any failed owner loses hot residency, including partially written KV slots.
template<uint32_t Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void CompleteEarlyUpdate(__gm__ uint64_t* bindings, uint32_t batch, uint64_t error) {
  auto status = reinterpret_cast<__gm__ uint32_t*>(bindings[6]) + batch;
  uint32_t result = asc_ldcg(status) | (error ? 4U : 0U);
  if (!threadIdx.x) asc_stcg(status, result);
  if (result) {
    uint32_t owner = asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(bindings[9]) + batch);
    uint32_t resident = uint32_t(bindings[25]);
    auto valid = reinterpret_cast<__gm__ uint32_t*>(bindings[16]) + uint64_t(owner) * resident;
    for (uint32_t slot = threadIdx.x; slot < resident; slot += Threads)
      asc_stcg(valid + slot, uint32_t(0));
  }
  asc_threadfence();
}
