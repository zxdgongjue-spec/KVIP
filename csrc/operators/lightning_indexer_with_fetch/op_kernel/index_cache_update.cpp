// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
  #include "simt_api/asc_simt.h"
  #include "simt_api/device_functions.h"
  #include "simt_api/device_atomic_functions.h"
  #include "simt_api/device_sync_functions.h"
  #include "simt_api/device_warp_functions.h"
  #include "index_cache_update.h"
using namespace AscendC;
namespace {
constexpr uint32_t kCandidates = kExtendedCandidates, kThreads = 512;
// keys, scales, addresses, logical IDs, LRU slots, producer changes,
// status, address row stride, producer rows, hot validity. No KV payload input.
template <bool WaitForFetch, bool Extended>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void Update(__gm__ uint64_t* a, uint32_t b, uint32_t owner,
                                                            bool checkTransport, __ubuf__ uint32_t* work) {
  uint32_t tid = threadIdx.x;
  auto status = work + (Extended ? 2*kExtendedHash : kPlanWords);
  auto rows = reinterpret_cast<__gm__ uint32_t*>(a[8]) + b * 256;
  auto lru = reinterpret_cast<__gm__ uint32_t*>(a[4]) + b * kCandidates;
  auto changes = reinterpret_cast<__gm__ uint32_t*>(a[5]) + b * kCandidates;
  // Early updates consume the resolver's stable records while DMA is pending.
  // The standalone update entry still requires completed transport.
  if (!tid) {
    status[0] = asc_ldcg(rows + 11) ? 2U : 0U;
    if constexpr (WaitForFetch)
      if (asc_ldcg(rows + 211) || asc_ldcg(rows + 248)) status[0] |= 2U;
  }
  if (!tid && checkTransport) {
    auto rowStatus = reinterpret_cast<__gm__ uint32_t*>(a[10]) + b * 32;
    if (asc_ldcg(rowStatus)) status[0] |= 4U;
    auto posts = reinterpret_cast<__gm__ uint32_t*>(a[11]) + uint64_t(b) * a[14] * 32;
    for (uint32_t worker = 0; worker < a[14]; ++worker) {
      auto post = posts + worker * 32;
      if (asc_ldcg(post + 2)) status[0] |= 4U;
      if constexpr (WaitForFetch)
        if (asc_ldcg(post + 1) || asc_ldcg(post + 3)) status[0] |= 4U;
    }
    if constexpr (WaitForFetch) {
      for (uint32_t array = 12; array <= 13; ++array) {
        auto states = reinterpret_cast<__gm__ uint64_t*>(a[array]);
        for (uint32_t lane = 0; lane < 2; ++lane) {
          auto state = states + uint64_t(b * 2 + lane) * (a[15] / 8);
          if (asc_ldcg(state) != 0x4B5641464247444FULL || asc_ldcg(state + 2)) status[0] |= 4U;
        }
      }
    }
  }
  asc_syncthreads();
  if (!status[0]) {
    if constexpr (!Extended) BuildUpdatePlan<kThreads>(lru,changes,uint32_t(a[24]),uint32_t(a[25]),work,status);
    else {
      uint32_t count=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[26])+b);
      auto extended=reinterpret_cast<__gm__ uint32_t*>(a[27])+uint64_t(b)*kExtendedWorkWords;
      BuildExtendedPlan<kThreads>(lru,changes,count,uint32_t(a[24]),uint32_t(a[25]),work,status,extended);
    }
  }
  asc_syncthreads();
  if (!tid) asc_stcg(reinterpret_cast<__gm__ uint32_t*>(a[6]) + b, status[0]);
  auto plan = reinterpret_cast<__gm__ uint32_t*>(a[checkTransport ? 17 : 10]) + uint64_t(b) * kExtendedTransferWords;
  if (status[0]) {
    if (!tid) asc_stcg(plan, uint32_t(0));
    asc_threadfence();
    return;
  }
  if constexpr (!Extended) PublishUpdatePlan<kThreads>(work,status,plan,uint32_t(a[25]));
  else PublishExtendedPlan<kThreads>(reinterpret_cast<__gm__ uint32_t*>(a[27])+uint64_t(b)*kExtendedWorkWords,
                                     status,plan,uint32_t(a[25]));
  asc_threadfence();
}
// Select the original small planner before entering SIMT: the extended
// planner must not increase its register footprint on the common path.
template <bool WaitForFetch>
__aicore__ inline void DispatchUpdate(__gm__ uint64_t* args, uint32_t batch,
                                    uint32_t owner, bool owned, __ubuf__ uint32_t* work) {
  auto counts = reinterpret_cast<__gm__ uint32_t*>(args[26]);
  if (counts[batch] <= kPlanCandidates)
    Simt::VF_CALL<Update<WaitForFetch, false>>(Simt::Dim3{kThreads, 1, 1}, args, batch, owner, owned, work);
  else
    Simt::VF_CALL<Update<WaitForFetch, true>>(Simt::Dim3{kThreads, 1, 1}, args, batch, owner, owned, work);
}
// Service output adaptation runs before Apply changes logical Index ownership.
// Four TND rows per request; tiles are partitioned so several AIVs share one
// request instead of one block walking all 8192 elements by itself.
constexpr uint32_t kPublishTokens = 2048;

struct OutputRange {
  uint32_t begin, end;
};

__aicore__ inline OutputRange PartitionOutputs(uint32_t split, uint32_t part) {
  constexpr uint32_t tiles = 4 * kPublishTokens / kThreads;
  return {tiles * part / split * kThreads, tiles * (part + 1) / split * kThreads};
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(__gm__ uint64_t* a, uint32_t b, uint32_t begin,
                                                                   uint32_t end) {
  auto topk = reinterpret_cast<__gm__ int32_t*>(a[18]);
  auto slots = reinterpret_cast<__gm__ int32_t*>(a[19]);
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto logicalOut = reinterpret_cast<__gm__ int32_t*>(a[22]);
  auto slotsOut = reinterpret_cast<__gm__ int32_t*>(a[23]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  auto logical = reinterpret_cast<__gm__ int64_t*>(a[3]);
  for (uint32_t q = begin / kPublishTokens; q * kPublishTokens < end; ++q) {
    uint32_t row = b * 4 + q;
    const uint32_t base = q * kPublishTokens;
    const uint32_t first = begin > base ? begin - base : 0;
    const uint32_t last = end - base < kPublishTokens ? end - base : kPublishTokens;
    bool active = asc_ldcg(valid + row) != 0;
    uint64_t sourceRow = active ? asc_ldcg(unpack + row) : 0;
    uint64_t owner = active ? asc_ldcg(owners + sourceRow / 4) : 0;
    for (uint32_t k = first + threadIdx.x; k < last; k += kThreads) {
      int32_t token = active ? asc_ldcg(topk + sourceRow * kPublishTokens + k) : -1;
      int32_t id = token >= 0 ? static_cast<int32_t>(asc_ldcg(logical + owner * uint32_t(a[24]) + token)) : -1;
      int32_t slot = active ? asc_ldcg(slots + sourceRow * kPublishTokens + k) : -1;
      asc_stcg(logicalOut + uint64_t(row) * kPublishTokens + k, id);
      asc_stcg(slotsOut + uint64_t(row) * kPublishTokens + k, slot);
    }
  }
}
template <bool Owned>
__aicore__ inline void Apply(GM_ADDR bindings) {
  uint32_t b = GetBlockIdx() / kCopyWorkers, group = GetBlockIdx() % kCopyWorkers;
  auto a = reinterpret_cast<__gm__ uint64_t*>(bindings);
  auto input = reinterpret_cast<__gm__ uint32_t*>(a[Owned ? 17 : 10]) + uint64_t(b) * kExtendedTransferWords;
  uint32_t owner = Owned ? reinterpret_cast<__gm__ uint32_t*>(a[9])[b] : b;
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, kMoveScratchBytes);
  auto local = storage.Get<uint32_t>();
  GlobalTensor<uint32_t> planGm;
  planGm.SetGlobalBuffer(input);
  DataCopy(local, planGm, kTransferHeaderWords);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  uint32_t count = local.GetValue(0);
  if (!count) return;
  if(count>kPlanRecords) {
    if(count>kExtendedRecords)return;
    Simt::VF_CALL<MoveExtended<kThreads>>(Simt::Dim3{kThreads,1,1},a,owner,group,Owned,input,
      reinterpret_cast<__ubuf__ uint64_t*>(local.GetPhyAddr()),count);
    return;
  }
  DataCopy(local[kTransferHeaderWords], planGm[kTransferHeaderWords], ((count * 2 + 7) / 8) * 8);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<Move<kThreads>>(Simt::Dim3{kThreads, 1, 1}, a, owner, group, Owned,
                                reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ uint64_t*>(local[kTransferPlanWords].GetPhyAddr()), count);
}

template <bool WaitForFetch>
__aicore__ inline void UpdateOutputs(GM_ADDR bindings, uint32_t split) {
  const uint32_t block = GetBlockIdx();
  const uint32_t batch = block / split, part = block % split;
  auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(args[9]);
  const auto range = PartitionOutputs(split, part);
  Simt::VF_CALL<PublishOutputs>(Simt::Dim3{kThreads, 1, 1}, args, batch, range.begin, range.end);
  if (part) return;
  // Only the first share of each request plans; the others are publish-only,
  // so the plan and the mapping run on different AIVs at the same time.
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, kExtendedScratchBytes);
  auto local = scratch.Get<uint32_t>();
  DispatchUpdate<WaitForFetch>(args, batch, owners[batch], true,
                                   reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
}

}  // namespace
#endif
extern "C" __global__ __aicore__ void kvop_direct_update_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, kExtendedScratchBytes);
    auto local = scratch.Get<uint32_t>();
    DispatchUpdate<true>(reinterpret_cast<__gm__ uint64_t*>(bindings), GetBlockIdx(),
                          GetBlockIdx(), false, reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
  }
#endif
}

// Same update body, with stable request-owner storage instead of compact batch
// storage. The test interface also carries its hot-valid pointer.
extern "C" __global__ __aicore__ void kvop_direct_update_owned_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    pipe.InitBuffer(scratch, kExtendedScratchBytes);
    auto local = scratch.Get<uint32_t>();
    auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
    auto owners = reinterpret_cast<__gm__ uint32_t*>(args[9]);
    uint32_t batch = GetBlockIdx();
    DispatchUpdate<true>(args, batch, owners[batch], true,
                          reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
  }
#endif
}
extern "C" __global__ __aicore__ void kvop_direct_update_outputs_kernel(GM_ADDR bindings, uint32_t split) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    UpdateOutputs<true>(bindings, split);
  }
#endif
}
extern "C" __global__ __aicore__ void kvop_direct_update_outputs_early_kernel(GM_ADDR bindings, uint32_t split) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    UpdateOutputs<false>(bindings, split);
  }
#endif
}

extern "C" __global__ __aicore__ void kvop_direct_apply_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    Apply<false>(bindings);
  }
#endif
}
extern "C" __global__ __aicore__ void kvop_direct_apply_owned_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    Apply<true>(bindings);
  }
#endif
}
