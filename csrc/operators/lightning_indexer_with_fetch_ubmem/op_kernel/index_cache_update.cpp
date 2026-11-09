// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
  #include "simt_api/asc_simt.h"
  #include "simt_api/device_functions.h"
  #include "simt_api/device_atomic_functions.h"
  #include "simt_api/device_sync_functions.h"
  #include "simt_api/device_warp_functions.h"
  #include "index_cache_update.h"
  #include "ubmem_abi.h"
using namespace AscendC;
namespace {
constexpr uint32_t kCandidates = kExtendedCandidates, kThreads = 512;
// keys, scales, addresses, logical IDs, LRU slots, producer changes,
// status, address row stride, producer rows, hot validity. No KV payload input.
template<bool Extended>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void Update(__gm__ uint64_t* a, uint32_t b, uint32_t owner,
                                                            bool checkTransport, __ubuf__ uint32_t* work) {
  uint32_t tid = threadIdx.x;
  auto status = work + (Extended ? 2*kExtendedHash : kPlanWords);
  auto rows = reinterpret_cast<__gm__ uint32_t*>(a[8]) + b * 256;
  auto lru = reinterpret_cast<__gm__ uint32_t*>(a[4]) + b * kCandidates;
  auto changes = reinterpret_cast<__gm__ uint32_t*>(a[5]) + b * kCandidates;
  // Do not publish Index ownership after a failed/incomplete producer fetch.
  // These are the existing Row error, pending and finalPending ABI fields.
  if (!tid) status[0] = (asc_ldcg(rows + 11) || asc_ldcg(rows + 211) || asc_ldcg(rows + 248)) ? 2U : 0U;
  if (!tid && checkTransport) {
    auto rowStatus = reinterpret_cast<__gm__ uint32_t*>(a[10]) + b * 32;
    if (asc_ldcg(rowStatus)) status[0] |= 4U;
    auto posts = reinterpret_cast<__gm__ uint32_t*>(a[11]) + uint64_t(b) * a[14] * 32;
    for (uint32_t worker = 0; worker < a[14]; ++worker)
      for (uint32_t field = 1; field < 4; ++field)
        if (asc_ldcg(posts + worker * 32 + field)) status[0] |= 4U;
    auto completed=reinterpret_cast<__gm__ uint64_t*>(a[12])+uint64_t(b)*rotating::kUbmemStatusWords;
    auto region=reinterpret_cast<__gm__ uint64_t*>(a[13]);
    if(asc_ldcg(region)!=rotating::kUbmemMagic || asc_ldcg(region+1)!=rotating::kUbmemVersion ||
       !asc_ldcg(region+5) || asc_ldcg(completed) ||
       asc_ldcg(completed+1)!=asc_ldcg(completed+2) ||
       asc_ldcg(completed+3)!=asc_ldcg(completed+4) ||
       asc_ldcg(completed+6)!=asc_ldcg(region+5))status[0]|=4U;
  }
  asc_syncthreads();
  if (!status[0]) {
    if constexpr (!Extended) BuildUpdatePlan<kThreads>(lru,changes,uint32_t(a[24]),uint32_t(a[25]),work,status);
    else BuildExtendedPlan<kThreads>(lru,changes,
        asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[26])+b),uint32_t(a[24]),uint32_t(a[25]),work,status,
        reinterpret_cast<__gm__ uint32_t*>(a[27])+uint64_t(b)*kExtendedWorkWords);
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
// Service output adaptation runs before Apply changes logical Index ownership.
// Each block owns four TND output rows, independently of their packed owners.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(__gm__ uint64_t* a, uint32_t b, uint32_t batchCount) {
  auto topk = reinterpret_cast<__gm__ int32_t*>(a[18]);
  auto slots = reinterpret_cast<__gm__ int32_t*>(a[19]);
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto logicalOut = reinterpret_cast<__gm__ int32_t*>(a[22]);
  auto slotsOut = reinterpret_cast<__gm__ int32_t*>(a[23]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  auto logical = reinterpret_cast<__gm__ int64_t*>(a[3]);
  for (uint32_t q = 0; q < 4; ++q) {
    uint32_t row = b * 4 + q;
    bool active = asc_ldcg(valid + row) != 0;
    uint64_t sourceRow = active ? asc_ldcg(unpack + row) : 0;
    // Plan statuses are complete on every packed row before this separate
    // launch. TND output row b need not come from packed row b.
    active = active && sourceRow<uint64_t(batchCount)*4 &&
        asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(a[6])+sourceRow/4)==0;
    uint64_t owner = active ? asc_ldcg(owners + sourceRow / 4) : 0;
    for (uint32_t k = threadIdx.x; k < 2048; k += 512) {
      int32_t token = active ? asc_ldcg(topk + sourceRow * 2048 + k) : -1;
      int32_t id = token >= 0 && uint32_t(token)<uint32_t(a[24]) ?
          static_cast<int32_t>(asc_ldcg(logical + owner * uint32_t(a[24]) + token)) : -1;
      int32_t slot = active ? asc_ldcg(slots + sourceRow * 2048 + k) : -1;
      asc_stcg(logicalOut + uint64_t(row) * 2048 + k, id);
      asc_stcg(slotsOut + uint64_t(row) * 2048 + k, slot);
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

}  // namespace
#endif


// Same update body, with stable request-owner storage instead of compact batch
// storage. The test interface also carries its hot-valid pointer.
extern "C" __global__ __aicore__ void kvip_ubmem_direct_update_owned_kernel(GM_ADDR bindings) {
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
    auto counts=reinterpret_cast<__gm__ uint32_t*>(args[26]);
    if(counts[batch]<=kPlanCandidates)
      Simt::VF_CALL<Update<false>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,
                                  reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
    else
      Simt::VF_CALL<Update<true>>(Simt::Dim3{kThreads,1,1},args,batch,owners[batch],true,
                                 reinterpret_cast<__ubuf__ uint32_t*>(local.GetPhyAddr()));
  }
#endif
}
extern "C" __global__ __aicore__ void kvip_ubmem_direct_update_outputs_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
    uint32_t batch = GetBlockIdx();
    Simt::VF_CALL<PublishOutputs>(Simt::Dim3{kThreads, 1, 1}, args, batch, GetBlockNum());
  }
#endif
}


extern "C" __global__ __aicore__ void kvip_ubmem_direct_apply_owned_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    Apply<true>(bindings);
  }
#endif
}
