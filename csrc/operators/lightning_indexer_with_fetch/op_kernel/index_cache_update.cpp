// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#ifndef KVIP_OPT_OUTPUT_BULK
#define KVIP_OPT_OUTPUT_BULK 0
#endif
#if KVIP_OPT_OUTPUT_BULK < 0 || KVIP_OPT_OUTPUT_BULK > 2
#error "KVIP_OPT_OUTPUT_BULK must be 0, 1 or 2"
#endif
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
// Each block owns four TND output rows, independently of their packed owners.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void PublishOutputs(__gm__ uint64_t* a, uint32_t b) {
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
    uint64_t owner = active ? asc_ldcg(owners + sourceRow / 4) : 0;
    for (uint32_t k = threadIdx.x; k < 2048; k += 512) {
      int32_t token = active ? asc_ldcg(topk + sourceRow * 2048 + k) : -1;
      int32_t id = token >= 0 ? static_cast<int32_t>(asc_ldcg(logical + owner * uint32_t(a[24]) + token)) : -1;
      int32_t slot = active ? asc_ldcg(slots + sourceRow * 2048 + k) : -1;
      asc_stcg(logicalOut + uint64_t(row) * 2048 + k, id);
      asc_stcg(slotsOut + uint64_t(row) * 2048 + k, slot);
    }
  }
}
#ifndef KVIP_OPT_OUTPUT_PINGPONG
#define KVIP_OPT_OUTPUT_PINGPONG 0
#endif
#if KVIP_OPT_OUTPUT_PINGPONG && KVIP_OPT_OUTPUT_BULK != 1
#error "KVIP_OPT_OUTPUT_PINGPONG requires per-query bulk output"
#endif
#if KVIP_OPT_OUTPUT_BULK
// Only logical-ID lookup remains indirect. Token/slot rows are staged by MTE2
// and both complete output rows are published by MTE3 below.
__simt_callee__ inline int32_t LookupOutputLogical(__gm__ int64_t* logical, int32_t token) {
  return token >= 0 ? static_cast<int32_t>(asc_ldcg(logical + token)) : -1;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherOutputLogical(
    __gm__ int64_t* logical, __ubuf__ int32_t* topk, __ubuf__ int32_t* ids) {
  for (uint32_t k = threadIdx.x; k < kPlanCandidates; k += kThreads)
    ids[k] = LookupOutputLogical(logical, topk[k]);
}
__aicore__ inline void PublishOutputsBulk(__gm__ uint64_t* a, uint32_t b,
                                         LocalTensor<uint32_t> scratch) {
  static_assert(3 * kPlanCandidates * sizeof(int32_t) <= kUpdateScratchBytes,
                "Output staging must reuse the existing Update scratch");
  auto topkLocal = scratch.ReinterpretCast<int32_t>();
  auto slotsLocal = scratch[kPlanCandidates].ReinterpretCast<int32_t>();
  auto logicalLocal = scratch[2 * kPlanCandidates].ReinterpretCast<int32_t>();
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  auto logical = reinterpret_cast<__gm__ int64_t*>(a[3]);
  GlobalTensor<int32_t> topk, slots, logicalOut, slotsOut;
  topk.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[18]));
  slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[19]));
  logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[22]));
  slotsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[23]));
  for (uint32_t q = 0; q < 4; ++q) {
    uint32_t row = b * 4 + q;
    bool active = valid[row] != 0;
    if (active) {
      uint64_t sourceRow = unpack[row], owner = owners[sourceRow / 4];
      DataCopy(topkLocal, topk[sourceRow * kPlanCandidates], kPlanCandidates);
      DataCopy(slotsLocal, slots[sourceRow * kPlanCandidates], kPlanCandidates);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Simt::VF_CALL<GatherOutputLogical>(Simt::Dim3{kThreads, 1, 1},
          logical + owner * uint32_t(a[24]),
          reinterpret_cast<__ubuf__ int32_t*>(topkLocal.GetPhyAddr()),
          reinterpret_cast<__ubuf__ int32_t*>(logicalLocal.GetPhyAddr()));
    } else {
      // Inactive unpack/source rows may be poison; never dereference them.
      Duplicate(logicalLocal, int32_t(-1), kPlanCandidates);
      Duplicate(slotsLocal, int32_t(-1), kPlanCandidates);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(logicalOut[uint64_t(row) * kPlanCandidates], logicalLocal, kPlanCandidates);
    DataCopy(slotsOut[uint64_t(row) * kPlanCandidates], slotsLocal, kPlanCandidates);
    // The next query can write these same buffers with either MTE2 or V.
    // Join MTE3 on scalar control before either path, and before Update reuses
    // all scratch. No output DMA remains live when this wrapper returns.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }
}
template <bool Pipeline>
__aicore__ inline void PublishOutputsPipelined(__gm__ uint64_t* a, uint32_t b,
                                         LocalTensor<uint32_t> scratch) {
  constexpr uint32_t bankWords = 3 * kPlanCandidates, banks = Pipeline ? 2 : 1;
  static_assert(banks * bankWords * sizeof(uint32_t) <= kExtendedScratchBytes,
                "Output banks must fit the existing Update allocation");
  auto pipe = GetTPipePtr();
  TEventID inputEvent = pipe->AllocEventID<HardEvent::MTE2_V>();
  TEventID publishEvent = pipe->AllocEventID<HardEvent::V_MTE3>();
  TEventID done[2];
  bool pending[2] = {false, false};
  for (uint32_t bank = 0; bank < banks; ++bank)
    done[bank] = pipe->AllocEventID<HardEvent::MTE3_S>();
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  auto logical = reinterpret_cast<__gm__ int64_t*>(a[3]);
  GlobalTensor<int32_t> topk, slots, logicalOut, slotsOut;
  topk.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[18]));
  slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[19]));
  logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[22]));
  slotsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[23]));
  for (uint32_t q = 0; q < 4; ++q) {
    uint32_t bank = Pipeline ? q % 2 : 0;
    if (pending[bank]) {
      WaitFlag<HardEvent::MTE3_S>(done[bank]);
      pending[bank] = false;
    }
    auto topkLocal = scratch[bank * bankWords].ReinterpretCast<int32_t>();
    auto slotsLocal = scratch[bank * bankWords + kPlanCandidates].ReinterpretCast<int32_t>();
    auto logicalLocal = scratch[bank * bankWords + 2 * kPlanCandidates].ReinterpretCast<int32_t>();
    uint32_t row = b * 4 + q;
    // This is AICore scalar control, not a SIMT callee. asc_ldcg is legal
    // inside GatherOutputLogical/PublishOutputs only.
    if (valid[row] != 0) {
      uint64_t sourceRow = unpack[row], owner = owners[sourceRow / 4];
      DataCopy(topkLocal, topk[sourceRow * kPlanCandidates], kPlanCandidates);
      DataCopy(slotsLocal, slots[sourceRow * kPlanCandidates], kPlanCandidates);
      SetFlag<HardEvent::MTE2_V>(inputEvent);
      WaitFlag<HardEvent::MTE2_V>(inputEvent);
      Simt::VF_CALL<GatherOutputLogical>(Simt::Dim3{kThreads, 1, 1},
          logical + owner * uint32_t(a[24]),
          reinterpret_cast<__ubuf__ int32_t*>(topkLocal.GetPhyAddr()),
          reinterpret_cast<__ubuf__ int32_t*>(logicalLocal.GetPhyAddr()));
    } else {
      // Padding may carry poison source/owner pointers.
      Duplicate(logicalLocal, int32_t(-1), kPlanCandidates);
      Duplicate(slotsLocal, int32_t(-1), kPlanCandidates);
    }
    SetFlag<HardEvent::V_MTE3>(publishEvent);
    WaitFlag<HardEvent::V_MTE3>(publishEvent);
    DataCopy(logicalOut[uint64_t(row) * kPlanCandidates], logicalLocal, kPlanCandidates);
    DataCopy(slotsOut[uint64_t(row) * kPlanCandidates], slotsLocal, kPlanCandidates);
    SetFlag<HardEvent::MTE3_S>(done[bank]);
    pending[bank] = true;
  }
  for (uint32_t bank = 0; bank < banks; ++bank) {
    if (pending[bank]) WaitFlag<HardEvent::MTE3_S>(done[bank]);
    pipe->ReleaseEventID<HardEvent::MTE3_S>(done[bank]);
  }
  pipe->ReleaseEventID<HardEvent::V_MTE3>(publishEvent);
  pipe->ReleaseEventID<HardEvent::MTE2_V>(inputEvent);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void GatherPackedOutputLogical(
    __gm__ int64_t* logical, uint32_t history, __ubuf__ uint32_t* metadata,
    __ubuf__ int32_t* topk, __ubuf__ int32_t* ids) {
  for (uint32_t i = threadIdx.x; i < 4 * kPlanCandidates; i += kThreads) {
    uint32_t q = i / kPlanCandidates;
    // An inactive bank of topk is uninitialized and must not be loaded.
    ids[i] = metadata[2 * q] ?
        LookupOutputLogical(logical + uint64_t(metadata[2 * q + 1]) * history, topk[i]) : -1;
  }
}
__aicore__ inline void PublishOutputsPacked(__gm__ uint64_t* a, uint32_t b,
                                           LocalTensor<uint32_t> scratch) {
  constexpr uint32_t rowWords = 4 * kPlanCandidates;
  static_assert(3 * rowWords + 8 <= kPlanWords + 128,
                "Packed output staging and metadata must fit Update scratch");
  auto topkLocal = scratch.ReinterpretCast<int32_t>();
  auto slotsLocal = scratch[rowWords].ReinterpretCast<int32_t>();
  auto logicalLocal = scratch[2 * rowWords].ReinterpretCast<int32_t>();
  auto metadata = scratch[3 * rowWords];
  auto unpack = reinterpret_cast<__gm__ int64_t*>(a[20]);
  auto valid = reinterpret_cast<__gm__ int64_t*>(a[21]);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(a[9]);
  GlobalTensor<int32_t> topk, slots, logicalOut, slotsOut;
  topk.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[18]));
  slots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[19]));
  logicalOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[22]));
  slotsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a[23]));
  for (uint32_t q = 0; q < 4; ++q) {
    uint32_t row = b * 4 + q;
    bool active = valid[row] != 0;
    metadata.SetValue(2 * q, uint32_t(active));
    metadata.SetValue(2 * q + 1, 0U);
    if (active) {
      uint64_t sourceRow = unpack[row];
      metadata.SetValue(2 * q + 1, owners[sourceRow / 4]);
      DataCopy(topkLocal[q * kPlanCandidates], topk[sourceRow * kPlanCandidates], kPlanCandidates);
      DataCopy(slotsLocal[q * kPlanCandidates], slots[sourceRow * kPlanCandidates], kPlanCandidates);
    } else {
      Duplicate(slotsLocal[q * kPlanCandidates], int32_t(-1), kPlanCandidates);
    }
  }
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Simt::VF_CALL<GatherPackedOutputLogical>(Simt::Dim3{kThreads, 1, 1},
      reinterpret_cast<__gm__ int64_t*>(a[3]), uint32_t(a[24]),
      reinterpret_cast<__ubuf__ uint32_t*>(metadata.GetPhyAddr()),
      reinterpret_cast<__ubuf__ int32_t*>(topkLocal.GetPhyAddr()),
      reinterpret_cast<__ubuf__ int32_t*>(logicalLocal.GetPhyAddr()));
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(logicalOut[uint64_t(b) * rowWords], logicalLocal, rowWords);
  DataCopy(slotsOut[uint64_t(b) * rowWords], slotsLocal, rowWords);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
#endif
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
__aicore__ inline void UpdateOutputs(GM_ADDR bindings) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, kExtendedScratchBytes);
  auto local = scratch.Get<uint32_t>();
  auto args = reinterpret_cast<__gm__ uint64_t*>(bindings);
  auto owners = reinterpret_cast<__gm__ uint32_t*>(args[9]);
  uint32_t batch = GetBlockIdx();
#if KVIP_OPT_OUTPUT_BULK == 1
#if KVIP_OPT_OUTPUT_PINGPONG
  PublishOutputsPipelined<true>(args, batch, local);
#else
  PublishOutputsBulk(args, batch, local);
#endif
#elif KVIP_OPT_OUTPUT_BULK == 2
  PublishOutputsPacked(args, batch, local);
#else
  Simt::VF_CALL<PublishOutputs>(Simt::Dim3{kThreads, 1, 1}, args, batch);
#endif
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
extern "C" __global__ __aicore__ void kvop_direct_update_outputs_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    UpdateOutputs<true>(bindings);
  }
#endif
}
extern "C" __global__ __aicore__ void kvop_direct_update_outputs_early_kernel(GM_ADDR bindings) {
#if defined(__CCE_AICORE__)
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if ASCEND_IS_AIV {
    UpdateOutputs<false>(bindings);
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
