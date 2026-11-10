#pragma once
// SPDX-License-Identifier: Apache-2.0

// Producer changes form an injective partial permutation. Close its paths by
// pairing outgoing destinations with freed source rows; cycles need no extras.
constexpr uint32_t kPlanCandidates = 2048, kPlanHash = 8192;
constexpr uint32_t kPlanRecords = 2 * kPlanCandidates;
constexpr uint32_t kPlanWords = 2 * kPlanHash + 2 * kPlanRecords;
constexpr uint32_t kUpdateScratchBytes = (kPlanWords + 128) * 4;
// Original v9 final_sparse.h::SparseHash; only the helper name is adapted.
__simt_callee__ inline uint32_t PlanHash(uint32_t token) { return (token * 0x9e3779b9U) >> 19; }
__simt_callee__ inline bool PlanInsert(__ubuf__ uint32_t* table, uint32_t key) {
  uint32_t i = PlanHash(key);
  for (uint32_t n = 0; n < kPlanHash; ++n) {
    uint32_t old = asc_atomic_cas(table + i, UINT32_MAX, key);
    if (old == UINT32_MAX) return true;
    if (old == key) return false;
    i = (i + 1) & (kPlanHash - 1);
  }
  return false;
}
__simt_callee__ inline bool PlanContains(__ubuf__ uint32_t* table, uint32_t key) {
  uint32_t i = PlanHash(key);
  for (uint32_t n = 0; n < kPlanHash; ++n) {
    uint32_t old = table[i];
    if (old == key) return true;
    if (old == UINT32_MAX) return false;
    i = (i + 1) & (kPlanHash - 1);
  }
  return false;
}
template <uint32_t Threads>
__simt_callee__ inline uint32_t PlanPrefix(uint32_t n, __ubuf__ uint32_t* scan) {
  if constexpr (Threads == 1) return 0;  // Host execution of the same planner.
  uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32, value = n;
  for (uint32_t d = 1; d < 32; d *= 2) {
    uint32_t prior = asc_shfl_up(value, d);
    if (lane >= d) value += prior;
  }
  if (lane == 31) scan[warp] = value;
  asc_syncthreads();
  if (!warp) {
    uint32_t v = lane < Threads / 32 ? scan[lane] : 0;
    for (uint32_t d = 1; d < 32; d *= 2) {
      uint32_t prior = asc_shfl_up(v, d);
      if (lane >= d) v += prior;
    }
    scan[lane] = v;
  }
  asc_syncthreads();
  return value - n + (warp ? scan[warp - 1] : 0);
}
template <uint32_t Threads>
__simt_callee__ inline void BuildUpdatePlan(__gm__ uint32_t* lru, __gm__ uint32_t* changes, uint32_t history,
                                            uint32_t resident, __ubuf__ uint32_t* work, __ubuf__ uint32_t* status) {
  uint32_t tid = threadIdx.x;
  auto dstSet = work, srcSet = work + kPlanHash;
  auto dst = work + 2 * kPlanHash, src = dst + kPlanRecords;
  for (uint32_t i = tid; i < kPlanWords; i += Threads) work[i] = UINT32_MAX;
  asc_syncthreads();
  for (uint32_t i = tid; i < kPlanCandidates; i += Threads) {
    uint32_t s = asc_ldcg(changes + i), d = asc_ldcg(lru + i);
    if (s == UINT32_MAX) continue;
    if (s >= history || d >= resident) {
      asc_atomic_or(status, 1U);
      continue;
    }
    src[i] = s;
    dst[i] = d;
    if (!PlanInsert(dstSet, d) || !PlanInsert(srcSet, s)) asc_atomic_or(status, 1U);
  }
  asc_syncthreads();
  if (status[0]) return;
  // Consecutive per-thread ranges retain the CPU oracle's record order.
  constexpr uint32_t per = kPlanCandidates / Threads;
  uint32_t out = 0, free = 0;
  for (uint32_t i = tid * per; i < (tid + 1) * per; ++i)
    if (src[i] != UINT32_MAX) {
      out += !PlanContains(srcSet, dst[i]);
      free += !PlanContains(dstSet, src[i]);
    }
  uint32_t outRank = kPlanCandidates + PlanPrefix<Threads>(out, status + 32);
  uint32_t freeRank = kPlanCandidates + PlanPrefix<Threads>(free, status + 64);
  for (uint32_t i = tid * per; i < (tid + 1) * per; ++i)
    if (src[i] != UINT32_MAX) {
      if (!PlanContains(srcSet, dst[i])) dst[outRank++] = dst[i];
      if (!PlanContains(dstSet, src[i])) src[freeRank++] = src[i];
    }
  // Extras move old outgoing destinations to the freed source rows.
  asc_syncthreads();
  for (uint32_t i = kPlanCandidates + tid; i < kPlanRecords; i += Threads) {
    uint32_t outgoing = dst[i];
    dst[i] = src[i];
    src[i] = outgoing;
  }
  asc_syncthreads();
}

// Extended records use the same partial-permutation planner. Only records
// spill to GM; hash tables remain in UB. The original <=2048 path is unchanged.
constexpr uint32_t kExtendedCandidates = 2048 + 8192;
constexpr uint32_t kExtendedRecords = 2 * kExtendedCandidates;
constexpr uint32_t kExtendedHash = 16384;
constexpr uint32_t kExtendedScratchBytes = (2 * kExtendedHash + 128) * 4;
constexpr uint32_t kExtendedWorkWords = 2 * kExtendedRecords;
__simt_callee__ inline bool ExtendedInsert(__ubuf__ uint32_t* table, uint32_t key) {
  uint32_t i = (key * 0x9e3779b9U) >> 18;
  for (uint32_t n = 0; n < kExtendedHash; ++n) {
    uint32_t old = asc_atomic_cas(table + i, UINT32_MAX, key);
    if (old == UINT32_MAX) return true;
    if (old == key) return false;
    i = (i + 1) & (kExtendedHash - 1);
  }
  return false;
}
__simt_callee__ inline bool ExtendedContains(__ubuf__ uint32_t* table, uint32_t key) {
  uint32_t i = (key * 0x9e3779b9U) >> 18;
  for (uint32_t n = 0; n < kExtendedHash; ++n) {
    uint32_t old = table[i];
    if (old == key) return true;
    if (old == UINT32_MAX) return false;
    i = (i + 1) & (kExtendedHash - 1);
  }
  return false;
}
template<uint32_t Threads>
__simt_callee__ inline void BuildExtendedPlan(__gm__ uint32_t* inputSlots, __gm__ uint32_t* changes,
    uint32_t count, uint32_t history, uint32_t resident, __ubuf__ uint32_t* hashes,
    __ubuf__ uint32_t* status, __gm__ uint32_t* records) {
  uint32_t tid = threadIdx.x;
  auto dstSet = hashes, srcSet = hashes + kExtendedHash;
  auto dst = records, src = records + kExtendedRecords;
  for (uint32_t i = tid; i < 2*kExtendedHash; i += Threads) hashes[i] = UINT32_MAX;
  for (uint32_t i = tid; i < 2*kExtendedRecords; i += Threads) asc_stcg(records+i, UINT32_MAX);
  asc_threadfence();asc_syncthreads();
  if(count > kExtendedCandidates) { if(!tid)status[0]=1; return; }
  for(uint32_t i=tid;i<count;i+=Threads) {
    uint32_t s=asc_ldcg(changes+i),d=asc_ldcg(inputSlots+i);
    if(s==UINT32_MAX)continue;
    if(s>=history || d>=resident) {asc_atomic_or(status,1U);continue;}
    asc_stcg(src+i,s);asc_stcg(dst+i,d);
    if(!ExtendedInsert(dstSet,d) || !ExtendedInsert(srcSet,s))asc_atomic_or(status,1U);
  }
  asc_threadfence();asc_syncthreads();
  if(status[0])return;
  uint32_t per=(count+Threads-1)/Threads,out=0,free=0;
  for(uint32_t i=tid*per;i<(tid+1)*per && i<count;++i) {
    uint32_t s=asc_ldcg(src+i),d=asc_ldcg(dst+i);
    if(s!=UINT32_MAX) {out+=!ExtendedContains(srcSet,d);free+=!ExtendedContains(dstSet,s);}
  }
  uint32_t outRank=kExtendedCandidates+PlanPrefix<Threads>(out,status+32);
  uint32_t freeRank=kExtendedCandidates+PlanPrefix<Threads>(free,status+64);
  for(uint32_t i=tid*per;i<(tid+1)*per && i<count;++i) {
    uint32_t s=asc_ldcg(src+i),d=asc_ldcg(dst+i);
    if(s!=UINT32_MAX) {
      if(!ExtendedContains(srcSet,d))asc_stcg(src+outRank++,d);
      if(!ExtendedContains(dstSet,s))asc_stcg(dst+freeRank++,s);
    }
  }
  asc_threadfence();asc_syncthreads();
}

// SPDX-License-Identifier: Apache-2.0

constexpr uint32_t kCopyWorkers = 4, kKeyFieldsPerWorker = 4, kMetadataFields = 3;
constexpr uint32_t kTransferHeaderWords = 8, kTransferPlanWords = kTransferHeaderWords + 2 * kPlanRecords;
constexpr uint32_t kInvalidateHot = 0x80000000U;
constexpr uint32_t kMoveScratchBytes = kTransferPlanWords * 4 + kPlanRecords * kKeyFieldsPerWorker * 8;
static_assert(kMoveScratchBytes <= 223232, "Move scratch exceeds reserved UB");
static_assert(kExtendedRecords * sizeof(uint64_t) <= kMoveScratchBytes,
              "Extended field snapshot exceeds move scratch");
template<uint32_t Threads>
__simt_callee__ inline void PublishUpdatePlan(__ubuf__ uint32_t* work, __ubuf__ uint32_t* status,
                                             __gm__ uint32_t* plan, uint32_t resident) {
  uint32_t tid = threadIdx.x;
  auto destinations = work + 2 * kPlanHash;
  auto sources = destinations + kPlanRecords;
  // Compact the original planner's records once, retaining their order.
  constexpr uint32_t per = kPlanRecords / Threads;
  uint32_t count = 0;
  for (uint32_t i = tid * per; i < (tid + 1) * per; ++i)
    count += sources[i] != UINT32_MAX && sources[i] != destinations[i];
  uint32_t rank = PlanPrefix<Threads>(count, status + 32);
  if (tid == Threads - 1) asc_stcg(plan, rank + count);
  for (uint32_t i = tid * per; i < (tid + 1) * per; ++i) {
    uint32_t src = sources[i], dst = destinations[i];
    if (src == UINT32_MAX || src == dst) continue;
    if (i >= kPlanCandidates && dst < resident) dst |= kInvalidateHot;
    asc_stcg(plan + kTransferHeaderWords + 2 * rank, dst);
    asc_stcg(plan + kTransferHeaderWords + 2 * rank + 1, src);
    ++rank;
  }
}
constexpr uint32_t kExtendedTransferWords = kTransferHeaderWords + 2*kExtendedRecords;
template<uint32_t Threads>
__simt_callee__ inline void PublishExtendedPlan(__gm__ uint32_t* records, __ubuf__ uint32_t* status,
    __gm__ uint32_t* plan,uint32_t resident) {
  uint32_t tid=threadIdx.x,per=kExtendedRecords/Threads,count=0;
  auto destinations=records,sources=records+kExtendedRecords;
  for(uint32_t i=tid*per;i<(tid+1)*per;++i) {
    uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(destinations+i);
    count+=src!=UINT32_MAX && src!=dst;
  }
  uint32_t rank=PlanPrefix<Threads>(count,status+32);
  if(tid==Threads-1)asc_stcg(plan,rank+count);
  for(uint32_t i=tid*per;i<(tid+1)*per;++i) {
    uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(destinations+i);
    if(src==UINT32_MAX || src==dst)continue;
    if(i>=kExtendedCandidates && dst<resident)dst|=kInvalidateHot;
    asc_stcg(plan+kTransferHeaderWords+2*rank,dst);
    asc_stcg(plan+kTransferHeaderWords+2*rank+1,src);++rank;
  }
}
// Snapshot one field across every record before scattering that field. This
// retains arbitrary cycles with the original four workers and bounded UB.
template<uint32_t Threads, bool BalancedMetadata = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void MoveExtended(__gm__ uint64_t* a,uint32_t owner,uint32_t group,bool owned,
    __gm__ uint32_t* plan,__ubuf__ uint64_t* staged,uint32_t count) {
  uint32_t tid=threadIdx.x;
  auto keys=reinterpret_cast<__gm__ uint64_t*>(a[0])+uint64_t(owner)*uint32_t(a[24])*16;
  for(uint32_t field=0;field<kKeyFieldsPerWorker;++field) {
    for(uint32_t i=tid;i<count;i+=Threads) {
      uint32_t src=asc_ldcg(plan+kTransferHeaderWords+2*i+1);
      staged[i]=asc_ldcg(keys+uint64_t(src)*16+group*kKeyFieldsPerWorker+field);
    }
    asc_syncthreads();
    for(uint32_t i=tid;i<count;i+=Threads) {
      uint32_t dst=asc_ldcg(plan+kTransferHeaderWords+2*i)&~kInvalidateHot;
      asc_stcg(keys+uint64_t(dst)*16+group*kKeyFieldsPerWorker+field,staged[i]);
    }
    asc_syncthreads();
  }
  // Metadata planes are independent of key fields and each other. The
  // opt-in policy assigns one whole plane to each of workers 0..2, instead
  // of appending all three planes to worker 3. Every plane still snapshots
  // ALL records before scatter, preserving arbitrary permutation cycles.
  if(BalancedMetadata ? group<kMetadataFields : group==kCopyWorkers-1) {
    auto scales=reinterpret_cast<__gm__ uint32_t*>(a[1])+uint64_t(owner)*uint32_t(a[24]);
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(a[2])+uint64_t(owner)*a[7];
    auto logical=reinterpret_cast<__gm__ uint64_t*>(a[3])+uint64_t(owner)*uint32_t(a[24]);
    const uint32_t first = BalancedMetadata ? group : 0;
    const uint32_t stop = BalancedMetadata ? group+1 : kMetadataFields;
    for(uint32_t field=first;field<stop;++field) {
      for(uint32_t i=tid;i<count;i+=Threads) {
        uint32_t src=asc_ldcg(plan+kTransferHeaderWords+2*i+1);
        staged[i]=field==0?uint64_t(asc_ldcg(scales+src)):(field==1?asc_ldcg(addresses+src):asc_ldcg(logical+src));
      }
      asc_syncthreads();
      for(uint32_t i=tid;i<count;i+=Threads) {
        uint32_t dst=asc_ldcg(plan+kTransferHeaderWords+2*i)&~kInvalidateHot;
        if(field==0)asc_stcg(scales+dst,uint32_t(staged[i]));
        else if(field==1)asc_stcg(addresses+dst,staged[i]);else asc_stcg(logical+dst,staged[i]);
      }
      asc_syncthreads();
    }
  }
  if(!group) {
    auto valid=reinterpret_cast<__gm__ uint32_t*>(a[owned?16:9])+uint64_t(owner)*uint32_t(a[25]);
    for(uint32_t i=tid;i<count;i+=Threads) {
      uint32_t dst=asc_ldcg(plan+kTransferHeaderWords+2*i);
      if(dst&kInvalidateHot)asc_stcg(valid+(dst&~kInvalidateHot),uint32_t(0));
    }
  }
  asc_threadfence();
}
// Each worker owns disjoint key fields over ALL records. Snapshotting its
// fields before scatter preserves arbitrary cycles without a cross-core wait.
template<uint32_t Threads, bool BalancedMetadata = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512)
inline void Move(__gm__ uint64_t* a, uint32_t owner, uint32_t group, bool owned,
                 __ubuf__ uint32_t* plan, __ubuf__ uint64_t* staged, uint32_t count) {
  uint32_t tid = threadIdx.x;
  auto keys = reinterpret_cast<__gm__ uint64_t*>(a[0]) + uint64_t(owner) * uint32_t(a[24]) * 16;
  for (uint32_t i = tid; i < count * kKeyFieldsPerWorker; i += Threads) {
    uint32_t row = i / kKeyFieldsPerWorker, field = i % kKeyFieldsPerWorker;
    uint32_t src = plan[kTransferHeaderWords + 2 * row + 1];
    staged[i] = asc_ldcg(keys + uint64_t(src) * 16 + group * kKeyFieldsPerWorker + field);
  }
  asc_syncthreads();
  for (uint32_t i = tid; i < count * kKeyFieldsPerWorker; i += Threads) {
    uint32_t row = i / kKeyFieldsPerWorker, field = i % kKeyFieldsPerWorker;
    uint32_t dst = plan[kTransferHeaderWords + 2 * row] & ~kInvalidateHot;
    asc_stcg(keys + uint64_t(dst) * 16 + group * kKeyFieldsPerWorker + field, staged[i]);
  }
  // Metadata lives in separate arrays, so this worker can overlap other key
  // workers without overwriting any of their source bytes.
  if (BalancedMetadata ? group < kMetadataFields : group == kCopyWorkers - 1) {
    asc_syncthreads();
    auto scales = reinterpret_cast<__gm__ uint32_t*>(a[1]) + uint64_t(owner) * uint32_t(a[24]);
    auto addresses = reinterpret_cast<__gm__ uint64_t*>(a[2]) + uint64_t(owner) * a[7];
    auto logical = reinterpret_cast<__gm__ uint64_t*>(a[3]) + uint64_t(owner) * uint32_t(a[24]);
    constexpr uint32_t fields = BalancedMetadata ? 1 : kMetadataFields;
    for (uint32_t i = tid; i < count * fields; i += Threads) {
      uint32_t src = plan[kTransferHeaderWords + 2 * (i / fields) + 1];
      uint32_t field = BalancedMetadata ? group : i % fields;
      staged[i] = field == 0 ? uint64_t(asc_ldcg(scales + src)) :
                  (field == 1 ? asc_ldcg(addresses + src) : asc_ldcg(logical + src));
    }
    asc_syncthreads();
    for (uint32_t i = tid; i < count * fields; i += Threads) {
      uint32_t dst = plan[kTransferHeaderWords + 2 * (i / fields)] & ~kInvalidateHot;
      uint32_t field = BalancedMetadata ? group : i % fields;
      if (field == 0) asc_stcg(scales + dst, uint32_t(staged[i]));
      else if (field == 1) asc_stcg(addresses + dst, staged[i]);
      else asc_stcg(logical + dst, staged[i]);
    }
  }
  if (!group) {
    auto valid = reinterpret_cast<__gm__ uint32_t*>(a[owned ? 16 : 9]) + uint64_t(owner) * uint32_t(a[25]);
    for (uint32_t i = tid; i < count; i += Threads) {
      uint32_t dst = plan[kTransferHeaderWords + 2 * i];
      if (dst & kInvalidateHot) asc_stcg(valid + (dst & ~kInvalidateHot), uint32_t(0));
    }
  }
  asc_threadfence();
}

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
