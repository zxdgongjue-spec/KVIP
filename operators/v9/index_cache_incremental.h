#pragma once
#include "owned_h0.h"
namespace rotating {
// Two bounded UB lists hold changed cold tokens and their destination holes.
// No steady-state traversal of the cold bank or candidate-sized dirty array.
constexpr uint32_t kIndexDeltaCapacity = 16384;
constexpr uint32_t kIndexScratchWords = 2 * kIndexDeltaCapacity + 16;
} // namespace rotating
#include "index_cache_exchange.h"
namespace rotating {

__simt_callee__ inline void SortIndexDelta(__ubuf__ uint32_t* values, uint32_t count) {
  uint32_t n = 1, tid = threadIdx.x;
  while (n < count) n *= 2;
  for (uint32_t i = count + tid; i < n; i += 256) values[i] = UINT32_MAX;
  asc_syncthreads();
  for (uint32_t width = 2; width <= n; width *= 2) {
    for (uint32_t stride = width / 2; stride; stride /= 2) {
      for (uint32_t i = tid; i < n; i += 256) {
        uint32_t other = i ^ stride;
        if (other > i) {
          uint32_t a = values[i], b = values[other];
          if ((a > b) == ((i & width) == 0)) {
            values[i] = b;
            values[other] = a;
          }
        }
      }
      asc_syncthreads();
    }
  }
}

__simt_callee__ inline void CopyIndexJobs(__gm__ Context* ctx, uint32_t b, uint32_t jobsCount) {
  using namespace contract_simt;
  auto c = ContractOf(ctx);
  uint32_t tid = threadIdx.x;
  auto jobs = IndexPositions(c, b) + c->candidate_capacity;
  auto hot = IndexIds(c, b, true), cold = IndexIds(c, b, false), dirty = IndexDirty(c, b, true);
  auto keys = ContractPtr<uint32_t>(IndexKey(c, b, true)), scales = ContractPtr<uint32_t>(IndexScale(c, b, true));
  for (uint32_t j = tid; j < jobsCount * 32; j += 256) {
    uint32_t index = uint32_t(asc_ldcg(jobs + j / 32)), word = j % 32;
    uint32_t t = uint32_t(asc_ldcg(index < c->resident_capacity ? hot + index : cold + index - c->resident_capacity));
    uint32_t page = ContractPtr<int32_t>(c->index_table)[b * c->index_columns + t / 128], source = page * 128 + t % 128;
    asc_stcg(keys + uint64_t(index) * 32 + word,
             asc_ldcg(ContractPtr<uint32_t>(c->key) + uint64_t(source) * 32 + word));
    if (!word) asc_stcg(scales + index, asc_ldcg(ContractPtr<uint32_t>(c->key_scale) + source));
  }
  asc_threadfence();
  asc_syncthreads();
  for (uint32_t j = tid; j < jobsCount; j += 256) asc_stcg(dirty + asc_ldcg(jobs + j), 0);
  asc_threadfence();
  asc_syncthreads();
}

__simt_callee__ inline bool UpdateIndexDelta(__gm__ Context* ctx, uint32_t b, bool tail, __ubuf__ uint32_t* scratch) {
  using namespace contract_simt;
  auto c = ContractOf(ctx);
  uint32_t tid = threadIdx.x, length = ContractLength(c, b), budget = ContractBudget(c, b);
  uint32_t pool = ContractPool(c, b);
  auto state = ContractPtr<int32_t>(IndexBase(c, b));
  if (!asc_ldcg(state) || asc_ldcg(state + 4) != int32_t(pool) || asc_ldcg(state + 5) != int32_t(c->layer_id) ||
      asc_ldcg(state + 6) != int32_t(budget))
    return false;
  uint32_t oldlen = uint32_t(asc_ldcg(state + 1)), oldh = uint32_t(asc_ldcg(state + 2));
  uint32_t oldc = uint32_t(asc_ldcg(state + 3));
  // Large source resizes take the complete rebuild path; ordinary decode
  // growth, arbitrary resident replacements, and same-length rewrites do not.
  if (length > oldlen + 1024 || oldlen > length + 1024) return false;
  auto incoming = scratch, holes = scratch + kIndexDeltaCapacity;
  auto scan = scratch + 2 * kIndexDeltaCapacity;
  auto reverse = ContractPtr<int32_t>(c->reverse) + uint64_t(pool) * c->resident_capacity;
  auto forward = ContractPtr<int32_t>(c->forward) + uint64_t(pool) * c->candidate_capacity;
  auto target = ContractPtr<int32_t>(c->next_reverse) + b * c->resident_capacity;
  auto bits = tail ? ContractPtr<uint32_t>(c->scores) + uint64_t(b) * 4 * c->score_stride / 2
                   : ContractPtr<uint32_t>(c->hot_bitmap) + b * ((c->candidate_capacity + 31) / 32);
  auto desired = ContractPtr<int32_t>(c->hot_tokens) + b * c->resident_capacity;
  auto hot = IndexIds(c, b, true), cold = IndexIds(c, b, false);
  auto dirty = IndexDirty(c, b, true), pos = IndexPositions(c, b);
  auto jobs = pos + c->candidate_capacity;
  uint32_t nh = 0, chunk = 0, begin = 0, end = 0, count = 0, rank = 0;
  ContractPrefix prefix{0, 0};
  if (!tail) {
    nh = uint32_t(asc_ldcg(ContractPtr<int32_t>(c->hot_counts) + b));
  } else {
    for (uint32_t i = tid; i < (c->candidate_capacity + 31) / 32; i += 256) asc_stcg(bits + i, 0U);
    for (uint32_t s = tid; s < budget; s += 256) {
      int32_t t = asc_ldcg(reverse + s);
      bool valid = t >= 0 && uint32_t(t) < length && asc_ldcg(forward + t) == int32_t(s);
      asc_stcg(target + s, valid ? t : -1);
    }
    asc_threadfence();
    asc_syncthreads();
    if (tail) {
      // Reserved slots are complete at the LI boundary. Do not consume the
      // touched bitmap while other final workers are still adding to it.
      uint32_t npre = ContractPtr<Row>(ctx->rows)[b].slots;
      for (uint32_t i = tid; i < npre; i += 256)
        asc_stcg(target + ContractPtr<int32_t>(c->reserved_slots)[b * 2048 + i], -1);
      asc_threadfence();
      asc_syncthreads();
      uint32_t n = ContractPtr<int32_t>(c->plan_count)[b];
      for (uint32_t i = tid; i < n; i += 256)
        asc_stcg(target + ContractPtr<int32_t>(c->planned_slots)[b * kSlots + i],
                 ContractPtr<int32_t>(c->planned_tokens)[b * kSlots + i]);
      for (uint32_t i = tid; i < npre; i += 256) {
        int32_t s = ContractPtr<int32_t>(c->extra_prefetch_slots)[b * 2048 + i];
        if (s >= 0) asc_stcg(target + s, ContractPtr<int32_t>(c->reserved_tokens)[b * 2048 + i]);
      }
    }
    asc_threadfence();
    asc_syncthreads();
    chunk = (budget + 255) / 256;
    begin = tid * chunk;
    end = begin + chunk;
    count = 0;
    if (end > budget) end = budget;
    for (uint32_t s = begin; s < end; ++s) count += asc_ldcg(target + s) >= 0;
    prefix = ContractPrefixUB(count, scan);
    nh = prefix.total;
    rank = prefix.begin;
    for (uint32_t s = begin; s < end; ++s) {
      int32_t t = asc_ldcg(target + s);
      if (t >= 0) {
        asc_stcg(desired + rank++, t);
        asc_atomic_or(bits + t / 32, 1U << (t % 32));
      }
    }
    asc_threadfence();
    asc_syncthreads();
  }
  uint32_t nc = length - nh;
  // The overwhelmingly common init case: valid resident membership is
  // unchanged. Refresh only the caller's dirty logical interval, via pos[].
  if (length == oldlen && nh == oldh) {
    count = 0;
    for (uint32_t i = tid; i < oldh; i += 256) {
      int32_t t = asc_ldcg(hot + i);
      count += !((asc_ldcg(bits + t / 32) >> (t % 32)) & 1U);
    }
    prefix = ContractPrefixUB(count, scan);
    if (!prefix.total) {
      uint32_t lo = tail ? 0 : uint32_t(ContractPtr<int32_t>(c->index_dirty)[b * 2]);
      uint32_t hi = tail ? 0 : uint32_t(ContractPtr<int32_t>(c->index_dirty)[b * 2 + 1]);
      if (lo > length) lo = length;
      if (hi > length) hi = length;
      for (uint32_t i = tid; i < hi - lo; i += 256) {
        int32_t at = asc_ldcg(pos + lo + i);
        asc_stcg(jobs + i, at <= -2 ? -at - 2 : int32_t(c->resident_capacity) + at);
      }
      asc_threadfence();
      asc_syncthreads();
      CopyIndexJobs(ctx, b, hi - lo);
      if (!tid) {
        asc_stcg(state + 7, int32_t(hi - lo));
        asc_stcg(state + 17, 2);  // Stable membership.
        if (!tail) asc_stcg(ContractPtr<int32_t>(c->cold_counts) + b, int32_t(nc));
        asc_stcg(state, tail ? 0 : 1);
      }
      asc_threadfence();
      return true;
    }
  }
  // Cold newcomers: demoted hot rows, survivors displaced by a shorter cold
  // bank, and new logical rows. These disjoint sources avoid a history scan.
  uint32_t displaced = oldc > nc ? oldc - nc : 0, added = length > oldlen ? length - oldlen : 0;
  uint32_t extent = oldh + displaced + added;
  count = 0;
  for (uint32_t i = tid; i < extent; i += 256) {
    int32_t t = i < oldh
                    ? asc_ldcg(hot + i)
                    : (i < oldh + displaced ? asc_ldcg(cold + nc + i - oldh) : int32_t(oldlen + i - oldh - displaced));
    if (t >= 0 && uint32_t(t) < length && !((asc_ldcg(bits + t / 32) >> (t % 32)) & 1U)) ++count;
  }
  prefix = ContractPrefixUB(count, scan);
  uint32_t coldChanges = prefix.total;
  if (coldChanges > kIndexDeltaCapacity) return false;
  rank = prefix.begin;
  for (uint32_t i = tid; i < extent; i += 256) {
    int32_t t = i < oldh
                    ? asc_ldcg(hot + i)
                    : (i < oldh + displaced ? asc_ldcg(cold + nc + i - oldh) : int32_t(oldlen + i - oldh - displaced));
    if (t >= 0 && uint32_t(t) < length && !((asc_ldcg(bits + t / 32) >> (t % 32)) & 1U)) incoming[rank++] = uint32_t(t);
  }
  // Cold holes: promoted rows, removed logical rows, and newly exposed slots.
  uint32_t removed = oldlen > length ? oldlen - length : 0, extended = nc > oldc ? nc - oldc : 0;
  extent = nh + removed + extended;
  count = 0;
  for (uint32_t i = tid; i < extent; i += 256) {
    int32_t at = i < nh ? asc_ldcg(pos + asc_ldcg(desired + i))
                        : (i < nh + removed ? asc_ldcg(pos + length + i - nh) : int32_t(oldc + i - nh - removed));
    if (at >= 0 && uint32_t(at) < nc) ++count;
  }
  prefix = ContractPrefixUB(count, scan);
  if (prefix.total != coldChanges) return false;  // Before any cache mutation.
  rank = prefix.begin;
  for (uint32_t i = tid; i < extent; i += 256) {
    int32_t at = i < nh ? asc_ldcg(pos + asc_ldcg(desired + i))
                        : (i < nh + removed ? asc_ldcg(pos + length + i - nh) : int32_t(oldc + i - nh - removed));
    if (at >= 0 && uint32_t(at) < nc) holes[rank++] = uint32_t(at);
  }
  asc_syncthreads();
  SortIndexDelta(incoming, coldChanges);
  SortIndexDelta(holes, coldChanges);
  // Preserve surviving hot positions and fill holes in resident-slot order.
  chunk = (nh + 255) / 256;
  begin = tid * chunk;
  end = begin + chunk;
  if (end > nh) end = nh;
  count = 0;
  for (uint32_t i = begin; i < end; ++i) {
    int32_t at = asc_ldcg(pos + asc_ldcg(desired + i));
    count += !(at <= -2 && uint32_t(-at - 2) < nh);
  }
  prefix = ContractPrefixUB(count, scan);
  rank = prefix.begin;
  for (uint32_t i = begin; i < end; ++i) {
    int32_t t = asc_ldcg(desired + i), at = asc_ldcg(pos + t);
    if (!(at <= -2 && uint32_t(-at - 2) < nh)) asc_stcg(jobs + rank++, t);
  }
  count = 0;
  for (uint32_t i = begin; i < end; ++i) {
    int32_t t = i < oldh ? asc_ldcg(hot + i) : -1;
    bool member = t >= 0 && uint32_t(t) < length && ((asc_ldcg(bits + t / 32) >> (t % 32)) & 1U);
    asc_stcg(dirty + i, member ? 0 : 1);
    count += !member;
  }
  prefix = ContractPrefixUB(count, scan);
  rank = prefix.begin;
  uint32_t hotChanges = prefix.total;
  asc_threadfence();
  asc_syncthreads();
  for (uint32_t i = begin; i < end; ++i)
    if (asc_ldcg(dirty + i)) {
      int32_t t = asc_ldcg(jobs + rank++);
      asc_stcg(hot + i, t);
      asc_stcg(pos + t, -int32_t(i) - 2);
    }
  for (uint32_t i = tid; i < coldChanges; i += 256) {
    uint32_t at = holes[i], t = incoming[i];
    asc_stcg(cold + at, int32_t(t));
    asc_stcg(pos + t, int32_t(at));
    asc_stcg(dirty + c->resident_capacity + at, 1);
  }
  for (uint32_t t = length + tid; t < oldlen; t += 256) asc_stcg(pos + t, -1);
  asc_threadfence();
  asc_syncthreads();
  rank = prefix.begin;
  for (uint32_t i = begin; i < end; ++i)
    if (asc_ldcg(dirty + i)) asc_stcg(jobs + rank++, int32_t(i));
  for (uint32_t i = tid; i < coldChanges; i += 256)
    asc_stcg(jobs + hotChanges + i, int32_t(c->resident_capacity + holes[i]));
  // Source rewrites address rows through the reverse position directory.
  // Already moved rows are copied once, not once for each reason to be dirty.
  uint32_t lo = tail ? 0 : uint32_t(ContractPtr<int32_t>(c->index_dirty)[b * 2]);
  uint32_t hi = tail ? 0 : uint32_t(ContractPtr<int32_t>(c->index_dirty)[b * 2 + 1]);
  if (hi > length) hi = length;
  count = 0;
  for (uint32_t t = lo + tid; t < hi; t += 256) {
    int32_t at = asc_ldcg(pos + t);
    uint32_t index = at <= -2 ? uint32_t(-at - 2) : c->resident_capacity + uint32_t(at);
    if (!asc_ldcg(dirty + index)) ++count;
  }
  prefix = ContractPrefixUB(count, scan);
  rank = hotChanges + coldChanges + prefix.begin;
  for (uint32_t t = lo + tid; t < hi; t += 256) {
    int32_t at = asc_ldcg(pos + t);
    uint32_t index = at <= -2 ? uint32_t(-at - 2) : c->resident_capacity + uint32_t(at);
    if (!asc_ldcg(dirty + index)) asc_stcg(jobs + rank++, int32_t(index));
  }
  uint32_t jobsCount = hotChanges + coldChanges + prefix.total;
  asc_threadfence();
  asc_syncthreads();
  CopyIndexJobs(ctx, b, jobsCount);
  if (!tid) {
    asc_stcg(state + 1, int32_t(length));
    asc_stcg(state + 2, int32_t(nh));
    asc_stcg(state + 3, int32_t(nc));
    asc_stcg(state + 7, int32_t(jobsCount));
    asc_stcg(state + 17, 1);  // Incremental path.
    if (!tail) asc_stcg(ContractPtr<int32_t>(c->cold_counts) + b, int32_t(nc));
    asc_stcg(state, tail ? 0 : 1);
  }
  asc_threadfence();
  return true;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ReconcileIndexCache(__gm__ Context* ctx, uint32_t b, bool tail,
                                                                         __ubuf__ uint32_t* scratch) {
  using namespace contract_simt;
  if (asc_ldcg(ContractPtr<int32_t>(ContractOf(ctx)->status) + b) != kPending) return;
  if(tail && ExchangeIndexTail(ctx,b,scratch,true)) return;
  auto c=ContractOf(ctx);
  if(IndexOwned(c,b)) {
    auto state=ContractPtr<int32_t>(IndexBase(c,b));
    uint32_t nh=uint32_t(asc_ldcg(state+2)), count=0;
    auto bits=ContractPtr<uint32_t>(c->hot_bitmap)+b*((c->candidate_capacity+31)/32);
    auto hot=IndexIds(c,b,true);
    // Init has already checked resident -> position -> hot ID while building
    // H0. Consume its one-call proof; standalone reconciliation still verifies.
    int32_t checked=tail?0:asc_ldcg(state+19);
    if(!checked)for(uint32_t i=threadIdx.x;i<nh;i+=256) {
      int32_t t=asc_ldcg(hot+i);
      count+=t<0 || uint32_t(t)>=ContractLength(c,b) || !((asc_ldcg(bits+t/32)>>(t%32))&1U);
    }
    ContractPrefix sum{0,0};
    if(!checked)sum=ContractPrefixUB(count,scratch);
    bool membership=checked?checked==1:!sum.total;
    asc_syncthreads();
    if(!threadIdx.x)asc_stcg(state+19,0);
    bool identity=!tail && asc_ldcg(state)==1 && asc_ldcg(state+1)==int32_t(ContractLength(c,b)) &&
      asc_ldcg(state+4)==int32_t(ContractPool(c,b)) && asc_ldcg(state+5)==int32_t(c->layer_id) &&
      asc_ldcg(state+6)==int32_t(ContractBudget(c,b));
    bool valid=identity && nh==uint32_t(ContractPtr<int32_t>(c->hot_counts)[b]) && membership;
    // A newly admitted tail token may already have a valid resident owner.
    // Init has zeroed the final plan/reservations, so this reconciles H0 only.
    bool exchanged=false;
    if(identity && !valid) { valid=ExchangeIndexTail(ctx,b,scratch); exchanged=valid; }
    if(!threadIdx.x) {
      if(valid) {
        if(!exchanged) { asc_stcg(state+7,0); asc_stcg(state+17,2); }
        asc_stcg(state,1);
        asc_stcg(ContractPtr<int32_t>(c->cold_counts)+b,asc_ldcg(state+3));
      } else {
        asc_stcg(ContractPtr<int32_t>(c->status)+b,int32_t(kInvalid));
        asc_stcg(ContractPtr<int32_t>(c->error_detail)+b*4,int32_t(kInvalid));
        asc_stcg(ContractPtr<int32_t>(c->sticky),int32_t(kInvalid));
      }
    }
    asc_threadfence();asc_syncthreads();
    if(valid)SnapshotOwnedH0(ctx,b);
    return;
  }
  if (!UpdateIndexDelta(ctx, b, tail, scratch)) {
    RebuildIndexCache(ctx, b, tail, scratch);
    if (!threadIdx.x) asc_stcg(ContractPtr<int32_t>(IndexBase(ContractOf(ctx), b)) + 17, 0);
    asc_threadfence();
  }
}
}  // namespace rotating
