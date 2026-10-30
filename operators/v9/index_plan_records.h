#pragma once
namespace rotating {
// A completed planner owns unique final slots. Emit (old hot row, new cold
// row) while its token/slot arrays are still in UB, before descriptor packing
// reuses them. Header 27 is a per-call ready bit; init always clears it.
__simt_callee__ inline bool StoreIndexPlanRecord(__gm__ Contract* c,uint32_t b,
    uint32_t record,uint32_t slot,uint32_t token) {
  using namespace contract_simt;
  uint32_t length=ContractLength(c,b),pool=ContractPool(c,b);
  auto state=ContractPtr<int32_t>(IndexBase(c,b));
  auto pos=IndexPositions(c,b),jobs=pos+c->candidate_capacity;
  auto reverse=ContractPtr<int32_t>(c->reverse)+uint64_t(pool)*c->resident_capacity;
  auto forward=ContractPtr<int32_t>(c->forward)+uint64_t(pool)*c->candidate_capacity;
  if(slot>=ContractBudget(c,b) || (token!=UINT32_MAX && token>=length))return false;
  int32_t incoming=-1,outgoing=-1;
  if(token!=UINT32_MAX) {
    incoming=asc_ldcg(pos+token);
    if(incoming<0 || incoming>=asc_ldcg(state+3))return false;
  }
  int32_t old=asc_ldcg(reverse+slot);
  if(old>=0 && uint32_t(old)<length && asc_ldcg(forward+old)==int32_t(slot)) {
    int32_t at=asc_ldcg(pos+old);
    if(at>-2 || -int64_t(at)-2>=asc_ldcg(state+2))return false;
    outgoing=int32_t(-int64_t(at)-2);
  }
  asc_stcg(jobs+2*record,outgoing);asc_stcg(jobs+2*record+1,incoming);
  return true;
}
} // namespace rotating
