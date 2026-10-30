// Included once per execution domain; keep pointer arithmetic identical.
template<class T>
V9_CONTRACT_ACCESS __gm__ T* ContractPtr(uint64_t address) {
    return reinterpret_cast<__gm__ T*>(address);
}
V9_CONTRACT_ACCESS __gm__ Contract* ContractOf(__gm__ Context* ctx) {
    return ContractPtr<Contract>(ctx->contract);
}
V9_CONTRACT_ACCESS uint32_t ContractLength(__gm__ Contract* c, uint32_t b) {
    return uint32_t(ContractPtr<int32_t>(c->candidate_lens)[b]);
}
V9_CONTRACT_ACCESS uint32_t ContractBudget(__gm__ Contract* c, uint32_t b) {
    return uint32_t(ContractPtr<int32_t>(c->budgets)[b]);
}
V9_CONTRACT_ACCESS uint32_t ContractPool(__gm__ Contract* c, uint32_t b) {
    return uint32_t(ContractPtr<int32_t>(c->pool_rows)[b]);
}
V9_CONTRACT_ACCESS bool ContractHot(__gm__ Contract* c, uint32_t b, uint32_t t) {
    uint32_t stride = (c->candidate_capacity + 31) / 32;
    return (ContractPtr<uint32_t>(c->hot_bitmap)[b * stride + t / 32] >> (t % 32)) & 1U;
}
V9_CONTRACT_ACCESS bool ContractVictim(__gm__ Contract* c, uint32_t b, uint32_t s) {
    return (ContractPtr<uint32_t>(c->victim_bitmap)[b * 512 + s / 32] >> (s % 32)) & 1U;
}
V9_CONTRACT_ACCESS uint32_t ContractDirectory(__gm__ Contract* c, uint32_t b, uint32_t t) {
    return (ContractPool(c, b) * c->num_layers + c->layer_id) * c->qcap + t / 128;
}
V9_CONTRACT_ACCESS uint64_t ContractRemote(__gm__ Contract* c, uint32_t b, uint32_t t) {
    return ContractPtr<uint64_t>(c->address_table)[ContractDirectory(c, b, t)] + uint64_t(t % 128) * 656;
}
V9_CONTRACT_ACCESS uint32_t ContractPhysical(__gm__ Contract* c, uint32_t b, uint32_t s) {
    return uint32_t(ContractPtr<int32_t>(c->resident_table)[b * c->resident_columns + s / 128]) * 128 + s % 128;
}
#include "index_cache_access.inc"
