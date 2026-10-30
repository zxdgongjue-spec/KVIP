// Packed FP8 banks have 128-byte aligned starts. Scales remain FP32.
V9_CONTRACT_ACCESS uint64_t IndexBase(__gm__ Contract* c, uint32_t b) {
    return ContractPtr<uint64_t>(c->index_cache)[b];
}
V9_CONTRACT_ACCESS bool IndexOwned(__gm__ Contract* c, uint32_t b) {
    return ContractPtr<int32_t>(IndexBase(c,b))[18] == 1;
}
V9_CONTRACT_ACCESS uint32_t IndexPayloadRows(__gm__ Contract* c, uint32_t b) {
    return c->resident_capacity + (IndexOwned(c,b) ? 0 : c->candidate_capacity);
}
V9_CONTRACT_ACCESS uint64_t IndexKey(__gm__ Contract* c, uint32_t b, bool hot) {
    return IndexBase(c,b) + 128 + (hot ? 0 : uint64_t(c->resident_capacity)*128);
}
V9_CONTRACT_ACCESS uint64_t IndexScale(__gm__ Contract* c, uint32_t b, bool hot) {
    return IndexBase(c,b) + 128 + uint64_t(IndexPayloadRows(c,b))*128 +
        (hot ? 0 : uint64_t(c->resident_capacity)*4);
}
V9_CONTRACT_ACCESS __gm__ int32_t* IndexIds(__gm__ Contract* c, uint32_t b, bool hot) {
    return ContractPtr<int32_t>(IndexBase(c,b)+128+uint64_t(IndexPayloadRows(c,b))*132) +
        (hot ? 0 : c->resident_capacity);
}
V9_CONTRACT_ACCESS __gm__ int32_t* IndexDirty(__gm__ Contract* c, uint32_t b, bool hot) {
    return IndexIds(c,b,true)+c->resident_capacity+c->candidate_capacity+(hot?0:c->resident_capacity);
}
V9_CONTRACT_ACCESS __gm__ int32_t* IndexPositions(__gm__ Contract* c, uint32_t b) {
    return IndexDirty(c,b,true)+c->resident_capacity+c->candidate_capacity;
}
// In owned mode the original page pool IS the cold bank, not a logical source.
V9_CONTRACT_ACCESS uint32_t IndexColdPhysical(__gm__ Contract* c, uint32_t b, uint32_t p) {
    return uint32_t(ContractPtr<int32_t>(c->index_table)[b*c->index_columns+p/128])*128+p%128;
}
V9_CONTRACT_ACCESS __gm__ uint32_t* IndexRowKey(__gm__ Contract* c, uint32_t b, bool hot, uint32_t p) {
    if (!hot && IndexOwned(c,b)) return ContractPtr<uint32_t>(c->key)+uint64_t(IndexColdPhysical(c,b,p))*32;
    return ContractPtr<uint32_t>(IndexKey(c,b,hot))+uint64_t(p)*32;
}
V9_CONTRACT_ACCESS __gm__ uint32_t* IndexRowScale(__gm__ Contract* c, uint32_t b, bool hot, uint32_t p) {
    if (!hot && IndexOwned(c,b)) return ContractPtr<uint32_t>(c->key_scale)+IndexColdPhysical(c,b,p);
    return ContractPtr<uint32_t>(IndexScale(c,b,hot))+p;
}
V9_CONTRACT_ACCESS uint32_t ColdCount(__gm__ Contract* c, uint32_t b) {
    return uint32_t(ContractPtr<int32_t>(c->cold_counts)[b]);
}
V9_CONTRACT_ACCESS uint32_t ColdExtent(__gm__ Contract* c) {
    // Every scheduled service core must reach its epoch, including all-hot rows.
    uint32_t extent=128;
    for(uint32_t b=0;b<c->batch;++b) if(ColdCount(c,b)>extent) extent=ColdCount(c,b);
    return (extent+127)/128*128;
}
