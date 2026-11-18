// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included in high_post_index; emit the same src -> dst convention as EmitEdges.
__simt_callee__ inline void VisitColdSlot(uint32_t hot, uint32_t cold, bool fresh,
        __ubuf__ uint32_t* first, __ubuf__ uint32_t* last, __ubuf__ uint32_t* touched,
        __gm__ Edge* edges, uint32_t newHotIndex, uint32_t repeatIndex) {
    if (fresh) {
        first[hot]=cold;
        touched[newHotIndex]=hot;
    } else {
        auto edge=edges+kCapacity-1-repeatIndex;
        asc_stcg(&edge->src,last[hot]);
        asc_stcg(&edge->dst,cold);
    }
    last[hot]=cold;
}

__simt_callee__ inline void CloseColdSlot(uint32_t hot, uint32_t first, uint32_t last,
        __gm__ Edge* edges, uint32_t pairIndex, uint32_t singleIndex) {
    if (first==last) {
        asc_stcg(&edges[pairIndex].src,first);
        asc_stcg(&edges[pairIndex].dst,hot);
    } else {
        auto lead=edges+kCapacity-1-singleIndex;
        asc_stcg(&lead->src,hot);
        asc_stcg(&lead->dst,first);
        asc_stcg(&(lead-1)->src,last);
        asc_stcg(&(lead-1)->dst,hot);
    }
}
