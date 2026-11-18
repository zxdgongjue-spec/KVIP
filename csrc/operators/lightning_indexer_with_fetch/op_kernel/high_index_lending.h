// SPDX-License-Identifier: Apache-2.0
#pragma once

struct FinalLoan {
    uint32_t donor=4, helper=4, count=0;
};

// Keep the original number of SQ doorbells. KVAF signals only the final WQE
// of a Post, so a loan still incurs one additional CQ checkpoint.
__aicore__ inline uint32_t LoanSize(uint32_t heavy,uint32_t light,uint32_t spare) {
    constexpr uint32_t bank=kvaf_device::kWqesPerPost;
    if(heavy<=bank || light>=heavy || heavy-light<bank) return 0;
    uint32_t target=(heavy-light)/2;
    if(spare<target) target=spare;
    const uint32_t rem=target%bank,tail=heavy%bank;
    // If both pieces need the final partial bank, round down to a full bank.
    if(rem && (!tail || rem<tail)) target-=rem;
    return target>=bank/2 ? target : 0;
}

// Count and eligibility are immutable after each lane publishes Validated.
// Use at most one pair, break ties by Q, and leave a busy/blocked FIFO alone.
__aicore__ inline FinalLoan ChooseLoan(__gm__ BatchFlag* peers,bool pending) {
    FinalLoan loan;
    uint32_t heavy=0,light=UINT32_MAX,spare=0;
    for(uint32_t q=0;q<4;++q) {
        auto p=peers+q;
        uint32_t n=p->count,capacity=p->reserved[12];
        if(!capacity || (pending && uint32_t(p->reserved[11])!=n)) continue;
        if(n>heavy) { heavy=n; loan.donor=q; }
        if(n<capacity && n<light) { light=n; loan.helper=q; spare=capacity-n; }
    }
    if(loan.donor!=loan.helper) loan.count=LoanSize(heavy,light,spare);
    return loan;
}

// Different Jettys may represent different MRs or remote peers. Both lane
// owners make the same decision from their immutable capture-time bindings.
template<class QueueArgs>
__aicore__ inline bool SameTransferMemory(const QueueArgs& a,const QueueArgs& b) {
    if(a.output!=b.output || a.output_rows!=b.output_rows || a.token_bytes!=b.token_bytes)
        return false;
    auto x=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(a.channel_entity);
    auto y=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(b.channel_entity);
    if(x->remote_host_ddr_address!=y->remote_host_ddr_address ||
       x->remote_host_ddr_size!=y->remote_host_ddr_size ||
       x->remote_token_id!=y->remote_token_id || x->remote_token_value!=y->remote_token_value ||
       x->local_hbm_address!=y->local_hbm_address || x->local_hbm_size!=y->local_hbm_size ||
       x->local_token_id!=y->local_token_id || x->local_token_value!=y->local_token_value)
        return false;
    for(uint32_t i=0;i<sizeof(x->remote_eid);++i)
        if(x->remote_eid[i]!=y->remote_eid[i]) return false;
    return true;
}
