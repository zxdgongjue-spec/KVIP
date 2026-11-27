// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#if defined(__CCE_AICORE__)
#define KVIP_RESUME __aicore__ inline
#else
#define KVIP_RESUME inline
#endif

namespace joint_tail {
enum class Phase : uint32_t { Empty, Preparing, Prepared, Retiring, Done };
enum class Probe : uint32_t { Empty, Ready, Error, Stale };
struct Span { uint64_t base, bytes; };
KVIP_RESUME bool Valid(Span s) { return s.base && s.bytes && s.bytes <= UINT64_MAX - s.base; }
KVIP_RESUME bool Disjoint(Span a, Span b) {
    return Valid(a) && Valid(b) && (a.base+a.bytes<=b.base || b.base+b.bytes<=a.base);
}
struct Saved {
    Phase phase{};
    uint32_t row{}, query{}, owner{}, generation{}, total{}, first{}, n{}, own{}, scan{}, topError{}, packError{}, indexError{};
    uint32_t ends[4]{}, count{}, issued{}, seen{}, success{}, credit{}, attempts{};
    uint64_t sqHead{}, cqHead{}, key{}, error{};
    bool stop{}, prefetched{}, checkedPrefetch{};
};
KVIP_RESUME bool Begin(Saved& s,uint32_t row,uint32_t query) {
    if (s.phase!=Phase::Empty || query>=4) return false;
    s.phase=Phase::Preparing; s.row=row; s.query=query; return true;
}
KVIP_RESUME bool Arm(Saved& s,uint64_t head,uint64_t cursor,uint32_t count,const uint32_t* ends,
                     uint32_t own,bool posted,uint32_t credit,uint64_t key) {
    if (s.phase!=Phase::Preparing || !key || count>4 || (credit!=1 && credit!=2) ||
        (own && (!count || ends[count-1]!=own)) || (!own && count) ||
        (posted && !own) || head>UINT64_MAX-own || cursor>UINT64_MAX-count) return false;
    for(uint32_t i=0;i<count;++i) {
        if(!ends[i] || (i && ends[i]<=ends[i-1])) return false;
        s.ends[i]=ends[i];
    }
    s.own=own; s.count=count; s.issued=posted; s.sqHead=head; s.cqHead=cursor;
    s.key=key;s.credit=credit;s.phase=Phase::Prepared;return true;
}
// Probe owns no mutable queue state. KVAF marks an available CQE with the inverse ring generation.
template<class Port>
KVIP_RESUME Probe Peek(const Saved& s, Port& p) {
    if(s.phase!=Phase::Prepared || s.seen>=s.issued || p.Key()!=s.key || p.Depth()<2 ||
       s.issued-s.seen>=p.Depth() || p.Cursor()!=s.cqHead+s.seen) return Probe::Stale;
    const uint64_t cursor=s.cqHead+s.seen;
    const uint32_t word=p.Read(uint32_t(cursor%p.Depth()));
    if (((word>>2)&1U)==((cursor/p.Depth())&1U)) return Probe::Empty;
    return word>>16 ? Probe::Error : Probe::Ready;
}
template<class Port>
KVIP_RESUME bool PostOne(Saved& s,Port& p) {
    if(s.phase!=Phase::Prepared || s.stop || s.error || s.issued==s.count || s.issued-s.seen>=s.credit)
        return false;
    const auto previous=s.sqHead+(s.issued?s.ends[s.issued-1]:0);
    if(p.Key()!=s.key || p.Producer()!=previous){s.error=p.Invalid();s.stop=true;return false;}
    p.Issue(s.sqHead+s.ends[s.issued]); ++s.issued;return true;
}
KVIP_RESUME void Poison(Saved& s,uint64_t err) { if(!s.error) s.error=err; }
// Error CQEs must be acknowledged once. A timeout or ownership discrepancy cannot reclaim a slot.
template<class Port>
KVIP_RESUME void PollOne(Saved& s,Port& p) {
    if(s.phase!=Phase::Prepared || s.stop || s.seen==s.issued) return;
    const auto r=Peek(s,p);
    if(r==Probe::Empty) {
        if(++s.attempts>=p.Limit()){Poison(s,p.Timeout());s.stop=true;} return;
    }
    if(r==Probe::Stale){Poison(s,p.Invalid());s.stop=true;return;}
    p.Consume(s.cqHead+s.seen+1,r==Probe::Ready);
    ++s.seen;s.attempts=0;
    if(r==Probe::Ready)++s.success;
    else Poison(s,p.Failure());
}
// If the exclusive prefetch owner were to start the original drain now, can it finish
// from already visible CQEs? An observed error after a ready prefix also stops that drain.
template<class Reader>
KVIP_RESUME bool PrefetchVisible(Reader& p,uint64_t cursor,uint32_t depth,uint32_t pending) {
    if(!pending)return true;
    if(depth<2 || pending>=depth || pending>4 || cursor>UINT64_MAX-pending)return false;
    for(uint32_t i=0;i<pending;++i){
        const auto at=cursor+i;const auto word=p.Read(uint32_t(at%depth));
        if(((word>>2)&1U)==((at/depth)&1U))return false;
        if(word>>16)return true;
    }
    return true;
}
KVIP_RESUME bool Gate(const Saved& s) {
    return !s.error && !s.stop && s.seen >= (s.count<3?s.count:3);
}
KVIP_RESUME bool Settled(const Saved& s) {
    return s.phase==Phase::Prepared && (s.stop || (s.seen==s.issued && (s.issued==s.count || s.error)));
}
KVIP_RESUME bool StartRetirement(Saved& s) {
    if(!Settled(s)) return false;
    s.phase=Phase::Retiring;return true;
}
}
