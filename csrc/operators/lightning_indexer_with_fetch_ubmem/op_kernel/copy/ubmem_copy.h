#pragma once
#include "../device_context.h"
namespace rotating {
__aicore__ inline bool UbmemCopyRow(__gm__ Context* ctx, uint64_t peer,
    uint64_t output, uint32_t slot, LocalTensor<uint8_t> scratch) {
    auto region = UbmemRegionAt(ctx);
    uint64_t source = 0;
    if (!UbmemRegionValid(region) || slot >= SIM_RESIDENT ||
        !UbmemTranslate(region->peerBase, region->mappedBase, region->bytes,
                        peer, kTokenBytes, source) || output < region->outputBase)
        return false;
    uint64_t offset = output-region->outputBase;
    uint64_t rowOffset = uint64_t(slot)*kTokenBytes;
    if (offset > region->outputBytes || rowOffset > region->outputBytes-offset ||
        kTokenBytes > region->outputBytes-offset-rowOffset) return false;
    GlobalTensor<uint8_t> from, to;
    from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source));
    to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output+rowOffset));
    // Reuse only dead scratch. Do not allocate a new TPipe or consume LI's
    // persistent V_MTE2 ping-pong credits inside the borrowed service window.
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopyExtParams copy{1, kTokenBytes, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad<uint8_t, PaddingMode::Normal>(scratch, from, copy, pad);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopyPad<uint8_t, PaddingMode::Normal>(to, scratch, copy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    return true;
}

// The service SIMT task plans disjoint eight-row groups in parallel while
// retaining raw peers for irregular/partial-group fallback. No DMA or mapping
// publication depends on these flags until the task's V_S join has completed.
constexpr uint32_t kUbmemPlanValid=1, kUbmemPlanFirstFour=2,
    kUbmemPlanLastFour=4, kUbmemPlanOneRun=8;
#if defined(__CCE_AICORE__)
#define V9_GROUP_PLAN_FN __simt_callee__
#else
#define V9_GROUP_PLAN_FN
#endif
V9_GROUP_PLAN_FN inline uint32_t UbmemPlanGroup8(uint64_t base,uint64_t maxOffset,
    uint64_t p0,uint64_t p1,uint64_t p2,uint64_t p3,
    uint64_t p4,uint64_t p5,uint64_t p6,uint64_t p7) {
    if(p0-base>maxOffset || p1-base>maxOffset || p2-base>maxOffset || p3-base>maxOffset ||
        p4-base>maxOffset || p5-base>maxOffset || p6-base>maxOffset || p7-base>maxOffset)return 0;
    if(p1<p0 || p3<p2 || p5<p4 || p7<p6)return 0;
    uint64_t d0=p1-p0,d1=p3-p2,d2=p5-p4,d3=p7-p6;
    if(d0<kTokenBytes || d0-kTokenBytes>UINT32_MAX || d1<kTokenBytes || d1-kTokenBytes>UINT32_MAX ||
        d2<kTokenBytes || d2-kTokenBytes>UINT32_MAX || d3<kTokenBytes || d3-kTokenBytes>UINT32_MAX)return 0;
    bool first=p2>=p1 && p2-p1==d0 && d1==d0;
    bool last=p6>=p5 && p6-p5==d2 && d3==d2;
    bool one=first && last && p4>=p3 && p4-p3==d0 && d2==d0;
    return kUbmemPlanValid|(first?kUbmemPlanFirstFour:0)|
        (last?kUbmemPlanLastFour:0)|(one?kUbmemPlanOneRun:0);
}
#undef V9_GROUP_PLAN_FN

// Eight 32-byte-aligned UB stages borrow 5376 dead bytes. The free credit
// (MTE3->MTE2) protects reuse, and ready (MTE2->MTE3) protects each store.
// These event directions are separate from LI's persistent V/MTE credits.
// Like DeepEP Engram's ring, issue eight loads before the first store. Unlike
// its kernel-long ring, this service must drain every credit before returning
// the scratch and event IDs to LI (including on a bad address).
class UbmemCopyPipeline {
public:
    static constexpr uint32_t kDepth = 8;
    static constexpr uint32_t kStageBytes = (kTokenBytes+31)/32*32;
    static constexpr uint32_t kScratchBytes=kDepth*kStageBytes, kOverflowBytes=0;
    static_assert(kScratchBytes<=23808,"single-row copy ring exceeds retired candidate scratch");
    __aicore__ inline UbmemCopyPipeline(__gm__ Context* ctx, uint64_t output,
        LocalTensor<uint8_t> scratch): region_(UbmemRegionAt(ctx)), output_(output), scratch_(scratch) {
        valid_ = UbmemRegionValid(region_);
        if (valid_) {
            peerBase_=region_->peerBase; mappedBase_=region_->mappedBase; bytes_=region_->bytes;
            outputBase_=region_->outputBase; outputBytes_=region_->outputBytes;
        }
    }
    __aicore__ inline UbmemCopyPipeline(__gm__ Context* ctx,uint64_t output,
        LocalTensor<uint8_t> scratch,LocalTensor<uint8_t>) : UbmemCopyPipeline(ctx,output,scratch) {}
    __aicore__ inline bool Add(uint64_t peer,uint32_t slot) { return AddAt(peer,output_,slot); }
    __aicore__ inline bool AddAt(uint64_t peer,uint64_t output,uint32_t slot) {
        uint64_t source=0;
        if (!valid_ || slot>=SIM_RESIDENT || output<outputBase_ ||
            !UbmemTranslate(peerBase_,mappedBase_,bytes_,peer,kTokenBytes,source)) return false;
        uint64_t offset=output-outputBase_, rowOffset=uint64_t(slot)*kTokenBytes;
        if (offset>outputBytes_ || rowOffset>outputBytes_-offset ||
            kTokenBytes>outputBytes_-offset-rowOffset) return false;
        if (!issued_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for (uint32_t i=0;i<kDepth;++i) SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        uint32_t stage=issued_%kDepth;
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        GlobalTensor<uint8_t> from;
        from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source));
        DataCopyExtParams copy{1,kTokenBytes,0,0,0};
        DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
        DataCopyPad<uint8_t,PaddingMode::Normal>(scratch_[stage*kStageBytes],from,copy,pad);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        destinations_[stage]=output+rowOffset;
        ++issued_;
        if (issued_-stored_==kDepth) StoreOne();
        return true;
    }
    __aicore__ inline uint64_t PlanPeerBase() const {return peerBase_;}
    __aicore__ inline uint64_t PlanMaxPeerOffset() const {return bytes_>=kTokenBytes?bytes_-kTokenBytes:0;}
    __aicore__ inline bool BeginRunAt(uint64_t output,uint32_t slot,uint32_t count) {
        runCount_=0;
        if(!valid_ || !count || slot>=SIM_RESIDENT || count>SIM_RESIDENT-slot || output<outputBase_)return false;
        uint64_t offset=output-outputBase_,rowOffset=uint64_t(slot)*kTokenBytes;
        if(offset>outputBytes_ || rowOffset>outputBytes_-offset ||
            count>(outputBytes_-offset-rowOffset)/kTokenBytes)return false;
        runOutput_=output;runSlot_=slot;runCount_=count;return true;
    }
    __aicore__ inline bool AddNext(uint64_t peer) {
        if(!runCount_ || !AddAt(peer,runOutput_,runSlot_))return false;
        --runCount_;++runSlot_;return true;
    }
    template<class... T> __aicore__ inline bool TryAddNext4(T...) {return false;}
    template<class... T> __aicore__ inline bool TryAddNext6(T...) {return false;}
    template<class... T> __aicore__ inline bool TryAddNext8(T...) {return false;}
    __aicore__ inline void SaveFinalCommandCounts(__gm__ uint64_t*) const {}
    __aicore__ inline uint32_t Finish() {
        if (!issued_) return 0;
        while (stored_<issued_) StoreOne();
        // Consume *all* final free credits, including untouched stages. A
        // final MTE2->scalar barrier proves both writes and credit consumption
        // complete, so another window can safely initialize the same IDs.
        for (uint32_t i=0;i<kDepth;++i) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return issued_;
    }
private:
    __aicore__ inline void StoreOne() {
        uint32_t stage=stored_%kDepth;
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        GlobalTensor<uint8_t> to;
        to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destinations_[stage]));
        DataCopyExtParams copy{1,kTokenBytes,0,0,0};
        DataCopyPad<uint8_t,PaddingMode::Normal>(to,scratch_[stage*kStageBytes],copy);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        ++stored_;
    }
    __gm__ UbmemRegion* region_;
    uint64_t output_,peerBase_=0,mappedBase_=0,bytes_=0,outputBase_=0,outputBytes_=0;
    LocalTensor<uint8_t> scratch_;
    uint64_t destinations_[kDepth]{},runOutput_=0;
    uint32_t runSlot_=0,runCount_=0;
    uint32_t issued_=0,stored_=0;
    bool valid_=false;
};

#if V9_UBMEM_GROUPED_COPY
// Four-layer rows are already 32-byte aligned. Pair independently validated
// rows in four stages: the same eight-row/20992-byte footprint as the original
// ring, but one ready/free exchange per pair. Both source and destination gaps
// are represented by exact strided copies, never by reading/writing the gaps.
// Reversed/repeated addresses retain two ordered copies within the same stage.
class UbmemPairCopyPipeline {
public:
    static constexpr uint32_t kDepth=4,kStageBytes=2*kTokenBytes;
    static constexpr uint32_t kScratchBytes=kDepth*kStageBytes,kOverflowBytes=0;
    static_assert(kTokenBytes==2624 && kTokenBytes%32==0 && kScratchBytes<=23808,
                  "Paired pipeline requires aligned four-layer rows and retired scratch");
    __aicore__ inline UbmemPairCopyPipeline(__gm__ Context* ctx,uint64_t output,LocalTensor<uint8_t> scratch)
        :output_(output),scratch_(scratch) {
        auto region=UbmemRegionAt(ctx);
        valid_=UbmemRegionValid(region);
        if(valid_) {
            peerBase_=region->peerBase;mappedBase_=region->mappedBase;bytes_=region->bytes;
            outputBase_=region->outputBase;outputBytes_=region->outputBytes;
        }
    }
    __aicore__ inline UbmemPairCopyPipeline(__gm__ Context* ctx,uint64_t output,
        LocalTensor<uint8_t> scratch,LocalTensor<uint8_t>):UbmemPairCopyPipeline(ctx,output,scratch) {}
    __aicore__ inline uint64_t PlanPeerBase()const {return peerBase_;}
    __aicore__ inline uint64_t PlanMaxPeerOffset()const {return bytes_>=kTokenBytes?bytes_-kTokenBytes:0;}
    __aicore__ inline bool Add(uint64_t peer,uint32_t slot) {return AddAt(peer,output_,slot);}
    __aicore__ inline bool AddAt(uint64_t peer,uint64_t output,uint32_t slot) {
        uint64_t source=0;
        if(!valid_ || slot>=SIM_RESIDENT || output<outputBase_ || (output&31) ||
           !UbmemTranslate(peerBase_,mappedBase_,bytes_,peer,kTokenBytes,source))return false;
        uint64_t offset=output-outputBase_,rowOffset=uint64_t(slot)*kTokenBytes;
        if(offset>outputBytes_ || rowOffset>outputBytes_-offset ||
           kTokenBytes>outputBytes_-offset-rowOffset)return false;
        if(!accepted_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for(uint32_t i=0;i<kDepth;++i)SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        sources_[pending_]=source;destinations_[pending_]=output+rowOffset;
        ++pending_;++accepted_;
        if(pending_==2)FlushPair();
        return true;
    }
    __aicore__ inline bool BeginRunAt(uint64_t output,uint32_t slot,uint32_t count) {
        runCount_=0;
        if(!valid_ || !count || slot>=SIM_RESIDENT || count>SIM_RESIDENT-slot ||
           output<outputBase_ || (output&31))return false;
        uint64_t offset=output-outputBase_,rowOffset=uint64_t(slot)*kTokenBytes;
        if(offset>outputBytes_ || rowOffset>outputBytes_-offset ||
           count>(outputBytes_-offset-rowOffset)/kTokenBytes)return false;
        runOutput_=output;runSlot_=slot;runCount_=count;return true;
    }
    __aicore__ inline bool AddNext(uint64_t peer) {
        if(!runCount_ || !AddAt(peer,runOutput_,runSlot_))return false;
        --runCount_;++runSlot_;return true;
    }
    template<class... T> __aicore__ inline bool TryAddNext4(T...) {return false;}
    template<class... T> __aicore__ inline bool TryAddNext6(T...) {return false;}
    template<class... T> __aicore__ inline bool TryAddNext8(T...) {return false;}
    __aicore__ inline void SaveFinalCommandCounts(__gm__ uint64_t*)const {}
    __aicore__ inline uint32_t Finish() {
        if(!accepted_)return 0;
        if(pending_)FlushPair();
        while(stored_<issued_)StoreOldest();
        for(uint32_t i=0;i<kDepth;++i)WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return accepted_;
    }
private:
    __aicore__ static inline bool ForwardPair(uint64_t first,uint64_t second) {
        return second>=first && second-first>=kTokenBytes && second-first-kTokenBytes<=UINT32_MAX;
    }
    __aicore__ inline void FlushPair() {
        uint32_t stage=issued_%kDepth,rows=pending_;
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        GlobalTensor<uint8_t> from;
        DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
        if(rows==2 && ForwardPair(sources_[0],sources_[1])) {
            from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sources_[0]));
            DataCopyExtParams copy{2,kTokenBytes,static_cast<uint32_t>(sources_[1]-sources_[0]-kTokenBytes),0,0};
            DataCopyPad<uint8_t,PaddingMode::Normal>(scratch_[stage*kStageBytes],from,copy,pad);
        } else {
            for(uint32_t i=0;i<rows;++i) {
                from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sources_[i]));
                DataCopyExtParams copy{1,kTokenBytes,0,0,0};
                DataCopyPad<uint8_t,PaddingMode::Normal>(scratch_[stage*kStageBytes+i*kTokenBytes],from,copy,pad);
            }
        }
        for(uint32_t i=0;i<rows;++i)storeDestinations_[stage][i]=destinations_[i];
        storeRows_[stage]=rows;
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        pending_=0;++issued_;
        // Preserve eight-row load lookahead before the first MTE3 store.
        if(issued_-stored_==kDepth)StoreOldest();
    }
    __aicore__ inline void StoreOldest() {
        uint32_t stage=stored_%kDepth,rows=storeRows_[stage];
        auto first=storeDestinations_[stage][0],second=storeDestinations_[stage][1];
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        GlobalTensor<uint8_t> to;
        if(rows==2 && ForwardPair(first,second)) {
            to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(first));
            DataCopyExtParams copy{2,kTokenBytes,0,static_cast<uint32_t>(second-first-kTokenBytes),0};
            DataCopyPad<uint8_t,PaddingMode::Normal>(to,scratch_[stage*kStageBytes],copy);
        } else {
            for(uint32_t i=0;i<rows;++i) {
                to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(storeDestinations_[stage][i]));
                DataCopyExtParams copy{1,kTokenBytes,0,0,0};
                DataCopyPad<uint8_t,PaddingMode::Normal>(to,scratch_[stage*kStageBytes+i*kTokenBytes],copy);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));++stored_;
    }
    LocalTensor<uint8_t> scratch_;
    uint64_t output_,peerBase_=0,mappedBase_=0,bytes_=0,outputBase_=0,outputBytes_=0;
    uint64_t sources_[2]{},destinations_[2]{},storeDestinations_[kDepth][2]{};
    uint64_t runOutput_=0;
    uint32_t storeRows_[kDepth]{},pending_=0,accepted_=0,issued_=0,stored_=0,runSlot_=0,runCount_=0;
    bool valid_=false;
};
#endif

#if TOKEN_BYTES == 656
// Keep the one-row ring above intact as the measured reference. This ring
// groups only consecutive destinations: the Host source of every row remains
// independent. Even-row Host runs can use Compact padding: two 656-byte rows
// occupy 1312 aligned bytes, allowing one contiguous MTE3 store for the group.
// Irregular runs that would start at an unaligned UB address retain Normal
// 672-byte row padding and exact 656-byte per-block stores.
static_assert(V9_WINDOW_UBPIPE_DEPTH==4 || V9_WINDOW_UBPIPE_DEPTH==8,"Invalid window pipeline depth");
template<uint32_t CompactRows=4,bool PairGaps=false,uint32_t Depth=8,bool PrefillLoads=false>
class UbmemGroupedCopyPipelineT {
public:
    static constexpr uint32_t kDepth = Depth, kRowsPerStage = 4;
    static constexpr uint32_t kLoadPrefill = PrefillLoads ? kDepth : 1;
    static_assert(Depth==4 || Depth==8,"Grouped ring depth must fit the eight event IDs");
    static_assert(!PrefillLoads || (CompactRows==8 && !PairGaps),"Load prefill is window-only");
    static constexpr uint32_t kRowBytes = (kTokenBytes+31)/32*32;
    // Wide groups use Compact; irregular inputs retain <=4-row Normal. The
    // depth-eight window has six stages in retired candidate scratch and two
    // in a disjoint dead work segment. Depth four fits entirely in retired
    // candidates, reducing live requests without changing final-copy storage.
    static_assert(CompactRows==4 || CompactRows==6 || CompactRows==8,"Unsupported compact group width");
    static_assert(!PairGaps || CompactRows==4,"Sector-pair groups use the original final scratch layout");
    static constexpr uint32_t kStageBytes = CompactRows>=6?CompactRows*kTokenBytes:kRowsPerStage*kRowBytes;
    static constexpr uint32_t kPrimaryStages = CompactRows==8 && kDepth>6?6:kDepth;
    static constexpr uint32_t kScratchBytes = kPrimaryStages*kStageBytes;
    static constexpr uint32_t kOverflowBytes = (kDepth-kPrimaryStages)*kStageBytes;
    static constexpr uint32_t kTotalScratchBytes = kScratchBytes+kOverflowBytes;
    static_assert(CompactRows!=4 || kScratchBytes<=23808,"Grouped copy overlaps live prefetch candidate IDs");
    static_assert(kRowsPerStage*kRowBytes<=kStageBytes,"Normal fallback exceeds compact stage");
    static_assert(kStageBytes%32==0 && kScratchBytes<=32000,"Window copy exceeds retired candidate scratch");
    static_assert(kScratchBytes<=kSlots*sizeof(uint32_t),"Grouped copy overlaps final locations");
    __aicore__ inline UbmemGroupedCopyPipelineT(__gm__ Context* ctx,uint64_t output,LocalTensor<uint8_t> scratch)
        :UbmemGroupedCopyPipelineT(ctx,output,scratch,scratch) {
        static_assert(CompactRows!=8,"Eight-row windows require the disjoint overflow segment");
    }
    __aicore__ inline UbmemGroupedCopyPipelineT(__gm__ Context* ctx,uint64_t output,
        LocalTensor<uint8_t> scratch,LocalTensor<uint8_t> overflow)
        :region_(UbmemRegionAt(ctx)),output_(output),scratch_(scratch),overflow_(overflow) {
        valid_ = UbmemRegionValid(region_) && region_->bytes>=kTokenBytes;
        if (valid_) {
            peerBase_=region_->peerBase; mappedBase_=region_->mappedBase; bytes_=region_->bytes;
            outputBase_=region_->outputBase; outputBytes_=region_->outputBytes;
            maxPeerOffset_=bytes_-kTokenBytes;
        }
    }
    __aicore__ inline bool Add(uint64_t peer, uint32_t slot) {
        return AddAt(peer,output_,slot);
    }
    // One borrowed service lane may copy several batch outputs before its
    // single drain. Stage destinations are absolute, so rebasing later rows
    // cannot change a store already in flight or merge across an output gap.
    __aicore__ inline bool AddAt(uint64_t peer, uint64_t output, uint32_t slot) {
        uint64_t source=0;
        if (!valid_ || slot>=SIM_RESIDENT || output<outputBase_ ||
            !TranslatePeer(peer,source)) return false;
        uint64_t offset=output-outputBase_, rowOffset=uint64_t(slot)*kTokenBytes;
        if (offset>outputBytes_ || rowOffset>outputBytes_-offset ||
            kTokenBytes>outputBytes_-offset-rowOffset) return false;
        Append(source,output+rowOffset);
        return true;
    }
    // Window slots are contiguous. Validate their complete destination range
    // once; a false result leaves the caller on AddAt so malformed ranges
    // retain the exact accepted-prefix behavior. Host peers are still checked
    // individually by AddNext, and this run never accepts beyond count rows.
    __aicore__ inline bool BeginRunAt(uint64_t output,uint32_t slot,uint32_t count) {
        runRemaining_=0;
        if(!valid_ || !count || slot>=SIM_RESIDENT ||
            count>SIM_RESIDENT-slot || output<outputBase_)return false;
        uint64_t offset=output-outputBase_,rowOffset=uint64_t(slot)*kTokenBytes;
        if(offset>outputBytes_ || rowOffset>outputBytes_-offset ||
            count>(outputBytes_-offset-rowOffset)/kTokenBytes)return false;
        // A new batch gap would flush this already-accepted prefix on its
        // first AddNext anyway. Queue that same partial store now so the new
        // run starts at an aligned prepared group instead of shifting by 1..3
        // rows. Invalid ranges above still have no DMA side effects.
        if constexpr(CompactRows==8) {
            if(pending_ && output+rowOffset!=nextDestination_)FlushGroup();
        }
        runDestination_=output+rowOffset;runRemaining_=count;
        return true;
    }
    __aicore__ inline bool AddNext(uint64_t peer) {
        uint64_t source=0;
        if(!runRemaining_ || !TranslatePeer(peer,source)) {
            runRemaining_=0;return false;
        }
        Append(source,runDestination_);runDestination_+=kTokenBytes;--runRemaining_;
        return true;
    }
    // Four consecutive window rows share one destination range check and
    // one fixed group submission. A false result changes no state and queues
    // no DMA; the caller uses AddNext to preserve exact bad-peer prefixes and
    // to complete partial groups before retrying this fixed-group path.
    __aicore__ inline bool TryAddNext4(uint64_t p0,uint64_t p1,uint64_t p2,uint64_t p3) {
        uint64_t s0=0,s1=0,s2=0,s3=0;
        if(runRemaining_<4 || pending_ || !TranslatePeer(p0,s0) || !TranslatePeer(p1,s1) ||
            !TranslatePeer(p2,s2) || !TranslatePeer(p3,s3) ||
            !ForwardPairValues(s0,s1) || !ForwardPairValues(s2,s3)) {
            return false;
        }
        if (!accepted_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for(uint32_t i=0;i<kDepth;++i)SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        uint32_t stage=issued_%kDepth;
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        uint64_t delta=s1-s0;
        bool oneRun=s2>=s1 && s2-s1==delta && s3-s2==delta;
        bool loopLoaded=false;
        if(!loopLoaded) {
            LoadCompact(stage,0,s0,oneRun?4:2,static_cast<uint32_t>(delta-kTokenBytes));
            if(!oneRun)LoadCompact(stage,2,s2,2,static_cast<uint32_t>(s3-s2-kTokenBytes));
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        ++issued_;accepted_+=4;
        QueueStore(stage,runDestination_,4,true);
        runDestination_+=4*kTokenBytes;runRemaining_-=4;nextDestination_=runDestination_;
        return true;
    }
    // Only the lane-wide window phase may borrow the retired candidate IDs.
    // Validate all six sources before any credit or DMA side effect. Falling
    // back to TryAddNext4/AddNext keeps the first-invalid-peer prefix exact.
    __aicore__ inline bool TryAddNext6(uint64_t p0,uint64_t p1,uint64_t p2,uint64_t p3,uint64_t p4,uint64_t p5) {
        if constexpr(CompactRows<6)return false;
        uint64_t s0=0,s1=0,s2=0,s3=0,s4=0,s5=0;
        if(runRemaining_<6 || pending_ || !TranslatePeer(p0,s0) || !TranslatePeer(p1,s1) ||
            !TranslatePeer(p2,s2) || !TranslatePeer(p3,s3) || !TranslatePeer(p4,s4) || !TranslatePeer(p5,s5) ||
            !ForwardPairValues(s0,s1) || !ForwardPairValues(s2,s3) || !ForwardPairValues(s4,s5))return false;
        if(!accepted_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for(uint32_t i=0;i<kDepth;++i)SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        uint32_t stage=issued_%kDepth;
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        uint64_t d0=s1-s0,d1=s3-s2,d2=s5-s4;
        bool firstFour=s2>=s1 && s2-s1==d0 && d1==d0;
        bool lastFour=s4>=s3 && s4-s3==d1 && d2==d1;
        bool loopLoaded=false;
        // Merge adjacent pairs only if their complete arithmetic run is
        // proven. Every command start is a 1312-byte aligned pair boundary.
        if(loopLoaded) {
            // All six exact rows are already queued in one loop-mode DMA.
        } else if(firstFour) {
            LoadCompact(stage,0,s0,lastFour?6:4,static_cast<uint32_t>(d0-kTokenBytes));
            if(!lastFour)LoadCompact(stage,4,s4,2,static_cast<uint32_t>(d2-kTokenBytes));
        } else {
            LoadCompact(stage,0,s0,2,static_cast<uint32_t>(d0-kTokenBytes));
            LoadCompact(stage,2,s2,lastFour?4:2,static_cast<uint32_t>(d1-kTokenBytes));
            if(!lastFour)LoadCompact(stage,4,s4,2,static_cast<uint32_t>(d2-kTokenBytes));
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        ++issued_;accepted_+=6;
        QueueStore(stage,runDestination_,6,true);
        runDestination_+=6*kTokenBytes;runRemaining_-=6;nextDestination_=runDestination_;
        return true;
    }
    // Four independently validated source pairs fill one 5248-byte stage.
    // Never inspect a ninth peer or issue a read before the whole group is
    // known valid. The caller can safely retry 6/4/1 rows after any rejection.
    __aicore__ inline bool TryAddNext8(uint64_t p0,uint64_t p1,uint64_t p2,uint64_t p3,
        uint64_t p4,uint64_t p5,uint64_t p6,uint64_t p7) {
        if constexpr(CompactRows!=8)return false;
        uint64_t s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
        if(runRemaining_<8 || pending_ || !TranslatePeer(p0,s0) || !TranslatePeer(p1,s1) ||
            !TranslatePeer(p2,s2) || !TranslatePeer(p3,s3) || !TranslatePeer(p4,s4) || !TranslatePeer(p5,s5) ||
            !TranslatePeer(p6,s6) || !TranslatePeer(p7,s7) || !ForwardPairValues(s0,s1) ||
            !ForwardPairValues(s2,s3) || !ForwardPairValues(s4,s5) || !ForwardPairValues(s6,s7))return false;
        return SubmitForwardEight(s0,s1,s2,s3,s4,s5,s6,s7);
    }
private:
    __aicore__ inline bool SubmitForwardEight(uint64_t s0,uint64_t s1,uint64_t s2,uint64_t s3,
        uint64_t s4,uint64_t s5,uint64_t s6,uint64_t s7) {
        uint64_t d0=s1-s0,d1=s3-s2,d2=s5-s4,d3=s7-s6;
        bool first=s2>=s1 && s2-s1==d0 && d1==d0;
        bool last=s6>=s5 && s6-s5==d2 && d3==d2;
        bool one=first && last && s4>=s3 && s4-s3==d0 && d2==d0;
        uint32_t plan=kUbmemPlanValid|(first?kUbmemPlanFirstFour:0)|
            (last?kUbmemPlanLastFour:0)|(one?kUbmemPlanOneRun:0);
        SubmitNext8(s0,s1,s2,s3,s4,s5,s6,s7,plan);
        return true;
    }
public:
    __aicore__ inline uint64_t PlanPeerBase() const { return peerBase_; }
    __aicore__ inline uint64_t PlanMaxPeerOffset() const { return maxPeerOffset_; }
    // Only flags produced by UbmemPlanGroup8 for these same raw peers may be
    // used here. Keeping raw values intact is essential: a pending partial
    // group from another batch can reject this path and require 6/4/1 fallback.
    __aicore__ inline bool TryAddPlannedNext8(uint32_t plan,
        uint64_t p0,uint64_t p1,uint64_t p2,uint64_t p3,
        uint64_t p4,uint64_t p5,uint64_t p6,uint64_t p7) {
        if constexpr(CompactRows!=8)return false;
        if(!valid_ || runRemaining_<8 || pending_ || !(plan&kUbmemPlanValid))return false;
        SubmitNext8(mappedBase_+(p0-peerBase_),mappedBase_+(p1-peerBase_),
            mappedBase_+(p2-peerBase_),mappedBase_+(p3-peerBase_),
            mappedBase_+(p4-peerBase_),mappedBase_+(p5-peerBase_),
            mappedBase_+(p6-peerBase_),mappedBase_+(p7-peerBase_),plan);
        return true;
    }
    __aicore__ inline uint64_t SavePrepareCount(uint32_t prepares) const {
        return prepares;
    }
private:
    __aicore__ inline void SubmitNext8(uint64_t s0,uint64_t s1,uint64_t s2,uint64_t s3,
        uint64_t s4,uint64_t s5,uint64_t s6,uint64_t s7,uint32_t plan) {
        if(!accepted_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for(uint32_t i=0;i<kDepth;++i)SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        uint32_t stage=issued_%kDepth;
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        uint64_t d0=s1-s0,d1=s3-s2,d2=s5-s4,d3=s7-s6;
        bool firstFour=plan&kUbmemPlanFirstFour,lastFour=plan&kUbmemPlanLastFour,oneRun=plan&kUbmemPlanOneRun;
        bool loopLoaded=false;
        if(!loopLoaded) {
            bool firstLoop=false,lastLoop=false;
            if(!firstLoop) {
                LoadCompact(stage,0,s0,oneRun?8:(firstFour?4:2),static_cast<uint32_t>(d0-kTokenBytes));
                if(!oneRun && !firstFour)LoadCompact(stage,2,s2,2,static_cast<uint32_t>(d1-kTokenBytes));
            }
            if(!oneRun) {
                if(!lastLoop) {
                    LoadCompact(stage,4,s4,lastFour?4:2,static_cast<uint32_t>(d2-kTokenBytes));
                    if(!lastFour)LoadCompact(stage,6,s6,2,static_cast<uint32_t>(d3-kTokenBytes));
                }
            }
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        ++issued_;accepted_+=8;
        QueueStore(stage,runDestination_,8,true);
        runDestination_+=8*kTokenBytes;runRemaining_-=8;nextDestination_=runDestination_;
    }
    __aicore__ inline LocalTensor<uint8_t> Stage(uint32_t stage) const {
        if constexpr(CompactRows==8) {
            if(stage>=kPrimaryStages)return overflow_[(stage-kPrimaryStages)*kStageBytes];
        }
        return scratch_[stage*kStageBytes];
    }
    __aicore__ inline bool TranslatePeer(uint64_t peer,uint64_t& source) const {
        // The constructor proved both complete ranges do not wrap. A peer
        // below peerBase_ underflows beyond maxPeerOffset_, so one unsigned
        // comparison proves both source bounds and the full 656-byte row.
        uint64_t offset=peer-peerBase_;
        if(!valid_ || offset>maxPeerOffset_)return false;
        source=mappedBase_+offset;return true;
    }
    __aicore__ static inline bool ForwardPairValues(uint64_t first,uint64_t second) {
        if(second<first)return false;
        uint64_t delta=second-first;
        return delta>=kTokenBytes && delta-kTokenBytes<=UINT32_MAX;
    }
    __aicore__ inline void LoadCompact(uint32_t stage,uint32_t first,uint64_t source,uint32_t count,uint32_t stride) {
        GlobalTensor<uint8_t> from;from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source));
        DataCopyExtParams copy{static_cast<uint16_t>(count),kTokenBytes,stride,0,0};
        DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
        DataCopyPad<uint8_t,PaddingMode::Compact>(Stage(stage)[first*kTokenBytes],from,copy,pad);
    }
    __aicore__ inline void Append(uint64_t source,uint64_t destination) {
        // Final workers own two-row sectors separated by the other workers'
        // sectors. Join at most two aligned sectors in one existing stage;
        // the strided GM store skips the gap and never touches another owner.
        // All other gaps retain the original flush, including a one-row tail,
        // backward/repeated destinations and oversized strides.
        if (pending_ && destination!=nextDestination_) {
            bool joinPair=false;
            if constexpr(PairGaps) {
                joinPair=pending_==2 && !pairDestination_ && !(groupDestination_&31) &&
                    !(destination&31) && destination>nextDestination_ &&
                    destination-nextDestination_<=UINT32_MAX;
            }
            if(joinPair)pairDestination_=destination;
            else FlushGroup();
        }
        if (!accepted_) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0); WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            for (uint32_t i=0;i<kDepth;++i) SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        }
        uint32_t stage=issued_%kDepth;
        if (!pending_) {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
            groupDestination_=destination;
        }
        // Defer at most four individually validated addresses so one MTE2
        // instruction can express a strided run without reading its gaps.
        sources_[pending_]=source;
        ++pending_; ++accepted_; nextDestination_=destination+kTokenBytes;
        if (pending_==kRowsPerStage) FlushGroup();
    }
public:
    __aicore__ inline uint32_t Finish() {
        if (!accepted_) return 0;
        // Invalid input never joins the group. Complete its accepted prefix,
        // including a partial last group, before publishing any completion.
        if (pending_) FlushGroup();
        // A partial/error prefix must drain the same accepted rows. The
        // prefill policy may still own up to depth-1 ready but unstored stages.
        if constexpr(PrefillLoads)while(stored_<issued_)StoreOldest();
        // Returned credits prove every store completed, including untouched
        // initial credits. Consume them before LI borrows these event IDs.
        for (uint32_t i=0;i<kDepth;++i) WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return accepted_;
    }
    __aicore__ inline void SaveCommandCounts(__gm__ uint64_t* out) const {
    }
    __aicore__ inline void SaveFinalCommandCounts(__gm__ uint64_t* out) const {
    }
private:
    __aicore__ inline bool ForwardPair(uint32_t first) const {
        if(sources_[first+1]<sources_[first])return false;
        uint64_t delta=sources_[first+1]-sources_[first];
        return delta>=kTokenBytes && delta-kTokenBytes<=UINT32_MAX;
    }
    __aicore__ inline void FlushGroup() {
        uint32_t stage=issued_%kDepth;
        uint32_t rows=pending_;
        // Every Compact load starts at an even row (1312-byte multiples).
        // A final odd row may pad only the end of the complete group. The
        // fallback keeps arbitrary/repeated/reversed peers fully supported.
        bool compact=pending_==1 || (ForwardPair(0) && (pending_<4 || ForwardPair(2)));
        for (uint32_t first=0;first<pending_;) {
            uint32_t count=1,stride=0;
            // Use only forward, nonoverlapping rows. Limit the byte gap to
            // uint32 even on 950's int64 stride API, preserving older SDKs.
            if (first+1<pending_ && sources_[first+1]>=sources_[first]) {
                uint64_t delta=sources_[first+1]-sources_[first];
                if (delta>=kTokenBytes && delta-kTokenBytes<=UINT32_MAX) {
                    stride=static_cast<uint32_t>(delta-kTokenBytes);
                    count=2;
                    while (first+count<pending_ && sources_[first+count]>=sources_[first+count-1] &&
                        sources_[first+count]-sources_[first+count-1]==delta) ++count;
                }
            }
            // A 3+1 pattern becomes 2+2: both starts stay 32-byte aligned,
            // with the same two MTE2 commands and no extra Host rows read.
            if(compact && (count&1) && first+count<pending_)--count;
            GlobalTensor<uint8_t> from;
            from.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sources_[first]));
            DataCopyExtParams copy{static_cast<uint16_t>(count),kTokenBytes,stride,0,0};
            DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
            // Both modes skip the same GM gaps. Compact pads only the end of
            // this command; Normal pads every row before the grouped store.
            if(compact) {
                DataCopyPad<uint8_t,PaddingMode::Compact>(Stage(stage)[first*kTokenBytes],from,copy,pad);
            } else {
                DataCopyPad<uint8_t,PaddingMode::Normal>(Stage(stage)[first*kRowBytes],from,copy,pad);
            }
            first+=count;
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        pending_=0; ++issued_;
        QueueStore(stage,groupDestination_,rows,compact,pairDestination_);
        pairDestination_=0;
    }
    __aicore__ inline void QueueStore(uint32_t stage,uint64_t destination,uint32_t rows,bool compact,
        uint64_t pairDestination=0) {
        if constexpr(PrefillLoads) {
            // Admit the full MTE2 ring before placing waits/stores into MTE3.
            // Only scalar metadata is retained; each UB stage keeps its own
            // ready/free credits and cannot be reused before StoreOldest.
            storeDestinations_[stage]=destination;
            storeDescriptors_[stage]=rows|(compact?256:0);
            if(issued_-stored_==kDepth)StoreOldest();
        } else {
            StoreOne(stage,destination,rows,compact,pairDestination);
        }
    }
    __aicore__ inline void StoreOldest() {
        uint32_t stage=stored_%kDepth,descriptor=storeDescriptors_[stage];
        StoreOne(stage,storeDestinations_[stage],descriptor&255,descriptor&256);
    }
    __aicore__ inline void StoreOne(uint32_t stage,uint64_t destination,uint32_t rows,bool compact,
        uint64_t pairDestination=0) {
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
        GlobalTensor<uint8_t> to;
        to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destination));
        if constexpr(PairGaps) {
            if(pairDestination) {
                if(compact && rows==4) {
                    // GM stride is bytes; UB block stride is rounded to 32B.
                    // Two 656B rows form one aligned1312B sector payload.
                    DataCopyExtParams copy{2,2*kTokenBytes,0,
                        static_cast<uint32_t>(pairDestination-destination-2*kTokenBytes),0};
                    DataCopyPad<uint8_t,PaddingMode::Normal>(to,Stage(stage),copy);
                } else {
                    DataCopyExtParams first{static_cast<uint16_t>(compact?1:2),
                        compact?2*kTokenBytes:kTokenBytes,0,0,0};
                    DataCopyPad<uint8_t,PaddingMode::Normal>(to,Stage(stage),first);
                    to.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(pairDestination));
                    DataCopyExtParams second{static_cast<uint16_t>(compact?1:rows-2),
                        compact?(rows-2)*kTokenBytes:kTokenBytes,0,0,0};
                    DataCopyPad<uint8_t,PaddingMode::Normal>(to,
                        Stage(stage)[2*(compact?kTokenBytes:kRowBytes)],second);
                }
            } else {
                DataCopyExtParams copy{static_cast<uint16_t>(compact?1:rows),
                    compact?rows*kTokenBytes:kTokenBytes,0,0,0};
                DataCopyPad<uint8_t,PaddingMode::Normal>(to,Stage(stage),copy);
            }
        } else {
            DataCopyExtParams copy{static_cast<uint16_t>(compact?1:rows),
                compact?rows*kTokenBytes:kTokenBytes,0,0,0};
            DataCopyPad<uint8_t,PaddingMode::Normal>(to,Stage(stage),copy);
        }
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
        ++stored_;
    }
    __gm__ UbmemRegion* region_;
    uint64_t output_,peerBase_=0,mappedBase_=0,bytes_=0,outputBase_=0,outputBytes_=0,maxPeerOffset_=0;
    LocalTensor<uint8_t> scratch_,overflow_;
    uint64_t storeDestinations_[PrefillLoads?kDepth:1];
    uint32_t storeDescriptors_[PrefillLoads?kDepth:1];
    uint64_t sources_[kRowsPerStage]{},groupDestination_=0,nextDestination_=0,runDestination_=0,pairDestination_=0;
    uint32_t issued_=0,stored_=0,accepted_=0,pending_=0,runRemaining_=0;
    bool valid_=false;
};

using UbmemGroupedCopyPipeline=UbmemGroupedCopyPipelineT<4>;
using UbmemFinalCopyPipeline=UbmemGroupedCopyPipelineT<4,true>;
using UbmemWindowCopyPipeline=UbmemGroupedCopyPipelineT<8,false,8,false>;
#elif V9_UBMEM_GROUPED_COPY
using UbmemGroupedCopyPipeline=UbmemPairCopyPipeline;
using UbmemFinalCopyPipeline=UbmemPairCopyPipeline;
using UbmemWindowCopyPipeline=UbmemPairCopyPipeline;
#else
using UbmemGroupedCopyPipeline=UbmemCopyPipeline;
using UbmemFinalCopyPipeline=UbmemCopyPipeline;
using UbmemWindowCopyPipeline=UbmemCopyPipeline;
#endif

struct UbmemCopyTrace {
    __aicore__ inline void Begin() {
    }
    __aicore__ inline void IssueEnd() {
    }
    __aicore__ inline void Done() {
    }
    __aicore__ inline void AccumulatePrefetch(__gm__ uint64_t* status,uint32_t rows) {
    }
    __aicore__ inline void SaveFinal(__gm__ FinalPost* post,uint32_t rows) {
    }
};
}
