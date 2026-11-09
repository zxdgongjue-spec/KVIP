// Shared candidate scan; included inside FastService with its local tensors.
        uint32_t candidateCount=0,stop=last,wordCount=0;
        bool aggregate=first%128==0 && last%128==0;
        for(uint32_t q=0;q<4;++q) {
            uint32_t consumed=state.GetValue(12+q);if(consumed<lower)consumed=lower;
            if(consumed!=first || end[q]!=last)aggregate=false;
        }
        GlobalTensor<uint8_t> bitmap;
        bitmap.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(ctx->hits)+b*KV_WINDOW_HISTORY);
        for(uint32_t base=(first/128)*128;first<last && base<last;) {
            // Compact victim coordinates occupy [0,2048); the remaining
            // hot bitmap is not written or read. Cold coordinates are unchanged.
            if(ctx->lruSlots && base>=kLruSlots && base<KV_WINDOW_RESIDENT) {
                base=KV_WINDOW_RESIDENT;continue;
            }
            uint32_t span=((last-base+127)/128)*128;
            if(span>KV_WINDOW_SCAN_TOKENS)span=KV_WINDOW_SCAN_TOKENS;
            if(ctx->lruSlots && base<kLruSlots && span>kLruSlots-base)span=kLruSlots-base;
            if(aggregate) {
                uint32_t room=((1792-wordCount)/8)*128;
                if(!room) {stop=base;break;}
                if(span>room)span=room;
            }
            // ready ends are 128-aligned; a capacity cutoff can leave a
            // half-tile cursor, which is masked on the next service.
            bool full=true;
            for(uint32_t q=0;q<4;++q) {
                uint32_t lo=state.GetValue(12+q)>base?state.GetValue(12+q)-base:0;
                if(base<lower && lo<lower-base)lo=lower-base;
                uint32_t hi=end[q]>base?end[q]-base:0;
                cfg.SetValue(28+q,lo<span?lo:span);cfg.SetValue(24+q,hi<span?hi:span);
                if(lo || hi<span)full=false;
#if V9_PARTIAL_SCAN_SIMD
                uint32_t clippedLo=lo<span?lo:span,clippedHi=hi<span?hi:span;
                cfg.SetValue(80+q*4,clippedLo/16);cfg.SetValue(81+q*4,clippedHi/16);
                cfg.SetValue(82+q*4,0x55555555U<<((clippedLo%16)*2));
                cfg.SetValue(83+q*4,((1U<<((clippedHi%16)*2))-1U)&0x55555555U);
#endif
            }
            DataCopyExtParams copy;copy.blockCount=4;copy.blockLen=span/4;
            copy.srcStride=(KV_WINDOW_HISTORY-span)/4;copy.dstStride=0;
            V9_FETCH_PROFILE(bitmapBytes+=span;)
            DataCopyPadExtParams<uint8_t> padding{false,0,0,0};
            DataCopyPad(scratch.ReinterpretCast<uint8_t>(),bitmap[base/4],copy,padding);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if(full || V9_PARTIAL_SCAN_SIMD) {
                uint32_t prior=aggregate?wordCount:0;
                if(full)UnionWords<false>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                else UnionWords<true>(reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(work[prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work[2304+prior].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),span/16,aggregate?base/16:0);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                uint32_t words=cfg.GetValue(72);wordCount+=words;
                if(!aggregate && words)Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount,words);
                else {cfg.SetValue(40,candidateCount);cfg.SetValue(48,span);}
            } else Simt::VF_CALL<ScanPartial>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),base,span,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);uint32_t scanned=cfg.GetValue(48);
            // V_S above joins the scan; forward that dependency to MTE2.
            // Do not consume LI's persistent V_MTE2 ping-pong credits.
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            if(scanned<span) { stop=base+scanned;break; }
            base+=span;
        }
        if(aggregate && wordCount) {
            Simt::VF_CALL<ExpandWords>(Simt::Dim3{256,1,1},reinterpret_cast<__ubuf__ uint32_t*>(work[2304].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(work.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(cfg.GetPhyAddr()),0,stop,0,wordCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            candidateCount=cfg.GetValue(40);stop=cfg.GetValue(48);
        }
        if(ctx->lruSlots && first<kLruSlots && candidateCount) {
            Simt::VF_CALL<RestoreVictimTokens>(Simt::Dim3{256,1,1},
                reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),
                reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots,candidateCount);
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
