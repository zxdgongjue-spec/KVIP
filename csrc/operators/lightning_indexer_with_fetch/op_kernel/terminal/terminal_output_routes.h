#pragma once
// The caller has already joined V -> MTE3. These masks were produced by the
// template warp inside the existing packing task, not a separate dispatch.
template<bool LargeBatch>
__aicore__ inline void PublishTerminalOutputs(
    __gm__ uint64_t* update, uint32_t batch, uint32_t query,
    LocalTensor<uint32_t> logical, LocalTensor<uint32_t> slots,
    LocalTensor<uint32_t> masks, uint64_t internalSlots) {
    GlobalTensor<uint32_t> output, outputSlots;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[22]));
    outputSlots.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(update[23]));
    const uint32_t words=(batch*4+31)/32;
    for(uint32_t word=0;word<words;++word){
        uint32_t mask=masks.GetValue(word);
        while(mask){
            uint32_t row=word*32+__builtin_ctz(mask);
            DataCopy(output[row*2048],logical,2048);
            if(update[23]+uint64_t(row)*2048*4!=internalSlots)
                DataCopy(outputSlots[row*2048],slots,2048);
            mask&=mask-1;
        }
    }
    constexpr uint32_t paddingOffset = terminal_runtime::Routes<LargeBatch>::words;
    bool hasPadding = false;
    for(uint32_t word=0;word<words;++word)
        hasPadding |= masks.GetValue(paddingOffset+word) != 0;
    if(hasPadding){
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        Duplicate(logical,UINT32_MAX,2048);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for(uint32_t word=0;word<words;++word){
            uint32_t mask=masks.GetValue(paddingOffset+word);
            while(mask){
                uint32_t row=word*32+__builtin_ctz(mask);
                DataCopy(output[row*2048],logical,2048);
                DataCopy(outputSlots[row*2048],logical,2048);
                mask&=mask-1;
            }
        }
    }
    // IDs may now be reused by WQE construction; every public output DMA has
    // consumed its UB input. Kernel completion still gates consumers of KV.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}
