#pragma once
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "rotation_abi.h"
#include "contract_device.h"
namespace rotating {
using namespace AscendC;
__simd_vf__ inline void PrepareProgress(__ubuf__ uint32_t* table, uint32_t baseTile) {
    using namespace MicroAPI;
    auto all = CreateMask<uint32_t, MaskPattern::ALL>();
    RegTensor<int32_t> sequence;
    Arange(sequence, int32_t(0));
    auto& index = reinterpret_cast<RegTensor<uint32_t>&>(sequence);
    RegTensor<uint32_t> value;
    ShiftRights(value, index, int16_t(4), all);
    Adds(value, value, uint32_t(baseTile + 1), all);
    Muls(value, value, uint32_t(128), all);
    for (uint16_t offset = 0; offset < 64 * 16; offset += 64) {
        StoreAlign(table + offset, value, all);
        Adds(value, value, uint32_t(512), all);
    }
}
__simd_callee__ inline void Visible(MicroAPI::RegTensor<uint16_t>& value,
    MicroAPI::RegTensor<uint16_t>& index, __ubuf__ uint16_t* input,
    __ubuf__ uint32_t* control, uint32_t q, MicroAPI::MaskReg& all) {
    using namespace MicroAPI;
    RegTensor<uint16_t> low, high, zero;
    MaskReg lower, upper, valid;
    LoadAlign(value, input);
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(low, reinterpret_cast<__ubuf__ uint16_t*>(control + 4 + q));
    LoadAlign<uint16_t, LoadDist::DIST_BRC_B16>(high, reinterpret_cast<__ubuf__ uint16_t*>(control + q));
    Compare<uint16_t, CMPMODE::GE>(lower, index, low, all);
    Compare<uint16_t, CMPMODE::LT>(upper, index, high, all);
    MaskAnd(valid, lower, upper, all);
    Duplicate(zero, uint16_t(0));
    Select(value, value, zero, valid);
}
}
#include "window_fetch.h"
namespace rotating {
__aicore__ inline void Service(GM_ADDR pointer, uint32_t epoch, uint32_t core,
    uint32_t lane, LocalTensor<uint32_t> scratch, LocalTensor<uint16_t> flags,
    LocalTensor<uint32_t> indices) {
    FastService(pointer, epoch, core, ServiceLane(core,lane), scratch, flags, indices);
}
}
