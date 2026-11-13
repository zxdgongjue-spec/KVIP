// SPDX-License-Identifier: Apache-2.0
#pragma once

// The Context ABI is not necessarily a multiple of a 32-byte DMA block.
// Copy exactly its bytes, including the optional admissionControls pointer;
// callers join MTE2 before a SIMT consumer reads this local snapshot.
__aicore__ inline void CopyContextSnapshot(
    LocalTensor<uint32_t> destination, __gm__ Context* context) {
    GlobalTensor<uint8_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(context));
    DataCopyExtParams copy{1, uint32_t(sizeof(Context)), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    DataCopyPad(destination.ReinterpretCast<uint8_t>(), source, copy, pad);
}
