// SPDX-License-Identifier: Apache-2.0
#pragma once

// There are 256 positions by the initial window and 192 more at each cold
// boundary. Any row with unused earlier credit may consume it at this boundary.
namespace lifetch_earned_window {
__aicore__ inline uint32_t RowLimit(uint32_t epoch, uint32_t alreadyPosted) {
    if (epoch >= 5) return 0;
    const uint32_t earned = 256U + 192U * epoch;
    const uint32_t available = earned > alreadyPosted ? earned - alreadyPosted : 0U;
    const uint32_t ceiling = epoch == 4 ? 192U : 256U + 64U * epoch;
    return available < ceiling ? available : ceiling;
}
}
