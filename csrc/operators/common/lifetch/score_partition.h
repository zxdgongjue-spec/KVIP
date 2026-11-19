#pragma once
#include <cstdint>
#if defined(__CCE_AICORE__)
#define KVIP_SCORE_DEVICE __aicore__
#else
#define KVIP_SCORE_DEVICE
#endif
namespace kvip_score_t2 {
struct Slice { uint32_t owner, beginTile, endTile; };
// Partition whole 256-key pages. Each producer writes real packed-row/Q/H
// addresses; physical core IDs must never become output row IDs.
KVIP_SCORE_DEVICE inline Slice Partition(uint32_t task, uint32_t parts, uint32_t history) {
    uint32_t shard = task % parts, pages = history / 256;
    return {task / parts, (pages * shard / parts) * 2,
                         (pages * (shard + 1) / parts) * 2};
}
KVIP_SCORE_DEVICE inline Slice WindowPartition(uint32_t task, uint32_t parts, uint32_t begin, uint32_t end) {
    const auto local = Partition(task, parts, end - begin);
    return {local.owner, begin / 128 + local.beginTile, begin / 128 + local.endTile};
}
KVIP_SCORE_DEVICE inline bool Valid(uint32_t batch, uint32_t history, uint32_t parts, uint32_t aics) {
    return batch >= 1 && batch <= 16 && parts >= 1 && parts <= 32 &&
           history % 256 == 0 && history / 256 >= parts && batch * parts <= aics;
}
KVIP_SCORE_DEVICE inline bool ValidWindow(uint32_t batch, uint32_t begin, uint32_t end,
                                         uint32_t parts, uint32_t aics) {
    return begin % 256 == 0 && end > begin && Valid(batch, end - begin, parts, aics);
}
}
#undef KVIP_SCORE_DEVICE
