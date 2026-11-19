#pragma once
#include "score_partition.h"
// RuntimeShape and SequenceHalfWindow are supplied by context.h. The producer
// uses MR14 direct canonical/predicate output; ready cursors retain MR23 T2.
namespace kvip_score_t2 {
struct Window { uint32_t begin, end, end0, end1, parts; };
inline Window PlanWindow(const rotating::RuntimeShape& shape, uint32_t epoch, uint32_t requested) {
    const uint32_t half = rotating::SequenceHalfWindow(shape);
    const uint32_t begin = epoch ? shape.resident + 2 * half * (epoch - 1) : 0;
    const uint32_t end = epoch ? rotating::SequenceWindowEnd(shape.history, shape.resident, half, epoch) : shape.resident;
    const uint32_t end0 = epoch ? (begin + half < shape.history ? begin + half : shape.history) : shape.resident;
    const uint32_t pages = (end - begin) / 256;
    return {begin, end, end0, end, requested < pages ? requested : pages};
}
}
