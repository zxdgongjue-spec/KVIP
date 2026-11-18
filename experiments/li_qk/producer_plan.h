#pragma once
#include <stdint.h>
namespace li_producer {
constexpr uint32_t Parts=16,K=2048,Shard=8192;
struct Bindings {
    uint64_t canonical,tasks,high,candidateScores,candidateIds,selected;
    uint32_t history,route;
};
}
