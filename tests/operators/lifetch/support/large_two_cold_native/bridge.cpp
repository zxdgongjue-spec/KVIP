#include <cstdint>

#include "runtime_shape.h"
#if defined(KVIP_EXPERIMENT_SPLIT_ROLE)
constexpr uint32_t kWindows = 4;
#define test_forward kvip_test_bs32_split_role_forward
#if KVIP_TEST_SPLIT_ROLE_TRACE
#define launch_cold aclrtlaunch_v9_li_large_bs32_split_role_trace
#else
#define launch_cold aclrtlaunch_v9_li_large_bs32_split_role
#endif
#else
constexpr uint32_t kWindows = 3;
#define test_forward kvip_test_two_cold_forward
#if KVIP_TEST_SIBLING_TRACE
#define launch_cold aclrtlaunch_v9_li_large_two_cold_trace
#else
#define launch_cold aclrtlaunch_v9_li_large_two_cold
#endif
#endif
extern "C" uint32_t launch_cold(uint32_t, void*, void*, void*);
using Launch = uint32_t (*)(uint32_t, void*, void*, void*);
using TailLaunch = uint32_t (*)(uint32_t, void*, void*);

// Hot and final kernels come directly from the loaded candidate library.
extern "C" uint32_t test_forward(
    void* bindings, void* terminal, void* stream, const rotating::RuntimeShape* shape, Launch hot, TailLaunch tail) {
    if (!bindings || !terminal || !stream || !shape || !hot || !tail ||
        !rotating::NativeLargeBatch(shape->batch, shape->aicCount) || shape->serviceLanes != shape->batch ||
        shape->rounds != kWindows || shape->liCores != 32 || shape->finalWorkers != 4) {
        return 100000;
    }
#if defined(KVIP_EXPERIMENT_SPLIT_ROLE)
    if (shape->batch != 32 || shape->indexHeads != 32 || shape->history != 131328 ||
        shape->resident != 16384 || shape->tokenBytes != 656 || shape->splitQ != 0) {
        return 100000;
    }
#endif
    uint32_t rc = hot(shape->liCores, stream, bindings, terminal);
    if (!rc) rc = launch_cold(shape->liCores, stream, bindings, terminal);
    if (!rc) rc = tail(64, stream, terminal);
    return rc;
}
