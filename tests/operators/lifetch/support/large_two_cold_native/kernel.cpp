// Keep all experiment symbols separate from the already loaded standard kernels.
#if defined(KVIP_EXPERIMENT_SPLIT_ROLE)
#include "indexer_large_dualrole.cpp"
#if defined(__CCE_AICORE__) && KVIP_BS32_SCORE_WINDOW_ROLES != 1
#error "Split-role device implementation is required"
#endif
#else
#define KVIP_EXPERIMENT_TWO_COLD 1
#if KVIP_TEST_SIBLING_TRACE
#define QLIKernel TwoColdTraceQLIKernel
#define v9_li_large_hot v9_li_two_cold_trace_unused_hot
#define v9_li_large_threshold_cold v9_li_large_two_cold_trace
#else
#define QLIKernel TwoColdQLIKernel
#define v9_li_large_hot v9_li_two_cold_unused_hot
#define v9_li_large_threshold_cold v9_li_large_two_cold
#endif
#include "indexer_large.cpp"
#endif
