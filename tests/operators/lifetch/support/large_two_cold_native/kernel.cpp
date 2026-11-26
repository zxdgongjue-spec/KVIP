// This optional translation unit is built only by the large-batch two-cold replay driver.
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
