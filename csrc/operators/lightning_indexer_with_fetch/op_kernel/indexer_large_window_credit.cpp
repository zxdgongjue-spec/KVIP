// Compile the four-window cold entry independently so the production hot kernel
// and both scoring lanes in the comparison keep their original machine code.
#define KVIP_LARGE_FOUR_WINDOW_CREDIT 1
#define QLIKernel QLIKernelLargeWindowCredit
#define v9_li_large_threshold_cold v9_li_large_four_window_credit
#include "indexer_large.cpp"
