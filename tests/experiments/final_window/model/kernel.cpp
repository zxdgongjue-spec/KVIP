#define KVIP_FINAL_WINDOW_INCLUDE
#include "indexer.cpp"
#if defined(__CCE_AICORE__)
#include "pipeline_types.h"
#include "terminal_window.h"
#endif
extern "C" __global__ __aicore__ void test_service(GM_ADDR arg,uint32_t mode){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 auto a=reinterpret_cast<__gm__ pipeline_ab::Args*>(arg);
 uint32_t q=GetBlockIdx();
 if(!q)RunSharedFinalWindow(a,0);
 AscendC::SyncAll<true>();
 PublishSavedTopK(a,q);
 AscendC::SyncAll<true>();
 if(mode==5 && !q)PublishTerminalReady(a,0);
#endif
}
