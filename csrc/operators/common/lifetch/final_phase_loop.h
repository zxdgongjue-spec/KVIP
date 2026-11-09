// Both transports submit nonconflicting work, join outstanding prefetch, then
// submit conflicts. UBmem arrives here after its synchronous prefetch completes.
for(uint32_t phase=0;phase<2;++phase) {
    if(phase==1) { LIFETCH_FINAL_JOIN; }
#include LIFETCH_FINAL_PHASE
}
