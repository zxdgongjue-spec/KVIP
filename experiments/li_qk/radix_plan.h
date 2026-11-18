#pragma once
#include <stdint.h>
namespace li_radix {
constexpr uint32_t K=2048, Bins=256;
struct Bindings {
    uint64_t scores,ends,selected,high,low,params,offsets;
    uint32_t batch,history,parts;
};
inline bool Valid(uint32_t b,uint32_t h,uint32_t p) {
    return (b==1 || b==16) && h>=2048 && h<=131072 && h%256==0 && p>=1 && p<=32 && h/128>=p;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline uint32_t Begin(uint32_t h,uint32_t p,uint32_t i){return (h/128)*i/p*128;}
}
