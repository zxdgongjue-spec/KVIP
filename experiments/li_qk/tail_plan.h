#pragma once
#include <stdint.h>
namespace li_tail {
constexpr uint32_t K=2048;
struct Bindings {
    uint64_t scores,selected,residentSelected,localIndices,candidateScores,candidateTokens,mergeIndices;
    uint64_t prefix,hits,context,plainContext,locations,queryBitmaps,queryErrors,unionBitmap;
    uint32_t batch,history,resident,parts,residentParts,coldParts;
};
inline bool Valid(uint32_t b,uint32_t h,uint32_t r,uint32_t p,uint32_t rp) {
    return (b==1 || b==16) && h>=4096 && h<=131072 && h%256==0 &&
      r>=2048 && r<=16384 && r<h && r%256==0 &&
      (p==1 || p==2 || p==4 || p==8 || p==16) && h%p==0 && h/p>=K && (h/p)%256==0 &&
      (rp==1 || rp==2 || rp==4) && r%rp==0 && r/rp>=K && (r/rp)%256==0;
}
}
