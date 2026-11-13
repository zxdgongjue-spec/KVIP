"""Execute the register-only Kth VF and the production Threshold dispatch.

MicroAPI operations model their documented lanes and cumulative-histogram
semantics. Reductions poison every lane except lane zero so missing broadcasts
cannot silently pass. Device instruction/compiler behavior needs CANN/NPU runs.
"""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER
from test_recent_threshold_only import threshold_model_source


def register_model_source(source):
    return source.replace(
        '#include "recent_index.h"',
        r"""
#define __simd_vf__
#define __simd_callee__
namespace CMPMODE {constexpr int GE=0,LT=1,EQ=2;}
namespace AscendC::MicroAPI {
enum class MaskPattern {ALL};enum class LoadDist {DIST_DINTLV_B8};
enum class HighLowPart {LOWEST};enum class HistogramsBinType {BIN0,BIN1};
enum class HistogramsType {ACCUMULATE};
struct MaskReg {std::array<bool,256> v{};};
template<class T> struct alignas(32) RegTensor {std::array<T,256/sizeof(T)> v;};
unsigned loads=0,stores=0,histograms=0,reductions=0;
template<class T,MaskPattern> MaskReg CreateMask(){MaskReg m;std::fill_n(m.v.begin(),256/sizeof(T),true);return m;}
template<class T> void Arange(RegTensor<T>& r,T start){for(unsigned i=0;i<r.v.size();++i)r.v[i]=T(start+i);}
template<class T> void Duplicate(RegTensor<T>& r,T value){r.v.fill(value);}
template<class T,HighLowPart> void Duplicate(RegTensor<T>& r,RegTensor<T>& src,MaskReg mask){
 T value=src.v[0];for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=value;
}
template<class T> void Adds(RegTensor<T>& r,RegTensor<T>& a,T value,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=T(a.v[i]+value);
}
template<class T> void Muls(RegTensor<T>& r,RegTensor<T>& a,T value,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=T(a.v[i]*value);
}
template<class T> void Add(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=T(a.v[i]+b.v[i]);
}
template<class T> void Sub(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=T(a.v[i]-b.v[i]);
}
template<class T,int C> void Compare(MaskReg& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<a.v.size();++i)r.v[i]=mask.v[i]&&(C==CMPMODE::GE?a.v[i]>=b.v[i]:C==CMPMODE::LT?a.v[i]<b.v[i]:a.v[i]==b.v[i]);
}
template<class T> void Select(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)r.v[i]=mask.v[i]?a.v[i]:b.v[i];
}
template<class T> void Min(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=std::min(a.v[i],b.v[i]);
}
template<class T> void Max(RegTensor<T>& r,RegTensor<T>& a,RegTensor<T>& b,MaskReg mask){
 for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])r.v[i]=std::max(a.v[i],b.v[i]);
}
template<class T> void ReduceMin(RegTensor<T>& r,RegTensor<T>& a,MaskReg mask){
 T value=0xffff;for(unsigned i=0;i<a.v.size();++i)if(mask.v[i])value=std::min(value,a.v[i]);
 r.v.fill(T(0xa55a));r.v[0]=value;++reductions;
}
template<class T> void ReduceMax(RegTensor<T>& r,RegTensor<T>& a,MaskReg mask){
 T value=0;for(unsigned i=0;i<a.v.size();++i)if(mask.v[i])value=std::max(value,a.v[i]);
 r.v.fill(T(0x5aa5));r.v[0]=value;++reductions;
}
template<class T,LoadDist> void LoadAlign(RegTensor<T>& low,RegTensor<T>& high,T* src){
 static_assert(std::is_same_v<T,uint16_t>);++loads;
 auto lo=reinterpret_cast<uint8_t*>(low.v.data()),hi=reinterpret_cast<uint8_t*>(high.v.data());
 for(unsigned i=0;i<256;++i){lo[i]=uint8_t(src[i]);hi[i]=uint8_t(src[i]>>8);}
}
template<class T,class U,HistogramsBinType B,HistogramsType>
void Histograms(RegTensor<U>& dst,RegTensor<T>& src,MaskReg mask){
 static_assert(std::is_same_v<T,uint8_t>&&std::is_same_v<U,uint16_t>);++histograms;
 std::array<unsigned,256> freq{};for(unsigned i=0;i<256;++i)if(mask.v[i])++freq[src.v[i]];
 unsigned cumulative=0,first=B==HistogramsBinType::BIN0?0:128;
 for(unsigned bin=0;bin<256;++bin){
  cumulative+=freq[bin];if(bin>=first&&bin<first+128){
   unsigned sum=dst.v[bin-first]+cumulative;assert(sum<=65535);dst.v[bin-first]=uint16_t(sum);
  }
 }
}
template<class T> void StoreAlign(T* dst,RegTensor<T>& r,MaskReg mask){
 ++stores;for(unsigned i=0;i<r.v.size();++i)if(mask.v[i])dst[i]=r.v[i];
}
}
#include "recent_threshold.h"
#include "recent_index.h"
""",
    )


@pytest.mark.parametrize("ubmem", [0, 1])
def test_register_kth_exact_keys_and_original_fallback(tmp_path, ubmem):
    source = register_model_source(threshold_model_source().split("int main(){", 1)[0])
    source += r"""
using namespace AscendC::MicroAPI;
void Check(std::vector<uint16_t> values,unsigned k){
 const unsigned n=values.size();assert(n%256==0&&n<=18432&&k&&k<=n);
 const auto original=values;auto partitioned=values;
 std::nth_element(partitioned.begin(),partitioned.begin()+n-k,partitioned.end());
 uint16_t expected=partitioned[n-k];
 std::array<uint16_t,256> output;output.fill(0xa55a);
 loads=stores=histograms=reductions=0;
 kvip_recent_threshold::RegisterThreshold(values.data(),output.data()+64,uint16_t(n/256),uint16_t(n-k+1));
 for(unsigned i=0;i<64;++i)assert(output[i]==0xa55a&&output[192+i]==0xa55a);
 for(unsigned i=64;i<192;++i)assert(output[i]==expected);
 assert(values==original&&loads==2*n/256&&histograms==4*n/256&&stores==1&&reductions==3);
 // Run the actual production dispatcher. Only the UR early-threshold path bypasses
 // the original four VF contracts; both must return exactly the same key.
 TopK top;top.topK=k;topkb16gather::stage=topkb16gather::passes=0;
 output.fill(0xa55a);loads=stores=histograms=reductions=0;
 top.Threshold({values.data(),n},{output.data()+64,128},n);
 assert(output[64]==expected&&values==original);
 bool fused=!V9_UBMEM;
 assert(topkb16gather::passes==unsigned(!fused));assert(stores==unsigned(fused));
 for(unsigned i=0;i<64;++i)assert(output[i]==0xa55a&&output[192+i]==0xa55a);
}
int main(){
 std::mt19937 rng(117);
 for(unsigned n:{256U,2048U,2304U,4096U,8192U,16384U,16640U,18176U,18432U}){
  for(unsigned pattern=0;pattern<9;++pattern){
   std::vector<uint16_t> values(n);
   for(unsigned i=0;i<n;++i){
    switch(pattern){
     case 0:values[i]=uint16_t(rng());break;
     case 1:values[i]=0;break;
     case 2:values[i]=0xffff;break;
     case 3:values[i]=i%2?0x7fff:0x8000;break;
     case 4:values[i]=i%2?0x007f:0x0080;break;
     case 5:values[i]=uint16_t(0xff00+(rng()%256));break;
     case 6:values[i]=i<n-127?uint16_t(rng()):0;break; // 256 padding.
     case 7:values[i]=i%17?0x8080:0x807f;break;
     default:values[i]=uint16_t(i%2?0:0xffff);break;
    }
   }
   for(unsigned k:{1U,std::min(2048U,n),n/2,n})Check(values,k);
  }
 }
 // Exercise every sortable-u16 key and every selected high bin. No BF16
 // decoding or signed comparison is permitted anywhere in this selector.
 for(unsigned high=0;high<256;++high){
  std::vector<uint16_t> values(256);
  for(unsigned low=0;low<256;++low)values[low]=uint16_t(high*256+low);
  for(unsigned k:{1U,17U,128U,256U})Check(values,k);
 }
 // Larger and unaligned inputs retain the established four-VF fallback.
 for(unsigned valid:{18688U,32768U,4097U}){
  unsigned padded=(valid+255)/256*256;
  std::vector<uint16_t> values(padded);for(auto& x:values)x=uint16_t(rng());
  std::array<uint16_t,128> output{};TopK top;top.topK=2048;
  topkb16gather::stage=topkb16gather::passes=0;stores=0;
  top.Threshold({values.data(),padded},{output.data(),128},valid);
  assert(topkb16gather::passes==1&&stores==0);
 }
}
"""
    cpp = tmp_path / "register_kth.cpp"
    cpp.write_text(source)
    exe = tmp_path / "register_kth"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DV9_UBMEM={ubmem}",
            "-I",
            str(COMMON),
            "-I",
            str(INDEXER),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
