// SPDX-License-Identifier: Apache-2.0
// Bounded SIMD instruction probes for the production stable compactor.
#define __simd_vf__
namespace AscendC {
enum class SpecialPurposeReg {AR};
inline uint32_t arBytes=0,squeezedCount=0;
template<SpecialPurposeReg>uint32_t GetSpr(){return arBytes;}
namespace MicroAPI {
enum class MaskPattern {ALL};
enum class MemType {VEC_STORE,VEC_LOAD};
enum class LoadDist {DIST_BRC_B16};
enum class GatherMaskMode {STORE_REG};
enum class PostLiteral {POST_MODE_UPDATE};
struct MaskReg {std::array<bool,128> bits{};};
template<class T>struct RegTensor {T data[128]{};};
struct UnalignRegForStore {};
template<class T,MaskPattern>MaskReg CreateMask(){MaskReg m;m.bits.fill(true);return m;}
template<class T>MaskReg UpdateMask(uint32_t& left){MaskReg m;for(unsigned i=0;i<128;++i)m.bits[i]=i<left;left=left>128?left-128:0;return m;}
template<class T>void Duplicate(RegTensor<T>& r,T v){std::fill_n(r.data,128,v);}
template<class T>void LoadAlign(RegTensor<T>& r,T* p){std::copy_n(p,128,r.data);}
template<class T,LoadDist>void LoadAlign(RegTensor<T>& r,T* p){std::fill_n(r.data,128,*p);}
template<class T>void StoreAlign(T* p,const RegTensor<T>& r,MaskReg m){for(unsigned i=0;i<128;++i)if(m.bits[i])p[i]=r.data[i];}
template<class T>void Min(RegTensor<T>& out,const RegTensor<T>& a,const RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<128;++i)if(m.bits[i])out.data[i]=std::min(a.data[i],b.data[i]);}
template<class T>void ReduceMin(RegTensor<T>& out,const RegTensor<T>& in,MaskReg){out.data[0]=*std::min_element(in.data,in.data+128);}
template<MemType,MemType>void LocalMemBar(){}
template<SpecialPurposeReg>void ClearSpr(){arBytes=0;}
template<class T>void Arange(RegTensor<T>& out,T first){for(unsigned i=0;i<128;++i)out.data[i]=T(first+i);}
template<class T,CMPMODE Mode>void Compare(MaskReg& out,const RegTensor<T>& a,const RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<128;++i)out.bits[i]=m.bits[i]&&(Mode==CMPMODE::GT?a.data[i]>b.data[i]:a.data[i]>=b.data[i]);}
template<class T,CMPMODE Mode>void Compares(MaskReg& out,const RegTensor<T>& a,T b,MaskReg m){RegTensor<T> rhs;Duplicate(rhs,b);Compare<T,Mode>(out,a,rhs,m);}
template<class T,GatherMaskMode>void Squeeze(RegTensor<T>& out,const RegTensor<T>& in,MaskReg m){squeezedCount=0;for(unsigned i=0;i<128;++i)if(m.bits[i])out.data[squeezedCount++]=in.data[i];}
template<class T,PostLiteral>void StoreUnAlign(T* p,const RegTensor<T>& in,UnalignRegForStore&){std::copy_n(in.data,squeezedCount,p+arBytes/sizeof(T));arBytes+=squeezedCount*sizeof(T);}
template<class T>void StoreUnAlignPost(T*,UnalignRegForStore&){}
template<class T>void Gather(RegTensor<T>& out,T* p,const RegTensor<uint16_t>& offsets,MaskReg m){for(unsigned i=0;i<128;++i)if(m.bits[i]){assert(offsets.data[i]<16384);out.data[i]=p[offsets.data[i]];}}
template<class T>void Select(RegTensor<T>& out,const RegTensor<T>& a,const RegTensor<T>& b,MaskReg m){for(unsigned i=0;i<128;++i)out.data[i]=m.bits[i]?a.data[i]:b.data[i];}
template<class T>void Adds(RegTensor<T>& out,const RegTensor<T>& in,T v,MaskReg m){for(unsigned i=0;i<128;++i)if(m.bits[i])out.data[i]=T(in.data[i]+v);}
}
}
#include "pruned_topk.h"
