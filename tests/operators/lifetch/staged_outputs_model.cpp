// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>
#define __CCE_AICORE__ 1
#define V9_STAGED_OUTPUTS 1
#define __aicore__
#define __simt_vf__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {uint32_t x;} threadIdx;
constexpr uint32_t B=MODEL_BATCH,R=MODEL_RESIDENT,H=R+1024,Q=B*4,K=2048;
uint32_t currentCore=0,coldLoads=0,scalarWords=0,hotStages=0;
uintptr_t logicalBegin=0,logicalEnd=0,ownerBegin=0,statusBegin=0;
int32_t* outputBase=nullptr;
std::vector<uint32_t> writes(2*Q*K),writers(2*Q*K);
template<class T>T asc_ldcg(T* p) {
    assert(uintptr_t(p)>=logicalBegin && uintptr_t(p)<logicalEnd);
    uint32_t token=uint32_t((uintptr_t(p)-logicalBegin)/8)%H;
    assert(token>=R); // Every hot gather must use the staged UB copy.
    ++coldLoads;return *p;
}
namespace AscendC {
constexpr uint32_t EVENT_ID0=0;
enum class HardEvent {MTE2_S,MTE2_V,V_MTE3,MTE3_S};
inline bool flags[4]{};
inline std::vector<std::function<void()>> mte2,mte3;
inline bool vfReady=false,pipeLive=false;
alignas(32) std::array<uint64_t,223232/8> ub;
template<class T>struct LocalTensor {
    T* p;
    LocalTensor operator[](uint32_t index)const{return {p+index};}
    template<class U>LocalTensor<U> ReinterpretCast()const{return {reinterpret_cast<U*>(p)};}
    uintptr_t GetPhyAddr()const{return uintptr_t(p);}
    T GetValue(uint32_t index)const{assert(mte2.empty());return p[index];}
};
template<class T>struct GlobalTensor {
    T* p=nullptr;
    void SetGlobalBuffer(T* value){p=value;}
    GlobalTensor operator[](uint32_t index)const{return {p+index};}
};
enum class TPosition {VECCALC};
template<TPosition>struct TBuf {
    template<class T>LocalTensor<T> Get(){return {reinterpret_cast<T*>(ub.data())};}
};
struct TPipe {
    TPipe(){assert(!pipeLive);pipeLive=true;}
    template<class T>void InitBuffer(T&,uint32_t bytes){assert(bytes<=sizeof(ub) && bytes%32==0);}
    void Destroy(){assert(mte2.empty() && mte3.empty());for(bool flag:flags)assert(!flag);pipeLive=false;}
};
template<HardEvent E>void SetFlag(uint32_t){assert(!flags[unsigned(E)]);flags[unsigned(E)]=true;}
template<HardEvent E>void WaitFlag(uint32_t) {
    assert(flags[unsigned(E)]);flags[unsigned(E)]=false;
    if constexpr(E==HardEvent::MTE2_S || E==HardEvent::MTE2_V){for(auto& op:mte2)op();mte2.clear();}
    if constexpr(E==HardEvent::V_MTE3)vfReady=true;
    if constexpr(E==HardEvent::MTE3_S){for(auto& op:mte3)op();mte3.clear();}
}
template<class T>void DataCopy(LocalTensor<T> output,GlobalTensor<T> input,uint32_t count) {
    assert(mte3.empty() && (count*sizeof(T))%32==0);
    if(uintptr_t(input.p)>=logicalBegin && uintptr_t(input.p)<logicalEnd) {
        assert(count==R && (uintptr_t(input.p)-logicalBegin)%(H*8)==0);++hotStages;
    }
    mte2.push_back([=]{std::copy_n(input.p,count,output.p);});
}
template<class T>void DataCopy(GlobalTensor<T> output,LocalTensor<T> input,uint32_t count) {
    assert(vfReady && mte2.empty() && count==K);
    // Keep the source pointer, not a copied value: missing MTE3 completion
    // before the URMA planner reuses UB must corrupt output and fail this test.
    mte3.push_back([=]{
        for(uint32_t i=0;i<count;++i) {
            auto index=reinterpret_cast<int32_t*>(output.p+i)-outputBase;
            assert(index>=0 && uint64_t(index)<writes.size());
            assert(++writes[index]==1);writers[index]=currentCore;output.p[i]=input.p[i];
        }
    });
}
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,rsv;};
template<class T>struct DataCopyPadExtParams {bool pad;uint32_t left,right;T value;};
template<class T>void DataCopyPad(LocalTensor<T> output,GlobalTensor<T> input,
    DataCopyExtParams params,DataCopyPadExtParams<T>) {
    assert(params.blockCount==1 && params.blockLen==4 && !params.srcStride && !params.dstStride);
    uintptr_t address=uintptr_t(input.p);
    assert((address>=ownerBegin && address<ownerBegin+B*4) || (address>=statusBegin && address<statusBegin+B*4));
    ++scalarWords;mte2.push_back([=]{std::memcpy(output.p,input.p,4);});
}
namespace Simt {
struct Dim3{uint32_t x,y,z;};
template<auto F,class...Args>void VF_CALL(Dim3 grid,Args...args) {
    assert(mte2.empty() && mte3.empty());vfReady=false;
    for(threadIdx.x=0;threadIdx.x<grid.x;++threadIdx.x)F(args...);
}
}
}
#include "staged_outputs.h"

int main() {
    assert(kvip_output::CanStageOutputs(R,B));
    assert(!kvip_output::CanStageOutputs(32768,B));
    assert(!kvip_output::CanStageOutputs(R+1,B));
    assert(!kvip_output::CanStageOutputs(0,B));
    assert(!kvip_output::CanStageOutputs(R,33));
    std::vector<int32_t> topk(Q*K),slots(Q*K),output(2*Q*K+32,9191);
    std::vector<int64_t> logical((B+1)*H),unpack(Q),valid(Q);
    std::vector<uint32_t> owners(B),status(B);
    outputBase=output.data();logicalBegin=uintptr_t(logical.data());logicalEnd=logicalBegin+logical.size()*8;
    ownerBegin=uintptr_t(owners.data());statusBegin=uintptr_t(status.data());
    for(uint32_t b=0;b<B;++b)owners[b]=B-b;
    for(uint32_t owner=0;owner<=B;++owner)
        for(uint32_t token=0;token<H;++token)logical[owner*H+token]=owner*100000+token;
    for(uint32_t row=0;row<Q;++row)for(uint32_t k=0;k<K;++k) {
        topk[row*K+k]=k==0?-1:k==1?H:k%13==0?int32_t(R+(k+row)%1024):int32_t((k*17+row)%R);
        slots[row*K+k]=(k*31+row*17)%R;
    }
    const auto originalLogical=logical;
    const auto originalTopk=topk,originalSlots=slots;
    uint64_t a[28]{};
    a[3]=uint64_t(logical.data());a[6]=uint64_t(status.data());a[9]=uint64_t(owners.data());
    a[18]=uint64_t(topk.data());a[19]=uint64_t(slots.data());a[20]=uint64_t(unpack.data());a[21]=uint64_t(valid.data());
    a[22]=uint64_t(output.data());a[23]=uint64_t(output.data()+Q*K);a[24]=H;a[25]=R;
    for(uint32_t trial=0;trial<6;++trial) {
        for(uint32_t row=0;row<Q;++row) {
            unpack[row]=(row*3+5)%Q;
            if(trial==4)unpack[row]=Q-1; // Many TND rows may name one query.
            valid[row]=trial==2?0:trial==1?row%3!=0:1;
        }
        for(uint32_t b=0;b<B;++b)status[b]=trial==1 && b%2==0?2:0;
        if(trial>=2){unpack[0]=-1;unpack[Q-1]=Q;}
        if(trial==5)std::fill(status.begin(),status.end(),4);
        std::fill(output.begin(),output.end(),9191);std::fill(writes.begin(),writes.end(),0);
        uint32_t expectedCold=0,expectedStages=0;
        coldLoads=hotStages=scalarWords=0;
        for(uint32_t b=0;b<B;++b) {
            bool used[4]{};
            for(uint32_t row=0;row<Q;++row)if(valid[row] && unpack[row]>=0 && unpack[row]<Q &&
                uint32_t(unpack[row]/4)==b && (!MODEL_CHECK_STATUS || !status[b]))used[unpack[row]%4]=true;
            expectedStages+=std::any_of(used,used+4,[](bool x){return x;});
            for(uint32_t q=0;q<4;++q)if(used[q])for(uint32_t k=0;k<K;++k) {
                int32_t token=topk[(b*4+q)*K+k];expectedCold+=token>=int32_t(R) && token<int32_t(H);
            }
        }
        for(uint32_t i=0;i<B;++i) {
            currentCore=B-1-i;
            kvip_output::PublishStagedOutputs<MODEL_CHECK_STATUS>(a,currentCore,B);
            // Model the later Index planner occupying all the same UB.
            std::fill(AscendC::ub.begin(),AscendC::ub.end(),0xdeadbeefdeadbeefULL);
        }
        assert(coldLoads==expectedCold && hotStages==expectedStages);
        assert(scalarWords==B*(MODEL_CHECK_STATUS?2:1));
        assert(logical==originalLogical && topk==originalTopk && slots==originalSlots);
        for(uint32_t row=0;row<Q;++row)for(uint32_t k=0;k<K;++k) {
            bool mapped=valid[row] && unpack[row]>=0 && unpack[row]<Q;
            bool active=mapped && (!MODEL_CHECK_STATUS || !status[unpack[row]/4]);
            int32_t token=active?topk[unpack[row]*K+k]:-1;
            int32_t id=token>=0 && token<int32_t(H)?int32_t(logical[owners[unpack[row]/4]*H+token]):-1;
            assert(output[row*K+k]==id);
            assert(output[Q*K+row*K+k]==(active?slots[unpack[row]*K+k]:-1));
            for(uint32_t plane=0;plane<2;++plane) {
                uint32_t index=plane*Q*K+row*K+k;
                assert(writes[index]==1 && writers[index]==(mapped?uint32_t(unpack[row]/4):row/4));
            }
        }
        assert(std::all_of(output.end()-32,output.end(),[](int32_t x){return x==9191;}));
    }
}
