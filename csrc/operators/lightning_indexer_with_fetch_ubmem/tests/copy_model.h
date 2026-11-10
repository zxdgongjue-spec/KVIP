#pragma once
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

#define __aicore__
#define __gm__
#define __simt_callee__
namespace copy_model {
inline bool events[7][8]{};
inline size_t fences[7][8]{};
inline std::vector<std::function<void()>> loads,stores;
inline size_t loadsDone=0,storesDone=0,loadCommands=0,storeCommands=0;
inline unsigned worker=0;
inline std::map<uintptr_t,unsigned> owners;
inline void Write(void* output,size_t bytes) {
    auto begin=reinterpret_cast<uintptr_t>(output);
    for(uintptr_t sector=begin/32;sector<=(begin+bytes-1)/32;++sector) {
        auto result=owners.emplace(sector,worker);
        assert(result.second || result.first->second==worker);
    }
}
inline void Complete(std::vector<std::function<void()>>& commands,size_t& done,size_t fence) {
    assert(fence<=commands.size());
    while(done<fence)commands[done++]();
}
inline void Drained() {
    for(auto& direction:events)for(bool credit:direction)assert(!credit);
    assert(loadsDone==loads.size() && storesDone==stores.size());
    loads.clear();stores.clear();loadsDone=storesDone=0;
}
}
namespace AscendC {
using event_t=unsigned;
constexpr event_t EVENT_ID0=0;
enum class HardEvent {S_MTE2,MTE2_S,S_MTE3,MTE3_S,MTE3_MTE2,MTE2_MTE3};
template<HardEvent Direction> void SetFlag(event_t event) {
    assert(event<8);auto& credit=copy_model::events[static_cast<unsigned>(Direction)][event];assert(!credit);credit=true;
    auto& fence=copy_model::fences[static_cast<unsigned>(Direction)][event];
    if constexpr(Direction==HardEvent::MTE2_MTE3 || Direction==HardEvent::MTE2_S)fence=copy_model::loads.size();
    if constexpr(Direction==HardEvent::MTE3_MTE2 || Direction==HardEvent::MTE3_S)fence=copy_model::stores.size();
}
template<HardEvent Direction> void WaitFlag(event_t event) {
    assert(event<8);auto& credit=copy_model::events[static_cast<unsigned>(Direction)][event];assert(credit);credit=false;
    auto fence=copy_model::fences[static_cast<unsigned>(Direction)][event];
    if constexpr(Direction==HardEvent::MTE2_MTE3 || Direction==HardEvent::MTE2_S)
        copy_model::Complete(copy_model::loads,copy_model::loadsDone,fence);
    if constexpr(Direction==HardEvent::MTE3_MTE2 || Direction==HardEvent::MTE3_S)
        copy_model::Complete(copy_model::stores,copy_model::storesDone,fence);
}
template<class T> struct LocalTensor {
    T* data;size_t bytes;
    LocalTensor operator[](size_t index)const {assert(index*sizeof(T)<=bytes);return {data+index,bytes-index*sizeof(T)};}
    T GetValue(size_t index)const {assert((index+1)*sizeof(T)<=bytes);T result;std::memcpy(&result,data+index,sizeof(T));return result;}
    void SetValue(size_t index,T value)const {assert((index+1)*sizeof(T)<=bytes);std::memcpy(data+index,&value,sizeof(T));}
    template<class U> LocalTensor<U> ReinterpretCast()const {return {reinterpret_cast<U*>(data),bytes};}
};
template<class T> struct GlobalTensor {
    T* data;
    void SetGlobalBuffer(T* pointer) {data=pointer;}
};
enum class PaddingMode {Normal,Compact};
struct DataCopyExtParams {uint16_t blockCount;uint32_t blockLen,srcStride,dstStride,reserved;};
template<class T> struct DataCopyPadExtParams {bool enabled;uint8_t left,right;T value;};
template<class T,PaddingMode Mode>
void DataCopyPad(LocalTensor<T> destination,GlobalTensor<T> source,DataCopyExtParams copy,DataCopyPadExtParams<T>) {
    size_t localStride=(Mode==PaddingMode::Compact?copy.blockLen:(copy.blockLen+31)/32*32)+copy.dstStride*32;
    auto* out=reinterpret_cast<uint8_t*>(destination.data);
    auto* in=reinterpret_cast<uint8_t*>(source.data);
    ++copy_model::loadCommands;
    copy_model::loads.push_back([=] {
        for(unsigned block=0;block<copy.blockCount;++block) {
            assert(block*localStride+copy.blockLen<=destination.bytes);
            std::memcpy(out+block*localStride,in+block*(copy.blockLen+copy.srcStride),copy.blockLen);
        }
    });
}
template<class T,PaddingMode>
void DataCopyPad(GlobalTensor<T> destination,LocalTensor<T> source,DataCopyExtParams copy) {
    size_t localStride=(copy.blockLen+31)/32*32+copy.srcStride*32;
    auto* out=reinterpret_cast<uint8_t*>(destination.data);
    auto* in=reinterpret_cast<uint8_t*>(source.data);
    const auto worker=copy_model::worker;
    ++copy_model::storeCommands;
    copy_model::stores.push_back([=] {
        assert(copy_model::worker==worker);
        for(unsigned block=0;block<copy.blockCount;++block) {
            assert(block*localStride+copy.blockLen<=source.bytes);
            copy_model::Write(out+block*(copy.blockLen+copy.dstStride),copy.blockLen);
            std::memcpy(out+block*(copy.blockLen+copy.dstStride),in+block*localStride,copy.blockLen);
        }
    });
}
}
using namespace AscendC;
