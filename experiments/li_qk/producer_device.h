// Private experimental output sidecar; virtual score inputs remain unchanged.
#pragma once
#include "producer_plan.h"
#include "plan.h"
namespace li_producer {
class DeviceState {
    __gm__ const Bindings* a_=nullptr;
    li_qk::Task task_{};
public:
    __aicore__ inline void Init(GM_ADDR binding,AscendC::TPipe*,uint32_t block) {
        a_=reinterpret_cast<__gm__ const Bindings*>(binding);
        auto t=reinterpret_cast<__gm__ const li_qk::Task*>(a_->tasks)+block/2;
        task_={t->queryBegin,t->queryCount,t->begin,t->end,t->owner};
    }
    __aicore__ inline bool Enabled() const {return a_!=nullptr;}
    __aicore__ inline void Tile(AscendC::LocalTensor<uint16_t> scores,uint32_t stride,
        uint32_t first,uint32_t count,uint32_t token) {
        using namespace AscendC;
        GlobalTensor<uint16_t> dst;dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(a_->canonical));
        DataCopyExtParams cp{uint16_t(count),256,(stride*2-256)/32,(a_->history-128)*2,0};
        DataCopyPad(dst[(uint64_t(task_.owner)*4+task_.queryBegin+first)*a_->history+task_.begin+token],scores,cp);
    }
    __aicore__ inline void Finish() {}
};
}
