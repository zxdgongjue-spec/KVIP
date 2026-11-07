// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
// Simulator substitutes only MMIO; the native commit/copy functions are real.
__aicore__ inline void TestDoorbell(uint32_t value,__gm__ uint32_t* address,uint32_t offset) {
  address[2+address[1]]=value;
  ++address[1];
  *address=value;
  AscendC::GlobalTensor<uint32_t> target;target.SetGlobalBuffer(address);
  AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
      AscendC::DcciDst::CACHELINE_OUT>(target);
}
#define st_dev(value, address, offset) TestDoorbell(value, address, offset)
#include "resolver.h"
#undef st_dev
using namespace AscendC;
using namespace rotating;
#endif
extern "C" __global__ __aicore__ void test_transport_submit(GM_ADDR context) {
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
  TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,kFinalScratchBytes);auto local=scratch.Get<uint32_t>();
  auto low=local[16384],high=local[24576],slots=local[45056],cfg=local[54272];
  auto ctx=reinterpret_cast<__gm__ Context*>(context);auto row=reinterpret_cast<__gm__ Row*>(ctx->rows);
  uint32_t count=row->finalRequests;
  for(unsigned j=0;j<count;++j){low.SetValue(j,0x100000+j);high.SetValue(j,0x1234);slots.SetValue(j,j);}
  SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
  PublishFinalWorker(1,1,0,656,ctx,row,0,0,count,count,0,0,local,low,high,slots,local[53248],false);
 }
#endif
}
