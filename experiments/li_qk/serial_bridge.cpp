#include <cstring>
#include "context.h"
extern "C" uint32_t aclrtlaunch_li_serial_prepare(uint32_t,void*,void*);
extern "C" uint32_t aclrtlaunch_li_qk_score_producer(uint32_t,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_li_qk_topk(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t li_serial_context_bytes(){return sizeof(rotating::Context);}
extern "C" uint32_t li_serial_context_init(void* out,uint32_t bytes,uint32_t batch,uint32_t history,uint32_t heads){
 if(!out||bytes!=sizeof(rotating::Context)||!batch||batch>16||history%256||(heads!=32&&heads!=64))return 100000;
 rotating::Context c{};c.shape.batch=batch;c.shape.history=c.shape.resident=history;c.shape.indexHeads=heads;
 std::memcpy(out,&c,sizeof(c));return 0;
}
extern "C" uint32_t li_serial_score(void* stream,void* inputs,void* bindings,void* output,uint32_t rows,uint32_t stride,uint32_t heads){
 if(!stream||!inputs||!bindings||!output||!rows||rows>32||stride%2048)return 100000;
 auto r=aclrtlaunch_li_serial_prepare(rows,stream,inputs);
 return r?r:aclrtlaunch_li_qk_score_producer(rows,stream,bindings,output,rows,stride,heads,4);
}
extern "C" uint32_t li_serial_topk(void* stream,void* scores,void* selected,void* context,uint32_t history,uint32_t heads,uint32_t batch,uint32_t aivs){
 if(!stream||!scores||!selected||!context||!batch||batch>16||aivs<4||aivs>64)return 100000;
 uint32_t capacity=(aivs/4)*4;
 for(uint32_t first=0;first<4*batch;first+=capacity){
  uint32_t n=4*batch-first;if(n>capacity)n=capacity;
  auto r=aclrtlaunch_li_qk_topk(n,stream,static_cast<uint8_t*>(scores)+uint64_t(first)*history*2,
   static_cast<uint8_t*>(selected)+uint64_t(first)*2048*4,context,history,heads,n);
  if(r)return r;
 }
 return 0;
}
