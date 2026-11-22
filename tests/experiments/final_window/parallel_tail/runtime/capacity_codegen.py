"""Candidate overflow allocation and larger per-worker update plans."""
from pathlib import Path
import shutil

from private_topk import replace_once


def adapt_capacity(destination):
    shutil.copy2(Path(__file__).with_name('capacity.h'), destination/'capacity.h')
    path=destination/'kernel.cpp'
    text=replace_once(path.read_text(), '#include "helpers.h"',
                      '#include "helpers.h"\n#include "capacity.h"')
    # 10240 records / four workers. The ordinary per-worker plan previously
    # overlapped its destination array at entry 512. TopK scratch has 48 KiB.
    text=replace_once(text, 'TAIL_PLAN_DST=4608, TAIL_CONTROL=5120;',
                      'TAIL_PLAN_DST=6656, TAIL_CONTROL=9216;')
    text=replace_once(text, ' if(profile)ts[0]=GetSystemCycle();',
                      ' uint32_t priorSlots=(reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+b)->slots;\n if(profile)ts[0]=GetSystemCycle();')
    marker=' // Consume the original compact UB result directly; compact destinations use postWork.'
    prep='''
 // All final prefixes must be visible before an overflow allocator protects
 // their hits. The same join also makes the final row count authoritative.
 if(profile)ts[18]=GetSystemCycle();
 auto activeLru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*rotating::kLruSlots;
 uint32_t slotCapacity=rotating::kLruSlots;bool capacityCounted=false;
 auto headers=reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+b*128;
 Simt::VF_CALL<PrepareTerminalCapacity>(Simt::Dim3{32,1,1},
  reinterpret_cast<__ubuf__ uint32_t*>(postIds.GetPhyAddr()),candidates,mapping,
  headers,flags,generation,q,priorSlots,
  reinterpret_cast<__ubuf__ uint32_t*>(postScratch[bitmapWords+16].GetPhyAddr()));
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 uint32_t upper=postScratch.GetValue(bitmapWords+16);
 capacityCounted=postScratch.GetValue(bitmapWords+18)!=0;
 if(capacityCounted){candidates=postScratch.GetValue(bitmapWords+17);counts[q]=candidates;width=(candidates+31)/32*32;}
 if(upper>rotating::kLruSlots){
  auto extended=reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[0])+uint64_t(b)*(rotating::kChangeCapacity+4*((s.resident+31)/32));
  auto savedIds=reinterpret_cast<__gm__ uint32_t*>(ctx->incremental+uint64_t(s.batch)*512+uint64_t(b)*(4*2048*6+1024)+4*2048*2);
  auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident;
  Simt::VF_CALL<ExtendTerminalLru>(Simt::Dim3{512,1,1},activeLru,extended,savedIds,inverse,
   ctx->queryEnds?reinterpret_cast<__gm__ uint32_t*>(ctx->queryEnds)+b*4:nullptr,
   s.resident,upper<rotating::kChangeCapacity?upper:rotating::kChangeCapacity,q,flags,extended+rotating::kChangeCapacity,
   reinterpret_cast<__ubuf__ uint32_t*>(postIds[2048].GetPhyAddr()));
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  slotCapacity=postIds.GetValue(2048);activeLru=extended;
 }
 if(profile)ts[19]=GetSystemCycle();
'''
    text=replace_once(text, marker, prep+marker)
    text=replace_once(text, '(reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+uint64_t(b)*rotating::kLruSlots),(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)',
                      'activeLru,slotCapacity,capacityCounted,(reinterpret_cast<__gm__ uint32_t*>(ctx->changes)')
    text=text.replace('n=total<=rotating::kLruSlots?last-first:0;',
                      'n=total<=rotating::kChangeCapacity?last-first:0;')
    text=text.replace('indexError=total>rotating::kLruSlots?1U:',
                      'indexError=total>rotating::kChangeCapacity?1U:')
    # Wait for all admissions before partitioning a shared count; Q1's own
    # output publication alone does not imply Q4 has allocated its slots.
    marker=' uint32_t total=(reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+b)->slots;'
    text=replace_once(text, marker, ' uint32_t total=postScratch.GetValue(bitmapWords+13);')
    # Probe inside the existing pack VF. Only an actual capacity boundary
    # returns to the scalar path for exact counting/extension and retries.
    start=text.index(' // All final prefixes')
    end=text.index(' // Consume the original compact UB result', start)
    block=text[start:end]
    decl_start=block.index(' auto activeLru=')
    work_start=block.index(' Simt::VF_CALL<PrepareTerminalCapacity>')
    declarations=block[decl_start:work_start]
    slow=block[work_start:]
    text=text[:start]+declarations+' for(uint32_t capacityPass=0;capacityPass<2;++capacityPass){\n if(capacityPass){\n'+slow+' }\n'+text[end:]
    text=replace_once(text, 'activeLru,slotCapacity,capacityCounted,',
                     'activeLru,slotCapacity,capacityCounted,capacityPass==0,headers,priorSlots,')
    text=replace_once(text, ' if(profile)ts[11]=GetSystemCycle();',
                     ' if(!postScratch.GetValue(bitmapWords+14))break;\n }\n if(profile)ts[11]=GetSystemCycle();')
    path.write_text(text)
    helper=destination/'helpers.h'
    text=replace_once(helper.read_text(), ' if(t<count){\n  uint32_t i=t;',
                      ' for(uint32_t i=t;i<count;i+=THREADS){')
    text=text.replace('m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;',
                      'm[TAIL_PLAN_SRC+begin+i]=src;m[TAIL_PLAN_DST+begin+i]=dst;')
    # The imported general planner uses SET=1024, although this terminal's
    # two tables already reserve 2048 words each. At 4000 changes the old
    # per-worker table was 97.7% full. Keep load below two thirds.
    text=replace_once(text, ' uint32_t t=threadIdx.x;\n if constexpr(!Restore){',
        ' uint32_t t=threadIdx.x,setSize=count<=512?SET:2048;\n if constexpr(!Restore){')
    text=text.replace('i<SET;i+=THREADS', 'i<setSize;i+=THREADS')
    text=text.replace('insert(m+TAIL_SRCSET,src)', '(count<=512?insert(m+TAIL_SRCSET,src):InsertTerminalRecord(m+TAIL_SRCSET,src,setSize))')
    text=text.replace('insert(m+TAIL_DSTSET,dst)', '(count<=512?insert(m+TAIL_DSTSET,dst):InsertTerminalRecord(m+TAIL_DSTSET,dst,setSize))')
    text = """__simt_callee__ inline bool InsertTerminalRecord(__ubuf__ uint32_t* table,uint32_t key,uint32_t size){
 uint32_t h=((key*0x9e3779b9U)>>21)&(size-1);
 for(uint32_t i=0;i<size;++i){uint32_t old=asc_atomic_cas(table+h,EMPTY,key);
  if(old==EMPTY)return true;if(old==key)return false;h=(h+1)&(size-1);}
 return false;
}
""" + text
    helper.write_text(text)
    post=destination/'parallel_post.h'
    text=replace_once(post.read_text(), '__gm__ uint32_t* lru,__gm__ uint32_t* changes,__gm__ uint32_t* changeSlots,uint64_t',
                      '__gm__ uint32_t* lru,uint32_t slotCapacity,bool preclaimed,__gm__ uint32_t* changes,__gm__ uint32_t* changeSlots,uint64_t')
    text=replace_once(text, 'slotBase>rotating::kLruSlots || count>rotating::kLruSlots-slotBase',
                      'slotBase>slotCapacity || count>slotCapacity-slotBase')
    text=replace_once(text, 'for(uint32_t i=tid;i<counts[q];i+=96)asc_atomic_min(mapping+input[i],claim);',
                      'if(!preclaimed)for(uint32_t i=tid;i<counts[q];i+=96)asc_atomic_min(mapping+input[i],claim);')
    text=replace_once(text, '  if(counts[q]>64 && counts[q]<=96){',
                      '  if(preclaimed){count=counts[q];for(uint32_t i=lane;i<count;i+=32)ids[i]=input[i];}\n  else if(counts[q]>64 && counts[q]<=96){')
    # Exact counting has already joined all four claim publishers. All Qs
    # take the same preclaimed branch, so flags20 and its producer-warp join
    # are redundant here. Keep descriptor publication (flags24) unchanged.
    text=replace_once(text, ' if(tid<96){\n  if(!preclaimed)',
                      ' if(tid<96 && !preclaimed){\n  if(!preclaimed)')
    text=replace_once(text, '  uint32_t claimed;do{',
                      '  if(!preclaimed){\n  uint32_t claimed;do{')
    text=replace_once(text, '  uint32_t count=0;\n  if(preclaimed)',
                      '  }\n  uint32_t count=0;\n  if(preclaimed)')
    text=replace_once(text, 'uint32_t slotCapacity,bool preclaimed,__gm__ uint32_t* changes',
                     'uint32_t slotCapacity,bool preclaimed,bool capacityProbe,__gm__ uint32_t* headers,uint32_t priorSlots,__gm__ uint32_t* changes')
    marker=' if(tid>=2 && tid<9 && tid!=5)counter[tid]=0; // The original block barrier publishes all control words.'
    text=replace_once(text, marker, marker+"""
 if(tid<32){uint32_t upper=0;
  if(capacityProbe){uint32_t ready;
   do{ready=asc_ballot(lane>=4 || asc_ldcg(flags+lane*32)>=generation);}while(ready!=UINT32_MAX);
   upper=lane<4?asc_ldcg(headers+lane*32+6):0;
   upper+=asc_shfl_down(upper,2);upper+=asc_shfl_down(upper,1);
   upper+=priorSlots;
  }
  if(!lane)counter[14]=capacityProbe && upper>2048;
 }
""")
    text=replace_once(text, ' asc_syncthreads();\n // High-bit owner tags',
                     ' asc_syncthreads();\n if(counter[14])return;\n // High-bit owner tags')
    # Reuse the pack task to publish the final allocation count. Only warp
    # zero joins all descriptor producers; no extra VF/scalar round trip.
    marker=' Emit(fullIds,outputSlots,error,mapping,lru,logical,history,resident,history-512,256,512,nullptr,nullptr,flags,generation,q+1);'
    text=replace_once(text, marker, marker+"""
 if(tid<32){uint32_t ready;
  do{ready=asc_ballot(lane>=4 || asc_ldcg(flags+24*32+lane)>=generation);}while(ready!=UINT32_MAX);
  if(!lane)counter[13]=asc_ldcg(&row->slots);
 }
""")
    post.write_text(text)
    finish=destination/'terminal_finish.h'
    text=replace_once(finish.read_text(), '    if(t<32){',
                      '    if(!t)control[4]=total>2048?asc_ldcg(flags+9*32):0U;\n    if(t<32){')
    text=replace_once(text, '    uint32_t error=control[0];', '''
    uint32_t extendedCount=control[4];
    if(extendedCount>2048){
        auto extended=reinterpret_cast<__gm__ uint32_t*>(runtime->reserved[0])+uint64_t(b)*(rotating::kChangeCapacity+4*((resident+31)/32));
        auto inverse=reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident;
        for(uint32_t i=2048+q*512+t;i<extendedCount;i+=2048)asc_stcg(inverse+asc_ldcg(extended+i),int32_t(-1));
    }
    // Snapshot the count on all workers before resetting its shared word.
    // The monotonic per-Q generation is published only after this retirement.
    if(!q && !t){asc_stcg(flags+8*32,0U);asc_stcg(flags+9*32,0U);asc_stcg(flags+10*32,0U);}
    uint32_t error=control[0];''')
    # The count was loaded before the existing cross-core completion join.
    finish.write_text(text)


def bind_model_capacity(destination):
    path=destination/'driver.cpp'
    text=replace_once(path.read_text(), ' binding.version=1;',
                      ' binding.reserved[0]=put(std::vector<uint32_t>(10240+4*((R+31)/32),UINT32_MAX));\n binding.version=1;')
    path.write_text(text)
