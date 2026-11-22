#pragma once
// Read every logical ID before Index metadata starts changing. Each Q writes
// its own final output row; this makes later metadata commits independent.
namespace rotating {
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void TerminalLogicalSnapshot(
 __gm__ uint32_t* ids,__gm__ uint64_t* logical,__gm__ uint32_t* output,
 __gm__ uint32_t* error,uint32_t history){
 for(uint32_t i=threadIdx.x;i<2048;i+=512){
  uint32_t tagged=asc_ldcg(ids+i),token=tagged&0x3fffffffU;
  if(tagged==UINT32_MAX){asc_stcg(output+i,UINT32_MAX);continue;}
  if(token>=history){asc_atomic_or(error,1U);asc_stcg(output+i,UINT32_MAX);continue;}
  asc_stcg(output+i,uint32_t(asc_ldcg(logical+token)));
 }
 asc_threadfence();
}
}
