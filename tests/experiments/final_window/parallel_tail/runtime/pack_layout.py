"""Apply the validated 512-thread packing/output layout to native sources."""
from private_topk import replace_once


def adapt_pack_layout(destination):
    p=destination/'kernel.cpp';s=p.read_text()
    s=replace_once(s,'VF_CALL<PackUniqueLocal>(Simt::Dim3{1024,1,1}',
                   'VF_CALL<PackUniqueLocal>(Simt::Dim3{512,1,1}')
    p.write_text(s)
    p=destination/'parallel_post.h';s=p.read_text()
    s=replace_once(s,'LAUNCH_BOUND(1024) inline void PackUniqueLocal(',
                   'LAUNCH_BOUND(512) inline void PackUniqueLocal(')
    # Keep the queue/route warp out of the output loop: it was the last
    # arrival at the existing barrier on all four Qs in the native trace.
    s=replace_once(s,'for(uint32_t i=tid-96;i<2048;i+=928)',
                   'if(tid>=128)for(uint32_t i=tid-128;i<2048;i+=384)')
    p.write_text(s)
    p=destination/'emit.h';s=p.read_text()
    begin=s.index(' uint32_t i0=threadIdx.x');end=s.index(' asc_threadfence();',begin)
    s=s[:begin]+''' bool use[4];uint32_t token[4],logicalId[4];
 #pragma unroll
 for(uint32_t part=0;part<4;++part){
  uint32_t i=threadIdx.x+part*512;
  use[part]=loc[i]==UINT32_MAX && ids[i]!=UINT32_MAX;
  token[part]=ids[i]&0x3fffffffU;
  logicalId[part]=use[part]?asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(logical)+2*token[part]):0;
 }
 if(threadIdx.x<32){uint32_t mask;do{bool ready=threadIdx.x>=producers || asc_ldcg(flags+24*32+threadIdx.x)>=generation;mask=asc_ballot(ready);}while(mask!=UINT32_MAX);}
 asc_syncthreads();
 #pragma unroll
 for(uint32_t part=0;part<4;++part){
  uint32_t i=threadIdx.x+part*512;
  uint32_t m=use[part]?asc_ldcg(mapping+token[part]):UINT32_MAX;
  uint32_t slot=m<resident?m:UINT32_MAX;
  if(use[part] && m-resident<2048)slot=asc_ldcg(lru+m-resident);
  if(use[part]){if(slot==UINT32_MAX)asc_atomic_or(error,2U);loc[i]=slot;ids[i]=logicalId[part];}
 }
'''+s[end:]
    p.write_text(s)
