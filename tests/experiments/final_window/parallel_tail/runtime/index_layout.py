"""Distribute validated Index swaps across disjoint worker groups."""
from private_topk import replace_once


def adapt_index_split(destination):
    """Apply the same disjoint worker layout to model or native generated code."""
    p=destination/'helpers.h';s=p.read_text()
    s=replace_once(s,'__ubuf__ uint32_t* m){\n uint32_t t=threadIdx.x;\n if constexpr(!Restore){',
                   '__ubuf__ uint32_t* m,__gm__ uint64_t* addresses,__gm__ uint64_t* logical){\n uint32_t t=threadIdx.x;\n if constexpr(!Restore){')
    start=s.index(' for(uint32_t i=t;i<count*16;');end=s.index('\n}\n__simt_callee__ inline void SwapColdMetadata',start)
    s=s[:start]+''' // The validated plan is immutable; keys and metadata occupy disjoint arrays.
 if(t<448){
  for(uint32_t i=t;i<count*16;i+=448){
   uint32_t src=m[TAIL_PLAN_SRC+begin+i/16],dst=m[TAIL_PLAN_DST+begin+i/16],j=i%16;
   auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);
   asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);
  }
 }else{
  for(uint32_t i=t-448;i<count;i+=64){
   uint32_t src=m[TAIL_PLAN_SRC+begin+i],dst=m[TAIL_PLAN_DST+begin+i];
   auto sx=asc_ldcg(scales+src),sy=asc_ldcg(scales+dst);
   auto ax=asc_ldcg(addresses+src),ay=asc_ldcg(addresses+dst);
   auto lx=asc_ldcg(logical+src),ly=asc_ldcg(logical+dst);
   asc_stcg(scales+src,sy);asc_stcg(scales+dst,sx);
   asc_stcg(addresses+src,ay);asc_stcg(addresses+dst,ax);
   asc_stcg(logical+src,ly);asc_stcg(logical+dst,lx);
  }
 }
 asc_threadfence();'''+s[end:]
    p.write_text(s)
    p=destination/'kernel.cpp';s=p.read_text()
    s=replace_once(s,'SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work);',
                   'SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work,addresses,logical);')
    s=replace_once(s,' if(count && !work[TAIL_CONTROL])SwapColdMetadata(addresses,logical,count,work);',' // Metadata swaps run alongside key swaps in the same validated task.')
    p.write_text(s)



def restore_index_baseline(destination):
    """Restore the qualified A layout for capacity AB generation."""
    p = destination / "helpers.h"
    p.write_text(replace_once(p.read_text(), 'template<bool Restore=false>\n__simt_callee__ inline void SwapColdIndexRows(\n __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t begin,uint32_t count,\n uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,\n __ubuf__ uint32_t* m,__gm__ uint64_t* addresses,__gm__ uint64_t* logical){\n uint32_t t=threadIdx.x;\n if constexpr(!Restore){\n if(!t)m[TAIL_CONTROL]=0;\n if(!begin)for(uint32_t i=t;i<SET;i+=THREADS){m[TAIL_SRCSET+i]=EMPTY;m[TAIL_DSTSET+i]=EMPTY;}\n asc_syncthreads();\n if(t<count){\n  uint32_t i=t;\n  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(dest+i);\n  m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;\n  if(src<resident || src>=history || dst>=resident ||\n     !insert(m+TAIL_SRCSET,src) || !insert(m+TAIL_DSTSET,dst))asc_atomic_or(m+TAIL_CONTROL,1U);\n }\n asc_syncthreads();\n if(m[TAIL_CONTROL])return;\n }\n // The validated plan is immutable; keys and metadata occupy disjoint arrays.\n if(t<448){\n  for(uint32_t i=t;i<count*16;i+=448){\n   uint32_t src=m[TAIL_PLAN_SRC+begin+i/16],dst=m[TAIL_PLAN_DST+begin+i/16],j=i%16;\n   auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);\n   asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);\n  }\n }else{\n  for(uint32_t i=t-448;i<count;i+=64){\n   uint32_t src=m[TAIL_PLAN_SRC+begin+i],dst=m[TAIL_PLAN_DST+begin+i];\n   auto sx=asc_ldcg(scales+src),sy=asc_ldcg(scales+dst);\n   auto ax=asc_ldcg(addresses+src),ay=asc_ldcg(addresses+dst);\n   auto lx=asc_ldcg(logical+src),ly=asc_ldcg(logical+dst);\n   asc_stcg(scales+src,sy);asc_stcg(scales+dst,sx);\n   asc_stcg(addresses+src,ay);asc_stcg(addresses+dst,ax);\n   asc_stcg(logical+src,ly);asc_stcg(logical+dst,lx);\n  }\n }\n asc_threadfence();\n}\n__simt_callee__ inline void SwapColdMetadata(\n __gm__ uint64_t* addr,__gm__ uint64_t* logical,uint32_t n,__ubuf__ uint32_t* m){\n uint32_t t=threadIdx.x;\n if(t<n){\n  uint32_t src=m[TAIL_PLAN_SRC+t],dst=m[TAIL_PLAN_DST+t];\n  auto x=asc_ldcg(addr+src),y=asc_ldcg(addr+dst);\n  auto a=asc_ldcg(logical+src),b=asc_ldcg(logical+dst);\n  asc_stcg(addr+src,y);asc_stcg(addr+dst,x);asc_stcg(logical+src,b);asc_stcg(logical+dst,a);\n }\n asc_threadfence();\n}\n', 'template<bool Restore=false>\n__simt_callee__ inline void SwapColdIndexRows(\n __gm__ uint32_t* sources,__gm__ uint32_t* dest,uint32_t begin,uint32_t count,\n uint32_t resident,uint32_t history,__gm__ uint64_t* keys,__gm__ uint32_t* scales,\n __ubuf__ uint32_t* m){\n uint32_t t=threadIdx.x;\n if constexpr(!Restore){\n if(!t)m[TAIL_CONTROL]=0;\n if(!begin)for(uint32_t i=t;i<SET;i+=THREADS){m[TAIL_SRCSET+i]=EMPTY;m[TAIL_DSTSET+i]=EMPTY;}\n asc_syncthreads();\n if(t<count){\n  uint32_t i=t;\n  uint32_t src=asc_ldcg(sources+i),dst=asc_ldcg(dest+i);\n  m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;\n  if(src<resident || src>=history || dst>=resident ||\n     !insert(m+TAIL_SRCSET,src) || !insert(m+TAIL_DSTSET,dst))asc_atomic_or(m+TAIL_CONTROL,1U);\n }\n asc_syncthreads();\n if(m[TAIL_CONTROL])return;\n }\n for(uint32_t i=t;i<count*16;i+=THREADS){\n  uint32_t src=m[TAIL_PLAN_SRC+begin+i/16],dst=m[TAIL_PLAN_DST+begin+i/16],j=i%16;\n  auto x=asc_ldcg(keys+16ULL*src+j),y=asc_ldcg(keys+16ULL*dst+j);\n  asc_stcg(keys+16ULL*src+j,y);asc_stcg(keys+16ULL*dst+j,x);\n }\n if(t<count){\n  uint32_t src=m[TAIL_PLAN_SRC+begin+t],dst=m[TAIL_PLAN_DST+begin+t];\n  auto x=asc_ldcg(scales+src),y=asc_ldcg(scales+dst);\n  asc_stcg(scales+src,y);asc_stcg(scales+dst,x);\n }\n // The following metadata phase fences both sets of stores.\n}\n__simt_callee__ inline void SwapColdMetadata(\n __gm__ uint64_t* addr,__gm__ uint64_t* logical,uint32_t n,__ubuf__ uint32_t* m){\n uint32_t t=threadIdx.x;\n if(t<n){\n  uint32_t src=m[TAIL_PLAN_SRC+t],dst=m[TAIL_PLAN_DST+t];\n  auto x=asc_ldcg(addr+src),y=asc_ldcg(addr+dst);\n  auto a=asc_ldcg(logical+src),b=asc_ldcg(logical+dst);\n  asc_stcg(addr+src,y);asc_stcg(addr+dst,x);asc_stcg(logical+src,b);asc_stcg(logical+dst,a);\n }\n asc_threadfence();\n}\n'))
    p = destination / "kernel.cpp"
    s = replace_once(p.read_text(), "SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work,addresses,logical);", "SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work);")
    s = replace_once(s, " // Metadata swaps run alongside key swaps in the same validated task.", " if(count && !work[TAIL_CONTROL])SwapColdMetadata(addresses,logical,count,work);")
    p.write_text(s)
