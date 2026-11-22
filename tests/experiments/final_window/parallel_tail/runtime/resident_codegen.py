"""Keep disjoint cold updates; reuse the original permutation planner for resident sources."""
from private_topk import replace_once


def _prepare_resident_updates(destination):
    p=destination/'kernel.cpp';s=p.read_text()
    start=s.index(' Simt::VF_CALL<WaitFour>(Simt::Dim3{32,1,1},flags,5,')
    end=s.index(' if(profile)ts[8]',start)
    s=s[:start]+''' auto allSources=reinterpret_cast<__gm__ uint32_t*>(ctx->changes)+uint64_t(b)*rotating::kChangeCapacity;
 auto liveRow=reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b;
 Simt::VF_CALL<JoinIndexReaders>(Simt::Dim3{32,1,1},flags,generation,allSources,
   s.resident,q,work,
   ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*s.resident:nullptr,liveRow);
 SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 bool residentUpdate=local.GetValue(0)!=0;
'''+s[end:]
    start=s.index(' Simt::VF_CALL<UpdateIndex>');end=s.index(' ownError|=(indexError<<4);',start)
    old=s[start:end]
    old=old.replace(' uint32_t indexError=', ' indexError=')
    generic=''' uint32_t indexError=0;
 if(!residentUpdate){
'''+old+'''
 }else{
  // No old TopK buffer is read after this point. Reuse its UB allocation for
  // the production planner, which snapshots aliases before writing any row.
  if(!q){
   pipe.Reset();
   TBuf<TPosition::VECCALC> indexScratch;
   pipe.InitBuffer(indexScratch,SCRATCH_WORDS*sizeof(uint32_t));
   auto indexLocal=indexScratch.Get<uint32_t>();auto indexWork=reinterpret_cast<__ubuf__ uint32_t*>(indexLocal.GetPhyAddr());
   auto keys=reinterpret_cast<__gm__ uint64_t*>(a->keys)+uint64_t(owner)*s.history*16;
   auto scales=reinterpret_cast<__gm__ uint32_t*>(a->scales)+uint64_t(owner)*s.history;
   auto addresses=reinterpret_cast<__gm__ uint64_t*>(a->addresses)+uint64_t(owner)*addressStride;
   auto logical=reinterpret_cast<__gm__ uint64_t*>(a->logical)+uint64_t(owner)*s.history;
   auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(owner)*s.resident;
   Simt::VF_CALL<InitShadow>(Simt::Dim3{512,1,1},indexWork);
   SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
   if(total<=rotating::kLruSlots){
    Simt::VF_CALL<ApplyBatch<false>>(Simt::Dim3{512,1,1},allSources,dst-first,0,total,keys,scales,valid,s.resident,indexWork,nullptr,nullptr,nullptr,s.history);
    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    indexError=indexLocal.GetValue(CONTROL);
    if(!indexError){
     Simt::VF_CALL<CommitMetadata>(Simt::Dim3{512,1,1},addresses,logical,indexWork);
     SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
   }else indexError=1;
   flags[7*32+1]=indexError;Signal(flags,7,generation);
  }else{Await(flags,7,generation);indexError=flags[7*32+1];}
  // Retirement is shared with the normal path; skip the already applied swap.
  Simt::VF_CALL<UpdateIndex>(Simt::Dim3{512,1,1},src,dst,n,s.resident,s.history,
   reinterpret_cast<__gm__ uint64_t*>(a->keys),reinterpret_cast<__gm__ uint32_t*>(a->scales),
   reinterpret_cast<__gm__ uint64_t*>(a->addresses),reinterpret_cast<__gm__ uint64_t*>(a->logical),
   work,ctx,total,q,first,reinterpret_cast<__gm__ uint32_t*>(rotating::TopKHeader(ctx))+query*32,
   reinterpret_cast<__gm__ uint32_t*>(ctx->progress)+query*32,priorJournalCount,b,flags+(12+q)*32);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 }
'''
    s=s[:start]+generic+s[end:]
    s=replace_once(s, 'uint32_t b,__gm__ uint32_t* diagnostic){',
                   'uint32_t b,__gm__ uint32_t* diagnostic,bool applied=false){')
    s=replace_once(s, ' if(count)SwapColdIndexRows<false>', ' if(count && !applied)SwapColdIndexRows<false>')
    s=replace_once(s, ' if(count && work[TAIL_CONTROL] && !threadIdx.x)',
                   ' if(count && !applied && work[TAIL_CONTROL] && !threadIdx.x)')
    s=replace_once(s, 'priorJournalCount,b,flags+(12+q)*32);\n  SetFlag',
                   'priorJournalCount,b,flags+(12+q)*32,true);\n  SetFlag')
    s=replace_once(s, 'priorJournalCount,b,flags+(12+q)*32);', 'priorJournalCount,b,flags+(12+q)*32,false);')
    start=s.index(' kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows)+uint64_t(b)*sizeof(rotating::Row),128);')
    end=s.index(' // Include the actual prior-prefetch',start)
    s=s[:start]+' uint32_t total=local.GetValue(1);\n'+s[end:]
    # An Index swap does not move the displaced KV back into the source hot
    # slot. Invalidate that source before publishing destination residency.
    s=replace_once(s, ' RetireShared(ctx,first,count,history,b);',
                   """ if(!applied && !work[TAIL_CONTROL]){
  for(uint32_t i=threadIdx.x;i<count;i+=THREADS){uint32_t src=work[TAIL_PLAN_SRC+i];if(src<resident){auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+uint64_t(rotating::OwnerRow(ctx,b))*resident;asc_stcg(valid+src,0U);}}
 }
 RetireShared(ctx,first,count,history,b);""")
    s=s.replace('if(src<resident)why|=1U;', '')
    p.write_text(s)
    finish=destination/'terminal_finish.h'
    finish.write_text(replace_once(finish.read_text(), '    uint32_t error=control[0];',
        '    if(!q && !t)asc_stcg(flags+5*32,0U);\n    uint32_t error=control[0];'))
    helper=destination/'helpers.h'
    h=replace_once(helper.read_text(), 'if(src<resident || src>=history || dst>=resident ||',
                   'if(src>=history || dst>=resident ||')
    helper.write_text(h)



def adapt_resident_update(destination):
    _prepare_resident_updates(destination)
    p = destination / 'kernel.cpp'
    s = p.read_text()
    start = s.index(' auto liveRow=')
    end = s.index(' if(profile)ts[8]', start)
    s = s[:start] + s[end:]
    s = replace_once(s, ' uint32_t total=local.GetValue(1);',
        ' kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(ctx->rows)+uint64_t(b)*sizeof(rotating::Row),128);\n uint32_t total=(reinterpret_cast<__gm__ volatile rotating::Row*>(ctx->rows)+b)->slots;')
    s = replace_once(s, ' if(!residentUpdate){\n', '')
    s = replace_once(s, '\n }else{\n  // No old TopK',
        '\n bool residentUpdate=(indexError&0x80000000U)!=0;\n if(residentUpdate){\n  // No old TopK')
    s = replace_once(s,
        ' if(count && !applied)SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work,addresses,logical);',
        ''' if(!applied)SwapColdIndexRows<false>(sources,dest,0,count,resident,history,keys,scales,work,addresses,logical,
  ctx->reservedHot?reinterpret_cast<__gm__ int32_t*>(ctx->reservedHot)+uint64_t(b)*resident:nullptr,
  total,diagnostic-(12+q)*32);
 if(!applied && (work[TAIL_CONTROL]&0x80000000U))return;''')
    # Even an empty partition participates in validation and the reader join.
    s = replace_once(s, '(n?local.GetValue(TAIL_CONTROL):0)', 'local.GetValue(TAIL_CONTROL)')
    s=replace_once(s, ' ts[20]=own;ts[21]=all;ts[22]=offset;',
                   ' if(profile){ts[20]=own;ts[21]=all;ts[22]=offset;}')
    s=replace_once(s, ' if(!q)ts[30]=', ' if(profile && !q)ts[30]=')
    p.write_text(s)
    h = destination / 'helpers.h'
    s = h.read_text()
    s = replace_once(s, '__ubuf__ uint32_t* m,__gm__ uint64_t* addresses,__gm__ uint64_t* logical){',
        '__ubuf__ uint32_t* m,__gm__ uint64_t* addresses,__gm__ uint64_t* logical,\n __gm__ int32_t* inverse,uint32_t total,__gm__ uint32_t* flags){')
    s = replace_once(s, ' if(!t)m[TAIL_CONTROL]=0;', ' if(!t){m[TAIL_CONTROL]=0;m[TAIL_CONTROL+1]=0;}')
    s = replace_once(s, '  m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;',
        '''  m[TAIL_PLAN_SRC+begin+t]=src;m[TAIL_PLAN_DST+begin+t]=dst;
  if(src<resident && (!inverse || uint32_t(asc_ldcg(inverse+src))<total))asc_atomic_or(m+TAIL_CONTROL+1,1U);''')
    s = replace_once(s, ' if(m[TAIL_CONTROL])return;', ''' // This replaces the old output-reader join. Do not write any Index row
 // before every Q has finished its output reads and validated its partition.
 if(!t){asc_threadfence();asc_atomic_add(flags+5*32,m[TAIL_CONTROL+1]?257U:1U);asc_threadfence();}
 if(t<32){
  uint32_t value=0,ready;
  do{if(!t)value=asc_ldcg(flags+5*32);ready=asc_ballot(t!=0 || (value&255U)==4U);}while(ready!=UINT32_MAX);
  if(!t && (value>>8))m[TAIL_CONTROL]|=0x80000000U;
 }
 asc_syncthreads();
 if(m[TAIL_CONTROL])return;''')
    h.write_text(s)
