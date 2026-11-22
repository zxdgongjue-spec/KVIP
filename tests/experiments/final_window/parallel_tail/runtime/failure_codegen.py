"""Add failure-only records without an extra rendezvous or success-path scan."""
from private_topk import replace_once


def add_failure_records(destination):
    path = destination / 'kernel.cpp'
    s = path.read_text()
    s = replace_once(s, 'uint32_t priorJournalCount,uint32_t b){',
                     'uint32_t priorJournalCount,uint32_t b,__gm__ uint32_t* diagnostic){')
    marker = ' // The immutable plan is already visible; key and metadata stores are disjoint.'
    s = replace_once(s, marker, ''' // Validation has already synchronized all workers. Preserve evidence before retirement.
 if(count && work[TAIL_CONTROL] && !threadIdx.x){
   uint32_t bad=UINT32_MAX,reason=0,source=UINT32_MAX,target=UINT32_MAX,previous=UINT32_MAX;
   for(uint32_t i=0;i<count && bad==UINT32_MAX;++i){
     uint32_t src=work[TAIL_PLAN_SRC+i],dst=work[TAIL_PLAN_DST+i],why=0;
     if(src<resident)why|=1U;if(src>=history)why|=2U;if(dst>=resident)why|=4U;
     for(uint32_t j=0;j<i;++j){
       if(src==work[TAIL_PLAN_SRC+j]){why|=8U;previous=first+j;}
       if(dst==work[TAIL_PLAN_DST+j]){why|=16U;previous=first+j;}
     }
     if(why){bad=first+i;reason=why;source=src;target=dst;}
   }
   asc_stcg(diagnostic+12,bad);asc_stcg(diagnostic+13,reason);
   asc_stcg(diagnostic+14,source);asc_stcg(diagnostic+15,target);
   asc_stcg(diagnostic+16,previous);
 }
''' + marker)
    s = replace_once(s, 'query*32,priorJournalCount,b);', 'query*32,priorJournalCount,b,flags+(28+q)*32);')
    s = replace_once(s, 'reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]=topkError;',
                     'uint32_t ownError=topkError;')
    s = replace_once(s, 'reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]|=postScratch.GetValue(bitmapWords+8);',
                     'uint32_t packError=postScratch.GetValue(bitmapWords+8);ownError|=packError;')
    s = replace_once(s, 'reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]|=(indexError<<4);',
                     'ownError|=(indexError<<4);')
    s = replace_once(s, 'reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32]|=uint32_t(cqError);',
                     'ownError|=uint32_t(cqError);')
    s = replace_once(s, ''' if(q==1)cqError|=(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b)->error;
 uint32_t ownError=reinterpret_cast<__gm__ uint32_t*>(a->queryErrors)[query*32];''',
                     ''' uint32_t prefetchError=q==1?(reinterpret_cast<__gm__ rotating::Row*>(ctx->rows)+b)->error:0;
 cqError|=prefetchError;
 if(ownError || cqError){
   auto d=flags+(28+q)*32;
   d[1]=topkError;d[2]=packError;d[3]=indexError;d[4]=post->error;
   d[5]=prefetchError;d[6]=total;d[7]=first;d[8]=n;d[9]=own;
   d[10]=post->pending;d[11]=generation;
   if(!indexError || !n){for(uint32_t i=12;i<=16;++i)d[i]=UINT32_MAX;}
 }''')
    s = replace_once(s, 'priorJournalCount,b,flags+(28+q)*32);', 'priorJournalCount,b,flags+(12+q)*32);')
    path.write_text(s)
