// All publishers, Fetch acknowledgements and the Index update have joined.
// Validate every row first so the original error path can still consume it.
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void FinalizeJoinedRow(
 __gm__ Args* a,__gm__ Context* ctx,__gm__ BatchFlag* flags,uint32_t b,uint32_t batch,
 uint32_t resident,uint32_t postOffset,uint32_t workers,uint32_t prefix,__ubuf__ uint32_t* scratch){
 uint32_t t=threadIdx.x,q=t/64,l=t%64;
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 auto post=reinterpret_cast<__gm__ FinalPost*>(ctx->owners+postOffset)+b*4+q;
 auto info=reinterpret_cast<__gm__ uint32_t*>(ctx->rowStatus)+b*kRowStatusWords;
 if(!l){scratch[q]=asc_ldcg(&post->count);scratch[4+q]=asc_ldcg(&post->pending);
  scratch[8+q]=asc_ldcg(&post->error)|asc_ldcg(&post->deferred);}
 asc_syncthreads();
 if(!t){
  uint32_t total=0,completed=0,err=0;
  for(uint32_t i=0;i<4;i++){total+=scratch[i];completed+=scratch[4+i];err|=scratch[8+i];}
  auto update=reinterpret_cast<__gm__ uint64_t*>(a->updateBindings);
  err|=asc_ldcg(reinterpret_cast<__gm__ uint32_t*>(asc_ldcg(update+6))+b);
  err|=asc_ldcg(info)|asc_ldcg(&row->error)|asc_ldcg(&row->pending);
  scratch[12]=uint32_t(err==0 && asc_ldcg(info+1)==total);scratch[13]=total;scratch[14]=completed;
 }
 asc_syncthreads();
 if(!scratch[12])return;
 auto valid=reinterpret_cast<__gm__ uint32_t*>(ctx->hotValid)+OwnerRow(ctx,b)*resident;
 if(prefix && ctx->changes && ctx->lruSlots){
  auto changes=reinterpret_cast<__gm__ int32_t*>(ctx->changes)+b*kChangeCapacity;
  auto lru=reinterpret_cast<__gm__ uint32_t*>(ctx->lruSlots)+b*kLruSlots;
  for(uint32_t i=t;i<prefix;i+=256)
   if(asc_ldcg(changes+i)>=0)asc_stcg(valid+asc_ldcg(lru+i),1U);
 }
 if(scratch[q] && asc_ldcg(&flags[8+q].reserved[0])){
  auto dst=reinterpret_cast<__gm__ int32_t*>(ctx->destinations)+asc_ldcg(&post->descriptor);
  uint32_t base=OwnerRow(ctx,b)*OutputStride(resident,ctx);
  for(uint32_t i=l;i<scratch[q];i+=64){
   uint32_t slot=uint32_t(asc_ldcg(dst+i))-base;
   if(slot<resident)asc_stcg(valid+slot,1U);
  }
 }
 if(!l)asc_stcg(&post->pending,0U);
 if(t<16){
  auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(ctx->finalStatus)+2*b+t/8;
  uint64_t value=0;
  switch(t%8){
   case 0:value=KVAF_AIV_BATCH_GET_STATUS_MAGIC;break;
   case 1:if(t<8)value=scratch[13];break;
   case 3:if(t<8)value=scratch[14];break;
   case 4:case 5:if(t<8){
    uint32_t pair=2*batch+2*(3*workers+b%workers);
    auto args=reinterpret_cast<__gm__ KvafExecuteBatchGetKernelArgs*>(ctx->args)+pair;
    auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(asc_ldcg(&args->queue_state));
    value=t==4?asc_ldcg(&queue->sq_producer_index):asc_ldcg(&queue->cq_consumer_index);
   }break;
   case 6:value=UINT64_MAX;break;
  }
  asc_stcg(reinterpret_cast<__gm__ uint64_t*>(status)+t%8,value);
 }
 if(!t){asc_stcg(info+3,0U);asc_stcg(info+4,scratch[14]);asc_stcg(&row->finalPending,0U);}
 asc_threadfence();
}
