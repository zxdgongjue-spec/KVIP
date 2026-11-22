#pragma once
// Simulator fixture only. Native prepare_native.py uses transport.h instead.
__aicore__ inline void CommitTerminalPost(
 const __gm__ KvafExecuteBatchGetKernelArgs* args,uint64_t head,uint32_t count,__gm__ FinalPost* post){
 post->count=count;post->pending=count?1:0;post->error=0;post->head=head;
 SetFlag<HardEvent::MTE3_S>(EVENT_ID0);WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
 auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
 queue->sq_producer_index=head+count;
 auto entity=reinterpret_cast<__gm__ uint64_t*>(args->channel_entity);
 *reinterpret_cast<__gm__ uint64_t*>(entity[8])=head+count;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(entity[8]),64);
 reinterpret_cast<__gm__ uint64_t*>(post)[4]=GetSystemCycle();
}
__aicore__ inline uint64_t CompleteTerminalPost(
 const __gm__ KvafExecuteBatchGetKernelArgs* args,__gm__ FinalPost* post){
 if(post->pending){
  uint64_t begin=reinterpret_cast<__gm__ uint64_t*>(post)[4];
  while(GetSystemCycle()-begin<9000){}
  auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
  ++queue->cq_consumer_index;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),64);
 }
 post->pending=0;post->error=0;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);
 return 0;
}
__aicore__ inline void CompleteTerminalPrefetch(__gm__ Context* ctx,uint32_t b){
 auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 row->pending=0;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));
}
