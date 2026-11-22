"""Include native status publication; retain only a software CQ arrival model."""
import argparse
from pathlib import Path
from prepare_native_model import prepare_model
from private_topk import replace_once


def prepare_status_model(destination):
    prepare_model(destination)
    transport=destination/'terminal_transport.h'
    text=transport.read_text()
    # DrainQueue updates both counters; CompleteTerminalPost performs the one
    # queue cache publication afterwards. Do not duplicate it in the CQ model.
    text=replace_once(text, """  ++queue->cq_consumer_index;
  kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),64);""",
                     """  ++queue->cq_consumer_index;
  ++queue->completed_batch_count;""")
    text=replace_once(text, ''' post->pending=0;post->error=0;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);''', ''' uint32_t completed=post->pending?1U:0U;
 post->pending=0;post->error=0;
 auto queue=reinterpret_cast<__gm__ KvafBatchGetQueueState*>(args->queue_state);
 auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args->core_status);
 kvaf_device::WriteStatus(status,0,post->count,completed,
     queue->sq_producer_index,queue->cq_consumer_index,UINT64_MAX,0);
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(queue),sizeof(*queue));
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(post),128);''')
    text=replace_once(text, ''' auto row=reinterpret_cast<__gm__ Row*>(ctx->rows)+b;
 row->pending=0;
 kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(row),sizeof(Row));''',
                     ''' // V9_SIMULATION suppresses the NIC CQ drain only. Keep the original
 // queue/status publication and precise cache-line operations.
 CompletePrefetch(ctx,b,false);''')
    transport.write_text(text)
    driver=destination/'driver.cpp';text=driver.read_text()
    injection='''
 auto terminalStatuses=put(std::vector<KvafBatchGetCoreStatus>(4));
 auto modelFinalArgs=perQArgs;
 for(uint32_t q=0;q<4;++q)modelFinalArgs[q].core_status=terminalStatuses+q*sizeof(KvafBatchGetCoreStatus);
 auto terminalQueueArgs=put(modelFinalArgs);
 auto prefetchQueues=put(std::vector<KvafBatchGetQueueState>(2));
 auto prefetchStatuses=put(std::vector<KvafBatchGetCoreStatus>(2));
 std::vector<KvafExecuteBatchGetKernelArgs> prefetchArgs(2,args);
 for(uint32_t q=0;q<2;++q){prefetchArgs[q].queue_state=prefetchQueues+q*sizeof(KvafBatchGetQueueState);
  prefetchArgs[q].core_status=prefetchStatuses+q*sizeof(KvafBatchGetCoreStatus);}
 c.args=put(prefetchArgs);a.context=put(std::vector<rotating::Context>{c});
'''
    text=replace_once(text,' auto deviceArg=put(std::vector<pipeline_ab::Args>{a});',injection+' auto deviceArg=put(std::vector<pipeline_ab::Args>{a});')
    text=replace_once(text,'binding.queueArgs=c.args;','binding.queueArgs=terminalQueueArgs;')
    # Check the status consumers' actual ABI, not merely the selected output.
    text=replace_once(text, ' printf("audit_errors=%zu\\n",bad-checkpoint);checkpoint=bad;',
        ' printf("audit_errors=%zu\\n",bad-checkpoint);checkpoint=bad;'+r'''
 auto completionRecords=get<rotating::FinalPost>(c.owners,4);
 auto transportStatuses=get<KvafBatchGetCoreStatus>(terminalStatuses,4);
 for(uint32_t q=0;q<4;++q){
  auto queueState=get<KvafBatchGetQueueState>(modelFinalArgs[q].queue_state,1)[0];
  const auto& status=transportStatuses[q];const auto& post=completionRecords[q];
  bad+=status.magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result!=0;
  bad+=status.processed_request_count!=post.count;
  bad+=status.completed_batch_count!=(post.count?1U:0U);
  bad+=status.sq_producer_index!=queueState.sq_producer_index;
  bad+=status.cq_consumer_index!=queueState.cq_consumer_index;
  bad+=queueState.completed_batch_count!=queueState.cq_consumer_index;
  bad+=post.pending!=0 || post.error!=0;
 }
 auto summary=get<uint32_t>(c.rowStatus,32);
 uint32_t expectedCompleted=0;for(const auto& record:completionRecords)expectedCompleted+=record.count!=0;
 bad+=summary[0]!=0 || summary[1]!=totalPosted || summary[3]!=0 || summary[4]!=expectedCompleted;
 bad+=get<uint32_t>(c.changeCount,1)[0]!=rows[0].slots;
 auto finalSummaries=get<KvafBatchGetCoreStatus>(c.finalStatus,2);
 auto lastQueue=get<KvafBatchGetQueueState>(modelFinalArgs[3].queue_state,1)[0];
 for(uint32_t bank=0;bank<2;++bank){
  const auto& status=finalSummaries[bank];
  bad+=status.magic!=KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.result!=0;
  bad+=status.processed_request_count!=(bank?0:totalPosted);
  bad+=status.completed_batch_count!=(bank?0:expectedCompleted);
  bad+=status.sq_producer_index!=(bank?0:lastQueue.sq_producer_index);
  bad+=status.cq_consumer_index!=(bank?0:lastQueue.cq_consumer_index);
 }
 printf("audit_transport_status=%zu\n",bad-checkpoint);checkpoint=bad;
''')
    driver.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    prepare_status_model(p.parse_args().output.resolve())
