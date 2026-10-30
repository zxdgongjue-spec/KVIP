// TEST ONLY. Interpret real generated GET descriptors against local NPU memory.
// No network/CQ emulation claims: copies finish before publishing completion.
#pragma once
__aicore__ inline uint64_t TestCopyToken(
    __gm__ const KvafBatchGetDirectUrmaEntity* e, uint64_t source, uint64_t target) {
    if (source < e->remote_host_ddr_address || target < e->local_hbm_address ||
        source - e->remote_host_ddr_address > e->remote_host_ddr_size - 656 ||
        target - e->local_hbm_address > e->local_hbm_size - 656 || (source|target)&3)
        return kvaf_device::kInvalidArgs;
    auto input = reinterpret_cast<__gm__ uint32_t*>(source);
    auto output = reinterpret_cast<__gm__ uint32_t*>(target);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(source), 656);
    for(uint32_t i=0;i<164;++i) output[i]=input[i];
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(target), 656);
    return 0;
}
__aicore__ inline CqDrainResult TestDrainQueue(
    __gm__ const KvafBatchGetDirectUrmaEntity* entity,
    __gm__ KvafBatchGetQueueState* queue,uint32_t pending) {
    uint32_t completed=0;
    uint64_t cursor=queue->reserved[0], end=queue->sq_producer_index;
    if(end<cursor || end-cursor>=entity->sq_depth) return {kvaf_device::kInvalidArgs,0,0};
    for(;cursor<end;++cursor) {
        auto wqe=reinterpret_cast<__gm__ uint32_t*>(entity->sq_va)+(cursor&(entity->sq_depth-1))*16;
        kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(wqe),64);
        uint64_t source=uint64_t(wqe[10])|(uint64_t(wqe[11])<<32);
        uint64_t target=uint64_t(wqe[14])|(uint64_t(wqe[15])<<32);
        uint64_t error=wqe[12]==656?TestCopyToken(entity,source,target):kvaf_device::kInvalidArgs;
        if(error) return {error,completed,0};
        if((wqe[0]&0x7fff0000U)==0x30360000U) ++completed;
    }
    queue->reserved[0]=cursor;
    queue->cq_consumer_index+=completed;
    queue->completed_batch_count+=completed;
    return {completed==pending?0:kvaf_device::kInvalidArgs,completed,0};
}
__aicore__ inline void TestBatchGet(KvafExecuteBatchGetKernelArgs args,
    uint32_t, uint32_t begin, uint32_t count) {
    auto e=reinterpret_cast<__gm__ const KvafBatchGetDirectUrmaEntity*>(args.channel_entity);
    auto indices=reinterpret_cast<__gm__ int32_t*>(args.indices);
    auto offsets=reinterpret_cast<__gm__ int32_t*>(args.offsets);
    auto slots=reinterpret_cast<__gm__ int32_t*>(args.output_indices);
    auto addresses=reinterpret_cast<__gm__ uint64_t*>(args.address_table);
    auto valid=reinterpret_cast<__gm__ uint8_t*>(args.valid_table);
    uint64_t error=0;uint32_t copied=0;
    if(begin>args.request_capacity || count>args.request_capacity-begin)error=kvaf_device::kInvalidArgs;
    for(uint32_t i=begin;!error && i<begin+count;++i) {
        if(uint32_t(indices[i])>=args.table_capacity || !valid[indices[i]] || offsets[i]<0 ||
           uint32_t(offsets[i])>128*656-656 || uint32_t(slots[i])>=args.output_rows) {
            error=kvaf_device::kInvalidArgs;break;
        }
        error=TestCopyToken(e,addresses[indices[i]]+offsets[i],args.output+uint64_t(slots[i])*656);
        if(!error)++copied;
    }
    auto status=reinterpret_cast<__gm__ KvafBatchGetCoreStatus*>(args.core_status);
    kvaf_device::WriteStatus(status,error,copied,!error&&count?1:0,0,0,UINT64_MAX,0);
    kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(status),sizeof(*status));
}
