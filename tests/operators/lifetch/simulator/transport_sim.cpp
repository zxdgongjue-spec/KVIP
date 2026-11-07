// SPDX-License-Identifier: Apache-2.0
// Native SQ copy/commit regression; MMIO is recorded by transport_kernel.cpp.
#include <acl/acl.h>
#include "context.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_test_transport_submit(uint32_t,void*,void*);
static void check(int rc) { if(rc)throw std::runtime_error("ACL failure"); }
template<class T> void* upload(const T& value) {
    void* p;check(aclrtMalloc(&p,sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));
    check(aclrtMemcpy(p,sizeof(T),&value,sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;
}
void run(aclrtStream stream,uint32_t count,uint64_t head) {
    constexpr uint32_t depth=32768,bytes=656;
    void* sq;check(aclrtMalloc(&sq,depth*64,ACL_MEM_MALLOC_HUGE_FIRST));
    uint32_t doorbells[64]{};auto db=upload(doorbells);
    KvafBatchGetDirectUrmaEntity entity{};
    entity.sq_va=(uint64_t)sq;entity.sq_db_va=(uint64_t)db;entity.sq_depth=depth;
    entity.remote_token_id=0x456;entity.local_token_id=0x789;entity.tp_id=0x123;
    auto e=upload(entity);KvafBatchGetQueueState queue{};
    queue.sq_producer_index=head;queue.cq_consumer_index=34;queue.completed_batch_count=6;
    auto q=upload(queue);KvafExecuteBatchGetKernelArgs args[2]{};
    args[1].channel_entity=(uint64_t)e;args[1].queue_state=(uint64_t)q;auto a=upload(args);
    rotating::Row row{};row.output=0x120080000000ULL;row.slots=512;row.finalRequests=count;
    auto r=upload(row);rotating::FinalPost post{};auto f=upload(post);
    rotating::Context ctx{};ctx.args=(uint64_t)a;ctx.rows=(uint64_t)r;ctx.owners=(uint64_t)f;auto c=upload(ctx);
    check(aclrtlaunch_test_transport_submit(1,stream,c));check(aclrtSynchronizeStream(stream));
    std::vector<uint32_t> output(depth*16);
    check(aclrtMemcpy(output.data(),depth*64,sq,depth*64,ACL_MEMCPY_DEVICE_TO_HOST));
    for(unsigned i=0;i<count;++i) {
        auto w=&output[((head+i)%depth)*16];
        uint32_t control=((head+i)&depth?0:0x80000000U)|(i+1==count?0x30360000U:0x30150000U);
        if(w[0]!=control || w[10]!=0x100000+i || w[11]!=0x1234 || w[12]!=bytes ||
           w[13]!=0x789 || w[14]!=0x80000000U+i*bytes || w[15]!=0x1200)
            throw std::runtime_error("SQ encoding/address mismatch");
    }
    check(aclrtMemcpy(&queue,sizeof(queue),q,sizeof(queue),ACL_MEMCPY_DEVICE_TO_HOST));
    check(aclrtMemcpy(doorbells,sizeof(doorbells),db,sizeof(doorbells),ACL_MEMCPY_DEVICE_TO_HOST));
    check(aclrtMemcpy(&post,sizeof(post),f,sizeof(post),ACL_MEMCPY_DEVICE_TO_HOST));
    unsigned posts=(count+127)/128;
    if(queue.sq_producer_index!=head+count || queue.cq_consumer_index!=34 || queue.completed_batch_count!=6 ||
       doorbells[0]!=head+count || doorbells[1]!=posts || post.count!=count || !post.pending || post.error || post.deferred)
        throw std::runtime_error("Queue/doorbell/publication mismatch");
    for(unsigned i=0;i<posts;++i) {
        auto end=(i+1)*128<count?(i+1)*128:count;
        if(doorbells[i+2]!=head+end)throw std::runtime_error("Missing intermediate doorbell");
    }
    std::printf("PASS native submit count=%u head=%llu posts=%u; SQ bytes, CQ state and doorbells checked\n",
                count,(unsigned long long)head,posts);std::fflush(stdout);
    for(auto p:{sq,db,e,q,a,r,f,c})check(aclrtFree(p));
}
int main() {
    try {
        check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
        run(stream,608,2091);run(stream,624,32740);run(stream,1025,65500);
        check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
