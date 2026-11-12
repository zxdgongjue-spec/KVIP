// Full Init->LI->Resolve->Outputs->Plan->Apply->Finalize replay. Transport is
// explicitly a synchronous GM-copy double; no URMA/KVAF timing is claimed.
#include <acl/acl.h>
#include "context.h"
#include "full_pool_launch.h"
#include "api.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_fusion3_stages_test_kernel(uint32_t,void*,void*,void*);
extern "C" uint32_t aclrtlaunch_v9_fusion3_tail_kernel(uint32_t,void*,void*,void*);
static void check(uint32_t c) { if(c)throw std::runtime_error("ACL/launch error "+std::to_string(c)); }
struct Memory {
    std::vector<void*> allocations;
    template<class T> uint64_t put(const std::vector<T>& v) {
        void* p=nullptr;size_t n=v.size()*sizeof(T);check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));
        allocations.push_back(p);check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);
    }
    template<class T> std::vector<T> get(uint64_t p,size_t n) {
        std::vector<T> v(n);check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    template<class T> void set(uint64_t p,const std::vector<T>& v) {
        check(aclrtMemcpy((void*)p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));
    }
    ~Memory(){for(auto p:allocations)aclrtFree(p);}
};
template<class T> void save(const std::string& p,const std::vector<T>& v) {
    std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));
    if(!f)throw std::runtime_error("write "+p);
}
static void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
#include "standard_host_helpers.h"
int main(int argc,char** argv) {
 try {
    require(argc>=3&&argc<=5,"fusion3_sim OUTPUT CASE [BATCH] [HEADS]");
    std::string out=argv[1],name=argv[2];uint32_t batch=argc>3?std::stoul(argv[3]):1,heads=argc>4?std::stoul(argv[4]):32;
    const std::vector<std::string> cases={"standard","zero","miss","fanout","padding","cycle","cycle-fanout","extended",
        "extended-odd","extended-4096","extended-4097","extended-sparse",
        "path","plan-error","plan-error-fanout","stale","transport-error","prefetch","bad-range",
        "prefetch-error","apply-error","binding-error","owner-error"};
    require(std::find(cases.begin(),cases.end(),name)!=cases.end(),"unknown fixture case");
    bool standard=name=="standard";
    require(batch>=1&&batch<=(standard?32U:2U)&&(heads==32||heads==64),"fixture shape");
    bool stage=!standard&&name!="zero"&&name!="miss"&&name!="fanout"&&name!="padding";
    bool extended=name=="extended"||name=="extended-odd"||name=="extended-4096"||name=="extended-4097"||name=="extended-sparse";
    bool failure=name=="plan-error"||name=="plan-error-fanout"||name=="stale"||name=="transport-error"||
        name=="bad-range"||name=="prefetch-error"||name=="apply-error"||name=="binding-error"||name=="owner-error";
    bool prefetch=name=="prefetch"||name=="bad-range"||name=="prefetch-error"||name=="plan-error";
    bool misses=name=="miss";
    const uint32_t history=standard?131072:(extended?32768:8192),resident=standard?16384:(extended?16384:4096),tokenBytes=656,fields=82,queries=batch*4;
    uint32_t outputStride=standard?resident+8192:resident;
    uint32_t moves=extended?3000:(name=="cycle"||name=="cycle-fanout"||name=="apply-error"?128:0);
    if(name=="extended-odd"||name=="extended-sparse")moves=2049;
    if(name=="extended-4096")moves=4096;
    if(name=="extended-4097")moves=4097;
    if(name=="plan-error"||name=="plan-error-fanout")moves=2;
    if(name=="path")moves=1;
    check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
    {
        Memory m;rotating::RuntimeShape s{};s.batch=batch;s.history=history;s.resident=resident;s.indexHeads=heads;
        s.aicCount=standard?(std::getenv("KVCORE_C")?std::stoul(std::getenv("KVCORE_C")):32):batch;s.splitQ=standard?(std::getenv("KVCORE_SPLIT")?std::stoul(std::getenv("KVCORE_SPLIT")):1):0;s.rounds=standard?8:1;s.finalWorkers=standard?0:2;s.tokenBytes=tokenBytes;
        check(rotating::PrepareShape(&s));s.rounds=0; // Explicit compute-only Service setting.
        uint32_t views=batch*(s.finalWorkers+1);
        std::vector<uint32_t> owners(batch);std::iota(owners.begin(),owners.end(),0);
        if(name=="padding"){require(batch==2,"padding needs B=2");owners[1]=0;}
        if(name=="owner-error"){require(batch==2,"owner-error needs B=2");owners={1,0};}
        auto ownerGm=m.put(owners);
        std::vector<uint8_t> keys(uint64_t(batch)*history*128+128,0x5a);
        for(uint32_t b=0;b<batch;++b)for(uint32_t t=0;t<history;++t) {
            uint8_t value=(misses?(t>=resident):(t<2048))?0x38:0;
            std::fill_n(keys.data()+(uint64_t(b)*history+t)*128,128,value);
        }
        if(standard){auto seed=ReadFixture<uint8_t>("arg1.bin");require(seed.size()==uint64_t(history)*128,"key seed size");
            for(uint32_t b=0;b<batch;++b)std::copy(seed.begin(),seed.end(),keys.begin()+uint64_t(b)*history*128);}
        std::vector<float> scales(uint64_t(batch)*history,1.f);
        if(standard){auto seed=ReadFixture<float>("arg4.bin");require(seed.size()==history,"scale seed size");
            for(uint32_t b=0;b<batch;++b)std::copy(seed.begin(),seed.end(),scales.begin()+uint64_t(b)*history);}
        if(stage)for(uint32_t b=0;b<batch;++b)for(uint32_t t=0;t<history;++t){
            scales[uint64_t(b)*history+t]=float(uint64_t(b)*history+t+1)*0.25f;
            for(uint32_t f=0;f<16;++f){uint64_t v=0x5600000000000000ULL|(uint64_t(b)<<48)|(uint64_t(t)<<8)|f;
                std::memcpy(keys.data()+(uint64_t(b)*history+t)*128+f*8,&v,8);}
        }
        std::vector<uint64_t> logical(uint64_t(batch)*history),addresses(logical.size());
        for(size_t i=0;i<logical.size();++i)logical[i]=standard?i%history:(uint64_t(1)<<40)+2000000+i;
        std::vector<uint64_t> payload(standard?0:(stage?1:uint64_t(batch)*history*fields));
        for(size_t i=0;i<payload.size();++i)payload[i]=0x1234000000000000ULL+i;
        uint64_t payloadWords=standard?uint64_t(batch)*history*fields:payload.size();
        auto payloadGm=standard?m.put(std::vector<uint64_t>(payloadWords)):m.put(payload);
        if(standard)for(uint32_t b=0;b<batch;++b)for(uint32_t begin=0;begin<history;begin+=1024){
            uint32_t count=std::min(1024U,history-begin);std::vector<uint64_t> chunk(uint64_t(count)*fields);
            for(uint32_t t=0;t<count;++t)for(uint32_t f=0;f<fields;++f)chunk[uint64_t(t)*fields+f]=ExpectedPayload(b,begin+t,f);
            m.set(payloadGm+(uint64_t(b)*history+begin)*tokenBytes,chunk);}
        for(size_t i=0;i<addresses.size();++i)addresses[i]=stage?1000000+i:payloadGm+i*tokenBytes;
        std::vector<uint64_t> outputPayload(uint64_t(batch)*outputStride*fields);
        if(standard){for(uint32_t b=0;b<batch;++b)for(uint32_t t=0;t<resident;++t)for(uint32_t f=0;f<fields;++f)
            outputPayload[(uint64_t(b)*outputStride+t)*fields+f]=ExpectedPayload(b,t,f);}
        else if(!stage)for(uint32_t b=0;b<batch;++b)
            std::copy_n(payload.data()+uint64_t(b)*history*fields,uint64_t(resident)*fields,
                        outputPayload.data()+uint64_t(b)*outputStride*fields);
        auto outputGm=m.put(outputPayload),keysGm=m.put(keys),scalesGm=m.put(scales),logicalGm=m.put(logical),addressesGm=m.put(addresses);
        std::vector<uint32_t> lru(batch*2048),changeSlots(batch*10240,UINT32_MAX),changes(batch*10240,UINT32_MAX),changeCount(batch,2048);
        for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<2048;++i)
            lru[b*2048+i]=changeSlots[b*10240+i]=stage?i:resident-2048+i;
        for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<moves;++i){
            changeSlots[b*10240+i]=i;changes[b*10240+i]=(i+1)%moves;
        }
        if(name=="extended-sparse")for(uint32_t b=0;b<batch;++b){
            std::fill_n(changes.data()+b*10240,10240,UINT32_MAX);
            changes[b*10240]=10;changes[b*10240+2048]=8192;
        }
        if(extended)std::fill(changeCount.begin(),changeCount.end(),moves);
        if(name=="path")for(uint32_t b=0;b<batch;++b)changes[b*10240]=10;
        if(name=="plan-error"||name=="plan-error-fanout")for(uint32_t b=0;b<batch;++b)changes[b*10240+1]=changes[b*10240];
        if(standard){auto seed=ReadFixture<uint32_t>("lru.bin");require(seed.size()==2048,"lru seed size");
            for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<2048;++i)lru[b*2048+i]=changeSlots[b*10240+i]=seed[i];}
        auto lruGm=m.put(lru),slotsGm=m.put(changeSlots),changesGm=m.put(changes),countsGm=m.put(changeCount);
        std::vector<int64_t> unpack(queries),valid(queries,1);std::iota(unpack.begin(),unpack.end(),0);
        if(name=="fanout"||name=="cycle-fanout"||name=="plan-error-fanout"){
            require(batch==2,"fanout needs B=2");unpack={7,0,7,1,0,6,4,4};valid[3]=0;unpack[3]=INT64_MIN;}
        if(name=="padding")for(uint32_t i=4;i<8;++i){valid[i]=0;unpack[i]=INT64_MIN;}
        auto unpackGm=m.put(unpack),validGm=m.put(valid);
        std::vector<uint32_t> queryEnds(queries,history);if(name=="padding")std::fill(queryEnds.begin()+4,queryEnds.end(),0);
        std::vector<uint32_t> hotValid(uint64_t(batch)*resident,1);
        auto hotValidGm=m.put(hotValid);
        std::vector<int32_t> topk(queries*2048+128,0x5a5a5a5a),locations(queries*2048+128,0x5a5a5a5a);
        if(stage)for(uint32_t q=0;q<queries;++q)for(uint32_t k=0;k<2048;++k)
            topk[q*2048+k]=locations[q*2048+k]=(q%4*97+k)%resident;
        auto topkGm=m.put(topk),locationsGm=m.put(locations);
        std::vector<uint32_t> cq(views*64*16,0);for(size_t i=0;i<cq.size();i+=16)cq[i]=4;
        auto cqGm=m.put(cq),sqGm=m.put(std::vector<uint8_t>(views*4096*64)),dbGm=m.put(std::vector<uint32_t>(views*64));
        std::vector<KvafBatchGetDirectUrmaEntity> entities(views);
        std::vector<KvafBatchGetQueueState> queues(views);
        std::vector<KvafBatchGetCoreStatus> statuses(views);
        for(uint32_t i=0;i<views;++i){
            auto& e=entities[i];e.magic=0x4b564155;e.version=1;e.sq_depth=4096;e.cq_depth=64;
            e.sq_va=sqGm+uint64_t(i)*4096*64;e.sq_db_va=dbGm+i*256;e.cq_va=cqGm+i*64*64;e.cq_db_va=dbGm+i*256+128;
            e.wqe_size=64;e.cqe_size=64;e.local_token_id=1;e.remote_token_id=1;e.local_hbm_address=outputGm;
            e.local_hbm_size=outputPayload.size()*8;e.remote_host_ddr_address=payloadGm;e.remote_host_ddr_size=payloadWords*8;
            queues[i].magic=0x4b564151;queues[i].initialized=1;queues[i].completed_batch_count=11;
            if(prefetch&&i<batch*2&&i%2==0)queues[i].sq_producer_index=3;
            statuses[i].magic=KVAF_AIV_BATCH_GET_STATUS_MAGIC;statuses[i].processed_request_count=2;
        }
        auto entitiesGm=m.put(entities),queuesGm=m.put(queues),statusesGm=m.put(statuses);
        auto blocksGm=m.put(std::vector<int32_t>(batch*8192)),offsetsGm=m.put(std::vector<int32_t>(batch*8192));
        auto destinationsGm=m.put(std::vector<int32_t>(batch*8192)),countGm=m.put(std::vector<uint32_t>(1));
        auto fallbackGm=m.put(std::vector<uint64_t>(batch*8192)),fallbackValidGm=m.put(std::vector<uint8_t>(batch*8192,1));
        std::vector<KvafExecuteBatchGetKernelArgs> args(views);
        for(uint32_t i=0;i<views;++i){auto& a=args[i];a.address_table=addressesGm;a.valid_table=fallbackValidGm;
            a.indices=blocksGm;a.offsets=offsetsGm;a.output_indices=destinationsGm;a.miss_count=countGm;a.output=outputGm;
            a.core_status=statusesGm+i*sizeof(KvafBatchGetCoreStatus);a.channel_entity=entitiesGm+i*sizeof(KvafBatchGetDirectUrmaEntity);
            a.queue_state=queuesGm+i*sizeof(KvafBatchGetQueueState);a.table_capacity=batch*8192;a.request_capacity=batch*8192;
            a.output_rows=batch*outputStride;a.token_bytes=tokenBytes;a.tokens_per_block=128;a.block_count=1;a.jettys_per_core=1;
            if(name=="stale")a.batch_get_max_inflight_total=0xfffffff1U;
            if(name=="transport-error")a.batch_get_max_inflight_total=0xfffffff2U;
        }
        rotating::Context ctx{};ctx.shape=s;ctx.rows=m.put(std::vector<rotating::Row>(batch));ctx.args=m.put(args);
        ctx.progress=m.put(std::vector<uint32_t>(batch*4*32));ctx.events=m.put(std::vector<uint8_t>((standard?8U:1U)*s.serviceLanes*128));
        ctx.owners=m.put(std::vector<uint8_t>(s.finalOwnerBytes));ctx.mapping=m.put(std::vector<uint32_t>(batch*history,UINT32_MAX));
        ctx.locations=locationsGm;ctx.selected=topkGm;ctx.rowOwners=ownerGm;ctx.queryEnds=m.put(queryEnds);
        ctx.rowStatus=m.put(std::vector<uint32_t>(batch*32));ctx.finalStatus=m.put(std::vector<KvafBatchGetCoreStatus>(batch*2));
        ctx.sourceBlocks=blocksGm;ctx.sourceOffsets=offsetsGm;ctx.destinations=destinationsGm;ctx.count=countGm;
        std::vector<int32_t> reserved(batch*resident,-1);
        for(uint32_t b=0;b<batch;++b)for(uint32_t i=0;i<2048;++i)reserved[b*resident+lru[b*2048+i]]=i;
        ctx.lruSlots=lruGm;ctx.reservedHot=m.put(reserved);ctx.changes=changesGm;
        ctx.changeSlots=slotsGm;ctx.changeCount=countsGm;ctx.hotValid=hotValidGm;ctx.tokenAddresses=addressesGm;
        ctx.addressStride=history;ctx.outputStride=outputStride;ctx.fallbackAddresses=fallbackGm;ctx.fallbackValid=fallbackValidGm;
        ctx.prefetchLimit=2048;ctx.windowLimit=448;
        std::vector<uint64_t> binding(15);
        std::vector<uint8_t> queryData(queries*heads*128,0x38);std::vector<uint16_t> weightData(queries*heads,heads==32?0x3d00:0x3c80);
        std::vector<float> queryScaleData(queries*heads,1);
        if(standard){auto q=ReadFixture<uint8_t>("arg0.bin");auto w=ReadFixture<uint16_t>("weights-bf16.bin");auto z=ReadFixture<float>("arg3.bin");
            require(q.size()==uint64_t(4)*heads*128&&w.size()==4*heads&&z.size()==4*heads,"query seed size");
            for(uint32_t b=0;b<batch;++b){std::copy(q.begin(),q.end(),queryData.begin()+uint64_t(b)*q.size());
                std::copy(w.begin(),w.end(),weightData.begin()+uint64_t(b)*w.size());std::copy(z.begin(),z.end(),queryScaleData.begin()+uint64_t(b)*z.size());}}
        binding[0]=m.put(queryData);binding[1]=keysGm;binding[2]=m.put(weightData);binding[3]=m.put(queryScaleData);binding[4]=scalesGm;
        std::vector<int32_t> lengths(batch),pages(batch*history/128);
        for(uint32_t b=0;b<batch;++b){lengths[b]=4;for(uint32_t p=0;p<history/128;++p)pages[b*history/128+p]=owners[b]*history/128+p;}
        binding[5]=m.put(lengths);binding[6]=m.put(std::vector<uint32_t>(batch,history));binding[7]=m.put(pages);binding[9]=topkGm;
        binding[12]=m.put(std::vector<uint16_t>(uint64_t(queries)*history+128,0x5a5a));
        uint64_t prefixBytes=uint64_t(batch)*(4*2048*6+1024+4*uint64_t(history)*2)+uint64_t(batch)*(history*8+8192*64+512);
        binding[13]=m.put(std::vector<uint8_t>(prefixBytes));ctx.hits=binding[13]+uint64_t(batch)*(4*2048*6+1024);
        if(name=="owner-error")ctx.rowOwners=0;
        binding[14]=m.put(std::vector<rotating::Context>{ctx});auto bindings=m.put(binding);
        std::vector<uint64_t> update(28);update[0]=keysGm;update[1]=scalesGm;update[2]=addressesGm;update[3]=logicalGm;
        update[4]=slotsGm;update[5]=changesGm;update[6]=m.put(std::vector<uint32_t>(batch));update[7]=history;
        update[8]=ctx.rows;update[9]=ownerGm;update[10]=ctx.rowStatus;update[11]=ctx.owners+s.finalPostOffset;
        update[12]=statusesGm;update[13]=ctx.finalStatus;update[14]=s.finalWorkers;update[15]=sizeof(KvafBatchGetCoreStatus);
        update[16]=hotValidGm;update[17]=m.put(std::vector<uint32_t>(batch*(8+2*20480),0x5a5a5a5a));
        update[18]=topkGm;update[19]=locationsGm;update[20]=unpackGm;update[21]=validGm;
        update[22]=m.put(std::vector<int32_t>((queries+1)*2048,0x5a5a5a5a));update[23]=m.put(std::vector<int32_t>((queries+1)*2048,0x5a5a5a5a));
        update[24]=history;update[25]=resident;update[26]=countsGm;update[27]=m.put(std::vector<uint32_t>(batch*40960,name=="extended-sparse"?0xa5a5a5a5U:0));
        if(name=="binding-error")update[9]=0;
        auto updateGm=m.put(update);
        if(stage){
            std::vector<rotating::Row> rows(batch);std::vector<uint32_t> info(batch*32);
            std::vector<rotating::FinalPost> posts(batch*s.finalWorkers);
            for(uint32_t b=0;b<batch;++b){auto& row=rows[b];row.entity=entitiesGm+b*2*sizeof(KvafBatchGetDirectUrmaEntity);
                row.queue=queuesGm+b*2*sizeof(KvafBatchGetQueueState);row.addresses=addressesGm+uint64_t(owners[b])*history*8;
                row.pending=prefetch?3:0;row.requests=row.pending;
                if(name=="prefetch-error")row.error=7;
                for(uint32_t w=0;w<s.finalWorkers;++w)posts[b*s.finalWorkers+w].descriptor=b*8192;
                auto& post=posts[b*s.finalWorkers];
                if(name=="stale"||name=="transport-error"){post.count=2;post.nonconflicting=2;post.deferred=1;}
                if(name=="bad-range"){post.descriptor=b*8192+8188;post.count=7;post.deferred=1;}
                if(name=="apply-error")post.error=1;
                info[b*32+1]=post.count;row.finalRequests=post.count;row.finalPending=post.count!=0;
            }
            m.set(ctx.rows,rows);m.set(ctx.rowStatus,info);m.set(ctx.owners+s.finalPostOffset,posts);
        }
        auto standardKeySeed=standard?ReadFixture<uint8_t>("arg1.bin"):std::vector<uint8_t>();
        if(standard){std::vector<uint8_t>().swap(keys);std::vector<uint64_t>().swap(outputPayload);}
        uint64_t poolGuardIds=0,poolGuardValues=0,poolGuardFinal=0;
        uint32_t resolvedPoolShards=std::getenv("KVCORE_POOL_SHARDS")?std::stoul(std::getenv("KVCORE_POOL_SHARDS")):0;
        uint32_t resolvedPoolV=std::getenv("KVCORE_V")?std::stoul(std::getenv("KVCORE_V")):64;
        {
            std::ofstream g(out+"/pool-geometry.json");
            g<<"{\"execution_mode\":\""<<(resolvedPoolShards?"independent_final_topk_pool":"original_mixed_li")
              <<"\",\"pool_shards\":"<<resolvedPoolShards<<",\"pool_vcap\":"<<resolvedPoolV
              <<",\"local_workers\":"<<(resolvedPoolShards?std::min(queries*resolvedPoolShards,resolvedPoolV):0)
              <<",\"merge_workers\":"<<(resolvedPoolShards>1?std::min(queries,resolvedPoolV):0)
              <<",\"li_aic\":"<<s.liCores<<",\"li_score_aiv\":"<<2*s.liCores<<"}\n";
        }
        auto start=std::chrono::steady_clock::now();
        if(stage){
#if KVIP_FUSION3
            if(name=="binding-error"||name=="owner-error")
                check(aclrtlaunch_v9_fusion3_tail_kernel(batch,stream,(void*)bindings,(void*)updateGm));
            else check(aclrtlaunch_v9_fusion3_stages_test_kernel(batch,stream,(void*)bindings,(void*)updateGm));
#else
            throw std::runtime_error("stage diagnostic requires candidate build");
#endif
        }else {
            uint32_t shards=std::getenv("KVCORE_POOL_SHARDS")?std::stoul(std::getenv("KVCORE_POOL_SHARDS")):0;
            if(!shards) check(kvip_lifetch_forward((void*)bindings,(void*)updateGm,stream,&s));
            else {
                uint32_t vcap=std::getenv("KVCORE_V")?std::stoul(std::getenv("KVCORE_V")):64;
                require(standard && (shards==1||shards==4||shards==8) && vcap>=1 && vcap<=64,"pooled shape");
                auto ids=m.put(std::vector<uint32_t>(uint64_t(queries)*shards*2048+128,0x5a5a5a5a));
                auto values=m.put(std::vector<uint16_t>(uint64_t(queries)*shards*2048+128,0x5a5a));
                auto finalValues=m.put(std::vector<uint16_t>(uint64_t(queries)*2048+128,0x5a5a));
                auto localTimes=m.put(std::vector<uint64_t>(64*4)),mergeTimes=m.put(std::vector<uint64_t>(64*4));
                poolGuardIds=ids;poolGuardValues=values;poolGuardFinal=finalValues;
                uint32_t lw=std::min(queries*shards,vcap),mw=std::min(queries,vcap);
                check(v9_init((void*)bindings,stream,&s));
                check(aclrtlaunch_v9_li_pool_kernel(s.liCores,stream,(void*)bindings));
                check(aclrtlaunch_split_topk_local(lw,stream,(void*)binding[12],
                    (void*)(shards==1?topkGm:ids),(void*)values,queries,history,shards,lw,(void*)localTimes));
                if(shards>1) check(aclrtlaunch_split_topk_merge(mw,stream,(void*)ids,(void*)values,
                    (void*)topkGm,(void*)finalValues,queries,shards,mw,(void*)mergeTimes));
                check(v9_resolve((void*)bindings,stream,&s));
                check(kvop_direct_update_outputs_early(batch,(void*)updateGm,stream));
                check(aclrtlaunch_v9_fetch_update_kernel(batch,stream,(void*)bindings,(void*)updateGm));
            }
        }
        check(aclrtSynchronizeStream(stream));
        if(resolvedPoolShards){
            auto gi=m.get<uint32_t>(poolGuardIds,uint64_t(queries)*resolvedPoolShards*2048+128);
            auto gv=m.get<uint16_t>(poolGuardValues,uint64_t(queries)*resolvedPoolShards*2048+128);
            auto gf=m.get<uint16_t>(poolGuardFinal,uint64_t(queries)*2048+128);
            auto gs=m.get<uint16_t>(binding[12],uint64_t(queries)*history+128);
            for(uint32_t i=0;i<128;++i) require(
                gi[uint64_t(queries)*resolvedPoolShards*2048+i]==0x5a5a5a5a &&
                gv[uint64_t(queries)*resolvedPoolShards*2048+i]==0x5a5a &&
                gf[uint64_t(queries)*2048+i]==0x5a5a &&
                gs[uint64_t(queries)*history+i]==0x5a5a,"pooled intermediate guard");
            std::ofstream proof(out+"/pool-intermediate-check.json");
            proof<<"{\"status\":\"passed\",\"candidate_allocation_guards\":true,\"score_gm_guard\":true}\n";
        }

        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        topk=m.get<int32_t>(topkGm,topk.size());locations=m.get<int32_t>(locationsGm,locations.size());
        auto outLogical=m.get<int32_t>(update[22],(queries+1)*2048),outSlots=m.get<int32_t>(update[23],(queries+1)*2048);
        auto errors=m.get<uint32_t>(update[6],batch),info=m.get<uint32_t>(ctx.rowStatus,batch*32);
        auto final=m.get<KvafBatchGetCoreStatus>(ctx.finalStatus,batch*2);auto rows=m.get<rotating::Row>(ctx.rows,batch);
        auto plans=m.get<uint32_t>(update[17],batch*(8+2*20480));
        auto outKeys=standard?std::vector<uint8_t>():m.get<uint8_t>(keysGm,keys.size());auto outScales=m.get<float>(scalesGm,scales.size());
        auto outAddresses=m.get<uint64_t>(addressesGm,addresses.size()),outIds=m.get<uint64_t>(logicalGm,logical.size());
        auto outValid=m.get<uint32_t>(hotValidGm,hotValid.size());
        auto newChanges=m.get<uint32_t>(changesGm,changes.size()),newSlots=m.get<uint32_t>(slotsGm,changeSlots.size());
        auto newCounts=m.get<uint32_t>(countsGm,batch);
        auto outPayload=standard?std::vector<uint64_t>():m.get<uint64_t>(outputGm,outputPayload.size());
        if(name=="extended-sparse")for(uint32_t b=0;b<batch;++b){
            uint64_t off=uint64_t(b)*(8+2*20480);
            const uint32_t expected[]={0,10,2048,8192,0x8000000aU,0,0x80002000U,2048};
            require(plans[off]==4,"sparse compact count");
            for(uint32_t j=0;j<8;++j)require(plans[off+8+j]==expected[j],"sparse closure/order/tag oracle");
            require(outValid[uint64_t(owners[b])*resident+10]==0&&
                    outValid[uint64_t(owners[b])*resident+8192]==0,"sparse extra hot invalidation");
        }
        for(size_t i=queries*2048;i<topk.size();++i)
            require(topk[i]==0x5a5a5a5a&&locations[i]==0x5a5a5a5a,"TopK/locations guard");
        if(!stage)for(uint32_t q=0;q<queries;++q){
            if(name=="padding"&&q>=4)continue;
            std::vector<bool> seen(history);
            for(uint32_t k=0;k<2048;++k){int32_t token=topk[q*2048+k];
                require(token>=0&&uint32_t(token)<history&&!seen[token],"LI TopK uniqueness/range");seen[token]=true;}
        }
        if(standard){auto mask=ReadFixture<uint8_t>("expected-selected.bin");require(mask.size()==uint64_t(4)*history,"mask size");
            for(uint32_t b=0;b<batch;++b){std::vector<bool> cold(history);for(uint32_t q=0;q<4;++q){uint32_t misses=0;
                for(uint32_t k=0;k<2048;++k){uint32_t t=topk[(b*4+q)*2048+k];require(mask[uint64_t(q)*history+t],"standard TopK set oracle");
                    if(t>=resident){++misses;cold[t]=true;}}require(misses==205,"standard per-query miss count");}
                require(std::count(cold.begin(),cold.end(),true)==410,"standard unique miss count");
                require(info[b*32+1]==410,"standard final fetch descriptor count");}}
        for(uint32_t q=0;q<=queries;++q)for(uint32_t k=0;k<2048;++k){
            if(q==queries){require(outLogical[q*2048+k]==0x5a5a5a5a&&outSlots[q*2048+k]==0x5a5a5a5a,"output guard");continue;}
            if(failure||!valid[q]){require(outLogical[q*2048+k]==-1&&outSlots[q*2048+k]==-1,"failed/padding output not poisoned");continue;}
            uint32_t source=unpack[q],owner=owners[source/4];int32_t token=topk[source*2048+k],slot=locations[source*2048+k];
            require(token>=0&&uint32_t(token)<history,"selected token range");
            if(!stage&&!standard)require(misses?uint32_t(token)>=resident:uint32_t(token)<resident,"LI TopK score-class oracle");
            require(outLogical[q*2048+k]==int32_t(logical[uint64_t(owner)*history+token]),"old logical output association");
            require(outSlots[q*2048+k]==slot,"physical output association");
            if(!stage)require(slot>=0&&uint32_t(slot)<resident,"physical slot range");
            if(!stage&&!standard){
                for(uint32_t f=0;f<fields;++f)require(outPayload[(uint64_t(owner)*outputStride+slot)*fields+f]==
                    (standard?ExpectedPayload(owner,token,f):payload[(uint64_t(owner)*history+token)*fields+f]),"mock payload mismatch");}
        }
        for(uint32_t b=0;b<batch;++b){
            require(failure?errors[b]!=0:errors[b]==0,"update error status");
            require(failure?final[b*2].result!=0:final[b*2].result==0,"aggregate final status");
            if(KVIP_FUSION3)require(info[b*32+5]==1,"phase_done missing");
            if(name=="prefetch-error")require(rows[b].pending==3,"uncertain prefetch was cleared");
            else require(rows[b].pending==0,"healthy prefetch not drained");
            if(failure)for(uint32_t t=0;t<resident;++t)require(outValid[uint64_t(owners[b])*resident+t]==0,"failed residency retained");
            uint32_t owner=owners[b],n=plans[b*(8+2*20480)];require(n<=20480,"plan count range");
            std::vector<uint32_t> permutation(history),written(history);std::iota(permutation.begin(),permutation.end(),0);
            std::vector<uint32_t> expectedValid(resident,1);
            for(uint32_t i=0;i<n;++i){uint32_t d=plans[b*(8+2*20480)+8+2*i]&0x7fffffffU,src=plans[b*(8+2*20480)+9+2*i];
                require(d<history&&src<history&&!written[d]++,"plan bounds/injectivity");permutation[d]=src;
                if(plans[b*(8+2*20480)+8+2*i]&0x80000000U){require(d<resident,"invalidate target range");expectedValid[d]=0;}}
            if(name!="plan-error"&&name!="plan-error-fanout"&&name!="padding")for(uint32_t i=0;i<newCounts[b];++i)
                if(newChanges[b*10240+i]!=UINT32_MAX)require(permutation[newSlots[b*10240+i]]==newChanges[b*10240+i],"producer edge missing");
            auto sorted=permutation;std::sort(sorted.begin(),sorted.end());for(uint32_t t=0;t<history;++t)require(sorted[t]==t,"not a full permutation");
            if(!failure){
                for(uint32_t i=0;i<2048;++i)if(newChanges[b*10240+i]!=UINT32_MAX)expectedValid[lru[b*2048+i]]=1;
                auto destinations=m.get<int32_t>(destinationsGm,batch*8192);
                for(uint32_t i=0;i<info[b*32+1];++i){uint32_t slot=destinations[b*8192+i]-owner*outputStride;
                    if(slot<resident)expectedValid[slot]=1;}
                for(uint32_t t=0;t<resident;++t)require(outValid[uint64_t(owner)*resident+t]==expectedValid[t],"hot validity/invalidate flag");
            }
            auto ownerKeys=standard?m.get<uint8_t>(keysGm+uint64_t(owner)*history*128,uint64_t(history)*128):std::vector<uint8_t>();
            auto ownerPayload=standard?m.get<uint64_t>(outputGm+uint64_t(owner)*outputStride*tokenBytes,uint64_t(outputStride)*fields):std::vector<uint64_t>();
            for(uint32_t t=0;t<history;++t){uint32_t src=permutation[t];
                for(uint32_t f=0;f<128;++f)require(
                    (standard?ownerKeys[uint64_t(t)*128+f]:outKeys[(uint64_t(owner)*history+t)*128+f])==
                    (standard?standardKeySeed[uint64_t(src)*128+f]:keys[(uint64_t(owner)*history+src)*128+f]),"Index key permutation");
                require(outScales[uint64_t(owner)*history+t]==scales[uint64_t(owner)*history+src]&&
                    outAddresses[uint64_t(owner)*history+t]==addresses[uint64_t(owner)*history+src]&&
                    outIds[uint64_t(owner)*history+t]==logical[uint64_t(owner)*history+src],"Index metadata permutation");
            }
            if(standard){
                for(uint32_t q=0;q<4;++q)for(uint32_t k=0;k<2048;++k){
                    uint32_t token=topk[(b*4+q)*2048+k],slot=locations[(b*4+q)*2048+k];
                    require(slot<resident&&outValid[uint64_t(owner)*resident+slot],"selected slot validity");
                    for(uint32_t f=0;f<fields;++f)require(ownerPayload[uint64_t(slot)*fields+f]==ExpectedPayload(owner,token,f),"selected payload ownership");}
                for(uint32_t t=0;t<resident;++t)if(outValid[uint64_t(owner)*resident+t])for(uint32_t f=0;f<fields;++f)
                    require(ownerPayload[uint64_t(t)*fields+f]==ExpectedPayload(owner,outIds[uint64_t(owner)*history+t],f),"all valid hot payload ownership");
                for(uint32_t t=resident;t<outputStride;++t)for(uint32_t f=0;f<fields;++f)
                    require(ownerPayload[uint64_t(t)*fields+f]==0,"unused output-tail guard");
            }
        }
        if(standard){auto guard=m.get<uint8_t>(keysGm+uint64_t(batch)*history*128,128);for(auto value:guard)require(value==0x5a,"key guard");}
        else for(size_t i=uint64_t(batch)*history*128;i<keys.size();++i)require(outKeys[i]==0x5a,"key guard");
        save(out+"/topk.bin",topk);save(out+"/locations.bin",locations);save(out+"/plans.bin",plans);
        save(out+"/changes.bin",newChanges);save(out+"/change-counts.bin",newCounts);
        save(out+"/final-status.bin",final);save(out+"/row-status.bin",info);save(out+"/update-status.bin",errors);
        save(out+"/logical-output.bin",outLogical);save(out+"/slot-output.bin",outSlots);save(out+"/rows.bin",rows);
        std::ofstream report(out+"/correctness.json");report<<"{\"status\":\"pass\",\"case\":\""<<name<<"\",\"batch\":"<<batch
            <<",\"heads\":"<<heads<<",\"history\":"<<history<<",\"resident\":"<<resident<<",\"fusion3\":"<<(KVIP_FUSION3?"true":"false")
            <<",\"full_compute_chain\":"<<(stage?"false":"true")<<",\"physical_npu\":false,\"native_transport\":false,\"transport\":\"synchronous GM copy double, warp per token\",\"host_simulator_wall_seconds\":"<<wall
            <<",\"failure_expected\":"<<(failure?"true":"false")<<",\"all_index_fields_checked\":true,\"outputs_checked\":true,\"guards_checked\":true}\n";
        report.close();
        if(standard){std::ofstream geometry(out+"/standard-geometry.json");geometry<<"{\"history\":"<<history<<",\"resident\":"<<resident<<",\"batch\":"<<batch
            <<",\"heads\":"<<heads<<",\"li_cores\":"<<s.liCores<<",\"split_q\":"<<s.splitQ<<",\"final_workers\":"<<s.finalWorkers
            <<",\"aic_budget\":"<<s.aicCount<<",\"output_stride\":"<<outputStride<<",\"service_rounds\":0,\"prepared_rounds\":8,\"miss_per_query\":205,\"unique_misses_per_request\":410}\n";}
        if (std::getenv("FUSION3_HOLD_FOR_RAW_SNAPSHOT")) {
            // Host-only handshake: all kernels completed before this point.
            // Delay SDK finalization until the collector has retained raw logs.
            auto holdStart=std::chrono::steady_clock::now();
            while (!std::ifstream(out+"/raw-snapshot.ready").good()) {
                require(std::chrono::steady_clock::now()-holdStart<std::chrono::seconds(120), "raw snapshot handshake timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        std::cout<<"PASS "<<name<<" B="<<batch<<" H="<<heads<<" "<<(stage?"tail stages":"full compute chain")<<"; synthetic GM transport only\n";
    }
    check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
