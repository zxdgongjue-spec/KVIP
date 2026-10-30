#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "rotation_abi.h"
extern "C" uint32_t v9_init(void*,void*),v9_li(void*,void*),v9_wait(void*,void*),v9_resolve(void*,void*),v9_fetch(void*,void*),v9_probe(void*,void*),v9_simulation();
static void Check(uint32_t code) { if(code) throw std::runtime_error("ACL error "+std::to_string(code)); }
struct Memory {
    std::vector<void*> allocations;
    void* Put(const void* data,size_t bytes) {
        void* device; Check(aclrtMalloc(&device,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); allocations.push_back(device);
        Check(aclrtMemcpy(device,bytes,data,bytes,ACL_MEMCPY_HOST_TO_DEVICE)); return device;
    }
    uint64_t File(const std::string& path,size_t bytes) {
        std::ifstream file(path,std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(file)),{});
        if(data.size()!=bytes) throw std::runtime_error("invalid replay file "+path);
        return reinterpret_cast<uint64_t>(Put(data.data(),bytes));
    }
    uint64_t Zero(size_t bytes,uint8_t fill=0) { std::vector<uint8_t> data(bytes,fill);return reinterpret_cast<uint64_t>(Put(data.data(),bytes)); }
    void Dump(uint64_t device,size_t bytes,const std::string& path) {
        std::vector<uint8_t> data(bytes); if(bytes) Check(aclrtMemcpy(data.data(),bytes,reinterpret_cast<void*>(device),bytes,ACL_MEMCPY_DEVICE_TO_HOST));
        std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<char*>(data.data()),bytes);
    }
    ~Memory() { for(void* p:allocations) aclrtFree(p); }
};
int main(int argc,char** argv) {
#if V9_CONTRACT_MODE
    // Historical binary fixtures do not carry the caller-owned ABI 8 tensors.
    // Reject before touching ACL/queues; the contract CPU runner is separate.
    std::cerr << "ABI 8: use benchmark_simulator.py for contract simulation; "
                 "historical CANN fixture requires -DV9_CONTRACT_MODE=OFF\n";
    return 2;
#endif
    try {
        if(argc<2 || argc>4 || !v9_simulation()) throw std::runtime_error("sim build only: v9_simulator INPUT [prefetch_limit [probe|final-only]]");
        using namespace rotating;
        const std::string root=argv[1]; const bool probe=argc==4 && std::string(argv[3])=="probe";
        const bool finalOnly=argc==4 && std::string(argv[3])=="final-only";
        if(argc==4 && !probe && !finalOnly)throw std::runtime_error("unknown simulator mode");
        std::remove((root+"/completed.json").c_str());
        std::ofstream execution(root+"/execution.json");
        execution << "{\"abi_version\":8,\"li_cores\":" << kLiCores << ",\"cores_per_batch\":" << kCoresPerBatch << ",\"service_lanes\":" << kServiceLanes << ",\"descriptor_strided\":" << (kFinalReplicated || kFinalWorkers==1) << ",\"final_workers\":" << kFinalWorkers << ",\"rounds\":" << kRounds << ",\"trigger_tokens\":[";
        for(uint32_t e=0;e<kRounds;++e) { if(e)execution << ",";execution << Trigger(e)*128; }
        execution << "],\"probe_epoch\":" << V9_PROBE_EPOCH << ",\"final_only\":" << (finalOnly?"true":"false") << "}";
        execution.close();
        Check(aclInit(nullptr));Check(aclrtSetDevice(0));aclrtStream stream;Check(aclrtCreateStream(&stream));
        {
            Memory m;uint64_t bindings[15]{};size_t sizes[14]{};
            for(uint32_t i=0;i<14;++i) {
                std::string name=i==2 ? "native_weights_bf16" : (finalOnly && (i==9 || i==13) ? "out" : "arg")+std::to_string(i);
                std::ifstream file(root+"/"+name+".bin",std::ios::binary);
                if(!file) throw std::runtime_error("missing fixture argument "+std::to_string(i));
                std::vector<char> data((std::istreambuf_iterator<char>(file)),{});sizes[i]=data.size();
                bindings[i]=reinterpret_cast<uint64_t>(m.Put(data.data(),data.size()));
            }
            Context ctx{};
            ctx.progress=m.Zero(SIM_BATCH*4*sizeof(Progress));ctx.rows=m.Zero(SIM_BATCH*sizeof(Row));
            ctx.mapping=m.Zero(SIM_BATCH*SIM_HISTORY*4,255);ctx.locations=m.Zero(SIM_BATCH*kSlots*4,255);
            ctx.events=m.Zero((kRounds?kRounds:1)*kServiceLanes*sizeof(Event));ctx.owners=m.Zero(kFinalOwnerBytes);
            ctx.hits=bindings[13]+SIM_BATCH*(4*2048*6+1024);ctx.selected=bindings[9];
            ctx.sourceBlocks=m.Zero(SIM_BATCH*kSlots*4);ctx.sourceOffsets=m.Zero(SIM_BATCH*kSlots*4);
            ctx.destinations=m.Zero(SIM_BATCH*kSlots*4);ctx.count=m.Zero(4);ctx.rowStatus=m.Zero(SIM_BATCH*kRowStatusWords*4);
            ctx.finalStatus=m.Zero(SIM_BATCH*2*sizeof(KvafBatchGetCoreStatus));
            ctx.prefetchLimit=argc>2 ? std::stoul(argv[2]) : 2048;ctx.windowLimit=480;
            std::vector<uint64_t> sqs,queues,dbs;
            std::vector<KvafExecuteBatchGetKernelArgs> args(kQueueViews);
            constexpr uint32_t depth=16384;
            const uint64_t output=0x1200ffffff00ULL;
            for(uint32_t i=0;i<kQueueViews;++i) {
                KvafBatchGetDirectUrmaEntity e{};
                e.sq_va=finalOnly && i<SIM_BATCH*2 && i%2==0 ? m.File(root+"/prefetch_sq"+std::to_string(i/2)+".bin",depth*64) : m.Zero(depth*64);
                e.sq_db_va=finalOnly && i<SIM_BATCH*2 && i%2==0 ? m.File(root+"/prefetch_db"+std::to_string(i/2)+".bin",128) : m.Zero(128);e.sq_depth=depth;e.cq_depth=depth;
                e.tp_id=0x1234;e.remote_token_id=0x54321;e.remote_token_value=0x2468ace0;e.local_token_id=0x12345;
                uint32_t eid[]={0x11223344,0x55667788,0x99aabbcc,0xddeeff00};memcpy(e.remote_eid,eid,16);
                KvafBatchGetQueueState queue{};queue.sq_producer_index=depth-2;
                args[i].channel_entity=reinterpret_cast<uint64_t>(m.Put(&e,sizeof(e)));
                args[i].queue_state=finalOnly && i<SIM_BATCH*2 && i%2==0 ? m.File(root+"/prefetch_queue"+std::to_string(i/2)+".bin",sizeof(queue)) : reinterpret_cast<uint64_t>(m.Put(&queue,sizeof(queue)));
                args[i].core_status=m.Zero(sizeof(KvafBatchGetCoreStatus));args[i].output=output;
                sqs.push_back(e.sq_va);queues.push_back(args[i].queue_state);dbs.push_back(e.sq_db_va);
            }
            ctx.args=reinterpret_cast<uint64_t>(m.Put(args.data(),args.size()*sizeof(args[0])));
            if(finalOnly) {
                std::ifstream file(root+"/rows.bin",std::ios::binary);
                std::vector<Row> rows(SIM_BATCH);
                file.read(reinterpret_cast<char*>(rows.data()),rows.size()*sizeof(Row));
                if(!file) throw std::runtime_error("invalid replay rows");
                for(uint32_t b=0;b<SIM_BATCH;++b) {
                    auto& row=rows[b];row.entity=args[b*2].channel_entity;row.queue=args[b*2].queue_state;
                    row.addresses=bindings[13]+SIM_BATCH*(4*2048*6+1024+4*uint64_t(SIM_HISTORY)*2)+
                        b*(uint64_t(SIM_HISTORY)*8+8192*64+512);
                    row.output=output+uint64_t(b)*(SIM_RESIDENT+kSlots)*kTokenBytes;
                    row.finalRequests=row.finalPending=0;row.finalHead=row.finalBegin=row.finalEnd=0;
                }
                ctx.rows=reinterpret_cast<uint64_t>(m.Put(rows.data(),rows.size()*sizeof(Row)));
                ctx.mapping=m.File(root+"/prefetch_mapping.bin",SIM_BATCH*SIM_HISTORY*4);
                ctx.progress=m.File(root+"/progress.bin",SIM_BATCH*4*sizeof(Progress));
                // Preserve earlier LI/service evidence for final-only replays.
                // No intermediate work is executed or timed by this launch.
                if(kRounds)ctx.events=m.File(root+"/events.bin",kRounds*kServiceLanes*sizeof(Event));
            }
            if(probe) {
                // AIV-only continuation fixture: no LI launch is needed to
                // exercise an unequal-cursor block followed by a full tail.
                std::ifstream cursorFile(root+"/probe_consumed.bin",std::ios::binary);
                if(cursorFile) {
                    std::vector<uint32_t> cursors(SIM_BATCH*4);
                    cursorFile.read(reinterpret_cast<char*>(cursors.data()),cursors.size()*4);
                    if(cursorFile.gcount()!=cursors.size()*4 || cursorFile.peek()!=EOF)
                        throw std::runtime_error("invalid probe_consumed size");
                    std::vector<Row> rows(SIM_BATCH);
                    for(uint32_t b=0;b<SIM_BATCH;++b)for(uint32_t q=0;q<4;++q) {
                        auto value=cursors[b*4+q];
                        if(value<SIM_RESIDENT || value>Trigger(V9_PROBE_EPOCH)*128 || value%128)
                            throw std::runtime_error("invalid probe_consumed cursor");
                        rows[b].consumed[q]=value;
                    }
                    Check(aclrtMemcpy(reinterpret_cast<void*>(ctx.rows),rows.size()*sizeof(Row),rows.data(),rows.size()*sizeof(Row),ACL_MEMCPY_HOST_TO_DEVICE));
                }
                std::vector<Progress> progress(SIM_BATCH*4);for(auto& p:progress)p.end[0]=Trigger(V9_PROBE_EPOCH)*128;
                std::ifstream overrideProgress(root+"/probe_progress.bin",std::ios::binary);
                if(overrideProgress) {
                    overrideProgress.read(reinterpret_cast<char*>(progress.data()),progress.size()*sizeof(Progress));
                    if(overrideProgress.gcount()!=progress.size()*sizeof(Progress) || overrideProgress.peek()!=EOF)
                        throw std::runtime_error("invalid probe_progress size");
                    for(const auto& p:progress)if(p.end[0]<SIM_RESIDENT || p.end[0]>Trigger(V9_PROBE_EPOCH)*128 || p.end[0]%128)
                        throw std::runtime_error("invalid probe_progress end");
                }
                Check(aclrtMemcpy(reinterpret_cast<void*>(ctx.progress),progress.size()*sizeof(Progress),progress.data(),progress.size()*sizeof(Progress),ACL_MEMCPY_HOST_TO_DEVICE));
                std::ifstream file(root+"/expected_hits.bin",std::ios::binary);
                std::vector<char> hits((std::istreambuf_iterator<char>(file)),{});
                if(hits.size()!=4*SIM_HISTORY*2)throw std::runtime_error("invalid probe flags");
                std::vector<uint8_t> packed(SIM_HISTORY,0);
                const auto values=reinterpret_cast<const uint16_t*>(hits.data());
                for(uint32_t q=0;q<4;++q)for(uint32_t t=0;t<SIM_HISTORY;++t)
                    if(values[q*SIM_HISTORY+t])packed[q*(SIM_HISTORY/4)+t/4]|=1U<<((t%4)*2);
                std::ifstream oddFile(root+"/probe_odd_bits.bin",std::ios::binary);
                if(oddFile) {
                    std::vector<uint8_t> odd((std::istreambuf_iterator<char>(oddFile)),{});
                    if(odd.size()!=packed.size())throw std::runtime_error("invalid odd-bit fixture size");
                    for(size_t i=0;i<odd.size();++i) {
                        if(odd[i]&0x55)throw std::runtime_error("odd-bit fixture changes meaningful bits");
                        packed[i]|=odd[i];
                    }
                }
                for(uint32_t b=0;b<SIM_BATCH;++b)Check(aclrtMemcpy(reinterpret_cast<void*>(ctx.hits+b*packed.size()),packed.size(),packed.data(),packed.size(),ACL_MEMCPY_HOST_TO_DEVICE));
            }
            if(probe && V9_PROBE_EPOCH) {
                ctx.rows=m.File(root+"/probe_rows.bin",SIM_BATCH*sizeof(Row));
                ctx.mapping=m.File(root+"/probe_mapping.bin",SIM_BATCH*SIM_HISTORY*4);
                auto packed=m.File(root+"/probe_bitmap.bin",SIM_BATCH*SIM_HISTORY);
                Check(aclrtMemcpy(reinterpret_cast<void*>(ctx.hits),SIM_BATCH*SIM_HISTORY,
                    reinterpret_cast<void*>(packed),SIM_BATCH*SIM_HISTORY,ACL_MEMCPY_DEVICE_TO_DEVICE));
                std::vector<Event> done(kRounds*kServiceLanes);
                for(uint32_t e=0;e<V9_PROBE_EPOCH;++e)for(uint32_t l=0;l<kServiceLanes;++l)done[e*kServiceLanes+l].complete=1;
                ctx.events=reinterpret_cast<uint64_t>(m.Put(done.data(),done.size()*sizeof(Event)));
            }
            bindings[14]=reinterpret_cast<uint64_t>(m.Put(&ctx,sizeof(ctx)));auto bindingsGm=m.Put(bindings,sizeof(bindings));
            if(!finalOnly) {
                Check(v9_init(bindingsGm,stream));Check(probe?v9_probe(bindingsGm,stream):v9_li(bindingsGm,stream));
            }
            Check(aclrtSynchronizeStream(stream));
            m.Dump(ctx.mapping,SIM_BATCH*SIM_HISTORY*4,root+"/prefetch_mapping.bin");
            for(uint32_t b=0;b<SIM_BATCH;++b) {
                m.Dump(sqs[b*2],depth*64,root+"/prefetch_sq"+std::to_string(b)+".bin");
                m.Dump(queues[b*2],sizeof(KvafBatchGetQueueState),root+"/prefetch_queue"+std::to_string(b)+".bin");
                m.Dump(dbs[b*2],128,root+"/prefetch_db"+std::to_string(b)+".bin");
            }
            if(!probe) Check(v9_resolve(bindingsGm,stream));
            if(!probe) Check(v9_fetch(bindingsGm,stream));
            else Check(v9_wait(bindingsGm,stream));
            Check(aclrtSynchronizeStream(stream));
            m.Dump(ctx.owners,SIM_BATCH*128,root+"/final_trace.bin");
            m.Dump(ctx.finalStatus,SIM_BATCH*2*sizeof(KvafBatchGetCoreStatus),root+"/final_status.bin");
            m.Dump(ctx.rows,SIM_BATCH*sizeof(Row),root+"/rows.bin");m.Dump(ctx.progress,SIM_BATCH*4*sizeof(Progress),root+"/progress.bin");
            m.Dump(ctx.mapping,SIM_BATCH*SIM_HISTORY*4,root+"/mapping.bin");m.Dump(ctx.locations,SIM_BATCH*kSlots*4,root+"/locations.bin");
            m.Dump(ctx.events,kRounds*kServiceLanes*sizeof(Event),root+"/events.bin");m.Dump(ctx.count,4,root+"/count.bin");
            m.Dump(ctx.rowStatus,SIM_BATCH*kRowStatusWords*4,root+"/row_status.bin");m.Dump(ctx.sourceBlocks,SIM_BATCH*kSlots*4,root+"/blocks.bin");
            m.Dump(ctx.sourceOffsets,SIM_BATCH*kSlots*4,root+"/offsets.bin");m.Dump(ctx.destinations,SIM_BATCH*kSlots*4,root+"/destinations.bin");
            for(uint32_t b=0;b<SIM_BATCH;++b) {
                m.Dump(sqs[b*2],depth*64,root+"/sq"+std::to_string(b)+".bin");
                m.Dump(queues[b*2],sizeof(KvafBatchGetQueueState),root+"/queue"+std::to_string(b)+".bin");
                m.Dump(dbs[b*2],128,root+"/db"+std::to_string(b)+".bin");
            }
            m.Dump(ctx.owners+kFinalPostOffset,kFinalCores*sizeof(FinalPost),root+"/final_posts.bin");
            for(uint32_t b=0;b<SIM_BATCH;++b)for(uint32_t w=0;w<kFinalWorkers;++w){
                auto q=FinalQueue(b,w);auto suffix=std::to_string(b)+"_"+std::to_string(w)+".bin";
                m.Dump(sqs[q],depth*64,root+"/final_sq"+suffix);
                m.Dump(queues[q],sizeof(KvafBatchGetQueueState),root+"/final_queue"+suffix);
                m.Dump(dbs[q],128,root+"/final_db"+suffix);
            }
            for(uint32_t i:{9,12,13}) m.Dump(bindings[i],sizes[i],root+"/out"+std::to_string(i)+".bin");
        }
        Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(0));Check(aclFinalize());
        std::ofstream(root+"/completed.json") << "{\"status\":\"complete\",\"network\":false}";
        puts("V9 simulation completed; network/CQ/payload NOT executed; independent verification required");return 0;
    } catch(const std::exception& e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
