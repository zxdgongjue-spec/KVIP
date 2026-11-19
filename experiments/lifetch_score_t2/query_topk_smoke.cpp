// Real incremental TopK/ranked export control; no posting, URMA or Final.
#include <acl/acl.h>
#include "context.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_li_window_topk(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_window_topk_serial_probe(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t);
static void Check(uint32_t n){if(n)throw std::runtime_error("ACL/launch "+std::to_string(n));}
struct Memory {
    std::vector<void*> buffers;
    template<class T> uint64_t Put(const std::vector<T>& v){
        void* p=nullptr;auto bytes=v.size()*sizeof(T);Check(aclrtMalloc(&p,bytes,ACL_MEM_MALLOC_HUGE_FIRST));buffers.push_back(p);
        Check(aclrtMemcpy(p,bytes,v.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);
    }
    template<class T> std::vector<T> Get(uint64_t p,size_t n){
        std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    ~Memory(){for(auto p:buffers)aclrtFree(p);}
};
template<class T> static void Write(std::vector<uint8_t>& v,size_t offset,T value){std::memcpy(v.data()+offset,&value,sizeof(T));}
template<class T> static T Read(const std::vector<uint8_t>& v,size_t offset){T value;std::memcpy(&value,v.data()+offset,sizeof(T));return value;}
int main(int argc,char** argv){try{
    if(argc!=4)throw std::runtime_error("query_topk_smoke DEVICE BATCH serial|query");
    const uint32_t device=std::stoul(argv[1]),b=std::stoul(argv[2]);const std::string route=argv[3];
    if((b!=2&&b!=4)||(route!="serial"&&route!="query"))throw std::runtime_error("invalid probe route/B");
    constexpr uint32_t h=8192,r=2048,k=2048,prefixBytes=4*k*6+1024,guard=128;
    rotating::RuntimeShape shape{};shape.batch=b;shape.history=h;shape.resident=r;shape.indexHeads=32;
    shape.aicCount=32;shape.splitQ=1;shape.rounds=4;shape.tokenBytes=656;Check(rotating::PrepareShape(&shape));
    const auto half=rotating::SequenceHalfWindow(shape),cap=rotating::SequenceWindowEnd(h,r,half,1);
    Check(aclInit(nullptr));Check(aclrtSetDevice(device));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
    {
    Memory memory;rotating::Context ctx{};ctx.shape=shape;ctx.windowLimit=256;ctx.prefetchLimit=1024;
    std::vector<uint16_t> scores(b*4*h+guard,0x5a5a);
    for(uint32_t q=0;q<b*4;++q)for(uint32_t t=0;t<h;++t)scores[q*h+t]=0xc000+((t*37+q*151)%8192);
    auto score=memory.Put(scores);
    std::vector<uint8_t> bitmap(b*4*h/4+guard,0x5a);
    for(uint32_t q=0;q<b*4;++q){uint16_t cutoff=*std::min_element(scores.begin()+q*h,scores.begin()+q*h+r);
        for(uint32_t t=0;t<h;t+=4){uint8_t mask=0;for(uint32_t j=0;j<4;++j)if(t+j>=r&&scores[q*h+t+j]>cutoff)mask|=1U<<(2*j);
            bitmap[q*h/4+t/4]=mask;}}
    ctx.hits=memory.Put(bitmap);ctx.progress=memory.Put(std::vector<uint32_t>(b*128));
    std::vector<rotating::Row> rows(b);ctx.rows=memory.Put(rows);
    std::vector<rotating::Event> events(shape.rounds*shape.serviceLanes);
    for(auto& e:events)e.complete=1;ctx.events=memory.Put(events);
    ctx.locations=memory.Put(std::vector<uint32_t>(b*8192+guard,0x5a5a5a5a));
    const uint32_t coveredBytes=(h+255)/256*32;
    std::vector<uint8_t> state(b*(512+prefixBytes+coveredBytes)+guard,0);
    std::fill(state.end()-guard,state.end(),0x5a);
    for(uint32_t q=0;q<b*4;++q){
        Write<uint64_t>(state,q*128,score+q*h*2);
        Write<uint64_t>(state,q*128+8,uint64_t(r)|(uint64_t(r)<<32));
        for(uint32_t t=0;t<r;++t){
            auto base=b*512+(q/4)*prefixBytes;
            Write<uint16_t>(state,base+(q%4)*k*2+t*2,scores[q*h+t]);
            Write<uint32_t>(state,base+4*k*2+(q%4)*k*4+t*4,t|0x80000000U);
        }
    }
    ctx.incremental=memory.Put(state);auto context=memory.Put(std::vector<rotating::Context>{ctx});
    std::vector<uint64_t> args(16);args[11]=memory.Put(std::vector<uint64_t>(b*4*16,0));args[12]=score;args[14]=context;auto bindings=memory.Put(args);
    if(route=="serial")Check(aclrtlaunch_v9_li_window_topk_serial_probe(shape.serviceAics*2,stream,(void*)bindings,1,r+half,cap));
    else Check(aclrtlaunch_v9_li_window_topk(b*4,stream,(void*)bindings,1,r+half,cap));
    Check(aclrtSynchronizeStream(stream));
    auto out=memory.Get<uint8_t>(ctx.incremental,state.size());auto jobs=memory.Get<uint32_t>(ctx.locations,b*8192+guard);
    uint32_t totalCandidates=0;uint64_t candidateHash=1469598103934665603ULL;
    for(uint32_t q=0;q<b*4;++q){
        auto base=b*512+(q/4)*prefixBytes;
        std::vector<std::pair<uint16_t,uint32_t>> oracle;
        for(uint32_t t=0;t<cap;++t)oracle.push_back({scores[q*h+t],t});
        std::sort(oracle.begin(),oracle.end(),std::greater<std::pair<uint16_t,uint32_t>>());oracle.resize(k);
        std::vector<std::pair<uint16_t,uint32_t>> actual;
        for(uint32_t i=0;i<k;++i){auto id=Read<uint32_t>(out,base+4*k*2+(q%4)*k*4+i*4)&0x3fffffff;
            actual.push_back({Read<uint16_t>(out,base+(q%4)*k*2+i*2),id});}
        std::sort(actual.begin(),actual.end(),std::greater<std::pair<uint16_t,uint32_t>>());
        if(actual!=oracle)throw std::runtime_error("TopK value/ID mismatch Q="+std::to_string(q));
        const auto count=Read<uint64_t>(out,q*128+3*8);if(count>64)throw std::runtime_error("rank quota overflow");
        for(uint32_t i=0;i<count;++i){auto id=jobs[q*k+i];
            if(id<r||id>=cap||!std::binary_search(oracle.begin(),oracle.end(),std::make_pair(scores[q*h+id],id),std::greater<std::pair<uint16_t,uint32_t>>()))
                throw std::runtime_error("invalid ranked candidate");}
        std::vector<uint32_t> candidateIds(jobs.begin()+q*k,jobs.begin()+q*k+count);
        std::sort(candidateIds.begin(),candidateIds.end());
        candidateHash=(candidateHash^count)*1099511628211ULL;
        for(auto id:candidateIds)candidateHash=(candidateHash^id)*1099511628211ULL;
        auto cursor=Read<uint64_t>(out,q*128+8);
        if(uint32_t(cursor)!=r+half||uint32_t(cursor>>32)!=cap)throw std::runtime_error("cursor mismatch");
        if(Read<uint64_t>(out,q*128)!=score+q*h*2)throw std::runtime_error("source metadata overwritten");
        if(Read<uint16_t>(out,q*128+16)!=oracle.back().first)throw std::runtime_error("cutoff mismatch");
        totalCandidates+=count;
    }
    for(size_t i=state.size()-guard;i<state.size();++i)if(out[i]!=0x5a)throw std::runtime_error("state guard changed");
    for(size_t i=b*(512+prefixBytes);i<b*(512+prefixBytes+coveredBytes);++i)if(out[i])throw std::runtime_error("covered bitmap changed before post");
    auto rowOut=memory.Get<rotating::Row>(ctx.rows,b);if(std::memcmp(rowOut.data(),rows.data(),b*sizeof(rotating::Row)))throw std::runtime_error("request row changed");
    if(memory.Get<uint16_t>(score,scores.size())!=scores||memory.Get<uint8_t>(ctx.hits,bitmap.size())!=bitmap)throw std::runtime_error("inputs changed");
    for(size_t i=b*8192;i<jobs.size();++i)if(jobs[i]!=0x5a5a5a5a)throw std::runtime_error("jobs guard changed");
    auto clocks=memory.Get<uint64_t>(args[11],b*4*16);
    uint64_t begin=UINT64_MAX,end=0,maxLane=0;
    uint32_t active=route=="query"?b*4:std::min(b,shape.serviceLanes);
    for(uint32_t i=0;i<active;++i){begin=std::min(begin,clocks[i*16]);end=std::max(end,clocks[i*16+1]);
        maxLane=std::max(maxLane,clocks[i*16+1]-clocks[i*16]);}
    std::cout<<"QUERY_TOPK_PASS B"<<b<<" Q4 H"<<h<<" R"<<r<<" W4 route="<<route<<" topk_keys="<<b*4*k
             <<" candidate_count="<<totalCandidates<<" candidate_hash="<<candidateHash
             <<" work_span_cycles="<<end-begin<<" max_lane_cycles="<<maxLane
             <<" scope=incremental_topk_rank_only\n";
    }
    Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(device));Check(aclFinalize());
    std::cout<<"QUERY_TOPK_CLEANUP_PASS stream_destroyed=true device_reset=true acl_finalized=true\n"<<std::flush;
#if defined(KVIP_CANN_MODEL) && KVIP_CANN_MODEL
    // CANN 9.1 pure-AIV Model repeats teardown in an atexit destructor after
    // successful explicit ACL cleanup, causing a double free. All allocations
    // and runtime resources above are already released. Skip only that second
    // process-exit teardown; kernel errors and ACL failures still fail normally.
    std::cout<<"QUERY_TOPK_MODEL_EXIT skip_duplicate_atexit=true\n"<<std::flush;
    std::_Exit(0);
#endif
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
