// Real full-H TopK and original classification on identical canonical scores.
// Score mode additionally checks the actual aligned FP8 history producers.
#include <acl/acl.h>
#include "context.h"
#include "pipeline_types.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_li_full_topk(uint32_t,void*,void*,uint32_t);
extern "C" uint32_t aclrtlaunch_v9_li_kernel_score_aligned(uint32_t,void*,void*,uint32_t);
static void Check(uint32_t n){if(n)throw std::runtime_error("ACL/launch "+std::to_string(n));}
struct Memory {
    std::vector<void*> buffers;
    template<class T> uint64_t Put(const std::vector<T>& v){
        void* p=nullptr;size_t n=v.size()*sizeof(T);Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));buffers.push_back(p);
        Check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);
    }
    template<class T> std::vector<T> Get(uint64_t p,size_t n){
        std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    ~Memory(){for(auto p:buffers)aclrtFree(p);}
};
int main(int argc,char** argv){try{
    if(argc!=3)throw std::runtime_error("aligned_smoke BATCH numeric|classify|score");
    uint32_t b=std::stoul(argv[1]);std::string route=argv[2];
    if((b!=1&&b!=2&&b!=4&&b!=8&&b!=16)||(route!="numeric"&&route!="classify"&&route!="score"))throw std::runtime_error("invalid B/route");
    const bool scoreProducer=route=="score",classify=route!="numeric";
    const uint32_t h=scoreProducer?8192:32768,r=scoreProducer?2048:16384,k=2048,guard=128;
    rotating::Context ctx{};auto& s=ctx.shape;s.batch=b;s.history=h;s.resident=r;s.indexHeads=32;
    s.aicCount=32;s.splitQ=1;s.rounds=2;s.tokenBytes=656;Check(rotating::PrepareShape(&s));
    Check(aclInit(nullptr));Check(aclrtSetDevice(0));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
    {
    Memory memory;
    std::vector<uint16_t> scores(b*4*h+guard,0x5a5a);
    for(uint32_t q=0;q<b*4;++q)for(uint32_t t=0;t<h;++t){
        if(scoreProducer){float value=32.0f*(t%129);uint32_t bits;std::memcpy(&bits,&value,4);scores[q*h+t]=uint16_t((bits>>16)^0x8000);}
        else scores[q*h+t]=0x8000+((t*37+q*151)%h);
    }
    auto input=scores;if(scoreProducer)std::fill(input.begin(),input.begin()+b*4*h,0x5a5a);
    std::vector<uint64_t> args(16);args[12]=memory.Put(input);
    args[9]=memory.Put(std::vector<int32_t>(b*4*k+guard,-1431655766));
    args[11]=memory.Put(std::vector<uint64_t>(b*4*16,0));
    std::vector<rotating::Row> rows(b);ctx.rows=memory.Put(rows);
    ctx.mapping=memory.Put(std::vector<int32_t>(b*h+guard,-1));
    std::vector<uint32_t> lru(b*k);for(uint32_t row=0;row<b;++row)std::iota(lru.begin()+row*k,lru.begin()+(row+1)*k,0);
    ctx.lruSlots=memory.Put(lru);
    std::vector<uint32_t> owners(b);std::iota(owners.begin(),owners.end(),1);ctx.rowOwners=memory.Put(owners);
    std::vector<uint32_t> valid((b+1)*r+guard,0);for(uint32_t row=1;row<=b;++row)for(uint32_t t=0;t<r;++t)valid[row*r+t]=t%7!=0;
    ctx.hotValid=memory.Put(valid);ctx.reservedHot=memory.Put(std::vector<int32_t>(b*r+guard,-1));
    ctx.locations=memory.Put(std::vector<uint32_t>(b*4*k+guard,0x5a5a5a5a));
    std::vector<rotating::Event> events(s.rounds*s.serviceLanes);ctx.events=memory.Put(events);
    ctx.progress=memory.Put(std::vector<uint32_t>(b*128));
    pipeline_ab::Args work{};work.queryBitmaps=memory.Put(std::vector<uint32_t>(b*4*(r/32)+guard,0x5a5a5a5a));
    work.queryErrors=memory.Put(std::vector<uint32_t>(b*4*32+guard,0x5a5a5a5a));
    args[15]=memory.Put(std::vector<pipeline_ab::Args>{work});
    if(scoreProducer){
        args[0]=memory.Put(std::vector<uint8_t>(b*4*32*128,0x38));
        std::vector<uint8_t> keys(b*h*128+guard,0x5a);
        for(uint32_t row=0;row<b;++row)for(uint32_t t=0;t<h;++t)for(uint32_t d=0;d<128;++d)keys[(row*h+t)*128+d]=d<t%129?0x38:0;
        args[1]=memory.Put(keys);args[2]=memory.Put(std::vector<uint16_t>(b*4*32,0x3f80));
        args[3]=memory.Put(std::vector<float>(b*4*32,1));args[4]=memory.Put(std::vector<float>(b*h,1));
        args[5]=memory.Put(std::vector<uint32_t>(b,4));args[6]=memory.Put(std::vector<uint32_t>(b,h));
        std::vector<int32_t> pages(b*h/128);std::iota(pages.begin(),pages.end(),0);args[7]=memory.Put(pages);
    }
    args[14]=memory.Put(std::vector<rotating::Context>{ctx});auto bindings=memory.Put(args);
    if(scoreProducer){
#if ALIGNED_SCORE_MODEL
        uint32_t p=b==1?16:32/b;Check(aclrtlaunch_v9_li_kernel_score_aligned(b*p,stream,(void*)bindings,p));
#else
        throw std::runtime_error("Compile ALIGNED_SCORE_MODEL=ON for score route");
#endif
    }
    Check(aclrtlaunch_v9_li_full_topk(b*4,stream,(void*)bindings,classify));Check(aclrtSynchronizeStream(stream));
    if(memory.Get<uint16_t>(args[12],scores.size())!=scores)throw std::runtime_error("score producer/input/guard mismatch");
    auto chosen=memory.Get<int32_t>(args[9],b*4*k+guard);
    auto locations=memory.Get<uint32_t>(ctx.locations,b*4*k+guard);
    auto bitmap=memory.Get<uint32_t>(work.queryBitmaps,b*4*r/32+guard);
    auto errors=memory.Get<uint32_t>(work.queryErrors,b*4*32+guard);
    uint64_t hash=1469598103934665603ULL;
    for(uint32_t q=0;q<b*4;++q){
        std::vector<uint16_t> sorted(scores.begin()+q*h,scores.begin()+(q+1)*h);std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());
        std::vector<bool> seen(h,false);std::vector<uint16_t> values;std::vector<uint32_t> bits(r/32,0);
        for(uint32_t i=0;i<k;++i){auto id=chosen[q*k+i];if(id<0||uint32_t(id)>=h||seen[id])throw std::runtime_error("invalid/duplicate TopK ID");
            seen[id]=true;values.push_back(scores[q*h+id]);hash=(hash^uint32_t(id))*1099511628211ULL;
            uint32_t slot=uint32_t(id)<r&&valid[(q/4+1)*r+id]?uint32_t(id):UINT32_MAX;
            if(classify&&locations[q*k+i]!=slot)throw std::runtime_error("classification slot mismatch");
            if(slot<r)bits[slot/32]|=1U<<(slot%32);
        }
        std::sort(values.begin(),values.end(),std::greater<uint16_t>());
        if(!std::equal(values.begin(),values.end(),sorted.begin()))throw std::runtime_error("TopK cutoff/value mismatch");
        if(classify){if(errors[q*32])throw std::runtime_error("classification error");
            if(!std::equal(bits.begin(),bits.end(),bitmap.begin()+q*(r/32)))throw std::runtime_error("protection bitmap mismatch");}
    }
    for(uint32_t i=b*4*k;i<chosen.size();++i)if(chosen[i]!=-1431655766||locations[i]!=0x5a5a5a5a)throw std::runtime_error("output guard changed");
    for(uint32_t i=b*4*r/32;i<bitmap.size();++i)if(bitmap[i]!=0x5a5a5a5a)throw std::runtime_error("bitmap guard changed");
    for(uint32_t i=b*4*32;i<errors.size();++i)if(errors[i]!=0x5a5a5a5a)throw std::runtime_error("error guard changed");
    if(!classify){if(std::any_of(locations.begin(),locations.end(),[](uint32_t x){return x!=0x5a5a5a5a;})||
        std::any_of(bitmap.begin(),bitmap.end(),[](uint32_t x){return x!=0x5a5a5a5a;}))throw std::runtime_error("numeric route wrote classification");}
    auto rowOut=memory.Get<rotating::Row>(ctx.rows,b);auto eventOut=memory.Get<rotating::Event>(ctx.events,events.size());
    if(std::memcmp(rows.data(),rowOut.data(),b*sizeof(rotating::Row))||std::memcmp(events.data(),eventOut.data(),events.size()*sizeof(rotating::Event)))throw std::runtime_error("unexpected Service state writes");
    auto clocks=memory.Get<uint64_t>(args[11],b*4*16);uint64_t first=UINT64_MAX,last=0;
    for(uint32_t q=0;q<b*4;++q){first=std::min(first,clocks[q*16]);last=std::max(last,clocks[q*16+1]);}
    if(last<=first)throw std::runtime_error("invalid clocks");
    std::cout<<"ALIGNED_PASS B"<<b<<" route="<<route<<" H="<<h<<" R="<<r<<" selected="<<b*4*k<<" hash="<<hash<<" work_span_cycles="<<last-first<<" guards=true no_service=true\n";
    }
    Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(0));Check(aclFinalize());
    std::cout<<"ALIGNED_CLEANUP_PASS\n"<<std::flush;
#if KVIP_CANN_MODEL
    std::cout<<"ALIGNED_MODEL_EXIT skip_duplicate_atexit=true\n"<<std::flush;std::_Exit(0);
#else
    return 0;
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
