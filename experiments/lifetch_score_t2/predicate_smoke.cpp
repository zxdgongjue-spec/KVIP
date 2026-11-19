// Exact Q4 signed-key predicate and ping-pong check through the real consumer.
#include <acl/acl.h>
#include "context.h"
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_predicate_probe(uint32_t,void*,void*);
static void Check(uint32_t c){if(c)throw std::runtime_error("ACL "+std::to_string(c));}
struct Memory {
    std::vector<void*> blocks;
    template<class T> uint64_t Put(const std::vector<T>& v){void* p=nullptr;size_t n=v.size()*sizeof(T);
        Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));blocks.push_back(p);
        Check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);}
    template<class T> std::vector<T> Get(uint64_t p,size_t n){std::vector<T> v(n);
        Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
    ~Memory(){for(void* p:blocks)aclrtFree(p);}
};
int main(int argc,char** argv){try{
    uint32_t device=argc>1?std::stoul(argv[1]):0;
    Check(aclInit(nullptr));Check(aclrtSetDevice(device));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
    {Memory m;constexpr uint32_t H=131072,R=16384,Q=4,prefixBytes=Q*2048*6+Q*256;
    std::vector<uint16_t> scores(Q*H+128,0x5a5a);
    const uint16_t limits[Q]={15088,15236,32768,49152};
    for(uint32_t q=0;q<Q;++q)for(uint32_t t=0;t<H;++t)
        scores[q*H+t]= t%11==0 ? limits[q] : uint16_t((t*193+q*7919+(t/128)*37)%65536);
    std::vector<uint16_t> prefix((prefixBytes+Q*H/4+128)/2,0x5a5a);
    for(uint32_t q=0;q<Q;++q)prefix[Q*2048*3+q*128]=limits[q];
    rotating::Context ctx{};ctx.shape.batch=1;ctx.shape.history=H;ctx.shape.resident=R;
    ctx.shape.indexHeads=32;ctx.shape.aicCount=32;ctx.shape.splitQ=1;ctx.shape.rounds=2;
    ctx.shape.tokenBytes=656;Check(rotating::PrepareShape(&ctx.shape));
    ctx.progress=m.Put(std::vector<uint32_t>(128));
    std::vector<uint64_t> args(16);auto unused=m.Put(std::vector<uint8_t>(65536));
    for(uint32_t i=0;i<12;++i)args[i]=unused;
    args[5]=m.Put(std::vector<uint32_t>{4});args[6]=m.Put(std::vector<uint32_t>{H});
    args[12]=m.Put(scores);args[13]=m.Put(prefix);args[14]=m.Put(std::vector<rotating::Context>{ctx});
    auto bindings=m.Put(args);
    Check(aclrtlaunch_v9_predicate_probe(2,stream,(void*)bindings));Check(aclrtSynchronizeStream(stream));
    auto hits=m.Get<uint8_t>(args[13],prefix.size()*2);
    uint32_t mismatch=0;
    for(uint32_t q=0;q<Q;++q)for(uint32_t t=R;t<H;++t){
        bool actual=(hits[prefixBytes+q*H/4+t/4]>>(2*(t%4)))&1;
        if(actual!=(scores[q*H+t]>limits[q])){
            if(mismatch<8)std::cerr<<"Q="<<q<<" token="<<t<<" score="<<scores[q*H+t]<<" threshold="<<limits[q]<<" hit="<<actual<<"\n";
            ++mismatch;
        }
    }
    if(m.Get<uint16_t>(args[12],scores.size())!=scores)throw std::runtime_error("score input/guard changed");
    for(uint32_t i=prefixBytes+Q*H/4;i<hits.size();++i)
        if(hits[i]!=0x5a)throw std::runtime_error("predicate guard changed");
    if(mismatch)throw std::runtime_error("predicate mismatches="+std::to_string(mismatch));
    std::cout<<"PREDICATE_MODEL_PASS B1 Q4 H131072 T2 keys="<<Q*(H-R)<<" ties=strict_gt guards=true scope=consumer_only\n";
    }
    Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(device));Check(aclFinalize());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
