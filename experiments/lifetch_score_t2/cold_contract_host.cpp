#include <acl/acl.h>
#include "context.h"
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_cold_contract(uint32_t,void*,void*);
static void Check(uint32_t code){if(code)throw std::runtime_error("ACL/launch "+std::to_string(code));}
struct Session {
    uint32_t device; aclrtStream stream=nullptr;
    explicit Session(uint32_t d):device(d){
        Check(aclInit(nullptr));Check(aclrtSetDevice(device));Check(aclrtCreateStream(&stream));
    }
    ~Session(){
        if(stream)aclrtDestroyStream(stream);
        aclrtResetDevice(device);aclFinalize();
    }
};
struct Memory {
    std::vector<void*> buffers;
    template<class T> uint64_t Put(const std::vector<T>& values){
        void* p=nullptr;auto n=values.size()*sizeof(T);Check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));
        buffers.push_back(p);Check(aclrtMemcpy(p,n,values.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(p);
    }
    template<class T> std::vector<T> Get(uint64_t p,size_t n){
        std::vector<T> v(n);Check(aclrtMemcpy(v.data(),n*sizeof(T),(void*)p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;
    }
    ~Memory(){for(auto p:buffers)aclrtFree(p);}
};
int main(int argc,char** argv){
    try {
        uint32_t b=argc>1?std::stoul(argv[1]):1;if(b!=1&&b!=16)throw std::runtime_error("batch must be 1 or 16");
        uint32_t device=argc>2?std::stoul(argv[2]):0;
        Session session(device);auto stream=session.stream;
        {
            Memory memory;constexpr uint32_t h=32768,r=16384,prefix=4*2048*6+1024;
            std::vector<uint16_t> scores(b*4*h),storage(b*prefix/2+b*4*h/8+64,0xa5a5);
            for(uint32_t owner=0;owner<b;++owner)for(uint32_t q=0;q<4;++q){
                auto theta=uint16_t(15000+q*1000+owner*7);
                storage[owner*prefix/2+4*2048*3+q*128]=theta;
                for(uint32_t token=0;token<h;++token)
                    scores[(owner*4+q)*h+token]=uint16_t(theta+int((token*17+q*3+owner)%11)-5);
            }
            rotating::Context context{};context.shape={};
            context.shape.batch=b;context.shape.history=h;context.shape.resident=r;context.shape.indexHeads=32;
            context.shape.aicCount=32;context.shape.splitQ=1;context.shape.rounds=2;context.shape.tokenBytes=656;
            Check(rotating::PrepareShape(&context.shape));context.progress=memory.Put(std::vector<uint32_t>(b*128));
            std::vector<uint64_t> args(16);auto dummy=memory.Put(std::vector<uint8_t>(65536));
            for(auto& address:args)address=dummy;
            args[5]=memory.Put(std::vector<uint32_t>(b,4));args[6]=memory.Put(std::vector<uint32_t>(b,h));
            args[12]=memory.Put(scores);args[13]=memory.Put(storage);args[14]=memory.Put(std::vector<rotating::Context>{context});
            auto bindings=memory.Put(args);Check(aclrtlaunch_cold_contract(b*2,stream,(void*)bindings));
            Check(aclrtSynchronizeStream(stream));auto result=memory.Get<uint16_t>(args[13],storage.size());
            auto hits=reinterpret_cast<const uint8_t*>(result.data())+b*prefix;
            uint32_t errors=0;
            for(uint32_t owner=0;owner<b;++owner)for(uint32_t q=0;q<4;++q)for(uint32_t token=r;token<h;++token){
                auto actual=(hits[(owner*4+q)*h/4+token/4]>>((token%4)*2))&1;
                auto expected=scores[(owner*4+q)*h+token]>15000+q*1000+owner*7?1U:0U;
                if(actual!=expected&&errors++<8)std::cerr<<"bitmap mismatch B="<<owner<<" Q="<<q<<" H="<<token<<" actual="<<actual<<" expected="<<expected<<"\n";
            }
            for(uint32_t i=0;i<b*prefix/2;++i)if(result[i]!=storage[i])throw std::runtime_error("threshold/prefix overwritten");
            for(size_t i=storage.size()-64;i<storage.size();++i)if(result[i]!=storage[i])throw std::runtime_error("guard overwritten");
            if(memory.Get<uint16_t>(args[12],scores.size())!=scores)throw std::runtime_error("canonical scores modified");
            if(errors)throw std::runtime_error("bitmap errors="+std::to_string(errors));
            std::cout<<"COLD_CONTRACT_PASS B"<<b<<" Q4 H32768 tiles="<<b*128<<" strict_gt=true guards=true scope=consumer_bitmap_only\n";
        }
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
