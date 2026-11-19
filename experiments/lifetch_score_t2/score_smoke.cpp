// Score-producer Model smoke test: exact positive FP8 operands, canonical Q/H
// output and guard checks. This is not the full URMA LIFetch performance test.
#include <acl/acl.h>
#include "context.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_li_kernel_score_t2(uint32_t,void*,void*,uint32_t);
static void Check(uint32_t code){if(code)throw std::runtime_error("ACL/launch "+std::to_string(code));}
struct Memory {
    std::vector<void*> buffers;
    template<class T> uint64_t Put(const std::vector<T>& values){
        void* buffer=nullptr;size_t bytes=values.size()*sizeof(T);
        Check(aclrtMalloc(&buffer,bytes,ACL_MEM_MALLOC_HUGE_FIRST));buffers.push_back(buffer);
        Check(aclrtMemcpy(buffer,bytes,values.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));return uint64_t(buffer);
    }
    template<class T> std::vector<T> Get(uint64_t address,size_t count){
        std::vector<T> values(count);Check(aclrtMemcpy(values.data(),count*sizeof(T),(void*)address,count*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return values;
    }
    ~Memory(){for(auto p:buffers)aclrtFree(p);}
};
int main(int argc,char** argv){
    try {
        if(argc!=2)throw std::runtime_error("score_smoke PARTS (1..32)");
        uint32_t parts=std::stoul(argv[1]);if(!parts||parts>32)throw std::runtime_error("invalid parts");
        Check(aclInit(nullptr));Check(aclrtSetDevice(0));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
        {
            Memory memory;constexpr uint32_t h=131072,g=32;
            std::vector<uint8_t> query(4*g*128,0x38),key(h*128+128,0x5a);
            for(uint32_t token=0;token<h;++token)for(uint32_t d=0;d<128;++d)key[token*128+d]=d<token%129?0x38:0;
            std::vector<uint16_t> weights(4*g,0x3f80),scores(4*h+128,0x5a5a);
            std::vector<int32_t> pages(h/128);for(uint32_t p=0;p<pages.size();++p)pages[p]=p;
            rotating::Context context{};
            context.shape.batch=1;context.shape.history=h;context.shape.resident=16384;
            context.shape.indexHeads=g;context.shape.aicCount=32;context.shape.splitQ=1;
            context.shape.rounds=2;context.shape.tokenBytes=656;Check(rotating::PrepareShape(&context.shape));
            context.progress=memory.Put(std::vector<uint32_t>(128));
            std::vector<uint64_t> args(16);
            args[0]=memory.Put(query);args[1]=memory.Put(key);args[2]=memory.Put(weights);
            args[3]=memory.Put(std::vector<float>(4*g,1));args[4]=memory.Put(std::vector<float>(h,1));
            args[5]=memory.Put(std::vector<uint32_t>{4});args[6]=memory.Put(std::vector<uint32_t>{h});
            args[7]=memory.Put(pages);args[9]=memory.Put(std::vector<int32_t>(4*2048,-1));
            args[12]=memory.Put(scores);args[13]=memory.Put(std::vector<uint8_t>(65536));
            args[14]=memory.Put(std::vector<rotating::Context>{context});
            auto bindings=memory.Put(args);
            Check(aclrtlaunch_v9_li_kernel_score_t2(parts,stream,(void*)bindings,parts));
            Check(aclrtSynchronizeStream(stream));
            auto result=memory.Get<uint16_t>(args[12],scores.size());
            for(uint32_t q=0;q<4;++q)for(uint32_t token=0;token<h;++token){
                float score=float(g*(token%129));uint32_t bits;std::memcpy(&bits,&score,4);
                uint16_t expected=uint16_t((bits>>16)^0x8000);
                if(result[q*h+token]!=expected)throw std::runtime_error("canonical score mismatch Q="+std::to_string(q)+" H="+std::to_string(token)+" value="+std::to_string(result[q*h+token])+" expected="+std::to_string(expected));
            }
            for(size_t i=4*h;i<result.size();++i)if(result[i]!=0x5a5a)throw std::runtime_error("score guard overwritten");
            if(memory.Get<uint8_t>(args[1],key.size())!=key)throw std::runtime_error("key input/guard changed");
            std::cout<<"SCORE_MODEL_PASS B1 Q4 H131072 P"<<parts<<" keys="<<4*h<<" guards=true scope=score_producer_only\n";
        }
        Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(0));Check(aclFinalize());return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
