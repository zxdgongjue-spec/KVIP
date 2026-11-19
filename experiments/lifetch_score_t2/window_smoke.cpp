// Window direct-producer contract: exact positive FP8 operands, canonical B/Q/H
// output and guard checks. This is not the full URMA LIFetch performance test.
#include <acl/acl.h>
#include <algorithm>
#include "context.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
extern "C" uint32_t aclrtlaunch_v9_li_kernel_score_window_t2(uint32_t,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t);
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
        if(argc!=4 && argc!=6)throw std::runtime_error("window_smoke DEVICE BATCH PARTS [HISTORY RESIDENT]");
        uint32_t device=std::stoul(argv[1]),batch=std::stoul(argv[2]),parts=std::stoul(argv[3]);
        if(!batch||batch>16||!parts||parts>32||batch*parts>32)throw std::runtime_error("invalid B/P budget");
        Check(aclInit(nullptr));Check(aclrtSetDevice(device));aclrtStream stream=nullptr;Check(aclrtCreateStream(&stream));
        {
            Memory memory;constexpr uint32_t g=32;
            const uint32_t h=argc==6?std::stoul(argv[4]):32768,r=argc==6?std::stoul(argv[5]):16384;
            if(r<2048||r%512||h<=r||h%256||r/256<parts||(h-r)/256<parts)throw std::runtime_error("invalid window probe geometry");
            std::vector<uint8_t> query(batch*4*g*128,0x38),key(batch*h*128+128,0x5a);
            for(uint32_t b=0;b<batch;++b)for(uint32_t token=0;token<h;++token)for(uint32_t d=0;d<128;++d)
                key[(b*h+token)*128+d]=d<(token+b*7)%129?0x38:0;
            std::vector<uint16_t> weights(batch*4*g),scores(batch*4*h+128,0x5a5a);
            for(uint32_t row=0;row<batch*4;++row)for(uint32_t head=0;head<g;++head)weights[row*g+head]=0x3f80+(row%4)*128;
            std::vector<int32_t> pages(batch*h/128);for(uint32_t p=0;p<pages.size();++p)pages[p]=p;
            rotating::Context context{};
            context.shape.batch=batch;context.shape.history=h;context.shape.resident=r;
            context.shape.indexHeads=g;context.shape.aicCount=32;context.shape.splitQ=1;
            context.shape.rounds=2;context.shape.tokenBytes=656;Check(rotating::PrepareShape(&context.shape));
            context.progress=memory.Put(std::vector<uint32_t>(batch*128));
            std::vector<uint8_t> hits(batch*4*h/4+128,0x5a),prefix(batch*(4*2048*6+1024)+hits.size());
            std::vector<uint16_t> theta(batch*4);
            for(uint32_t row=0;row<batch*4;++row){
                float value=float((1U<<(row%4))*g*(40+row/4));uint32_t bits;std::memcpy(&bits,&value,4);
                theta[row]=uint16_t((bits>>16)^0x8000);
                for(uint32_t i=0;i<128;++i)std::memcpy(prefix.data()+(row/4)*(4*2048*6+1024)+4*2048*6+(row%4)*256+i*2,&theta[row],2);
            }
            std::copy(hits.begin(),hits.end(),prefix.begin()+batch*(4*2048*6+1024));
            std::vector<uint64_t> args(16);
            args[0]=memory.Put(query);args[1]=memory.Put(key);args[2]=memory.Put(weights);
            args[3]=memory.Put(std::vector<float>(batch*4*g,1));args[4]=memory.Put(std::vector<float>(batch*h,1));
            args[5]=memory.Put(std::vector<uint32_t>(batch,4));args[6]=memory.Put(std::vector<uint32_t>(batch,h));
            args[7]=memory.Put(pages);args[9]=memory.Put(std::vector<int32_t>(batch*4*2048,-1));
            args[12]=memory.Put(scores);args[13]=memory.Put(prefix);
            context.hits=args[13]+batch*(4*2048*6+1024);
            args[14]=memory.Put(std::vector<rotating::Context>{context});
            auto bindings=memory.Put(args);
            Check(aclrtlaunch_v9_li_kernel_score_window_t2(batch*parts,stream,(void*)bindings,parts,0,r,0));
            Check(aclrtlaunch_v9_li_kernel_score_window_t2(batch*parts,stream,(void*)bindings,parts,r,h,1));
            Check(aclrtSynchronizeStream(stream));
            auto result=memory.Get<uint16_t>(args[12],scores.size());
            for(uint32_t q=0;q<batch*4;++q)for(uint32_t token=0;token<h;++token){
                float score=float((1U<<(q%4))*g*((token+(q/4)*7)%129));uint32_t bits;std::memcpy(&bits,&score,4);
                uint16_t expected=uint16_t((bits>>16)^0x8000);
                if(result[q*h+token]!=expected)throw std::runtime_error("canonical score mismatch Q="+std::to_string(q)+" H="+std::to_string(token)+" value="+std::to_string(result[q*h+token])+" expected="+std::to_string(expected));
            }
            for(size_t i=batch*4*h;i<result.size();++i)if(result[i]!=0x5a5a)throw std::runtime_error("score guard overwritten");
            if(memory.Get<uint8_t>(args[1],key.size())!=key)throw std::runtime_error("key input/guard changed");
            auto bitmap=memory.Get<uint8_t>(context.hits,hits.size());
            for(uint32_t q=0;q<batch*4;++q)for(uint32_t token=0;token<h;token+=4){
                uint8_t expected=0x5a;
                if(token>=r){expected=0;for(uint32_t j=0;j<4;++j)if(result[q*h+token+j]>theta[q])expected|=1U<<(2*j);}
                if(bitmap[q*h/4+token/4]!=expected)throw std::runtime_error("bitmap mismatch Q="+std::to_string(q)+" token="+std::to_string(token));
            }
            for(size_t i=batch*4*h/4;i<bitmap.size();++i)if(bitmap[i]!=0x5a)throw std::runtime_error("bitmap guard overwritten");
            auto savedPrefix=memory.Get<uint8_t>(args[13],batch*(4*2048*6+1024));
            if(!std::equal(savedPrefix.begin(),savedPrefix.end(),prefix.begin()))throw std::runtime_error("theta changed");
            auto ready=memory.Get<uint32_t>(context.progress,batch*128);
            for(auto value:ready)if(value)throw std::runtime_error("producer published unjoined ready cursor");
            std::cout<<"WINDOW_PRODUCER_PASS B"<<batch<<" Q4 H"<<h<<" P"<<parts<<" score_keys="<<batch*4*h
                <<" predicate_keys="<<batch*4*(h-r)<<" guards=true scope=score_predicate_only\n";
        }
        Check(aclrtDestroyStream(stream));Check(aclrtResetDevice(device));Check(aclFinalize());return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
