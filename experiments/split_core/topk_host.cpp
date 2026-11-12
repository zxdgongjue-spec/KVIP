#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_split_topk_local(uint32_t,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,uint32_t,void*);
extern "C" uint32_t aclrtlaunch_split_topk_merge(uint32_t,void*,void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,void*);
void check(uint32_t x){if(x)throw std::runtime_error("ACL="+std::to_string(x));}
template<class T>std::vector<T> read(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("input "+path);
    size_t bytes=f.tellg();if(bytes%sizeof(T))throw std::runtime_error("unaligned input");
    std::vector<T> v(bytes/sizeof(T));f.seekg(0);f.read((char*)v.data(),bytes);return v;
}
struct Memory {
    std::vector<void*> ptrs;
    template<class T>void* put(const std::vector<T>& v){void* p=nullptr;size_t n=v.size()*sizeof(T);
        check(aclrtMalloc(&p,n,ACL_MEM_MALLOC_HUGE_FIRST));ptrs.push_back(p);
        check(aclrtMemcpy(p,n,v.data(),n,ACL_MEMCPY_HOST_TO_DEVICE));return p;}
    template<class T>std::vector<T> get(void* p,size_t n){std::vector<T> v(n);
        check(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
    ~Memory(){for(auto p:ptrs)aclrtFree(p);}
};
int main(int argc,char** argv) {
    try {
        if(argc!=9)throw std::runtime_error("topk_sim SCORE.bin OUT.json B H VALID MISS SHARDS VCAP");
        std::string file=argv[1],out=argv[2];
        uint32_t b=std::stoul(argv[3]),h=std::stoul(argv[4]),valid=std::stoul(argv[5]),
            miss=std::stoul(argv[6]),s=std::stoul(argv[7]),vcap=std::stoul(argv[8]),q=4*b,k=2048;
        if(!b||b>16||h%256||valid>h||!s||s>16||h/256/s*256<k||!vcap||vcap>64)
            throw std::runtime_error("shape");
        auto seed=read<uint16_t>(file);if(seed.size()!=4*h)throw std::runtime_error("seed size");
        std::vector<uint16_t> input(size_t(q)*h);
        for(uint32_t i=0;i<b;++i)std::copy(seed.begin(),seed.end(),input.begin()+size_t(i)*4*h);
        const uint32_t guard=64,marker=0x5a5a5a5a;std::vector<uint32_t> blank(size_t(q)*s*k+guard,marker);
        check(aclInit(nullptr));check(aclrtSetDevice(0));aclrtStream stream;check(aclrtCreateStream(&stream));
        {
            Memory m;void* in=m.put(input),*ids=m.put(blank),*scores=m.put(std::vector<uint16_t>(size_t(q)*s*k+guard,0x5a5a));
            void* result=ids,*resultScores=scores;
            void* localTimes=m.put(std::vector<uint64_t>(64*4));
            void* mergeTimes=m.put(std::vector<uint64_t>(64*4));
            uint32_t localWorkers=std::min(q*s,vcap),mergeWorkers=std::min(q,vcap);
            check(aclrtlaunch_split_topk_local(localWorkers,stream,in,ids,scores,q,h,s,localWorkers,localTimes));
            if(s>1){result=m.put(std::vector<uint32_t>(size_t(q)*k+guard,marker));
                resultScores=m.put(std::vector<uint16_t>(size_t(q)*k+guard,0x5a5a));
                check(aclrtlaunch_split_topk_merge(mergeWorkers,stream,ids,scores,result,resultScores,q,s,mergeWorkers,mergeTimes));}
            check(aclrtSynchronizeStream(stream));
            auto actual=m.get<uint32_t>(result,size_t(q)*k+guard);
            auto actualScores=m.get<uint16_t>(resultScores,size_t(q)*k+guard);
            auto allLocal=m.get<uint32_t>(ids,size_t(q)*s*k+guard);
            auto allLocalScores=m.get<uint16_t>(scores,size_t(q)*s*k+guard);
            if(m.get<uint16_t>(in,input.size())!=input)throw std::runtime_error("input mutated");
            auto lt=m.get<uint64_t>(localTimes,64*4),mt=m.get<uint64_t>(mergeTimes,64*4);
            uint64_t localMax=0,mergeMax=0;
            for(uint32_t i=0;i<localWorkers;++i){if(lt[4*i+1]<=lt[4*i]||lt[4*i+2]!=i)throw std::runtime_error("local timer");
                localMax=std::max(localMax,lt[4*i+1]-lt[4*i]);}
            if(s>1)for(uint32_t i=0;i<mergeWorkers;++i){if(mt[4*i+1]<=mt[4*i]||mt[4*i+2]!=i)throw std::runtime_error("merge timer");
                mergeMax=std::max(mergeMax,mt[4*i+1]-mt[4*i]);}
            {
                std::ofstream raw(out+".timers.bin",std::ios::binary);
                raw.write(reinterpret_cast<const char*>(lt.data()),lt.size()*sizeof(uint64_t));
                raw.write(reinterpret_cast<const char*>(mt.data()),mt.size()*sizeof(uint64_t));
                if(!raw)throw std::runtime_error("timer evidence write");
            }
            for(uint32_t i=0;i<guard;++i)
                if(actual[size_t(q)*k+i]!=marker||actualScores[size_t(q)*k+i]!=0x5a5a||
                    allLocal[size_t(q)*s*k+i]!=marker||allLocalScores[size_t(q)*s*k+i]!=0x5a5a)
                    throw std::runtime_error("guard");
            std::vector<uint16_t> sorted(h);
            for(uint32_t query=0;query<q;++query){
                for(uint32_t shard=0;shard<s;++shard){
                    uint32_t begin=h/256*shard/s*256,end=h/256*(shard+1)/s*256;
                    std::vector<uint16_t> local(input.begin()+size_t(query)*h+begin,input.begin()+size_t(query)*h+end);
                    std::nth_element(local.begin(),local.begin()+k-1,local.end(),std::greater<uint16_t>());
                    uint16_t threshold=local[k-1];std::set<uint32_t> seen;
                    for(uint32_t i=0;i<k;++i){size_t pos=(size_t(query)*s+shard)*k+i;auto token=allLocal[pos];
                        if(token<begin||token>=end||!seen.insert(token).second||
                           input[size_t(query)*h+token]<threshold||allLocalScores[pos]!=input[size_t(query)*h+token])
                            throw std::runtime_error("local candidate mismatch");}
                    for(uint32_t t=begin;t<end;++t)
                        if(input[size_t(query)*h+t]>threshold&&!seen.count(t))throw std::runtime_error("local winner omitted");
                }
                std::copy_n(input.begin()+size_t(query)*h,h,sorted.begin());
                std::nth_element(sorted.begin(),sorted.begin()+k-1,sorted.end(),std::greater<uint16_t>());
                uint16_t cutoff=sorted[k-1];std::set<uint32_t> expected,seen;
                for(uint32_t t=0;t<valid;++t)if(input[size_t(query)*h+t]>=cutoff)expected.insert(t);
                if(expected.size()!=k)throw std::runtime_error("oracle boundary tie");
                uint32_t misses=0;
                for(uint32_t i=0;i<k;++i){auto token=actual[size_t(query)*k+i];
                    if(token>=valid||!seen.insert(token).second||!expected.count(token))throw std::runtime_error("selected mismatch");
                    if(actualScores[size_t(query)*k+i]!=input[size_t(query)*h+token])throw std::runtime_error("selected score mismatch");
                    misses+=token>=16384;}
                if(misses!=miss)throw std::runtime_error("miss fixture mismatch");
            }
            std::ofstream f(out);f<<"{\"status\":\"passed\",\"scope\":\"AIV-only exact final TopK component\","
                <<"\"batch\":"<<b<<",\"history\":"<<h<<",\"valid\":"<<valid<<",\"miss_per_query\":"<<miss
                <<",\"shards\":"<<s<<",\"aiv_budget\":"<<vcap<<",\"local_workers\":"<<localWorkers
                <<",\"merge_workers\":"<<(s>1?mergeWorkers:0)<<",\"aic_workers\":0,\"guards\":true,"
                <<"\"local_max_worker_cycle_delta\":"<<localMax<<",\"merge_max_worker_cycle_delta\":"<<mergeMax
                <<",\"sum_phase_max_worker_cycle_delta\":"<<(localMax+mergeMax)<<",\"timing_scope\":\"worker-body only; excludes dispatch and init\","
                <<"\"selected_set_and_scores\":true,\"all_local_candidates\":true,\"input_unchanged\":true,\"full_LI\":false,\"transport\":false}\n";
            std::cout<<"TOPK_PASS B="<<b<<" H="<<h<<" S="<<s<<" V="<<vcap<<"\n";
        }
        check(aclrtDestroyStream(stream));check(aclrtResetDevice(0));check(aclFinalize());return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
