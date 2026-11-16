#include <acl/acl.h>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include "context.h"
extern "C" uint32_t aclrtlaunch_li_candidate_kernel(uint32_t,void*,void*);
void ck(int e){if(e)throw std::runtime_error("ACL "+std::to_string(e));}
template<class T>void* put(const std::vector<T>& v){void*p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
template<class T>std::vector<T> get(void*p,size_t n){std::vector<T> v(n);ck(aclrtMemcpy(v.data(),n*sizeof(T),p,n*sizeof(T),ACL_MEMCPY_DEVICE_TO_HOST));return v;}
int main(int argc,char**argv){try{
 constexpr int H=8192,R=4096,K=2048,Q=4,prefixBytes=Q*K*6+Q*256;
 std::mt19937 rng(43);std::vector<uint8_t> q(Q*32*128),key(H*128);
 for(auto&v:q)v=(rng()%2)?0x38:0xb8;for(auto&v:key)v=(rng()%2)?0x38:0xb8;
 std::vector<uint32_t> bt(H/128);std::iota(bt.begin(),bt.end(),0);
 ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 std::vector<void*> p;
 p.push_back(put(q));p.push_back(put(key));p.push_back(put(std::vector<uint16_t>(Q*32,0x3f80)));
 p.push_back(put(std::vector<float>(Q*32,1)));p.push_back(put(std::vector<float>(H,1)));
 p.push_back(put(std::vector<uint32_t>{4}));p.push_back(put(std::vector<uint32_t>{H}));p.push_back(put(bt));
 p.push_back(put(std::vector<uint32_t>(Q*K,UINT32_MAX)));p.push_back(put(std::vector<uint16_t>(Q*H,0)));
 p.push_back(put(std::vector<uint8_t>(prefixBytes+Q*(H/128)*544+256,0)));
 p.push_back(put(std::vector<uint32_t>(128,0)));
 rotating::Context c{};c.progress=(uint64_t)p[11];c.hits=(uint64_t)p[10]+prefixBytes;
 c.shape.batch=1;c.shape.history=H;c.shape.resident=R;c.shape.indexHeads=32;c.shape.aicCount=1;c.shape.liCores=1;c.shape.serviceAics=1;c.shape.rounds=0;
 p.push_back(put(std::vector<rotating::Context>{c}));std::vector<uint64_t> args;for(auto x:p)args.push_back((uint64_t)x);auto bindings=put(args);
 ck(aclrtlaunch_li_candidate_kernel(1,stream,bindings));ck(aclrtSynchronizeStream(stream));
 auto scores=get<uint16_t>(p[9],Q*H);auto top=get<uint32_t>(p[8],Q*K);auto prefix=get<uint16_t>(p[10],(prefixBytes+Q*(H/128)*544)/2);
 std::vector<uint16_t> canonical(Q*H,0);size_t count=0;
 for(int query=0;query<Q;++query){
  auto sorted=std::vector<uint16_t>(scores.begin()+query*H,scores.begin()+(query+1)*H);std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());
  std::vector<uint16_t> got;std::vector<bool> seen(H,false);
  for(int i=0;i<K;++i){auto id=top[query*K+i];if(id>=H||seen[id])throw std::runtime_error("bad TopK ID");seen[id]=true;got.push_back(scores[query*H+id]);}
  std::sort(got.begin(),got.end(),std::greater<uint16_t>());if(!std::equal(got.begin(),got.end(),sorted.begin()))throw std::runtime_error("bad TopK values");
  auto threshold=prefix[Q*K*3+query*128];auto hot=std::vector<uint16_t>(scores.begin()+query*H,scores.begin()+query*H+R);std::sort(hot.begin(),hot.end(),std::greater<uint16_t>());
  if(threshold!=hot[K-1])throw std::runtime_error("bad resident threshold");
  for(int tile=R/128;tile<H/128;++tile){int base=tile*128;
   if(EXPERIMENT_COMPACT){
    auto record=prefix.data()+prefixBytes/2+(query*(H/128)+tile)*272;
    auto n=record[256];if(n>128)throw std::runtime_error("bad count");
    for(int i=0;i<n;++i){auto id=record[128+i];if(id>=128||canonical[query*H+base+id])throw std::runtime_error("bad candidate ID");
     if(record[i]!=scores[query*H+base+id])throw std::runtime_error("bad candidate score");canonical[query*H+base+id]=1;}
   }else{
    auto bits=reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(prefix.data())+prefixBytes+query*(H/4)+base/4);
    for(int i=0;i<128;++i)canonical[query*H+base+i]=(bits[i/16]>>(2*(i%16)))&1;
   }
   for(int i=0;i<128;++i){bool expected=scores[query*H+base+i]>threshold;if(canonical[query*H+base+i]!=expected)throw std::runtime_error("candidate predicate mismatch");count+=expected;}
  }
 }
 if(argc>1){std::ofstream f(argv[1],std::ios::binary);f.write((char*)scores.data(),scores.size()*2);f.write((char*)top.data(),top.size()*4);f.write((char*)canonical.data(),canonical.size()*2);}
 printf("PASS mode=%s BS1 Q4 H8192 R4096 heads32 AIC1 AIV2 candidates=%zu/%d: score/TopK/threshold/candidates checked\n",EXPERIMENT_COMPACT?"compact":"bitmap",count,Q*(H-R));
 ck(aclrtFree(bindings));for(auto x:p)ck(aclrtFree(x));ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
 }catch(std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
