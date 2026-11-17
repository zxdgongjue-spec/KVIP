#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include "runtime_shape.h"
#include "staged_schedule.h"
#define __simt_vf__
#define __aicore__
#define __global__
#define __ubuf__
#define __simt_callee__
#define __gm__
#define LAUNCH_BOUND(x)
#define KERNEL_TASK_TYPE_DEFAULT(x)
#define ASCEND_IS_AIV (true)
#define EVENT_ID0 0
using GM_ADDR=uint8_t*;
struct {uint32_t x;} threadIdx;uint32_t block;
template<class T> T asc_ldcg(T* p){return *p;}
template<class T> void asc_stcg(T* p,T x){*p=x;}
void asc_threadfence(){}
namespace rotating {
struct Context{RuntimeShape shape;uint64_t hits,queryEnds;};
RuntimeShape LoadRuntimeShape(Context* p){return p->shape;}
}
namespace kvaf_device {void CleanCacheLines(uint8_t*,uint32_t){} }
namespace AscendC {
enum class TPosition {VECCALC};enum class HardEvent {MTE2_MTE3,MTE3_MTE2,V_S,MTE2_V,V_MTE3};
uint32_t GetBlockIdx(){return block;}
template<class T> struct LocalTensor {T* p;T* GetPhyAddr(){return p;}LocalTensor operator[](uint64_t x){return {p+x};}};
template<class T> struct GlobalTensor {T* p;void SetGlobalBuffer(T* q){p=q;}GlobalTensor operator[](uint64_t x){return {p+x};}};
struct TPipe{template<class B> void InitBuffer(B& b,uint32_t n){b.data.resize(n);}void Destroy(){}};
template<TPosition P> struct TBuf{std::vector<uint8_t> data;template<class T> LocalTensor<T> Get(){return {reinterpret_cast<T*>(data.data())};}};
template<class T> void DataCopy(LocalTensor<T> d,GlobalTensor<T> s,uint32_t n){std::memcpy(d.p,s.p,n*sizeof(T));}
template<class T> void DataCopy(GlobalTensor<T> d,LocalTensor<T> s,uint32_t n){std::memcpy(d.p,s.p,n*sizeof(T));}
template<HardEvent E>void SetFlag(uint32_t){}
template<HardEvent E>void WaitFlag(uint32_t){}
namespace Simt {struct Dim3{uint32_t x,y,z;};template<auto F,class... A>void VF_CALL(Dim3,A... a){for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)F(a...);}}
}
// ACTUAL_PREPARE
#include "staged_post_device.h"
// ACTUAL_SCATTER
void scatter_checks(){
 constexpr uint32_t h=8192,stride=4224,rows=2;
 uint64_t original[16]{};rotating::Context ctx{};ctx.shape.batch=2;ctx.shape.history=h;
 std::vector<uint16_t> canonical(2*4*h+64),scores(rows*4*stride),prefix(2*(4*2048*3+512));
 std::vector<uint8_t> bits(2*h+64);uint32_t ends[8];
 ctx.hits=(uint64_t)bits.data();ctx.queryEnds=(uint64_t)ends;original[13]=(uint64_t)prefix.data();original[14]=(uint64_t)&ctx;
 for(uint32_t qg:{2U,4U}){
  kvip_staged::ScoreTask tasks[2]={{0,qg==2?2U:0U,128,128+stride},{1,0,256,256+4096}};
  std::fill(canonical.begin(),canonical.end(),0xA5A5);std::fill(bits.begin(),bits.end(),0xA5);
  for(uint32_t q=0;q<8;++q){ends[q]=3000+q;prefix[(q/4)*(4*2048*3+512)+4*2048*3+(q%4)*128]=uint16_t(2000+q);}
  for(uint32_t i=0;i<scores.size();++i)scores[i]=(i*17)%4096;
  for(block=0;block<2*rows;++block)v9_staged_scatter_predicate((GM_ADDR)original,(GM_ADDR)scores.data(),(GM_ADDR)canonical.data(),(GM_ADDR)tasks,rows,stride,qg);
  for(uint32_t b=0;b<2;++b)for(uint32_t q=0;q<4;++q)for(uint32_t x=0;x<h;x+=4){
   auto t=tasks[b];bool touched=q>=t.queryBegin && q<t.queryBegin+qg && x>=t.begin && x<t.end;
   uint8_t expected=0xA5;
   if(touched){expected=0;for(uint32_t i=0;i<4;++i){auto key=scores[(b*qg+q-t.queryBegin)*stride+x-t.begin+i];assert(canonical[(b*4+q)*h+x+i]==key);if(x+i<ends[b*4+q] && key>uint16_t(2000+b*4+q))expected|=1U<<(2*i);}}
   else for(uint32_t i=0;i<4;++i)assert(canonical[(b*4+q)*h+x+i]==0xA5A5);
   assert(bits[(b*4+q)*(h/4)+x/4]==expected);
  }
  for(uint32_t i=2*h;i<bits.size();++i)assert(bits[i]==0xA5);
  for(uint32_t i=2*4*h;i<canonical.size();++i)assert(canonical[i]==0xA5A5);
 }
}
int main(){
 scatter_checks();
 constexpr uint32_t h=1024,heads=32,qg=2,rows=2,stride=384;
 rotating::Context ctx{};ctx.shape.batch=2;ctx.shape.history=h;ctx.shape.indexHeads=heads;
 uint64_t original[16]{},wave[16]{};
 std::vector<uint8_t> input[3],output[3];uint32_t fields[3]={0,2,3},widths[3]={128,2,4};
 for(uint32_t f=0;f<3;++f){input[f].resize(2*4*heads*widths[f]);output[f].resize(rows*qg*heads*widths[f],0xA5);original[fields[f]]=(uint64_t)input[f].data();wave[fields[f]]=(uint64_t)output[f].data();}
 uint32_t pages[2*h/128]{},outpages[rows*stride/128];std::fill(std::begin(outpages),std::end(outpages),0xA5);
 original[7]=(uint64_t)pages;wave[7]=(uint64_t)outpages;original[14]=(uint64_t)&ctx;
 kvip_staged::ScoreTask tasks[2]={{0,2,256,512},{1,0,0,384}};
 kvip_staged::PrepareWave desc[1]={{(uint64_t)wave,(uint64_t)tasks,rows,stride,qg,0}};
 for(uint32_t iteration=0;iteration<3;++iteration){
  for(uint32_t f=0;f<3;++f)for(size_t i=0;i<input[f].size();++i)input[f][i]=(i*17+iteration*31)%251;
  for(uint32_t i=0;i<2*h/128;++i)pages[i]=(31-i)+iteration*79;
  original[1]=123+iteration;original[4]=567+iteration;
  for(block=0;block<rows;++block)v9_staged_prepare_all((GM_ADDR)original,(GM_ADDR)desc,1);
  assert(wave[1]==original[1] && wave[4]==original[4]);
  for(uint32_t row=0;row<rows;++row){auto t=tasks[row];
   for(uint32_t f=0;f<3;++f){uint32_t bytes=qg*heads*widths[f];uint64_t off=(t.owner*4+t.queryBegin)*heads*widths[f];assert(std::memcmp(output[f].data()+row*bytes,input[f].data()+off,bytes)==0);}
   for(uint32_t i=0;i<(t.end-t.begin)/128;++i)assert(outpages[row*stride/128+i]==pages[t.owner*h/128+t.begin/128+i]);
  }
  assert(outpages[2]==0xA5);
 }
}
