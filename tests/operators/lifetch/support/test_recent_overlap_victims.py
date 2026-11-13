"""Run native score loading and SaveResident with an outstanding LRU DMA."""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function
from test_recent_local_victims import local_victims_model_source


@pytest.mark.parametrize("enabled,register,seed", [(0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)])
def test_overlap_keeps_lru_pending_during_kth_and_consumes_per_query(tmp_path, enabled, register, seed):
    vector = (INDEXER / "indexer_vector.h").read_text()
    topk = (INDEXER / "topk.h").read_text()
    loader = function(vector, "    __aicore__ inline bool TryUnifiedResidentTopK(")
    eligible = function(topk, "    __aicore__ inline bool CanRegisterThreshold(")
    source = local_victims_model_source().split("int main(){", 1)[0]
    source = source.replace("#include <random>", "#include <random>\n#include <type_traits>")
    source = source.replace(
        "constexpr unsigned SIM_Q=4,EVENT_ID0=0,TOPK_V_MTE3_EVENT=6,V_MTE2_EVENT3=5;",
        "constexpr unsigned SIM_Q=4,EVENT_ID0=0,TOPK_V_MTE3_EVENT=6,V_MTE2_EVENT3=5,TOPK_MTE2_V_EVENT=5;",
    )
    source = source.replace(
        "struct Context {uint64_t lruSlots=0,hits=0,queryEnds=0,progress=0;};",
        "struct Context {uint64_t lruSlots=0,hits=0,queryEnds=0,progress=0,thresholdBounds=0;};",
    )
    source = source.replace(
        "template<class T> using GlobalTensor=Tensor<T>;",
        """
template<class T> using GlobalTensor=Tensor<T>;
template<class T> T Min(T a,T b){return std::min(a,b);}
unsigned lruIssued=0,scoreIssued=0,kthWithPendingLru=0,kthWithoutPendingLru=0;
template<class T> void Duplicate(Tensor<T> dst,T value,unsigned count){
 assert(count<=dst.n);std::fill_n(dst.p,count,value);
}
""",
    )
    source = source.replace(
        "assert(count<=dst.n&&count<=src.n&&dst.gm!=src.gm);",
        "assert(count<=dst.n&&count<=src.n&&dst.gm!=src.gm);"
        "if constexpr(std::is_same_v<T,uint32_t>){if(src.gm){assert(count==2048);++lruIssued;}}",
    )
    pos = source.index("struct Model {")
    source = (
        source[:pos]
        + r"""
template<class T> struct DataCopyPadExtParams {bool pad;unsigned left,right;T value;};
template<class T> void DataCopyPad(Tensor<T> dst,Tensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T>){
 assert(src.gm&&!dst.gm&&p.blockCount==1&&!p.srcStride&&!p.dstStride);
 unsigned count=p.blockLen/sizeof(T);assert(count<=dst.n&&count<=src.n);
 ++scoreIssued;mte2.push_back([=](){std::copy_n(src.p,count,dst.p);});
}
namespace kvip_recent_index {
void RemapUnified(uint32_t*,uint32_t,uint32_t,uint32_t){assert(false);}
void MaskSeedPadding(uint16_t* scores,uint16_t count,uint16_t begin,uint16_t end){
 assert(mte2.empty());for(unsigned i=0;i<count;++i)if(i<begin||i>=end)scores[i]=0;
}
}
struct TopK {
 unsigned topK=2048;uint32_t* lruDestination=nullptr;
"""
        + eligible
        + r"""
 void Threshold(Tensor<uint16_t> input,Tensor<uint16_t> output,unsigned count){
  // Arithmetic is independently tested in test_recent_register_kth. Here the
  // VF boundary deliberately does NOT drain unrelated MTE2 work: its input
  // score copies must already be ready, but LRU must still be pending.
  if(!mte2.empty()){
   assert(CanRegisterThreshold(count)&&mte2.size()==1&&lruDestination);
   assert(lruDestination[0]==0xcccccccc);++kthWithPendingLru;
  }else ++kthWithoutPendingLru;
  std::vector<uint16_t> values(input.p,input.p+count);
  std::nth_element(values.begin(),values.begin()+count-topK,values.end());
  std::fill_n(output.p,128,values[count-topK]);
 }
 void operator()(Tensor<uint16_t>,Tensor<int32_t>,Tensor<uint16_t>,unsigned,unsigned,unsigned,bool){assert(false);}
};
"""
        + source[pos:]
    )
    source = source.replace(
        "struct Model {",
        """struct Model {
 using SCORE_T=uint16_t;
 unsigned topkCount_=2048,topkCountAlign256_=2048,trunkLen_=16384;
 uint32_t residentVictimsQuery_=UINT32_MAX;
 bool residentSeedActive_=false;uint32_t residentSeedQuery_=UINT32_MAX;
 kvip_recent_index::SeedLayout residentSeed_{};
 TopK topkOp_;
""",
        1,
    )
    pos = source.rfind("};")
    source = source[:pos] + loader + source[pos:]
    source += r"""
void CheckBitmap(Model& m,unsigned q,unsigned row){
 unsigned r=m.runtimeShape_.resident,h=m.runtimeShape_.history;
 auto actual=m.bitmap.data()+q*h/4;auto ids=m.victims.data()+(q/4)*2048;
 unsigned end=std::min(m.ends[q],r);uint16_t threshold=m.thresholds[(q%2)*128];
 for(unsigned i=0;i<2048;++i){
  bool selected=ids[i]<end&&m.scores[row*h+ids[i]]>threshold;
  assert(((actual[i/4]>>((i%4)*2))&3U)==unsigned(selected));
 }
}
int main(){
 std::mt19937 rng(91);
 for(unsigned mode=0;mode<8;++mode){
  Model m;unsigned r=m.runtimeShape_.resident,h=m.runtimeShape_.history;
  int32_t bounds[6]={20001,20300,30001,30300,40001,40300};
  m.ctx.thresholdBounds=uint64_t(bounds);
  for(auto& x:m.scores)x=uint16_t(rng());for(auto& id:m.victims)id=rng()%r;
  m.ends.fill(h);m.topkOp_.lruDestination=m.scratch.data()+1024;
  if(mode==3)for(unsigned b=0;b<3;++b)bounds[b*2+1]=bounds[b*2]+4096; // exceeds bank capacity.
  if(mode==4)m.ctx.lruSlots=0;
  if(mode==5)for(unsigned b=0;b<3;++b)bounds[b*2+1]=bounds[b*2];
  if(mode==6)m.ctx.thresholdBounds=0;
  if(mode==7)m.runtimeShape_.resident=2048;
  r=m.runtimeShape_.resident;for(auto& id:m.victims)id%=r;
  for(unsigned q:{0U,1U,6U,11U}){
   unsigned row=q%4,available=mode==2?r-1:h;
   m.ends[q]=mode==1?r-1:h;
   unsigned count=r; // mode1 deliberately supplies full count with shorter context bound.
   std::fill(m.scratch.begin(),m.scratch.end(),0xcccccccc);
   lruIssued=scoreIssued=kthWithPendingLru=kthWithoutPendingLru=0;
   gmReads=simtCalls=bitmapDma=cleanCalls=0;
   const auto scoresBefore=m.scores;
   bool ok=m.TryUnifiedResidentTopK(q,row,count,available,true);
   bool expectedQueued=KVIP_RECENT_OVERLAP_VICTIMS&&KVIP_RECENT_REGISTER_KTH&&ok&&
     m.ctx.lruSlots&&m.ends[q]>=r&&available>=r;
   if(!ok){
    assert(mode==3&&lruIssued==0&&mte2.empty()&&m.residentVictimsQuery_==UINT32_MAX);
    m.out.fill(32768);m.thresholds.fill(32768);
   }else{
    assert(kthWithPendingLru==unsigned(expectedQueued));
    assert(kthWithoutPendingLru==unsigned(!expectedQueued));
    assert(mte2.size()==unsigned(expectedQueued));
    assert(lruIssued==unsigned(expectedQueued));
    assert(scoreIssued>=1); // score DMA completed before the queued LRU.
   }
   assert(m.residentVictimsQuery_==(expectedQueued?q:UINT32_MAX));
   m.SaveResident(q,row,true,available,true,ok);
   assert(m.residentVictimsQuery_==UINT32_MAX&&mte2.empty()&&mte3.empty());
   assert(lruIssued==unsigned(bool(m.ctx.lruSlots))); // no duplicate DMA.
   if(m.ctx.lruSlots)CheckBitmap(m,q,row);
   assert(m.scores==scoresBefore);
   // A second Save cannot reuse a consumed marker, even for the same query.
   if(ok&&m.ctx.lruSlots){
    unsigned before=lruIssued;m.SaveResident(q,row,false,available,true,true);
    assert(lruIssued==before+1&&m.residentVictimsQuery_==UINT32_MAX);
   }
  }
 }
 #if KVIP_RECENT_OVERLAP_VICTIMS
 // Mismatched or fallback Save entries consume, but never trust a stale marker.
 for(bool ready:{false,true}){
  Model m;unsigned q=2;m.ends.fill(131328);m.out.fill(0);m.thresholds.fill(0);
  for(unsigned i=0;i<m.victims.size();++i)m.victims[i]=i%16384;
  m.residentVictimsQuery_=ready?q+1:q;
  lruIssued=0;m.SaveResident(q,q,false,131328,true,ready);
  assert(lruIssued==1&&m.residentVictimsQuery_==UINT32_MAX);
 }
 // Every Try resets stale state before empty/fallback checks.
 {
  Model m;m.residentVictimsQuery_=5;
  assert(!m.TryUnifiedResidentTopK(1,1,1,131328,true));
  assert(m.residentVictimsQuery_==UINT32_MAX);
 }
#endif
#if KVIP_RECENT_SEED_UB
 // The direct-score seed path remains conservative: no speculative LRU DMA.
 {
  Model m;unsigned q=6,row=2,r=16384,h=131328;
  m.ends.fill(h);for(auto& x:m.scores)x=uint16_t(rng());
  for(auto& id:m.victims)id=rng()%r;
  std::copy_n(m.scores.data()+row*h,r,m.merge.data());
  m.residentSeedActive_=true;m.residentSeedQuery_=q;
  m.residentSeed_={true,{0,0},{0,0},r};lruIssued=kthWithPendingLru=0;
  assert(m.TryUnifiedResidentTopK(q,row,r,h,true));
  assert(!m.residentSeedActive_&&m.residentVictimsQuery_==UINT32_MAX&&lruIssued==0&&kthWithPendingLru==0);
  m.SaveResident(q,row,true,h,true,true);assert(lruIssued==1);CheckBitmap(m,q,row);
 }
#endif
 for(auto& bank:credits)for(bool credit:bank)assert(!credit);
}
"""
    cpp = tmp_path / "overlap_victims.cpp"
    cpp.write_text(source)
    exe = tmp_path / "overlap_victims"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DKVIP_RECENT_OVERLAP_VICTIMS={enabled}",
            f"-DKVIP_RECENT_REGISTER_KTH={register}",
            f"-DKVIP_RECENT_SEED_UB={seed}",
            "-DKVIP_RECENT_UNIFIED_TOPK=1",
            "-DKVIP_RECENT_THRESHOLD_ONLY=1",
            "-DKVIP_RECENT_LOCAL_VICTIMS=1",
            "-DV9_UBMEM=0",
            "-I",
            str(COMMON),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
