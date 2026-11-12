"""Execute the native two-range driver and query-bank decisions on the CPU.

The model checks lifecycle/event ownership and physical tile order; it does not
simulate accelerator timing or matrix/vector numerical instructions.
"""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function

ROOT = Path(__file__).resolve().parents[4]
INDEXER = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer"
COMMON = ROOT / "csrc/operators/common/lifetch"


def native_model():
    source = (INDEXER / "indexer.h").read_text()
    methods = "\n".join(
        function(source, signature).replace("QuantLightningIndexerKernel<QLIT>::", "")
        for signature in (
            "__aicore__ inline bool QuantLightningIndexerKernel<QLIT>::ScoreRecent(",
            "__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ScanSequenceHalf(",
            "__aicore__ inline void QuantLightningIndexerKernel<QLIT>::ProcessBaseBlock(",
            "__aicore__ inline void QuantLightningIndexerKernel<QLIT>::FinishBaseBlock(",
        )
    )
    run_info = function((INDEXER / "indexer_types.h").read_text(), "struct RunInfo {") + ";"
    return (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
bool cubeMode;
#define ASCEND_IS_AIC (cubeMode)
#define PIPE_FIX 0
#define PIPE_V 0
template<unsigned M,unsigned P> void CrossCoreWaitFlag(unsigned) {assert(false);}
template<unsigned M,unsigned P> void CrossCoreSetFlag(unsigned) {assert(false);}
#include "recent_index.h"
constexpr unsigned SIM_Q=4,S2_BASE_SIZE=128;
namespace QLICommon {
constexpr unsigned KScaleTiles=MODEL_SCALE_TILES;
struct ConstInfo {static constexpr unsigned QLI_SYNC_MODE4=4;};
"""
        + run_info
        + r"""
}
namespace rotating {
struct Context {uint64_t thresholdBounds=0,queryEnds=0;};
template<class... Args> int EpochAt(Args...) {return -1;}
}
// These are the existing two query-bank credits. Native LoadQuery/ReleaseQuery
// decide exactly when the Cube acquires/releases them. Key ownership remains
// tied to the physical range, independently of query-bank retention.
struct Cube {
 bool credit[2]={true,true};
 unsigned owner[2]={0,0},queryIndex=0,chunks=1,loads=0,releases=0,physicalTails=0;
 std::vector<QLICommon::RunInfo> calls;
 void ComputeMm1(QLICommon::RunInfo info){
  for(unsigned part=0;part<chunks;++part){
   unsigned index;
   if(info.LoadQuery()){
    index=++queryIndex;unsigned bank=index%2;
    assert(credit[bank]);credit[bank]=false;owner[bank]=info.bIdx;++loads;
   }else{
    // Same queryL1Mte1BufIdx selection as ComputeMm1 for one/two M blocks.
    index=queryIndex-(chunks-1-(part>0));
   }
   unsigned bank=index%2;assert(!credit[bank] && owner[bank]==info.bIdx);
   if(info.ReleaseQuery()){credit[bank]=true;++releases;}
  }
  if(info.isLastS2InnerLoop)++physicalTails;
  calls.push_back(info);
 }
 void drained(){assert(credit[0] && credit[1] && loads==releases);}
};
struct Vector {
 bool qCredit[2]={true,true},kCredit[2]={true,true};
 int cachedOwner[2]={-1,-1};
 unsigned loads=0,emptyLanes=0,activeQueries=4,lane=0;
 std::vector<QLICommon::RunInfo> calls;
 void ProcessVec1(QLICommon::RunInfo info,unsigned=0,bool=false,bool=false){
  calls.push_back(info);
  // Same early return as ProcessVec1: an empty AIV participates in CV/VC
  // credits but must not acquire/load any query or scale bank.
  unsigned per=(activeQueries+1)/2;
  if(lane*per>=activeQueries){++emptyLanes;return;}
  unsigned q=info.qScaleLoop%2,k=info.kScaleLoop%2;
  if(info.LoadQuery()){
   assert(qCredit[q]);qCredit[q]=false;cachedOwner[q]=info.bIdx;++loads;
  }
  assert(cachedOwner[q]==int(info.bIdx));
  bool first=(info.s2Idx-info.s2Start)%QLICommon::KScaleTiles==0;
  if(first){assert(kCredit[k]);kCredit[k]=false;}
  if(info.LoadQuery()){assert(!qCredit[q]);qCredit[q]=true;}
#if MODEL_SCALE_TILES == 8
  if((info.s2Idx-info.s2Start+1)%QLICommon::KScaleTiles==0 ||
     (info.s2Idx+1)*128>=info.validS2Len || info.isLastS2InnerLoop){
#else
  if(first){
#endif
   assert(!kCredit[k]);kCredit[k]=true;
  }
 }
 void ProcessResidentTopK(QLICommon::RunInfo){}
 void PrepareRotationProgress(){}
 void RotateFetch(int) {assert(false);}
 void ProcessTopK(QLICommon::RunInfo){}
 void drained(){assert(qCredit[0]&&qCredit[1]&&kCredit[0]&&kCredit[1]);}
};
struct Model {
 unsigned resident_=16384,aiCoreIdx=0;
 bool rotationServed_=false;
 struct {unsigned history=131328,batch=2,resident=16384,splitQ=1,rounds=8,serviceAics=8;} runtimeShape_;
 rotating::Context ctx;
 void* rotationContext_=&ctx;
 struct {unsigned s2Start=0;} splitCoreInfo;
 struct {unsigned bIdx=0,actS2Size=131328,s2LoopEnd=1025,validS2Len=131328;} tempLoopInfo;
 Vector vectorService;Cube matmulService;
 void CalcRunInfo(unsigned loop,unsigned tile,QLICommon::RunInfo& i,unsigned q,unsigned k){
  i.loop=loop;i.s2Idx=tile;i.s2Start=splitCoreInfo.s2Start;i.validS2Len=tempLoopInfo.validS2Len;
  i.qScaleLoop=q;i.kScaleLoop=k;i.bIdx=tempLoopInfo.bIdx;
  i.isFirstS2InnerLoop=tile==splitCoreInfo.s2Start;
  i.isLastS2InnerLoop=tile==tempLoopInfo.s2LoopEnd;
  i.tensorKeyOffset=uint64_t(tile)*128*128;
 }
"""
        + methods
        + r"""
};
int main(){
 for(bool aic:{false,true})for(unsigned chunks:{1U,2U})for(unsigned shards:{1U,2U})
 for(unsigned length:{0U,1U,127U,128U,129U,1023U,2048U,2050U,8193U})
 for(unsigned active:{0U,1U,2U,3U,4U})for(unsigned lane:{0U,1U}){
  cubeMode=aic;
  for(unsigned shard=0;shard<shards;++shard){
   Model m;m.matmulService.chunks=chunks;m.vectorService.activeQueries=active;m.vectorService.lane=lane;
   int32_t bounds[4]={20001,int32_t(20001+length),65537,int32_t(65537+length)};
   uint32_t ends[8]={131000,130999,130998,130997,131000,130999,130998,130997};
   if(!active)std::fill_n(ends,8,0U);
   m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);
   m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
   unsigned loop=0;
   // Exercise consecutive owner/query groups, including bank wrap. The
   // sequence-split hot phase has qScaleLoop0; general scan alternates it.
   for(unsigned group=0;group<4;++group){
    m.tempLoopInfo.bIdx=group%2;
    unsigned qScale=shards==2?0:group;
    size_t previous=aic?m.matmulService.calls.size():m.vectorService.calls.size();
    bool had=m.ScoreRecent(loop,qScale,shard,shards);
    auto range=kvip_recent_index::Clip(bounds[2*(group%2)],bounds[2*(group%2)+1],
                                     m.resident_,m.runtimeShape_.history,ends[4*(group%2)]);
    auto recent=kvip_recent_index::Tiles(range,shard,shards);
    assert(had==(recent.begin!=recent.end));
    unsigned begin=shard*(128/shards),end=(shard+1)*(128/shards);
    if(shards==2){
     unsigned scaleLoop=0;
     // Real caller passes hadRecent to mark the hot range's query boundary.
     m.ScanSequenceHalf(begin,end,shard,loop,scaleLoop,true,had);
    }else{
     m.splitCoreInfo.s2Start=begin;m.tempLoopInfo.s2LoopEnd=end-1;m.tempLoopInfo.validS2Len=end*128;
     unsigned scale=0;
     for(unsigned tile=begin;tile<end;++tile,++loop){
      if((tile-begin)%QLICommon::KScaleTiles==0)++scale;
      QLICommon::RunInfo info;
      m.ProcessBaseBlock(loop,tile,info,qScale,scale,had&&tile==begin);
     }
    }
    const auto& calls=aic?m.matmulService.calls:m.vectorService.calls;
    assert(calls.size()-previous==(recent.end-recent.begin)+(end-begin));
    for(unsigned i=0;i<calls.size()-previous;++i){
     auto c=calls[previous+i];unsigned n=recent.end-recent.begin;
     unsigned tile=i<n?recent.begin+i:begin+i-n;
     assert(c.s2Idx==tile && c.tensorKeyOffset==uint64_t(tile)*128*128);
     if(i)assert(c.loop==calls[previous+i-1].loop+1);
     assert(c.isLastS2InnerLoop==(i+1==n || i+1==n+end-begin));
     if(i<n)assert(c.retainQuery==bool(KVIP_RECENT_PIPELINE));
     if(i==n && had)assert(c.LoadQuery()==!bool(KVIP_RECENT_PIPELINE));
    }
    m.matmulService.drained();m.vectorService.drained();
    m.tempLoopInfo.s2LoopEnd=1025;m.tempLoopInfo.validS2Len=131328;m.splitCoreInfo.s2Start=0;
    // Each sequence-split launch starts at loop0; general query groups keep
    // a continuous loop. Both forms must preserve all physical tile visits.
    if(shards==2)loop=0;
   }
   auto& calls=aic?m.matmulService.calls:m.vectorService.calls;
   unsigned expectedLoads=0;
   for(const auto& c:calls)expectedLoads+=c.LoadQuery();
   if(aic)assert(m.matmulService.loads==expectedLoads*chunks);
   else if(!active || lane*((active+1)/2)>=active)assert(m.vectorService.loads==0);
   else assert(m.vectorService.loads==expectedLoads);
  }
 }
 // Null bounds retain exactly the normal query lifetime.
 cubeMode=true;Model m;unsigned loop=0,scale=0;
 assert(!m.ScoreRecent(loop,0,0,1));
 m.ScanSequenceHalf(0,128,0,loop,scale,true,false);
 assert(m.matmulService.loads==1 && m.matmulService.physicalTails==1);m.matmulService.drained();
}
"""
    )


@pytest.mark.parametrize("pipeline", [0, 1])
@pytest.mark.parametrize("scale_tiles", [8, 16])
def test_native_two_range_query_lifetime(tmp_path, pipeline, scale_tiles):
    cpp = tmp_path / "pipeline.cpp"
    cpp.write_text(native_model())
    exe = tmp_path / "pipeline"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DKVIP_RECENT_PIPELINE={pipeline}",
            f"-DMODEL_SCALE_TILES={scale_tiles}",
            "-I",
            str(COMMON),
            str(cpp),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_query_lifetime_keeps_physical_key_boundaries():
    cube = function((INDEXER / "indexer_cube.h").read_text(), "__aicore__ inline void QLIMatmul<QLIT>::ComputeMm1(")
    vector = function(
        (INDEXER / "indexer_vector.h").read_text(), "__aicore__ inline void QLIVector<QLIT>::ProcessVec1("
    )
    assert cube.count("runInfo.LoadQuery()") == 2
    assert cube.count("runInfo.ReleaseQuery()") == 2
    assert "l1FullyUsed || runInfo.isLastS2InnerLoop" in cube
    assert vector.count("info.LoadQuery()") == 2
    assert "info.validS2Len || info.isLastS2InnerLoop" in vector
