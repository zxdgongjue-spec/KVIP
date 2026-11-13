"""Run the actual full-scan and sequence-split schedulers with skipped tiles."""

from pathlib import Path
import subprocess

import pytest

from test_recent_index import function
from test_recent_pipeline import native_model

ROOT = Path(__file__).resolve().parents[4]
INDEXER = ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer"
COMMON = ROOT / "csrc/operators/common/lifetch"


def skip_model():
    code = native_model().split("int main(){", 1)[0]
    source = (INDEXER / "indexer.h").read_text()
    methods = []
    for result, name in (
        ("kvip_recent_index::Range", "ReusableRecentTiles"),
        ("void", "ScanRemainingSequence"),
        ("void", "ScanRemainingMain"),
        ("void", "ProcessSequenceSplit"),
        ("void", "ProcessMain"),
    ):
        method = function(source, f"__aicore__ inline {result} QuantLightningIndexerKernel<QLIT>::{name}(")
        method = method.replace("QuantLightningIndexerKernel<QLIT>::", "")
        if name == "ProcessSequenceSplit":
            method = "template<SequenceLiPhase Phase>\n" + method
        methods.append(method)
    end = code.rfind("};")
    code = code[:end] + "\n".join(methods) + code[end:]
    code = code.replace(
        "#define ASCEND_IS_AIC (cubeMode)", "#define ASCEND_IS_AIC (cubeMode)\n#define ASCEND_IS_AIV (!cubeMode)"
    )
    code = code.replace("{assert(false);}", "{}", 2)  # Only the cross-core wait/set stubs.
    code = code.replace(
        "struct ConstInfo {static constexpr unsigned QLI_SYNC_MODE4=4;};",
        "struct ConstInfo {static constexpr unsigned QLI_SYNC_MODE4=4,CROSS_VC_EVENT=0; bool attenMaskFlag=false,isLDOpen=false;};",
    )
    code = code.replace(
        "template<class... Args> int EpochAt(Args...) {return -1;}",
        """
struct RuntimeShape {unsigned history=131584,batch=2,resident=16384,splitQ=1,rounds=8,serviceAics=1;};
bool IsTrigger(unsigned h,unsigned r,unsigned rounds,unsigned tile){
 for(unsigned e=0;e<rounds;++e)if(tile==r/128+((h-r)/128)*e/(rounds-1))return true;return false;
}
unsigned ServiceRank(unsigned batch,unsigned split,unsigned core){return split?(core%2)*batch+core/2:core;}
unsigned SequenceHalfWindow(const RuntimeShape& s){unsigned blocks=(s.history-s.resident)/512;return ((blocks+s.rounds-2)/(s.rounds-1))*256;}
int EpochAt(unsigned b,unsigned h,unsigned r,unsigned split,unsigned rounds,unsigned service,unsigned tile,unsigned core){
 unsigned epoch=ServiceRank(b,split,core)/service;
 return epoch<rounds && tile==r/128+((h-r)/128)*epoch/(rounds-1)?int(epoch):-1;
}
""",
    )
    code = code.replace(
        "constexpr unsigned SIM_Q=4,S2_BASE_SIZE=128;",
        "constexpr unsigned SIM_Q=4,S2_BASE_SIZE=128;\n"
        "enum class SequenceLiPhase {Hot,Threshold,Cold,Final};\n"
        "enum class LI_LAYOUT {BSND};\n"
        "template<class T> T Min(T a,T b){return std::min(a,b);}\n"
        "template<class T> T Max(T a,T b){return std::max(a,b);}",
    )
    code = code.replace("struct Cube {", "struct Cube {\n void AllocEventID(){} void FreeEventID(){drained();}")
    code = code.replace(
        "struct Vector {",
        """struct Vector {
 std::vector<unsigned> progress,services;unsigned residentCalls=0,finalCalls=0;
 std::vector<bool> ready=std::vector<bool>(2048,false);
 std::vector<std::pair<char,unsigned>> order;
 std::vector<unsigned> replayed;std::vector<std::pair<unsigned,unsigned>> replayRanges;
 void ReplayRecentScores(QLICommon::RunInfo info,unsigned begin,unsigned end){
  assert(begin<end && info.s2Idx==begin);
  // No later window is classified before the prior window's service boundary.
  order.push_back({'R',begin});
  replayRanges.push_back({begin,end});for(unsigned tile=begin;tile<end;++tile){replayed.push_back(tile);ready[tile]=true;}
 }

 void AllocEventID(){} void FreeEventID(){drained();}
 void LoadSequenceSplitThresholds(unsigned){}
 void PublishScoreProgress(QLICommon::RunInfo info,unsigned=0,bool=false){
#if !defined(V9_UBMEM) || !V9_UBMEM
  assert(ready[info.s2Idx]);
#endif
  order.push_back({'P',info.s2Idx+1});progress.push_back(info.s2Idx+1);
 }
""",
    )
    code = code.replace(
        "void ProcessVec1(QLICommon::RunInfo info,unsigned=0,bool=false,bool=false){\n  calls.push_back(info);",
        "void ProcessVec1(QLICommon::RunInfo info,unsigned=0,bool=false,bool scoreOnly=false){\n"
        "  calls.push_back(info);if(!scoreOnly && info.s2Idx>=128){ready[info.s2Idx]=true;PublishScoreProgress(info);}",
    )
    code = code.replace(
        "void ProcessResidentTopK(QLICommon::RunInfo){}",
        "void ProcessResidentTopK(QLICommon::RunInfo,unsigned=4){++residentCalls;}",
    )
    code = code.replace(
        "void ProcessTopK(QLICommon::RunInfo){}", "void ProcessTopK(QLICommon::RunInfo,unsigned=4){++finalCalls;}"
    )
    code = code.replace(
        "void RotateFetch(int) {assert(false);}",
        "void RotateFetch(int epoch,unsigned=0) {order.push_back({'S',unsigned(epoch)});services.push_back(epoch);}",
    )
    code = code.replace(
        "struct {unsigned history=131328,batch=2,resident=16384,splitQ=1,rounds=8,serviceAics=8;} runtimeShape_;",
        "rotating::RuntimeShape runtimeShape_;\nQLICommon::ConstInfo constInfo;\n"
        "static constexpr LI_LAYOUT Q_LAYOUT_T=LI_LAYOUT::BSND; unsigned tmpBlockIdx=0;",
    )
    code = code.replace(
        "struct {unsigned s2Start=0;} splitCoreInfo;",
        "struct {unsigned s2Start=0,bN2Start=0,bN2End=0,gS1Start=0;bool isCoreEnable=true;} splitCoreInfo;",
    )
    code = code.replace(
        "struct {unsigned bIdx=0,actS2Size=131328,s2LoopEnd=1025,validS2Len=131328;} tempLoopInfo;",
        "struct {unsigned bIdx=0,actS2Size=131584,s2LoopEnd=1027,validS2Len=131584,"
        "actS1Size=4,gS1LoopEnd=0,n2Idx=0;bool curActSeqLenIsZero=false,needDealActS1LessThanS1=false;} tempLoopInfo;",
    )
    code = code.replace(
        "Vector vectorService;Cube matmulService;",
        """Vector vectorService;Cube matmulService;
 void CalcGS1LoopParams(unsigned b){tempLoopInfo.bIdx=b;}
 void CalcS2LoopParams(unsigned,unsigned){splitCoreInfo.s2Start=0;tempLoopInfo.s2LoopEnd=tempLoopInfo.actS2Size/128-1;tempLoopInfo.validS2Len=tempLoopInfo.actS2Size;}
 void DealActSeqLenIsZero(unsigned,unsigned,unsigned){}
""",
    )
    code = code.replace(
        "i.qScaleLoop=q;i.kScaleLoop=k;i.bIdx=tempLoopInfo.bIdx;",
        "i.qScaleLoop=q;i.kScaleLoop=k;i.bIdx=tempLoopInfo.bIdx;i.actS1Size=tempLoopInfo.actS1Size;i.actS2Size=tempLoopInfo.actS2Size;",
    )
    # Real declarations supply these defaults; the extracted definitions do not.
    code = code.replace("uint32_t shard, uint32_t shards)", "uint32_t shard=0, uint32_t shards=1)")
    code = code.replace("bool last, bool newQuery)", "bool last, bool newQuery=false)")
    return (
        code
        + r"""
// Model ComputeMm1's actual 128/256-key cache acquisition/return protocol.
// Unlike query banks it must drain at every discontinuous physical segment.
void checkKeyBanks(const std::vector<QLICommon::RunInfo>& calls){
 for(unsigned width:{128U,256U}){
  bool credits[3]={true,true,true},valid=false;unsigned bank=0,start=0,loaded=0;
  for(const auto& info:calls){
   unsigned base=info.s2Idx*128;
   if(info.isFirstS2InnerLoop){assert(!valid);valid=false;}
   bool hit=width==256 && valid && base>=start && base+128<=start+loaded;
   if(!hit){assert(!valid && credits[bank]);credits[bank]=false;valid=true;
    start=base;loaded=std::min(width,info.actS2Size-base);
   }
   if(width==128 || base+128>=start+loaded || info.isLastS2InnerLoop){
    assert(valid && !credits[bank]);credits[bank]=true;bank=(bank+1)%3;valid=false;
   }
  }
  assert(!valid && credits[0] && credits[1] && credits[2]);
 }
}
void checkScaleBankOrder(const std::vector<QLICommon::RunInfo>& calls,bool earlyPass){
 bool normal=!earlyPass,seen=false;unsigned previous=0;
 for(const auto& info:calls){
  // ScoreRecent -> normal has its pre-existing reset. This change only
  // extends the established normal-scan bank sequence across skipped holes.
  if(earlyPass && info.s2Idx==0){normal=true;seen=false;earlyPass=false;}
  if(!normal)continue;
  bool first=(info.s2Idx-info.s2Start)%QLICommon::KScaleTiles==0;
  if(seen && first){
   assert(info.kScaleLoop==previous+1);
   // In the legacy 16-tile protocol the credit was returned at group start,
   // before its tail consumers. A newly split span must use the other bank,
   // not overwrite that tail merely because an old credit is available.
   assert(info.kScaleLoop%2!=previous%2);
  }else if(seen){assert(info.kScaleLoop==previous);}
  previous=info.kScaleLoop;seen=true;
 }
}
void setup(Model& m,int32_t* bounds,uint32_t* ends,unsigned core){
 m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
 m.aiCoreIdx=core;m.tmpBlockIdx=2*core;m.vectorService.activeQueries=4;
}
int main(){
 using namespace kvip_recent_index;
 for(unsigned h:{131328U,131584U})for(unsigned mode=0;mode<9;++mode)
 for(unsigned core=0;core<4;++core)for(bool aic:{false,true}){
  cubeMode=aic;Model m;m.runtimeShape_.history=h;m.tempLoopInfo.actS2Size=h;
  int32_t bounds[2]={int32_t(mode==0?20001:mode==1?16384:mode==2?h-1:mode==3?16384:mode==4?32763:mode==5?16384:mode==6?h-32769:20001),
   int32_t(mode==0?20513:mode==1?16385:mode==2?h:mode==3?h:mode==4?70003:mode==5?16384:mode==6?h-7:mode==7?20001:20777)};
  uint32_t ends[4]={h-4,h-3,h-2,h-1};if(mode==8)std::fill_n(ends,4,0U);
  setup(m,bounds,ends,core);
  if(mode==7)m.constInfo.attenMaskFlag=true; // Explicit fallback.
  m.CalcS2LoopParams(0,0);auto skip=m.ReusableRecentTiles();
  auto early=Tiles(Clip(bounds[0],bounds[1],16384,h,ends[3]));
  m.ProcessMain();
  const auto& calls=aic?m.matmulService.calls:m.vectorService.calls;
  checkKeyBanks(calls);
  checkScaleBankOrder(calls,true);
  std::vector<unsigned> visits(h/128);
  for(auto info:calls){assert(info.s2Idx<h/128);++visits[info.s2Idx];}
  for(unsigned tile=0;tile<h/128;++tile){
   bool prior=tile>=early.begin && tile<early.end;
   bool omitted=tile>=skip.begin && tile<skip.end;
   assert(visits[tile]==1U+prior-omitted);
  }
  m.matmulService.drained();m.vectorService.drained();
  if(!aic){
   assert(m.vectorService.residentCalls==1 && m.vectorService.finalCalls==1);
#if !defined(V9_UBMEM) || !V9_UBMEM
   for(auto range:m.vectorService.replayRanges){
    for(unsigned tile=range.first+1;tile<range.second;++tile)assert(!rotating::IsTrigger(h,16384,8,tile));
   }
   std::vector<unsigned> omitted;
   for(unsigned tile=skip.begin;tile<skip.end;++tile)omitted.push_back(tile);
   assert(m.vectorService.replayed==omitted);
   // Classification starts only when the logical scan reaches that span;
   // each original service boundary must already have been observed.
   unsigned lastProgress=128;
   for(auto event:m.vectorService.order){
    if(event.first=='P')lastProgress=event.second;
    if(event.first=='R')assert(event.second==lastProgress);
   }
#endif
   unsigned epoch=rotating::ServiceRank(2,1,core);
   assert(m.vectorService.services==std::vector<unsigned>{epoch});
   assert(m.vectorService.progress.size()==h/128-128);
   for(unsigned i=0;i<m.vectorService.progress.size();++i)assert(m.vectorService.progress[i]==129+i);
  }
 }
 // Run the actual cold phase. Preserve every original window/service even
 // when a whole window, the phase's head, or all its tail is skipped.
 for(unsigned mode=0;mode<6;++mode)for(unsigned core=0;core<8;++core)
 for(bool aic:{false,true}){
  cubeMode=aic;Model m;const unsigned h=m.runtimeShape_.history;
  const unsigned half=rotating::SequenceHalfWindow(m.runtimeShape_)/128;
  int32_t bounds[2]={int32_t(mode==0?20001:mode==1?16384:mode==2?h-32769:mode==3?16384+2*half*128:mode==4?h-1:20001),
                   int32_t(mode==0?20513:mode==1?h:mode==2?h:mode==3?16384+4*half*128:mode==4?h:20001)};
  uint32_t ends[4]={h,h,h,h};setup(m,bounds,ends,core);
  m.CalcS2LoopParams(0,0);auto skip=m.ReusableRecentTiles();
  m.ProcessSequenceSplit<SequenceLiPhase::Cold>();
  const auto& calls=aic?m.matmulService.calls:m.vectorService.calls;
  checkKeyBanks(calls);
  checkScaleBankOrder(calls,false);
  std::vector<unsigned> expected;
  unsigned shard=core%2,last=0;
  for(unsigned epoch=1,begin=128+shard*half;epoch<8;++epoch,begin+=2*half){
   unsigned end=std::min(begin+half,h/128);last=end;
   for(unsigned tile=begin;tile<end;++tile)if(tile<skip.begin || tile>=skip.end)expected.push_back(tile);
   if(!aic && begin<end)assert(std::find(m.vectorService.progress.begin(),m.vectorService.progress.end(),end)!=m.vectorService.progress.end());
  }
  assert(calls.size()==expected.size());
  for(unsigned i=0;i<calls.size();++i){assert(calls[i].s2Idx==expected[i] && calls[i].loop==i);}
  m.matmulService.drained();m.vectorService.drained();
  if(!aic){
   unsigned epoch=rotating::ServiceRank(2,1,core);
   assert(m.vectorService.services==std::vector<unsigned>{epoch});
   assert(std::is_sorted(m.vectorService.progress.begin(),m.vectorService.progress.end()));
#if !defined(V9_UBMEM) || !V9_UBMEM
   std::vector<unsigned> omitted;
   for(unsigned e=1,begin=128+shard*half;e<8;++e,begin+=2*half){
    unsigned end=std::min(begin+half,h/128);
    for(unsigned tile=begin;tile<end;++tile)if(tile>=skip.begin && tile<skip.end)omitted.push_back(tile);
    for(auto range:m.vectorService.replayRanges)if(range.first>=begin && range.first<end)assert(range.second<=end);
   }
   assert(m.vectorService.replayed==omitted);
#endif
  }
 }
 assert(ReusableTiles(16384,20001,16384,131584,131456,131000).end==0);
 assert(ReusableTiles(-1,20001,16384,131584,131584,131000).end==0);
}
"""
    )


@pytest.mark.parametrize("backend", ["urma", "ubmem"])
@pytest.mark.parametrize("skip", [0, 1])
@pytest.mark.parametrize("pipeline", [0, 1])
@pytest.mark.parametrize("scale_tiles", [8, 16])
def test_actual_main_and_sequence_skip(tmp_path, skip, pipeline, scale_tiles, backend):
    cpp = tmp_path / "skip.cpp"
    cpp.write_text(skip_model())
    exe = tmp_path / "skip"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            f"-DKVIP_RECENT_SKIP={skip}",
            f"-DV9_UBMEM={int(backend == 'ubmem')}",
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
