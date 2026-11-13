"""Execute the diagnostic packer/hooks and actual pruning driver on the CPU."""

import subprocess

from test_final_topk_prune import source
from test_recent_index import INDEXER, function


def probe_source():
    vector = (INDEXER / "indexer_vector.h").read_text()
    begin = function(vector, "    __aicore__ inline void BeginTopkPruneProbe(")
    publish = function(vector, "    __aicore__ inline void PublishTopkPruneProbe(")
    model = source().split("int main(){", 1)[0]
    model = model.replace(
        "#define KVIP_FINAL_TOPK_PRUNE 1",
        "#define KVIP_FINAL_TOPK_PRUNE 1\n#define KVIP_STREAMING_LI 1\n"
        "#define KVIP_TOPK_PRUNE_DIAGNOSTICS 1\n#define __gm__",
    )
    model = model.replace(
        "template<SpecialPurposeReg> uint64_t GetSpr(){return ar;}",
        "bool probeBadAr=false;\ntemplate<SpecialPurposeReg> uint64_t GetSpr(){return probeBadAr?65536:ar;}",
    )
    model = model.replace(
        "struct Model {",
        r"""
#include "topk_prune_probe.h"
constexpr unsigned SIM_Q=4;
namespace pipeline_ab {struct Args {uint32_t prefetchCount=0;uint64_t times=0;};}
unsigned flushes=0;
namespace kvaf_device {void CleanCacheLines(uint8_t* address,unsigned bytes){
 assert(bytes==128 && uintptr_t(address)%128==0);++flushes;}}
struct Model {
 uint8_t* streamingWorkspace_=nullptr;
 bool topkPruneProbeEnabled_=false;
 kvip_topk_probe::Record topkPruneProbe_;
"""
        + begin
        + publish
        + "\n#undef KVIP_FINAL_TOPK_PRUNE\n#define KVIP_FINAL_TOPK_PRUNE 0\n"
        + begin.replace("BeginTopkPruneProbe(", "BeginDisabledProbe(")
        + "\n#undef KVIP_FINAL_TOPK_PRUNE\n#define KVIP_FINAL_TOPK_PRUNE 1\n",
    )
    return model + r"""
constexpr uint64_t sentinel=0x1122334455667788ULL;
struct Fixture {
 Model model;
 alignas(128) std::array<uint64_t,16*64+64*16> times;
 pipeline_ab::Args args;
 Fixture(){times.fill(sentinel);args.prefetchCount=0x80000000U;
  args.times=reinterpret_cast<uint64_t>(times.data());
  model.streamingWorkspace_=reinterpret_cast<uint8_t*>(&args);}
 uint64_t* row(unsigned query){return times.data()+16*64+query*16;}
 void CheckRow(unsigned query,unsigned status,bool activated){
  auto* words=row(query);
  for(unsigned i=0;i<12;++i)assert(words[i]==sentinel);
  assert(uint32_t(words[12])==kvip_topk_probe::kMagic);
  assert(((words[12]>>48)&255)==status);
  assert(((words[12]>>56)&1)==activated && !(words[12]>>57));
 }
};
void CheckSuccessful(unsigned query,unsigned n){
 Fixture f;auto& m=f.model;auto* values=m.gm.data()+(query%2)*m.constInfo_.kSeqSize;
 for(unsigned i=0;i<m.constInfo_.kSeqSize;++i)values[i]=uint16_t(1+i%65535);
 m.prefixThreshold_.p[(query%2)*128]=10000;
 unsigned calls=selectCalls,elements=selectElements;
 m.BeginTopkPruneProbe(query,n);
 assert(m.TryPrunedFinalTopK(query,query%2,n));
 m.PublishTopkPruneProbe(query);m.check();f.CheckRow(query,kvip_topk_probe::Success,true);
 auto* words=f.row(query);
 assert(((words[12]>>32)&65535)==0);
 assert(uint32_t(words[13])==n);
 assert((words[13]>>32)==std::count_if(values,values+n,[](uint16_t v){return v>=10000;}));
 assert(uint32_t(words[14])==selectElements-elements);
 assert((words[14]>>32)==selectCalls-calls);
 assert(uint32_t(words[15])==n && ((words[15]>>32)&65535)==10000);
 assert((words[15]>>48)==(n+16383)/16384);
 unsigned prior=flushes;m.PublishTopkPruneProbe(query);assert(flushes==prior);
 // A second packed batch/query resets counts and cannot borrow the first row.
 auto saved=f.times;unsigned next=query+4;
 m.BeginTopkPruneProbe(next,0);m.PublishTopkPruneProbe(next);
 f.CheckRow(next,kvip_topk_probe::EmptyQuery,false);
 assert(f.row(next)[13]==0 && f.row(next)[14]==0 && f.row(next)[15]==0);
 assert(std::equal(saved.begin()+16*64+query*16,saved.begin()+16*64+(query+1)*16,f.row(query)));
}
int main(){
 for(unsigned n:{16384U,16385U,131328U})CheckSuccessful(5,n);
 const unsigned reasons[]={kvip_topk_probe::TopkOnly,kvip_topk_probe::LoadDistribution,
  kvip_topk_probe::QueryTile,kvip_topk_probe::UnsplitQuery,kvip_topk_probe::TaskCount,
  kvip_topk_probe::TopkSize,kvip_topk_probe::TrunkSize,kvip_topk_probe::ShortResident,
  kvip_topk_probe::ShortTopk,kvip_topk_probe::ScratchSize};
 for(unsigned i=0;i<10;++i){Fixture f;auto& m=f.model;unsigned n=131328;
  if(i==0)m.streamingTopkOnly_=true;if(i==1)m.constInfo_.isLDOpen=true;
  if(i==2)m.s1BaseSize_=4;if(i==3)m.runtimeShape_.splitQ=0;
  if(i==4)m.runtimeShape_.aicCount=31;if(i==5)m.topkCount_=1024;
  if(i==6)m.trunkLen_=8192;if(i==7)n=16000;if(i==8)n=1000;
  if(i==9)m.constInfo_.mBaseSize=32;
  unsigned before=dma;m.BeginTopkPruneProbe(9,n);
  assert(!m.TryPrunedFinalTopK(9,1,n));m.PublishTopkPruneProbe(9);
  f.CheckRow(9,kvip_topk_probe::GuardFallback,false);
  assert(((f.row(9)[12]>>32)&65535)&reasons[i]);
  assert(f.row(9)[13]==0 && f.row(9)[14]==0 && dma==before);m.check();
 }
 // Sequence-final records all applicable guards, not a silent empty slot.
 {Fixture f;auto& m=f.model;m.streamingTopkOnly_=true;m.s1BaseSize_=4;
  m.BeginTopkPruneProbe(63,131072);assert(!m.TryPrunedFinalTopK(63,1,131072));
  m.PublishTopkPruneProbe(63);f.CheckRow(63,kvip_topk_probe::GuardFallback,false);
  assert(((f.row(63)[12]>>32)&65535)==(kvip_topk_probe::TopkOnly|kvip_topk_probe::QueryTile));}
 for(unsigned high:{0U,2U}){Fixture f;auto& m=f.model;
  std::fill(m.gm.begin(),m.gm.end(),1);m.prefixThreshold_.p[0]=high;
  m.BeginTopkPruneProbe(4,131328);assert(!m.TryPrunedFinalTopK(4,0,131328));
  m.PublishTopkPruneProbe(4);m.check();
  f.CheckRow(4,high?kvip_topk_probe::InsufficientCandidates:kvip_topk_probe::ZeroThreshold,high!=0);
  assert(uint32_t(f.row(4)[13])==(high?131328U:0U));
  assert((f.row(4)[13]>>32)==0 && f.row(4)[14]==0);}
 {Fixture f;auto& m=f.model;std::fill(m.gm.begin(),m.gm.end(),3);
  m.prefixThreshold_.p[0]=2;m.BeginTopkPruneProbe(4,131328);AscendC::probeBadAr=true;
  assert(!m.TryPrunedFinalTopK(4,0,131328));AscendC::probeBadAr=false;
  m.PublishTopkPruneProbe(4);m.check();f.CheckRow(4,kvip_topk_probe::InvalidCompaction,true);
  assert(uint32_t(f.row(4)[13])==16384 && (f.row(4)[13]>>32)==0 && f.row(4)[14]==0);}
 {Fixture f;f.model.BeginDisabledProbe(4,131328);f.model.PublishTopkPruneProbe(4);
  f.CheckRow(4,kvip_topk_probe::Disabled,false);
  assert(f.row(4)[13]==0 && f.row(4)[14]==0 && f.row(4)[15]==131328);}
 // Runtime diagnostics off, prefix work and an unavailable times buffer must
 // neither write nor reuse the previous query's record.
 for(unsigned mode=0;mode<4;++mode){Fixture f;auto& m=f.model;unsigned prior=flushes;
  m.topkPruneProbeEnabled_=true;m.topkPruneProbe_.status=kvip_topk_probe::Success;
  if(mode==0)f.args.prefetchCount=0;if(mode==1)m.prefixActive_=true;
  if(mode==2)f.args.times=0;if(mode==3)m.streamingWorkspace_=nullptr;
  m.BeginTopkPruneProbe(4,131328);m.PublishTopkPruneProbe(4);
  assert(flushes==prior);assert(std::all_of(f.times.begin(),f.times.end(),[](uint64_t v){return v==sentinel;}));}
 assert(kvip_topk_probe::GuardReasons(false,true,false,false,2,true,16,32,2048,16384,131328,16384,32768)
  ==(kvip_topk_probe::ScoreType|kvip_topk_probe::Prefix));
}
"""


def test_actual_probe_hooks_driver_and_protocol(tmp_path):
    cpp = tmp_path / "probe.cpp"
    cpp.write_text(probe_source())
    exe = tmp_path / "probe"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
            "-I", str(INDEXER), str(cpp), "-o", str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_compile_disabled_probe_has_no_types_or_driver_code(tmp_path):
    # Compare preprocessed actual driver with/without an explicit zero: the
    # diagnostic fields/counter branches must not survive in timing binaries.
    driver = function(
        (INDEXER / "indexer_vector.h").read_text(),
        "__aicore__ inline bool QLIVector<QLIT>::TryPrunedFinalTopK(",
    )
    cpp = tmp_path / "disabled.cpp"
    cpp.write_text('#include "topk_prune_probe.h"\n' + driver)
    command = ["c++", "-E", "-P", "-x", "c++", "-I", str(INDEXER), str(cpp)]
    default = subprocess.check_output(command, text=True)
    explicit = subprocess.check_output(command + ["-DKVIP_TOPK_PRUNE_DIAGNOSTICS=0"], text=True)
    assert default == explicit
    assert "kvip_topk_probe" not in default
    assert "topkPruneProbe" not in default
