"""Execute the native producer/consumer permutation and its causal boundaries."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "csrc/operators/lightning_indexer_with_fetch"


def test_reverse_blocks_preserve_ids_scales_and_causality(tmp_path):
    source = (NATIVE / "op_kernel/prefetch.h").read_text()
    begin = source.index("__simt_callee__ inline uint32_t DecodeScanToken(")
    end = source.index("\n}", begin) + 2
    decoder = source[begin:end]
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#define __simt_callee__
#define __ubuf__
#define KVIP_REVERSE_COLD 1
#include "cold_scan.h"
DECODER
int main() {
 std::mt19937 random(15);
 for (unsigned trial=0;trial<80;++trial) {
  unsigned resident=2048+(random()%64)*256,history=resident+256+(random()%750)*256;
  unsigned end=resident+(random()%(history-resident+1));
  unsigned ends[4]={end,end+1,end+2,end+3};
  auto range=kvip_cold_scan::Bounds(resident,history,ends);
  unsigned cfg[64]{};cfg[60]=range.begin;cfg[61]=range.end;
  std::vector<unsigned> seen(history),forward(history),reverse(history);
  std::iota(forward.begin(),forward.end(),0);
  std::shuffle(forward.begin(),forward.end(),random);
  for(unsigned scan=0;scan<history;++scan) {
   unsigned token=kvip_cold_scan::Remap(scan,range);
   assert(token<history && !seen[token]++);
   assert(kvip_cold_scan::Remap(token,range)==scan);
   assert(DecodeScanToken(scan,cfg[60],cfg[61])==token);
   if(scan<resident || scan>=range.end)assert(scan==token);
   for(unsigned q=0;q<4;++q)assert((scan<ends[q])==(token<ends[q]));
   reverse[token]=forward[token]; // LI stores at physical, not scan coordinates.
   assert(scan%128==token%128 && scan%2048==token%2048);
   if(scan%2048==0 && scan>=range.begin && scan<range.end)
    for(unsigned j=0;j<2048;++j)assert(kvip_cold_scan::Remap(scan+j,range)==token+j);
  }
  assert(forward==reverse);
  // The actual consumer decodes only admitted IDs, never the resume cursor.
  for(unsigned q=0;q<4;++q) {
   std::vector<unsigned> expected,actual;
   unsigned qEnd=std::min(ends[q],history),threshold=history*3/4;
   for(unsigned i=resident;i<qEnd;++i)if(forward[i]>threshold)expected.push_back(i);
   for(unsigned window=0;window<8;++window) {
    unsigned lo=resident+(history-resident)*window/8;
    unsigned hi=resident+(history-resident)*(window+1)/8;
    for(unsigned scan=lo;scan<std::min(hi,qEnd);++scan) {
     auto physical=kvip_cold_scan::Remap(scan,range);
     if(forward[physical]>threshold)actual.push_back(DecodeScanToken(scan,cfg[60],cfg[61]));
    }
   }
   std::sort(actual.begin(),actual.end());assert(expected==actual);
  }
 }
}
'''.replace("DECODER", decoder)
    cpp = tmp_path / "cold_scan.cpp"
    cpp.write_text(program)
    binary = tmp_path / "cold_scan"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined", "-I", str(NATIVE / "include"),
         str(cpp), "-o", str(binary)], check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_native_driver_keeps_query_lifetime_and_progress(tmp_path):
    from test_recent_pipeline import native_model, COMMON

    # Execute the actual ProcessBaseBlock and FinishBaseBlock with the existing
    # host event-ownership model; only matrix/vector arithmetic is stubbed.
    program = native_model().split("int main(){", 1)[0]
    program = program.replace('#include "recent_index.h"', '#include "recent_index.h"\n#include "cold_scan.h"')
    program = program.replace("struct Model {", """struct Model {
 kvip_cold_scan::Range coldScan_{};
 struct {unsigned kHeadNum=1,headDim=128;} constInfo;
""")
    program = program.replace("i.loop=loop;i.s2Idx=tile;", "i.loop=loop;i.s2Idx=tile;i.keyS2Idx=tile;")
    program = program.replace("i.tensorKeyOffset=uint64_t(tile)*128*128;",
                              "i.tensorKeyOffset=uint64_t(tile)*128*128;i.tensorKeyScaleOffset=uint64_t(tile)*128;")
    program += r'''
int main(){
 for(bool aic:{false,true})for(bool recent:{false,true}){
  cubeMode=aic;Model m;uint32_t ends[4]={130155,130156,130157,130158};
  m.coldScan_=kvip_cold_scan::Bounds(16384,131328,ends);
  int32_t bounds[2]={128000,130048};
  if(recent){m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);}
  unsigned loop=0,scale=0;bool had=m.ScoreRecent(loop,0,0,1);
  auto previous=aic?m.matmulService.calls.size():m.vectorService.calls.size();
  for(unsigned tile=0;tile<131328/128;++tile,++loop){
   if(tile%16==0)++scale;
   QLICommon::RunInfo info;
   m.ProcessBaseBlock(loop,tile,info,0,scale,had&&tile==0);
  }
  auto& calls=aic?m.matmulService.calls:m.vectorService.calls;
  unsigned loads=0;
  for(unsigned i=0;i<calls.size();++i){
   auto& c=calls[i];loads+=c.LoadQuery();
   if(i<previous){assert(c.keyS2Idx==c.s2Idx);continue;}
   unsigned scan=i-previous,physical=kvip_cold_scan::Remap(scan*128,m.coldScan_)/128;
   assert(c.s2Idx==scan && c.keyS2Idx==physical);
   assert(c.tensorKeyOffset==uint64_t(physical)*128*128);
   assert(c.tensorKeyScaleOffset==uint64_t(physical)*128);
   assert(c.isLastS2InnerLoop==(scan+1==131328/128));
   assert(c.kScaleLoop==scan/16+1);
  }
  assert(loads==1);m.matmulService.drained();m.vectorService.drained();
 }
}
'''
    cpp = tmp_path / "driver.cpp"
    cpp.write_text(program)
    binary = tmp_path / "driver"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
         "-DMODEL_SCALE_TILES=16", "-DKVIP_REVERSE_COLD=1", "-I", str(COMMON),
         "-I", str(NATIVE / "include"), str(cpp), "-o", str(binary)], check=True,
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_window_runs_cover_exact_physical_keys_without_duplicates(tmp_path):
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>
#include "cold_scan.h"
int main(){
 std::mt19937 rng(32);
 for(unsigned trial=0;trial<2500;++trial){
  unsigned resident=(rng()%160)*128,history=resident+128+(rng()%2000)*128;
  unsigned ends[4]={history-87,history-83,history-62,history-1};
  auto range=kvip_cold_scan::Bounds(resident,history,ends);
  if(trial%3==0)range={0,0};
  unsigned lo=rng()%history,hi=lo+rng()%(history-lo+1);
  auto runs=kvip_cold_scan::WindowRuns(lo,hi,range);
  assert(runs.count<=5);
  std::vector<unsigned> expected,actual;
  for(unsigned i=lo;i<hi;++i)expected.push_back(kvip_cold_scan::Remap(i,range));
  for(unsigned r=0;r<runs.count;++r){
   auto span=runs.spans[r];assert(span.begin<span.end && span.end<=history);
   for(unsigned i=span.begin;i<span.end;++i)actual.push_back(i);
  }
  std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
  assert(expected==actual);assert(std::adjacent_find(actual.begin(),actual.end())==actual.end());
 }
}
'''
    cpp = tmp_path / "window_runs.cpp"
    cpp.write_text(program)
    binary = cpp.with_suffix("")
    subprocess.run(["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
                    "-I", str(NATIVE / "include"), str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)
