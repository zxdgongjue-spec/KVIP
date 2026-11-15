"""Execute admission arithmetic and the device gate preparation on the CPU."""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function


def test_admission_limits_and_independent_threshold(tmp_path):
    prepare = function(
        (INDEXER / "indexer_vector.h").read_text(),
        "    __aicore__ inline void PrepareAdmissionThreshold(",
    )
    cpp = tmp_path / "admission.cpp"
    cpp.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#define __aicore__
#define __gm__
#include "admission_policy.h"
#include "prefetch_budget.h"
constexpr unsigned SIM_Q=4, EVENT_ID0=0;
enum class HardEvent {V_S,S_V};
bool credits[2]{};
unsigned barriers=0;
template<HardEvent E> void SetFlag(unsigned){++barriers;assert(!credits[unsigned(E)]);credits[unsigned(E)]=true;}
template<HardEvent E> void WaitFlag(unsigned){assert(credits[unsigned(E)]);credits[unsigned(E)]=false;}
template<class T> struct Tensor {
 std::array<T,256>* p;
 T GetValue(unsigned i){return p->at(i);}
 void SetValue(unsigned i,T v){p->at(i)=v;}
};
struct Model {
 uint64_t admissionControls_=0;
 std::array<uint16_t,256> prefixStorage{},gateStorage{};
 Tensor<uint16_t> prefixThreshold_{&prefixStorage},admissionThreshold_{&gateStorage};
"""
        + prepare
        + r"""
};
int main(){
 using namespace kvip_admission;
 assert(Valid({0,1,0,0})&&Valid({2048,480,65535,1}));
 for(auto invalid:{Controls{-1,64,0,0},{2049,64,0,0},{1,0,0,0},
     {1,481,0,0},{1,64,-1,0},{1,64,65536,0},{1,64,0,2}})assert(!Valid(invalid));
 for(uint32_t key=0;key<=65535;++key){
  assert(LiftKey(uint16_t(key),0)==key);
  for(uint32_t steps:{1U,7U,128U,65535U,UINT32_MAX}){
   auto gate=LiftKey(uint16_t(key),steps);
   assert(gate>=key&&gate==std::min<uint64_t>(65535,uint64_t(key)+steps));
  }
 }
 assert(WindowAllowance(64,1,0,true)==128);
 assert(WindowAllowance(64,1,0,false)==64);
 assert(WindowAllowance(64,7,0,true)==480);
 assert(WindowAllowance(64,2,192,true)==0);
 assert(WindowAllowance(480,UINT32_MAX,0,true)==480);
 std::mt19937 rng(137);
 for(unsigned trial=0;trial<20000;++trial){
  uint32_t base=1+rng()%480,total=rng()%2049,queue=rng()%2049,slotLimit=rng()%2049;
  uint32_t posted=0,slots=0;bool carry=bool(rng()%2);
  for(uint32_t epoch=0;epoch<32;++epoch){
   auto window=WindowAllowance(base,epoch,posted,carry);
   auto allowance=rotating::PrefetchAllowance(slots,posted,slotLimit,total,queue,window);
   assert(allowance<=480&&allowance<=window);
   assert(allowance<=total-posted&&allowance<=queue-posted&&allowance<=slotLimit-slots);
   auto admitted=std::min<uint32_t>(allowance,rng()%641);
   posted+=admitted;slots+=admitted;
   assert(posted<=total&&posted<=queue&&slots<=slotLimit);
   if(carry)assert(posted<=uint64_t(epoch+1)*base);
   else assert(admitted<=base);
  }
 }
 // Zero total and exhausted resources forbid admission but do not change
 // any selection threshold. Completion publication lives outside the loop.
 assert(rotating::PrefetchAllowance(0,0,2048,0,2048,480)==0);
 assert(rotating::PrefetchAllowance(2048,0,2048,2048,2048,480)==0);
 assert(rotating::PrefetchAllowance(10,10,2048,2048,10,480)==0);
 std::array<Controls,16> controls;
 for(unsigned row=0;row<controls.size();++row)controls[row]={512,64,int32_t(row*4311),int32_t(row%2)};
 Model model;
 model.prefixStorage.fill(49000);model.gateStorage.fill(123);
 // A gate-enabled binary always initializes its fixed gate, including null
 // controls and zero steps. The baseline binary compiles this code out.
 model.PrepareAdmissionThreshold(0);
 assert(model.gateStorage[0]==49000&&barriers==2);
 model.admissionControls_=uint64_t(controls.data());
 for(unsigned query=0;query<64;++query){
  auto oldPrefix=model.prefixStorage,oldGate=model.gateStorage;unsigned oldBarriers=barriers;
  model.PrepareAdmissionThreshold(query);
  assert(model.prefixStorage==oldPrefix); // Exact Kth and victim protection are immutable.
  unsigned bank=(query%2)*128;
  assert(model.gateStorage[bank]==LiftKey(49000,controls[query/4].keySteps));
  assert(barriers==oldBarriers+2);
  for(unsigned i=0;i<256;++i)if(i!=bank)assert(model.gateStorage[i]==oldGate[i]);
  assert(!credits[0]&&!credits[1]);
 }
}
"""
    )
    exe = tmp_path / "admission"
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
         "-I", str(INDEXER.parents[1] / "include"), "-I", str(COMMON),
         str(cpp), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_gate_is_not_used_as_an_exact_selection_bound():
    source = (INDEXER / "indexer_vector.h").read_text()
    final = function(source, "__aicore__ inline bool QLIVector<QLIT>::TryPrunedFinalTopK(")
    victims = function(source, "    __aicore__ inline void SaveResident(")
    assert "admissionThreshold_" not in final
    assert "prefixThreshold_.GetValue" in final
    assert "admissionThreshold_" not in victims
    assert "ClassifyLocalResidentScores" in victims and "prefixThreshold_" in victims


@pytest.mark.parametrize("adaptive,gate", [(0, 0), (1, 0), (1, 1)])
def test_native_admission_compiles_out_of_baseline(adaptive, gate):
    """Preprocess real sources without CANN, checking the selected path."""
    kernel = INDEXER.parent
    sources = [kernel / "indexer.cpp", INDEXER / "indexer.h", INDEXER / "indexer_vector.h",
               kernel / "lifecycle.cpp", kernel / "prefetch.h"]
    expanded = []
    for path in sources:
        source = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith("#include")
        )
        result = subprocess.run(
            ["c++", "-E", "-P", "-x", "c++", "-D__CCE_AICORE__=1", "-DV9_UBMEM=0",
             "-DKVIP_STREAMING_LI=1", f"-DKVIP_ADAPTIVE_ADMISSION={adaptive}",
             f"-DKVIP_PREFETCH_GATE={gate}", "-"],
            input=source, text=True, capture_output=True, check=True,
        )
        expanded.append(result.stdout)
    li = "\n".join(expanded[:3])
    if not gate:
        assert "admissionControls" not in li
        assert "PrepareAdmissionThreshold" not in li
        assert "admissionThresholdBuf_" not in li
        assert "coldThreshold_" not in li
        assert "(__ubuf__ uint16_t *)prefixThreshold_.GetPhyAddr() + (curAivS1Idx % 2) * 128" in li
    else:
        assert "PrepareAdmissionThreshold" in li
        assert "coldThreshold_" not in li and "admissionEnabled_" not in li
        allocate = li.index("pipe->InitBuffer(admissionThresholdBuf_, 512)")
        assert allocate > li.index("pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize)")
        assert allocate > li.index("pipe->InitBuffer(streamingBitmapBuf_")
    tail = "\n".join(expanded[3:])
    if not adaptive:
        assert "admissionControls" not in tail
        assert "WindowAllowance" not in tail
    else:
        assert "admissionControls" in tail
        assert "WindowAllowance" in tail
