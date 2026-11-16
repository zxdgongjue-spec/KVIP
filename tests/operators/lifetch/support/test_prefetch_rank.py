# SPDX-License-Identifier: Apache-2.0
"""Execute both native rank outputs and the gate publication on CPU."""

import subprocess

import pytest

from test_recent_index import COMMON, INDEXER, function
from test_recent_register_kth import register_model_source
from test_recent_threshold_only import threshold_model_source


@pytest.mark.parametrize("rank", (2048, 1792, 1536))
def test_dual_rank_exact_keys_safe_bound_and_split_publication(tmp_path, rank):
    vector = (INDEXER / "indexer_vector.h").read_text()
    prepare = function(vector, "    __aicore__ inline void PrepareAdmissionThreshold(")
    loader = function(vector, "    __aicore__ inline bool TryResidentThreshold(")
    source = register_model_source(threshold_model_source().split("int main(){", 1)[0])
    source = source.replace(
        "enum class HardEvent {V_MTE2,MTE2_V,V_S};bool credits[3][8]{};",
        "enum class HardEvent {V_MTE2,MTE2_V,V_S,S_V};bool credits[4][8]{};",
    )
    source = source.replace(
        " T* GetPhyAddr()const{return p;}",
        " T* GetPhyAddr()const{return p;}\n T GetValue(unsigned i)const{assert(i<n);return p[i];}\n"
        " void SetValue(unsigned i,T v){assert(i<n);p[i]=v;}",
    )
    source = source.replace(
        " std::array<uint16_t,384> threshold{};",
        " std::array<uint16_t,384> threshold{},gate{};\n"
        " Tensor<uint16_t> admissionThreshold_{gate.data()+64,256};",
    )
    source = source.replace(loader, prepare + loader)
    source += r"""
#include "admission_policy.h"
using namespace AscendC::MicroAPI;
uint16_t kth(std::vector<uint16_t> v,unsigned rank){
 std::nth_element(v.begin(),v.end()-rank,v.end());return v[v.size()-rank];
}
void CheckPair(std::vector<uint16_t> values,unsigned gateRank){
 unsigned n=values.size();const auto original=values;
 std::array<uint16_t,256> safe,gate;safe.fill(0xa55a);gate.fill(0x5aa5);
 loads=stores=histograms=reductions=0;
 kvip_recent_threshold::RegisterThresholdPair(values.data(),safe.data()+64,gate.data()+64,
  uint16_t(n/256),uint16_t(n-2048+1),uint16_t(n-gateRank+1));
 uint16_t expectedSafe=kth(values,2048),expectedGate=kth(values,gateRank);
 assert(expectedGate>=expectedSafe);
 for(unsigned i=0;i<64;++i){assert(safe[i]==0xa55a&&safe[192+i]==0xa55a);assert(gate[i]==0x5aa5&&gate[192+i]==0x5aa5);}
 for(unsigned i=64;i<192;++i){assert(safe[i]==expectedSafe);assert(gate[i]==expectedGate);}
 assert(values==original&&loads==2*n/256&&histograms==6*n/256&&stores==2&&reductions==6);
}
int main(){
 assert(kvip_admission::Valid({512,64,0,1}));
 assert(!kvip_admission::Valid({512,64,1,1}));
 std::mt19937 random(918);
 for(unsigned n:{2048U,4096U,16384U,16640U,18432U})for(unsigned mode=0;mode<7;++mode){
  std::vector<uint16_t> values(n);
  for(auto& v:values)v=mode==0?0:mode==1?65535:mode==2?uint16_t(32760+random()%17):
    mode==3?uint16_t(random()%2?255:256):uint16_t(random());
  if(mode==4)std::fill(values.end()-127,values.end(),0); // Padded, not a candidate.
  for(unsigned rank:{1U,1536U,1792U,2048U})CheckPair(values,rank);
 }
 for(unsigned q=0;q<4;++q){
  Model m;
  for(auto& v:m.scores)v=uint16_t(32000+random()%1024);
  const auto original=m.scores;
  int32_t bounds[2]={130000,130257};m.ctx.thresholdBounds=reinterpret_cast<uint64_t>(bounds);
  uint32_t ends[4]={131328,130128,130003,131328};m.ctx.queryEnds=reinterpret_cast<uint64_t>(ends);
  std::vector<uint16_t> seed(m.scores.begin()+q*131328,m.scores.begin()+q*131328+16384);
  seed.insert(seed.end(),m.scores.begin()+q*131328+bounds[0],m.scores.begin()+q*131328+std::min(uint32_t(bounds[1]),ends[q]));
  assert(m.TryResidentThreshold(q,q,16384,131328));
  assert(m.scores==original);
  uint16_t safe=kth(seed,2048),gate=kth(seed,KVIP_PREFETCH_RANK);unsigned bank=(q%2)*128;
  assert(m.prefixThreshold_.GetValue(bank)==safe&&m.admissionThreshold_.GetValue(bank)==gate);
  auto previous=m.threshold;
  loads=stores=histograms=reductions=0;
  m.PrepareAdmissionThreshold(q,true);
  assert(loads==0&&histograms==0); // Fast path reuses both outputs.
  assert(m.prefixThreshold_.GetValue(bank)==safe&&m.prefixThreshold_.GetValue(bank+1)==gate);
  for(unsigned i=0;i<m.threshold.size();++i)if(i!=64+bank+1)assert(m.threshold[i]==previous[i]);
  // Actual sequence-split wire image: bank[0] is safe, bank[1] is admission.
  Model cold;cold.threshold=m.threshold;auto stored=cold.threshold;cold.gate.fill(0xaaaa);
  cold.PrepareAdmissionThreshold(q,false,true);
  assert(cold.threshold==stored&&cold.admissionThreshold_.GetValue(bank)==gate);
  assert(loads==0&&histograms==0);
  // Legacy fallback has the exact seed TopK list, not a full-resident copy.
  std::sort(seed.begin(),seed.end(),std::greater<uint16_t>());
  std::copy_n(seed.begin(),2048,m.out.begin());std::shuffle(m.out.begin(),m.out.end(),random);
  m.gate.fill(0xbbbb);m.prefixThreshold_.SetValue(bank+1,0xcccc);
  m.PrepareAdmissionThreshold(q,false);
  assert(m.prefixThreshold_.GetValue(bank)==safe&&m.prefixThreshold_.GetValue(bank+1)==gate);
  assert(m.admissionThreshold_.GetValue(bank)==gate&&loads==16);
  for(auto& row:credits)for(bool credit:row)assert(!credit);
 }
}
"""
    cpp, exe = tmp_path / "rank.cpp", tmp_path / "rank"
    cpp.write_text(source)
    subprocess.run(
        ["c++", "-std=c++17", "-O1", "-fsanitize=address,undefined",
         f"-DKVIP_PREFETCH_RANK={rank}", "-DV9_UBMEM=0",
         "-I", str(COMMON), "-I", str(INDEXER), "-I", str(INDEXER.parents[1] / "include"),
         str(cpp), "-o", str(exe)],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_gate_rank_never_changes_final_or_victim_selection():
    vector = (INDEXER / "indexer_vector.h").read_text()
    final = function(vector, "__aicore__ inline bool QLIVector<QLIT>::TryPrunedFinalTopK(")
    save = function(vector, "    __aicore__ inline void SaveResident(")
    assert "KVIP_PREFETCH_RANK" not in final and "admissionThreshold_" not in final
    assert save.index("PrepareAdmissionThreshold(query, thresholdReady)") < save.index("DataCopy(thresholds[")
    assert "ClassifyLocalResidentScores" in save and "prefixThreshold_[(query % 2) * 128]" in save
    assert "admissionThreshold_" not in save
