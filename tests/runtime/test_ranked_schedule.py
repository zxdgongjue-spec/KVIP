"""The URMA ranked experiment uses eight cold windows, not nine services."""

from pathlib import Path
import subprocess

from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens


ROOT = Path(__file__).resolve().parents[2]


def test_every_aiv_has_one_fetch_turn_and_seed_has_none():
    geometry = li_geometry(16, intermediate_fetches=8)
    groups = geometry["service_groups"]
    assert len(groups) == 8
    assert geometry["service_aivs_per_window"] == 8
    assert all(len(group) == 4 for group in groups)
    assert sorted(core for group in groups for core in group) == list(range(32))
    boundaries = trigger_tokens(16, 131328, 16384, intermediate_fetches=8)
    assert len(set(boundaries)) == 8
    assert boundaries[0] > 16384 and boundaries[-1] == 131328


def test_shared_sequence_schedule_keeps_ubmem_geometry(tmp_path):
    header = ROOT / "csrc/operators/common/lifetch/sequence_schedule.h"
    runtime = ROOT / "csrc/operators/lightning_indexer_with_fetch/include/runtime_shape.h"
    for cold_only in (0, 1):
        source = tmp_path / f"schedule{cold_only}.cpp"
        source.write_text(f'''
#include <cassert>
#include "{runtime}"
#define KVIP_COLD_FETCH_ONLY {cold_only}
namespace rotating {{
#include "{header}"
}}
int main() {{
    rotating::RuntimeShape s{{}};
    s.history=131072; s.resident=16384; s.splitQ=1; s.indexHeads=32;
    s.rounds={8 if cold_only else 9};
    assert(rotating::UseSequenceSplit(s));
    assert(rotating::ColdWindowCount(s)==8);
    auto half=rotating::SequenceHalfWindow(s);
    assert(rotating::SequenceWindowEnd(s.history,s.resident,half,{0 if cold_only else 1})>s.resident);
    assert(rotating::SequenceWindowEnd(s.history,s.resident,half,{7 if cold_only else 8})==s.history);
    s.history=131328;
    assert(!rotating::UseSequenceSplit(s));
}}
''')
        binary = tmp_path / f"schedule{cold_only}"
        subprocess.run(["c++", "-std=c++17", str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


def test_partial_seed_writeback_cannot_overwrite_a_concurrent_producer(tmp_path):
    source = (ROOT / "csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer/ranked_methods.h").read_text()
    start = source.index("        for(uint32_t b=lane,i=0;b<runtimeShape_.batch;b+=runtimeShape_.serviceLanes,++i) {")
    end = source.index("        SetFlag<HardEvent::MTE3_S>", start)
    # Execute the actual native copy loop with host tensors. The GM header of
    # an unready Q changes AFTER staging, as it can during concurrent seed work.
    cpp = tmp_path / "commit.cpp"
    cpp.write_text('''
#include <algorithm>
#include <cassert>
#include <cstdint>
struct Tensor { uint64_t* p; Tensor operator[](unsigned i) const {return {p+i};} };
unsigned calls=0, words=0;
void DataCopy(Tensor out,Tensor in,unsigned n) {++calls;words+=n;std::copy_n(in.p,n,out.p);}
struct {unsigned batch=16,serviceLanes=8;} runtimeShape_;
int main() {
    uint64_t gm[16*64]={}, staged[2*64]={};
    Tensor metadata{gm},incrementalMetadata_{staged};
    unsigned lane=0,readyMasks[2]={7,15};
    std::fill_n(staged,128,11);
    std::fill_n(gm+3*16,16,99); // The still-running producer publishes newer state.
''' + source[start:end] + '''
    for(unsigned i=48;i<64;++i)assert(gm[i]==99);
    assert(calls==4 && words==112);
    calls=words=0;readyMasks[0]=15;
''' + source[start:end] + '''
    assert(calls==2 && words==128); // One row copy/request in the common path.
}
''')
    binary = tmp_path / "commit"
    subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
