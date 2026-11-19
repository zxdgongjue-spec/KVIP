"""Run native ranked admission with hot hits, victims, ABC and Q4 reuse."""

from pathlib import Path
import subprocess


def test_native_ranked_admission_preserves_hot_hits_and_refetches_victims(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/operators/lightning_indexer_with_fetch/op_kernel/prefetch.h").read_text()
    begin = source.index("inline void GatherRankedCandidateMap(")
    end = source.index("\n#endif", begin)
    # Execute the production SIMT body on CPU, one logical lane at a time.
    cpp = tmp_path / "admit.cpp"
    cpp.write_text('''
#include <cassert>
#include <cstdint>
#include <vector>
#define __gm__
#define __ubuf__
struct {unsigned x;} threadIdx;
struct Context {uint64_t rowOwners,hotValid,reservedHot,tailBounds;};
template<class T>T asc_ldcg(T* p){return *p;}
uint32_t OwnerRow(Context* c,uint32_t b){return c->rowOwners?reinterpret_cast<uint32_t*>(c->rowOwners)[b]:b;}
''' + source[begin:end] + '''
int main() {
    constexpr unsigned R=16,H=640,B=2;
    uint32_t owners[B]={2,1},tail[B*2]={28,32,12,14};
    std::vector<uint32_t> valid(3*R,1),map(H,UINT32_MAX),ids(H),out(H);
    std::vector<int32_t> reserved(B*R,-1);
    for(unsigned i=0;i<H;++i)ids[i]=i;
    valid[2*R+3]=0;reserved[4]=0; // Invalid hot and a possible victim both need DDR.
    valid[0]=0;reserved[R+1]=7;  // Wrong-owner data must have no effect on row 0.
    map[5]=R+2;map[20]=R+3;     // Previously submitted IDs, including a hot victim.
    reserved[5]=2;
    for(unsigned t=28;t<32;++t)map[t]=R+8192+t%8;
    Context ctx{uint64_t(owners),uint64_t(valid.data()),uint64_t(reserved.data()),uint64_t(tail)};
    auto run=[&](unsigned b){for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
        GatherRankedCandidateMap(ids.data(),out.data(),map.data(),H,&ctx,b,R);};
    run(0);
    for(unsigned t=0;t<H;++t) {
        bool expectedMiss=t==3 || t==4 || (t>=R && t!=20 && !(t>=28 && t<32));
        assert((out[t]==UINT32_MAX)==expectedMiss);
    }
    // Original map-only admission wrongly takes hot token 0, whose Index
    // location is still used as a hit by the final resolver.
    assert(map[0]==UINT32_MAX && out[0]==0);
    // Q4 dedup: a reserved token stays eligible until actually submitted.
    assert(out[4]==UINT32_MAX);map[4]=R+6;run(0);assert(out[4]==R+6);
    assert(out[3]==UINT32_MAX);
    // A tail within the resident coordinate range follows the ABC mapping.
    map[12]=R+8192+4;map[13]=R+8192+5;run(1);
    assert(out[12]==map[12] && out[13]==map[13] && out[1]==UINT32_MAX);
    // Preserve the ordinary optional-metadata defaults.
    ctx.rowOwners=ctx.hotValid=ctx.reservedHot=ctx.tailBounds=0;run(0);
    for(unsigned t=0;t<R;++t)assert(out[t]==t);
}
''')
    binary = tmp_path / "admit"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
