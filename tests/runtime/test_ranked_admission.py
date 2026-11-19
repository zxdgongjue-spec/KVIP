"""Compile the production seed classifier and exercise transport admission rules."""

from pathlib import Path
import subprocess


def test_ranked_seed_tags_use_stable_owner_without_hiding_victims(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = (root / "csrc/operators/lightning_indexer_with_fetch/op_kernel/prefetch.h").read_text()
    begin = source.index("inline void TagRankedSeed(")
    end = source.index("\n#endif", begin)
    cpp = tmp_path / "seed.cpp"
    cpp.write_text("""
#include <cassert>
#include <cstdint>
#include <vector>
#define __gm__
#define __ubuf__
struct {unsigned x;} threadIdx;
struct Context {uint64_t rowOwners,hotValid,reservedHot,tailBounds;};
template<class T>T asc_ldcg(T* p){return *p;}
uint32_t OwnerRow(Context* c,uint32_t b){return c->rowOwners?reinterpret_cast<uint32_t*>(c->rowOwners)[b]:b;}
""" + source[begin:end] + """
int main() {
    constexpr unsigned R=16,H=640,B=2,tag=1U<<31;
    uint32_t owners[B]={2,1}, tail[B*2]={28,32,12,14};
    std::vector<uint32_t> valid(3*R,1),ids(H),out(H);
    std::vector<int32_t> reserved(B*R,-1);
    for(unsigned i=0;i<H;++i)ids[i]=i;
    ids.back()=UINT32_MAX;
    valid[2*R+3]=0;reserved[4]=0;
    valid[0]=0;reserved[R+1]=7;
    Context ctx{uint64_t(owners),uint64_t(valid.data()),uint64_t(reserved.data()),uint64_t(tail)};
    auto run=[&](unsigned b){for(threadIdx.x=0;threadIdx.x<256;++threadIdx.x)
        TagRankedSeed(ids.data(),out.data(),H,&ctx,b,R);};
    run(0);
    for(unsigned i=0;i<H;++i) {
        bool safe=i<R && i!=3 && i!=4;
        assert(out[i]==(safe?(tag|i):ids[i]));
    }
    run(1);
    assert(out[0]==tag && out[1]==1 && out[12]==12 && out[13]==13);
    assert(out[14]==(tag|14) && out.back()==UINT32_MAX);
    ctx.rowOwners=ctx.hotValid=ctx.reservedHot=ctx.tailBounds=0;run(0);
    for(unsigned t=0;t<R;++t)assert(out[t]==(tag|t));
}
""")
    binary = tmp_path / "seed"
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
