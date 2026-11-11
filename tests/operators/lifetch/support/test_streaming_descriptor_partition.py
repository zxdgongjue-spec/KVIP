"""Execute the actual per-Q packing body on a serial host SIMT stub."""
import subprocess
from pathlib import Path


def test_per_query_pack_does_not_overwrite_other_queries(tmp_path):
    root = Path(__file__).resolve().parents[4]
    header = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/staged_urma_pack.h').read_text()
    start = header.index('inline void PackStagedUrmaQuery(')
    body = header[start:header.rindex('\n}')].replace('(own+511)/512', 'own').replace('512/warpSize', '1')
    code = r'''
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cassert>
#define __gm__
#define __ubuf__
struct {unsigned x=0;} threadIdx;
constexpr unsigned warpSize=1, kSlots=8192,kTop=2048;
struct Row { uint64_t addresses; };
struct Context {uint64_t rows,reservedHot,sourceBlocks,sourceOffsets,destinations,fallbackAddresses,count;};
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_atomic_or(unsigned* p,unsigned v){*p|=v;}
unsigned asc_atomic_add(unsigned* p,unsigned v){auto old=*p;*p+=v;return old;}
void asc_syncthreads(){} void asc_threadfence(){}
unsigned OwnerRow(Context*,unsigned b){return b;}
unsigned OutputStride(unsigned r,Context*){return r;}
unsigned StagedPrefix(unsigned n,unsigned* ctl){ctl[544]=n;return 0;}
void StoreFallbackSource(unsigned,unsigned,Context* ctx,unsigned,unsigned dst,unsigned,uint64_t addr,int* blocks,int* offsets){
 reinterpret_cast<uint64_t*>(ctx->fallbackAddresses)[dst]=addr;blocks[dst]=dst;offsets[dst]=0;}
''' + body + r'''
int main(){
 std::vector<uint64_t> addr(32),remote(kSlots,0);for(unsigned i=0;i<32;i++)addr[i]=1000+i;
 std::vector<int> reserved(32,-1),blocks(kSlots,-1),offsets(kSlots,-1),dest(kSlots,-1);
 std::vector<unsigned> selected(32),slots(kSlots),ctl(3000),count(1);std::vector<uint16_t> ids(kSlots);
 for(unsigned i=0;i<32;i++){selected[i]=i;ids[i]=i;slots[i]=i;}
 Row row{uint64_t(addr.data())};Context c{uint64_t(&row),uint64_t(reserved.data()),uint64_t(blocks.data()),uint64_t(offsets.data()),uint64_t(dest.data()),uint64_t(remote.data()),uint64_t(count.data())};
 PackStagedUrmaQuery(32,32,656,0,0,0,0,3,&c,selected.data(),ids.data(),slots.data(),ctl.data());
 PackStagedUrmaQuery(32,32,656,0,1,0,3,3,&c,selected.data(),ids.data(),slots.data(),ctl.data());
 for(int i=0;i<3;i++){assert(dest[i]==i);assert(dest[2048+i]==i+3);assert(remote[i]==1000+i);assert(remote[2048+i]==1003+i);}
 assert(dest[3]==-1 && dest[2047]==-1 && count[0]==6);
 std::vector<int> direct(6*kTop+64,-1);direct[6*kTop]=0;direct[6*kTop+8]=0;
 reserved[1]=0;
 PackStagedUrmaQuery(32,32,656,0,0,1,0,3,&c,selected.data(),ids.data(),slots.data(),ctl.data(),direct.data(),kTop);
 assert(direct[6*kTop]==2 && direct[6*kTop+8]==1);
 assert(direct[2*kTop]==0 && direct[2*kTop+1]==2 && direct[5*kTop]==1);
 PackStagedUrmaQuery(32,32,656,0,1,1,3,3,&c,selected.data(),ids.data(),slots.data(),ctl.data(),direct.data(),kTop);
 assert(direct[6*kTop]==5 && direct[6*kTop+8]==1);
 for(int i=0;i<3;i++)assert(direct[2*kTop+2+i]==3+i);
 assert(ctl[1]==0);
}
'''
    cpp = tmp_path / 'pack.cpp'
    cpp.write_text(code)
    exe = tmp_path / 'pack'
    subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
