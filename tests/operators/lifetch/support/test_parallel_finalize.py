"""Run the actual SIMT publication code with host lanes (no transport model)."""
import subprocess
from pathlib import Path


def test_query_publication_shards_counts_and_failures(tmp_path):
    source = (Path(__file__).resolve().parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/streaming.cpp').read_text()
    body = source[source.index('__simt_vf__ __aicore__ LAUNCH_BOUND(256)\ninline void PublishQueryCompletion'):source.index('// Compact each ready request')]
    update = (Path(__file__).resolve().parents[4] / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/index_cache_update.h').read_text()
    rollback = update[update.index('template<uint32_t Threads>\n__simt_vf__ __aicore__ LAUNCH_BOUND(512)\ninline void CompleteEarlyUpdate'):]
    cpp = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#define __simt_vf__
#define __aicore__
#define __gm__
#define LAUNCH_BOUND(x)
struct {unsigned x;} threadIdx;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
void asc_threadfence(){} void asc_syncthreads(){}
constexpr unsigned kFlagCount=12,kChangeCapacity=2048,kLruSlots=2048;
struct BatchFlag{uint32_t ready,count,error;};
struct FinalPost{uint32_t count,pending,error,deferred,descriptor;};
struct Context{uint64_t owners,hotValid,destinations,changes,lruSlots;};
uint32_t OwnerRow(Context*,unsigned b){return b;}
uint32_t OutputStride(unsigned resident,Context*){return resident;}
''' + body + rollback + r'''
int main(){
 constexpr unsigned B=4,R=1024;
 std::vector<uint32_t> valid(B*R,0),lru(B*kLruSlots,0);
 std::vector<int32_t> dest(B*4*512,0),changes(B*kChangeCapacity,-1);
 FinalPost posts[B*4]{};BatchFlag flags[B*kFlagCount]{};
 Context c{uint64_t(posts),uint64_t(valid.data()),uint64_t(dest.data()),uint64_t(changes.data()),uint64_t(lru.data())};
 for(unsigned b=0;b<B;b++)for(unsigned q=0;q<4;q++){
  auto &p=posts[b*4+q];p.count=q<2?300:0;p.descriptor=(b*4+q)*512;p.deferred=1;
  for(unsigned i=0;i<p.count;i++)dest[p.descriptor+i]=b*R+q*300+i;
 }
 // Host lanes in reverse order: thread0 publishes after all payload lanes.
 auto run=[&](unsigned q,unsigned lane,uint64_t error){for(int t=255;t>=0;t--){threadIdx.x=t;PublishQueryCompletion(&c,flags,B,R,0,q,lane,2,error,7);}};
 run(0,0,0);
 for(unsigned b=0;b<B;b++)for(unsigned i=0;i<R;i++)assert(valid[b*R+i]==unsigned(b%2==0 && i<300));
 assert(posts[0].pending==7 && posts[8].pending==0);
 assert(flags[8].ready==1 && flags[2*kFlagCount+8].ready==1 && !flags[kFlagCount+8].ready);
 run(1,1,9); // failure reports completion/error but no successful hot residency
 assert(flags[kFlagCount+9].ready && flags[kFlagCount+9].error==9);
 for(unsigned b=1;b<B;b+=2)for(unsigned i=0;i<R;i++)assert(!valid[b*R+i]);
 run(3,0,0); // empty Q publishes completion
 assert(flags[11].ready && flags[2*kFlagCount+11].ready);
 // Prefetch publisher touches only listed prefetch slots, not final-only slots.
 changes[0]=12;lru[0]=900;
 for(unsigned t=0;t<256;t++){threadIdx.x=t;PublishPrefetchResidency(&c,0,R);}
 assert(valid[900]==1 && valid[800]==0);
 // A later failure clears even slots successfully published by another Q.
 uint32_t status[B]={},owners[B]={0,1,2,3};uint64_t bindings[26]={};
 bindings[6]=uint64_t(status);bindings[9]=uint64_t(owners);
 bindings[16]=uint64_t(valid.data());bindings[25]=R;
 for(unsigned t=0;t<256;t++){threadIdx.x=t;CompleteEarlyUpdate<256>(bindings,0,1);}
 assert(status[0]==4);for(unsigned i=0;i<R;i++)assert(!valid[i]);
 // Another request's successful publication is preserved.
 assert(valid[2*R]==1);
}
'''
    path = tmp_path / 'publish.cpp'
    path.write_text(cpp)
    exe = tmp_path / 'publish'
    subprocess.run(['c++', '-std=c++17', str(path), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_summary_accepts_old_and_parallel_timelines(tmp_path):
    import json
    from test_core_timeline import fixture
    from summarize_finalize_ab import summarize
    for version in ('A', 'B'):
        raw = fixture()
        if version == 'B':
            raw['fetch'][0][116:118] = [471, 479]
            for c in raw['cores'][2:]:
                c[12:16] = [600, 610, 620, 630]
        folder = tmp_path / version / 'case0-b1-prototype-graph'
        folder.mkdir(parents=True)
        (folder / 'report.json').write_text(json.dumps({'timing': {'prototype_with_update': {'p50_us': 100, 'p99_us': 110}}}))
        sample = {'round': 0, 'accounting': {'stages_us': {'resolve_update': 90}}, 'streaming_cycles': [{'core_timeline_raw': raw}]}
        (folder / 'full_attribution.json').write_text(json.dumps({'samples': [sample]}))
    text = summarize(tmp_path)
    assert 'A | 1 | 100.000' in text and 'B | 1 | 100.000' in text
    assert 'prefetch_residency' in text
    assert 'fetch Q4 lane0' in text
    assert 'time change=+0.00%' in text
