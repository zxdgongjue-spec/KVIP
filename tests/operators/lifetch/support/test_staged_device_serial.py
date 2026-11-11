import subprocess
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[4]
common=root/'csrc/operators/common/lifetch'
def fn(path,name):
 # Extract the default production branch after preprocessing conditional
 # signatures; slicing before #if/#else resolution creates invalid C++.
 s=subprocess.run(['c++', '-E', '-P', '-x', 'c++',
                   '-DKVIP_FUSE_QUERY_WORK=0', str(path)],
                  check=True, capture_output=True, text=True).stdout
 p=s.index(name+'(');start=s.rfind('\n',0,p)+1;b=s.index('{',p);i=b+1;d=1
 while d:
  d+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[start:i]
funcs='\n'.join(fn(common/'cache_metadata.h',n) for n in ['OwnerRow','OutputStride','IsTailToken','HotValid','ResolvePhysicalSlot','StoreFallbackSource'])
funcs+='\n'+fn(common/'resolver_candidates.h','SparseHash')
stage=(root/'tests/operators/lifetch/simulator/resolver_staged_dma2.h').read_text().replace('#pragma once','')
stage=stage.replace('512/warpSize','1/warpSize').replace('+=512','+=1').replace('tid*4','tid*2048').replace('(tid+1)*4','(tid+1)*2048').replace('(words+511)/512','words')
prefix=r'''
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstdio>
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define __gm__
#define __ubuf__
#define LAUNCH_BOUND(x)
struct {unsigned x=0;} threadIdx;
constexpr unsigned warpSize=1;
namespace AscendC {namespace Simt {unsigned GetThreadIdx(){return 0;} unsigned GetThreadNum(){return 1;}}}
#include "hot_slot_selection.h"
#include "context.h"
using namespace rotating;
template<class T>T asc_ldcg(T* p){return *p;}
template<class T>void asc_stcg(T* p,T v){*p=v;}
unsigned asc_atomic_cas(unsigned* p,unsigned a,unsigned b){auto old=*p;if(old==a)*p=b;return old;}
void asc_atomic_or(unsigned* p,unsigned v){*p|=v;}
void asc_atomic_min(unsigned* p,unsigned v){*p=std::min(*p,v);}
unsigned asc_shfl_up(unsigned v,unsigned){return v;}
void asc_syncthreads(){} void asc_threadfence(){}
namespace rotating {constexpr unsigned kLruSlots=2048,kChangeCapacity=10240;
'''+funcs+'\n}\n'+stage
main=r'''
int main(){constexpr unsigned h=8192,r=4096,cap=10240;Context c{};unsigned used=64;
 std::vector<unsigned> sel(8192),loc(8192),keys(16384),ranks(8192),slots(8192),scratch(8192),ctl(2560);
 std::vector<uint16_t> ids(8192);std::vector<unsigned> map(h,UINT32_MAX),lru(2048),change(cap,UINT32_MAX),cs(cap),valid(r,1),out(8192),status(32),count(1,2048);
 std::vector<int> reserved(r,-1),blocks(8192),offsets(8192),dst(8192);std::vector<uint64_t> addr(h),remote(8192);
 for(unsigned i=0;i<2048;i++){lru[i]=cs[i]=i;reserved[i]=i;}for(unsigned i=0;i<64;i++){map[r+i]=r+i;change[i]=r+i;}
 for(unsigned q=0;q<4;q++){for(unsigned i=0;i<205;i++)sel[q*2048+i]=r+(q%2)*205+i;for(unsigned i=205;i<2048;i++)sel[q*2048+i]=2048+i-205;}
 std::iota(addr.begin(),addr.end(),10000);rotating::Row row{};row.addresses=(uint64_t)addr.data();
 c.rows=(uint64_t)&row;c.mapping=(uint64_t)map.data();c.lruSlots=(uint64_t)lru.data();c.reservedHot=(uint64_t)reserved.data();c.hotValid=(uint64_t)valid.data();c.changes=(uint64_t)change.data();c.changeSlots=(uint64_t)cs.data();c.changeCount=(uint64_t)count.data();c.locations=(uint64_t)out.data();c.sourceBlocks=(uint64_t)blocks.data();c.sourceOffsets=(uint64_t)offsets.data();c.destinations=(uint64_t)dst.data();c.fallbackAddresses=(uint64_t)remote.data();c.rowStatus=(uint64_t)status.data();
 // Emulate the metadata DMA used by the wrapper.
 std::copy(valid.begin(),valid.end(),keys.begin());std::copy(reserved.begin(),reserved.end(),reinterpret_cast<int*>(ids.data()));
 std::copy(lru.begin(),lru.end(),scratch.begin()+4096);
 StagedInitialize(h,r,0,used,&c,sel.data(),loc.data(),keys.data(),ranks.data(),ids.data(),ctl.data());
 for(unsigned q=0;q<4 && !ctl[1];q++){
  StagedQuery(q,h,r,656,0,used,&c,sel.data(),loc.data(),keys.data(),ranks.data(),ids.data(),slots.data(),scratch.data(),ctl.data());
  if(ctl[1])break;
  status[4+q]=ctl[2+q];std::copy(loc.begin()+q*2048,loc.begin()+(q+1)*2048,out.begin()+q*2048);
  std::vector<unsigned> published(r);std::iota(published.begin(),published.end(),0);
  for(unsigned i=0;i<cap;i++)if(change[i]!=UINT32_MAX){assert(cs[i]<r);published[cs[i]]=change[i];}
  for(unsigned i=0;i<(q+1)*2048;i++){if(sel[i]==UINT32_MAX){assert(out[i]==UINT32_MAX);continue;}assert(out[i]<r);assert(published[out[i]]==sel[i]);}
 }
 status[0]=ctl[1];status[1]=ctl[0];
 printf("status=%u total=%u counts=%u,%u,%u,%u\n",status[0],status[1],status[4],status[5],status[6],status[7]);
 assert(status[0]==0 && status[1]==346);return 0;}
'''
# For this small host geometry reserved metadata is 4096 ints. ids + slots are contiguous in
# production; allocate a contiguous surrogate to preserve that arrangement.
main=main.replace('std::vector<uint16_t> ids(8192);','std::vector<uint16_t> ids(16384);')
@pytest.mark.parametrize('scenario', ['normal', 'fallback', 'cross_query', 'padding', 'exhaustion', 'prefetch512'])
@pytest.mark.parametrize('production', [False, True])
def test_staged_device_code_serial(tmp_path, scenario, production):
    code = main
    if scenario == 'fallback':
        code = code.replace('h=8192,r=4096', 'h=32768,r=8192')
        code = code.replace('for(unsigned q=0;q<4;q++){for(unsigned i=0;i<205;i++)sel[q*2048+i]=r+(q%2)*205+i;for(unsigned i=205;i<2048;i++)sel[q*2048+i]=2048+i-205;}',
                                'for(unsigned i=0;i<8192;i++)sel[i]=i<3000?r+i:4096+(i-3000)%2048;')
        code = code.replace('status[1]==346', 'status[1]==2936')
    elif scenario == 'cross_query':
        # Q2 repeats a Q1 miss at a different position, not the same-position fast path.
        code = code.replace('sel[q*2048+i]=r+(q%2)*205+i;',
                            'sel[q*2048+i]=(q==1 && i==0)?r+100:r+(q%2)*205+i;')
        code = code.replace('status[0]==0 && status[1]==346',
                            'status[0]==0 && status[1]==346 && status[4]==141 && status[5]==204 && status[6]==0 && status[7]==1')
    elif scenario == 'padding':
        code = code.replace('std::iota(addr.begin(),addr.end(),10000);',
                            'sel[8191]=UINT32_MAX;std::iota(addr.begin(),addr.end(),10000);')
    elif scenario == 'prefetch512':
        code = code.replace('h=8192,r=4096', 'h=32768,r=4096').replace('unsigned used=64;', 'unsigned used=512;')
        code = code.replace('for(unsigned i=0;i<64;i++){map[r+i]=r+i;change[i]=r+i;}',
                            'for(unsigned i=0;i<512;i++){unsigned token=r+(i<64?i:10000+i);map[token]=r+i;change[i]=token;}')
    elif scenario == 'exhaustion':
        code = code.replace('h=8192,r=4096', 'h=32768,r=4096')
        code = code.replace('for(unsigned q=0;q<4;q++){for(unsigned i=0;i<205;i++)sel[q*2048+i]=r+(q%2)*205+i;for(unsigned i=205;i<2048;i++)sel[q*2048+i]=2048+i-205;}',
                                'for(unsigned i=0;i<8192;i++)sel[i]=r+i;')
        code = code.replace('status[0]==0 && status[1]==346', 'status[0]==4')
    if scenario != 'exhaustion':
        code = code.replace('printf("status=%u total=%u counts=%u,%u,%u,%u\\n",status[0],status[1],status[4],status[5],status[6],status[7]);',
                            'std::vector<unsigned> ownership(r);std::iota(ownership.begin(),ownership.end(),0);'
                            'for(unsigned i=0;i<cap;i++)if(change[i]!=UINT32_MAX){assert(cs[i]<r);ownership[cs[i]]=change[i];}'
                            'for(unsigned i=0;i<8192;i++){if(sel[i]==UINT32_MAX){assert(out[i]==UINT32_MAX);continue;}assert(out[i]<r);assert(ownership[out[i]]==sel[i]);}')
    source, exe = tmp_path / 'staged.cpp', tmp_path / 'staged'
    current_prefix = prefix
    if production:
        body = (root / 'csrc/operators/lightning_indexer_with_fetch_streaming_qworkers/op_kernel/resolver_staged_dma2.h').read_text().replace('#pragma once', '')
        body = body.replace('512/warpSize', '1/warpSize').replace('+=512', '+=1').replace('tid*4', 'tid*2048').replace('(tid+1)*4', '(tid+1)*2048').replace('(words+511)/512', 'words')
        current_prefix = '#define KVIP_FUSE_QUERY_WORK 1\n' + prefix.replace(stage, body)
        # Exercise the actual ready-input initializer after emulating LI's
        # classification. This is an algorithm oracle, never a timing model.
        code = code.replace('for(unsigned q=0;q<4 && !ctl[1];q++){',
            'StagedInitializeReady(sel.data(),loc.data(),keys.data(),ranks.data(),ctl.data());'
            'for(unsigned q=0;q<4 && !ctl[1];q++){')
    source.write_text(current_prefix + code)
    (tmp_path / 'batch_get_abi.h').write_text('')
    subprocess.run(['c++', '-std=c++17', '-O0', str(source), '-I' + str(tmp_path),
                    '-I' + str(root/'csrc/operators/lightning_indexer_with_fetch_safe/include'),
                    '-I' + str(root/'csrc/operators/lidu/op_kernel'), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
