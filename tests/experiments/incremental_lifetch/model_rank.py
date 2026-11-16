"""Build the exact prefetch rank-selection helper in the CANN simulator."""
import argparse
from pathlib import Path
import re
from model_window import CMAKE

KERNEL = r'''
#include "kernel_operator.h"
#if defined(__CCE_AICORE__)
#include "ranked_candidates.h"
namespace rotating { using namespace AscendC;
#include "original_export.h"
}
#endif
using namespace AscendC;
extern "C" __global__ __aicore__ void rank_model(GM_ADDR inputs,GM_ADDR outputs,GM_ADDR times){
#if defined(__CCE_AICORE__)
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 if ASCEND_IS_AIV {
 TPipe pipe;TBuf<TPosition::VECCALC> ib,sb,ob,tmp;
 pipe.InitBuffer(ib,8192);pipe.InitBuffer(sb,4096);pipe.InitBuffer(ob,8192);pipe.InitBuffer(tmp,14336);
 auto ids=ib.Get<uint32_t>();auto scores=sb.Get<uint16_t>();auto out=ob.Get<uint32_t>();auto scratch=tmp.Get<uint32_t>();
 GlobalTensor<uint32_t> in,output;in.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(inputs));output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputs));
 auto timing=reinterpret_cast<__gm__ uint64_t*>(times);
 for(uint32_t q=0;q<8;++q){
 DataCopy(ids,in[q*3072],2048);DataCopy(scores,in[q*3072+2048].ReinterpretCast<uint16_t>(),2048);
 SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 for(uint32_t mode=0;mode<3;++mode){
 uint64_t stamp[5];uint32_t rank=1024+(1024*(q%4)+1)/3;
 uint32_t count;
 if(mode<2)count=kvip_prefetch_rank::Select(ids,scores,out,scratch,16384,131072,rank,32,mode!=0,stamp);
 else {stamp[0]=GetSystemCycle();rotating::CompactColdTopk(reinterpret_cast<__ubuf__ uint32_t*>(ids.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),reinterpret_cast<__ubuf__ uint32_t*>(scratch.GetPhyAddr()),16384,131072);
 SetFlag<HardEvent::V_S>(EVENT_ID4);WaitFlag<HardEvent::V_S>(EVENT_ID4);count=scratch.GetValue(0);stamp[1]=stamp[2]=stamp[3]=stamp[4]=GetSystemCycle();}
 for(uint32_t j=0;j<5;++j)timing[(q*3+mode)*6+j]=stamp[j];timing[(q*3+mode)*6+5]=count;
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
 DataCopy(output[(q*3+mode)*2048],out,2048);
 SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
 }
 }
 GlobalTensor<uint8_t> recorded;recorded.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(times));
 for(uint32_t offset=0;offset<24*6*8;offset+=64)
 DataCacheCleanAndInvalid<uint8_t,CacheLine::SINGLE_CACHE_LINE,DcciDst::CACHELINE_OUT>(recorded[offset]);
 pipe.Destroy();
 }
#endif
}
'''
HOST = r'''
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" uint32_t aclrtlaunch_rank_model(uint32_t,void*,void*,void*,void*);
void ck(int v){if(v)throw std::runtime_error("ACL "+std::to_string(v));}
template<class T>void* upload(std::vector<T>& v){void* p;ck(aclrtMalloc(&p,v.size()*sizeof(T),ACL_MEM_MALLOC_HUGE_FIRST));ck(aclrtMemcpy(p,v.size()*sizeof(T),v.data(),v.size()*sizeof(T),ACL_MEMCPY_HOST_TO_DEVICE));return p;}
int main(){try{
 ck(aclInit(nullptr));ck(aclrtSetDevice(0));aclrtStream stream;ck(aclrtCreateStream(&stream));
 std::vector<uint32_t> input(8*3072),output(24*2048);std::vector<uint64_t> timing(24*6);
 const uint32_t counts[8]={0,12,64,256,2048,256,12,64};
 for(uint32_t q=0;q<8;++q){auto ids=input.data()+q*3072;auto scores=reinterpret_cast<uint16_t*>(ids+2048);uint32_t kind=q;
 for(uint32_t i=0;i<2048;++i){uint32_t ordinal=(i*127)%2048;ids[i]=ordinal<counts[kind]?16384+ordinal:ordinal;
 scores[i]=kind==5?uint16_t(50000):uint16_t(40000+(ordinal*311)%20000);
 if(kind==1&&ordinal<counts[kind])scores[i]=60000+ordinal;
 if(kind==6&&ordinal<counts[kind])scores[i]=1+ordinal;
 if(kind==7&&ordinal<32)ids[i]=UINT32_MAX;
 }}
 void* in=upload(input);void* out=upload(output);void* t=upload(timing);
 ck(aclrtlaunch_rank_model(1,stream,in,out,t));ck(aclrtSynchronizeStream(stream));
 ck(aclrtMemcpy(output.data(),output.size()*4,out,output.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
 ck(aclrtMemcpy(timing.data(),timing.size()*8,t,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST));
 for(uint32_t q=0;q<8;++q){auto ids=input.data()+q*3072;auto scores=reinterpret_cast<uint16_t*>(ids+2048);uint32_t rank=1024+(1024*(q%4)+1)/3;
 std::vector<uint16_t> sorted(scores,scores+2048);std::sort(sorted.begin(),sorted.end(),std::greater<uint16_t>());auto cutoff=sorted[rank-1];
 std::vector<uint32_t> expected;for(uint32_t i=0;i<2048;++i)if(ids[i]>=16384&&ids[i]<131072&&scores[i]>=cutoff)expected.push_back(i);
 std::stable_sort(expected.begin(),expected.end(),[&](auto a,auto b){return scores[a]>scores[b];});if(expected.size()>32)expected.resize(32);
 std::vector<uint32_t> expectedIds;for(auto i:expected)expectedIds.push_back(ids[i]);std::sort(expectedIds.begin(),expectedIds.end());
 for(uint32_t mode=0;mode<3;++mode){auto ts=timing.data()+(q*3+mode)*6;std::vector<uint32_t> actual(output.begin()+(q*3+mode)*2048,output.begin()+(q*3+mode)*2048+ts[5]);std::sort(actual.begin(),actual.end());
 std::vector<uint32_t> wanted=expectedIds;if(mode==2){wanted.clear();for(uint32_t i=0;i<2048;++i)if(ids[i]>=16384&&ids[i]<131072)wanted.push_back(ids[i]);std::sort(wanted.begin(),wanted.end());}
 if(actual!=wanted)throw std::runtime_error("selection q="+std::to_string(q)+" mode="+std::to_string(mode)+" count="+std::to_string(ts[5])+" wanted="+std::to_string(expected.size()));
 printf("PASS kind=%u rank=%u mode=%u count=%llu cycles pack=%llu rank_cycles=%llu cap=%llu emit=%llu total=%llu\n",q,rank,mode,(unsigned long long)ts[5],(unsigned long long)(ts[1]-ts[0]),(unsigned long long)(ts[2]-ts[1]),(unsigned long long)(ts[3]-ts[2]),(unsigned long long)(ts[4]-ts[3]),(unsigned long long)(ts[4]-ts[0]));
 }}
 ck(aclrtDestroyStream(stream));ck(aclrtResetDevice(0));ck(aclFinalize());return 0;
 }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--bitmap',action='store_true',help='Indexed miss vs bitmap miss; default: only representative case 3')
    p.add_argument('--case',type=int,choices=range(12),help='Run one fixture instead of the full correctness matrix')
    p.add_argument('--miss',action='store_true',help='Compare cold floor vs true-miss floor, including indexed metadata loads')
    p.add_argument('--floor',action='store_true',help='Compare no lower bound vs quota/2; preserve old-export reference')
    p.add_argument('--summarize',action='store_true',help='Read output/model.log; all times use 1800 model cycles/us')
    args=p.parse_args();dest=args.output.resolve()
    if args.bitmap:
        args.miss=True
        if args.case is None: args.case=3
    if args.miss: args.floor=True
    if args.summarize:
        rows=[]
        for line in (dest/'model.log').read_text().splitlines():
            if line.startswith('PASS kind='):
                prefix, separator, stages = line.partition(' cycles ')
                line = prefix + separator + stages.replace(' rank=', ' rank_cycles=')
                rows.append({k:int(v) for k,v in re.findall(r'(kind|rank|rank_cycles|mode|count|pack|cap|emit|total)=(\d+)',line)})
        expected=3 if args.case is not None else (36 if args.floor else 24)
        if len(rows)!=expected:
            raise RuntimeError(f'Expected {expected} verified samples, got {len(rows)}')
        print('CANN simulator; 1800 cycles/us. One Q/sample; 4Q is 4 x the per-Q delta, not end-to-end LI.')
        if args.bitmap: print('A=indexed true-miss; B=covered bitmap, including 16 KiB GM load. Compare pack cycles as state check + packing.')
        elif args.miss: print('A=cold floor; B=true-miss floor. B total includes indexed map/hot/reserved filtering. Counts are before cross-Q dedup.')
        print('cold/Q | rank | before count | after count | before us/Q | after us/Q | added us/4Q' if args.floor else 'cold/Q | rank | v1 us/Q | optimized us/Q | old export us/Q | added us/4Q')
        counts=(0,12,64,256,2048,256,12,32,1024,20,16,1) if args.floor else (0,12,64,256,2048,256,12,32)
        for kind in ([args.case] if args.case is not None else range(len(counts))):
            for rank in (1024 if args.floor and kind==3 else 1024+(1024*(kind%4)+1)//3,):
                r={v['mode']:v for v in rows if v['kind']==kind and v['rank']==rank}
                a,b,old=(r[m]['total']/1800 for m in (0,1,2))
                if args.floor:
                    print(f"{counts[kind]:6} | {rank:4} | {r[0]['count']:12} | {r[1]['count']:11} | {a:11.2f} | {b:10.2f} | {(b-a)*4:11.2f}")
                    if args.bitmap:
                        print(f"  check+pack us/Q: A={r[0]['pack']/1800:.3f} B={r[1]['pack']/1800:.3f}; B cycles={r[1]['pack']}")
                        if r[1]['pack']>=1800: raise RuntimeError('State check + pack exceeds 1 us target')
                    continue
                print(f'{counts[kind]:6} | {rank:4} | {a:7.2f} | {b:14.2f} | {old:15.2f} | {(b-old)*4:11.2f}')
        return
    dest.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    for name in ('ranked_candidates.h','rank_threshold.h'):
        (dest/name).write_text((here/name).read_text())
    (dest/'original_export.h').write_text((here/'topk_candidates.h').read_text().split('// One UB bit')[0])
    kernel,host=KERNEL,HOST
    if args.floor:
        kernel=kernel.replace('rank,32,mode!=0,stamp)', 'rank,32,true,stamp,mode==1?16:0)')
        # Warm both measured paths after MTE2 input loads. Otherwise the first
        # path pays cold instruction/pipeline costs in unrelated PackCold work.
        kernel=kernel.replace(' uint32_t count;',
            ' if(mode<2){kvip_prefetch_rank::Select(ids,scores,out,scratch,16384,131072,rank,32,true,stamp,mode==1?16:0);PipeBarrier<PIPE_ALL>();}\n uint32_t count;')
        kernel=kernel.replace('q<8', 'q<12').replace('offset<24*6*8','offset<36*6*8')
        host=host.replace('q<8','q<12').replace('input(8*3072),output(24*2048)','input(12*3072),output(36*2048)').replace('timing(24*6)','timing(36*6)')
        host=host.replace('counts[8]={0,12,64,256,2048,256,12,64}', 'counts[12]={0,12,64,256,2048,256,12,64,1024,20,16,1}')
        host=host.replace('if(kind==6&&ordinal<counts[kind])', 'if((kind==3||kind==6||kind==8||kind==9||kind==10)&&ordinal<counts[kind])')
        host=host.replace('if(kind==7&&ordinal<32)', 'if(kind==5&&ordinal<counts[kind])scores[i]=1;\n if(kind==7&&ordinal<32)')
        kernel=kernel.replace('uint32_t rank=1024+', 'uint32_t rank=q==3?1024:1024+')
        host=host.replace('uint32_t rank=1024+', 'uint32_t rank=q==3?1024:1024+')
        anchor='std::vector<uint32_t> wanted=expectedIds;if(mode==2)'
        replacement=r'''std::vector<uint32_t> wanted=expectedIds;
 if(mode==1){
 std::vector<uint32_t> allCold;for(uint32_t i=0;i<2048;++i)if(ids[i]>=16384&&ids[i]<131072)allCold.push_back(i);
 std::stable_sort(allCold.begin(),allCold.end(),[&](auto a,auto b){return scores[a]>scores[b];});
 uint16_t effective=cutoff;if(allCold.size()<=16)effective=0;else effective=std::min(effective,scores[allCold[15]]);
 wanted.clear();for(auto i:allCold)if(scores[i]>=effective&&wanted.size()<32)wanted.push_back(ids[i]);std::sort(wanted.begin(),wanted.end());
 }
 if(mode==2)'''
        assert anchor in host
        host=host.replace(anchor,replacement)
    if args.miss:
        kernel=kernel.replace('GM_ADDR times)', 'GM_ADDR times,GM_ADDR mapping)')
        kernel=kernel.replace('pipe.InitBuffer(tmp,14336)', 'pipe.InitBuffer(tmp,22528)')
        call='kvip_prefetch_rank::Select(ids,scores,out,scratch,16384,131072,rank,32,true,stamp,mode==1?16:0)'
        newcall='''(mode==1?kvip_prefetch_rank::SelectMiss(ids,scores,out,scratch,
 reinterpret_cast<__gm__ uint32_t*>(mapping)+q*163840,
 reinterpret_cast<__gm__ uint32_t*>(mapping)+q*163840+131072,
 reinterpret_cast<__gm__ int32_t*>(mapping)+q*163840+147456,
 16384,131072,rank,32,stamp,16):kvip_prefetch_rank::Select(ids,scores,out,scratch,16384,131072,rank,32,true,stamp,16))'''
        assert kernel.count(call)==2
        kernel=kernel.replace(call,newcall)
        # Measured total includes indexed metadata filtering, not just Select.
        kernel=kernel.replace(' if(mode<2)count=', ' uint64_t whole=GetSystemCycle();\n if(mode<2)count=')
        kernel=kernel.replace(' for(uint32_t j=0;j<5;++j)timing', ' if(mode<2)stamp[0]=whole;\n for(uint32_t j=0;j<5;++j)timing')
        host=host.replace('uint32_t,void*,void*,void*,void*);', 'uint32_t,void*,void*,void*,void*,void*);')
        host=host.replace(' void* in=upload(input);',''' std::vector<uint32_t> maps(12*163840,UINT32_MAX);
 for(uint32_t q=0;q<12;++q){auto m=maps.data()+q*163840;
 std::fill(m+131072,m+147456,1);
 auto ids=input.data()+q*3072;auto scores=reinterpret_cast<uint16_t*>(ids+2048);
 std::vector<uint32_t> cold;for(uint32_t i=0;i<2048;++i)if(ids[i]>=16384&&ids[i]<131072)cold.push_back(i);
 std::stable_sort(cold.begin(),cold.end(),[&](auto a,auto b){return scores[a]>scores[b];});
 for(uint32_t j=0;j<cold.size();++j)if(q==4||j<32)m[ids[cold[j]]]=16384+j;
 // Cases 0/6 cover invalid hot and reserved-but-previously-valid hot.
 if(q==0||q==6){for(uint32_t t=0;t<12;++t)m[131072+t]=0;for(uint32_t t=12;t<24;++t)m[147456+t]=0;m[0]=16384;}
 // ABC-local tag is excluded just like an in-flight admission.
 if(q==8&&!cold.empty())m[ids[cold.back()]]=16384+8192;
 }
 void* mapD=upload(maps);
 void* in=upload(input);''')
        host=host.replace('stream,in,out,t)', 'stream,in,out,t,mapD)')
        host=host.replace(' if(mode==1){', ' if(mode<2){')
        host=host.replace('if(ids[i]>=16384&&ids[i]<131072)allCold.push_back(i);','''if(ids[i]<131072){uint32_t token=ids[i];auto m=maps.data()+q*163840;
 bool local=token<16384&&m[131072+token]!=0&&int32_t(m[147456+token])<0;
 if(mode==0?token>=16384:(!local&&m[token]==UINT32_MAX))allCold.push_back(i);}''')
        # Verify original TopK input stays untouched after both paths.
        host=host.replace(' ck(aclrtMemcpy(timing.data(),timing.size()*8,t,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST));',''' ck(aclrtMemcpy(timing.data(),timing.size()*8,t,timing.size()*8,ACL_MEMCPY_DEVICE_TO_HOST));
 std::vector<uint32_t> unchanged(input.size());ck(aclrtMemcpy(unchanged.data(),unchanged.size()*4,in,unchanged.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
 if(unchanged!=input)throw std::runtime_error("TopK input modified");
''')
    if args.bitmap:
        kernel=kernel.replace('pipe.InitBuffer(ob,8192)','pipe.InitBuffer(ob,16384)').replace('pipe.InitBuffer(tmp,22528)','pipe.InitBuffer(tmp,25600)')
        kernel=kernel.replace('q*163840','q*167936')
        indexed='kvip_prefetch_rank::SelectMiss(ids,scores,out,scratch,reinterpret_cast<__gm__ uint32_t*>(mapping)+q*167936,reinterpret_cast<__gm__ uint32_t*>(mapping)+q*167936+131072,reinterpret_cast<__gm__ int32_t*>(mapping)+q*167936+147456,16384,131072,rank,32,stamp,16)'
        kernel=kernel.replace('kvip_prefetch_rank::Select(ids,scores,out,scratch,16384,131072,rank,32,true,stamp,16)',indexed)
        old='''kvip_prefetch_rank::SelectMiss(ids,scores,out,scratch,
 reinterpret_cast<__gm__ uint32_t*>(mapping)+q*167936,
 reinterpret_cast<__gm__ uint32_t*>(mapping)+q*167936+131072,
 reinterpret_cast<__gm__ int32_t*>(mapping)+q*167936+147456,
 16384,131072,rank,32,stamp,16)'''
        if kernel.count(old)!=2: raise RuntimeError('Miss model call pattern changed')
        kernel=kernel.replace(old,'kvip_prefetch_rank::SelectBitmap(ids,scores,out,scratch,reinterpret_cast<__gm__ uint32_t*>(mapping)+q*167936+163840,131072,131072,rank,32,stamp,16)')
        host=host.replace('12*163840','12*167936').replace('q*163840','q*167936')
        host=host.replace(' void* mapD=upload(maps);',''' for(uint32_t q=0;q<12;++q){auto m=maps.data()+q*167936;auto bits=m+163840;std::fill(bits,bits+4096,0);
 for(uint32_t t=0;t<131072;++t)if(m[t]!=UINT32_MAX||(t<16384&&m[131072+t]!=0&&int32_t(m[147456+t])<0))bits[t/32]|=1U<<(t%32);}
 void* mapD=upload(maps);''')
        host=host.replace('if(mode==0?token>=16384:(!local&&m[token]==UINT32_MAX))','if(!local&&m[token]==UINT32_MAX)')
    if args.case is not None:
        if not args.floor: raise ValueError('--case requires --floor, --miss or --bitmap')
        kernel=kernel.replace('for(uint32_t q=0;q<12;++q)',f'for(uint32_t q={args.case};q<{args.case+1};++q)')
        anchor='for(uint32_t q=0;q<12;++q){auto ids=input.data()+q*3072;auto scores=reinterpret_cast<uint16_t*>(ids+2048);uint32_t rank='
        host=host.replace(anchor,anchor.replace('q=0;q<12',f'q={args.case};q<{args.case+1}'))
    (dest/'kernel.cpp').write_text(kernel);(dest/'main.cpp').write_text(host)
    cmake=CMAKE.replace('@OP@',str(dest)).replace('@KVAF@',str(dest)).replace('window_model','rank_model')
    (dest/'CMakeLists.txt').write_text(cmake)
    print(dest)
if __name__=='__main__':main()
