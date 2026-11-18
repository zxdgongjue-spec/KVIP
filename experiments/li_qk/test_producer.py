"""Actual host oracle and boundary validation with CPU launches; no timing claims."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from run import HERE,ROOT,make_fixture
from test_controls import ACL_STUB,LAUNCH_STUB
class ProducerControls(unittest.TestCase):
 def test_actual_host_direct_oracle_and_unwritten_scratch(self):
  stub='''\n#include "producer_plan.h"\nextern "C" unsigned aclrtlaunch_li_qk_score_producer(unsigned cores,void* stream,void* binding,void* sidecar,unsigned rows,unsigned stride,unsigned heads,unsigned group){auto a=(li_producer::Bindings*)sidecar;auto b=(unsigned long long*)binding;std::vector<unsigned short> raw(rows*group*stride);auto old=b[12];b[12]=(unsigned long long)raw.data();aclrtlaunch_li_qk_score(cores,stream,b,rows,stride,heads,group);b[12]=old;aclrtlaunch_li_qk_scatter(cores,stream,raw.data(),(void*)a->canonical,(void*)a->tasks,rows,stride,a->history,group);if(a->route==2){auto high=(unsigned*)a->high;auto s=(unsigned short*)a->canonical;for(unsigned q=0;q<4;++q)for(unsigned part=0;part<16;++part){unsigned counts[256]={};for(unsigned x=part*8192;x<(part+1)*8192;++x)++counts[s[q*a->history+x]>>8];unsigned sum=0;for(unsigned b=0;b<256;++b){sum+=counts[b];high[(q*16+part)*256+b]=sum;}}}if(a->route==3){auto s=(unsigned short*)a->canonical;auto vals=(unsigned short*)a->candidateScores;auto tokens=(unsigned*)a->candidateIds;for(unsigned q=0;q<4;++q)for(unsigned part=0;part<16;++part){std::vector<unsigned> order(8192);std::iota(order.begin(),order.end(),part*8192);std::partial_sort(order.begin(),order.begin()+2048,order.end(),[&](unsigned x,unsigned y){return s[q*a->history+x]>s[q*a->history+y]||(s[q*a->history+x]==s[q*a->history+y]&&x<y);});for(unsigned k=0;k<2048;++k){tokens[(q*16+part)*2048+k]=order[k];vals[(q*16+part)*2048+k]=s[q*a->history+order[k]];}}}return 0;}\nextern "C" unsigned aclrtlaunch_li_producer_high_tail(unsigned cores,void* stream,void* sidecar){auto a=(li_producer::Bindings*)sidecar;return aclrtlaunch_li_qk_topk(cores,stream,(void*)a->canonical,(void*)a->selected,nullptr,a->history,32,4);}\nextern "C" unsigned aclrtlaunch_li_producer_candidate_tail(unsigned cores,void* stream,void* sidecar){return aclrtlaunch_li_producer_high_tail(cores,stream,sidecar);}\n'''
  with tempfile.TemporaryDirectory() as folder:
   d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'batch_get_abi.h').write_text('#pragma once\n');(d/'launch.cpp').write_text(LAUNCH_STUB+stub)
   subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="sim"','-I'+str(d),'-I'+str(HERE),'-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),str(HERE/'producer_host.cpp'),str(d/'launch.cpp'),'-o',str(d/'host')],check=True,capture_output=True)
   make_fixture(d/'fixture',1,2048,32);
   for name in ('key','kscale'):
    path=d/'fixture'/(name+'.bin');path.write_bytes(path.read_bytes()*64)
   import numpy as np
   path=d/'fixture/oracle.bin';v=np.fromfile(path,dtype=np.uint16).reshape(4,2048);np.tile(v,(1,64)).tofile(path)
   env=dict(os.environ,LI_QK_ORACLE=str(d/'fixture/oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'))
   for route in (0,1,2,3):
    r=subprocess.run([str(d/'host'),str(d/'fixture'),str(d/f'r{route}'),str(route),'0','1','0'],env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    p=json.loads((d/f'r{route}/report.json').read_text());self.assertTrue(p['oracle']);self.assertEqual(p['full_li_launches'],3 if not route else 2)
if __name__=='__main__':unittest.main()
