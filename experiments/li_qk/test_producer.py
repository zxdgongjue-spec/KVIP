"""Actual host oracle and boundary validation with CPU launches; no timing claims."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
from run import HERE,ROOT,make_fixture
from test_controls import ACL_STUB,LAUNCH_STUB
class ProducerControls(unittest.TestCase):
 def test_actual_host_direct_oracle_and_unwritten_scratch(self):
  stub='''\n#include "producer_plan.h"\nextern "C" unsigned aclrtlaunch_li_qk_score_producer(unsigned cores,void* stream,void* binding,void* sidecar,unsigned rows,unsigned stride,unsigned heads,unsigned group){auto a=(li_producer::Bindings*)sidecar;auto b=(unsigned long long*)binding;std::vector<unsigned short> raw(rows*group*stride);auto old=b[12];b[12]=(unsigned long long)raw.data();aclrtlaunch_li_qk_score(cores,stream,b,rows,stride,heads,group);b[12]=old;aclrtlaunch_li_qk_scatter(cores,stream,raw.data(),(void*)a->canonical,(void*)a->tasks,rows,stride,a->history,group);return 0;}\n'''
  with tempfile.TemporaryDirectory() as folder:
   d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'batch_get_abi.h').write_text('#pragma once\n');(d/'launch.cpp').write_text(LAUNCH_STUB+stub)
   subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="sim"','-I'+str(d),'-I'+str(HERE),'-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),str(HERE/'producer_host.cpp'),str(d/'launch.cpp'),'-o',str(d/'host')],check=True,capture_output=True)
   make_fixture(d/'fixture',1,131072,32);env=dict(os.environ,LI_QK_ORACLE=str(d/'fixture/oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'))
   for route in (0,1):
    r=subprocess.run([str(d/'host'),str(d/'fixture'),str(d/f'r{route}'),str(route),'0','1','0'],env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    p=json.loads((d/f'r{route}/report.json').read_text());self.assertTrue(p['oracle']);self.assertEqual(p['full_li_launches'],3 if not route else 2)
if __name__=='__main__':unittest.main()
