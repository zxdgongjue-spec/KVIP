"""Verify actual full LI host with CPU kernel stubs; no hardware timing claims."""
import json,os,subprocess,tempfile,unittest
from pathlib import Path
import li_radix_compare as compare
from test_controls import ACL_STUB,LAUNCH_STUB
from test_radix import STUB
from run import HERE,ROOT,make_fixture,sha

class FullRadixControls(unittest.TestCase):
    def test_actual_host_three_chains_and_canonical(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'batch_get_abi.h').write_text('#pragma once\n')
            # Use the original score/scatter/topk stubs, append only radix entrypoints.
            radix=STUB[STUB.index('extern "C" unsigned aclrtlaunch_li_radix_high('):]
            helper='''\n#include "radix_plan.h"\nstatic void choose(uint16_t* s,int32_t* out,uint32_t b,uint32_t h,uint32_t*){aclrtlaunch_li_qk_topk(b*4,nullptr,s,out,nullptr,h,32,b*4);}\n'''
            (d/'launch.cpp').write_text(LAUNCH_STUB+helper+radix)
            subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="npu"','-DLI_QK_RADIX_COMPARE=1','-I'+str(d),'-I'+str(HERE),'-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),str(HERE/'host.cpp'),str(d/'launch.cpp'),'-o',str(d/'host')],check=True,capture_output=True)
            fixture=make_fixture(d/'fixture',1,8192,32)
            env=dict(os.environ,LI_QK_ORACLE=str(d/'fixture/oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'))
            evidence=[]
            for c in compare.definitions():
                (d/'calls').write_text('');out=d/c['name'];cmd=[str(d/'host'),str(d/'fixture'),str(out),'8192','32','32','64',str(c['score_parts']),'1','1','2','0','1','waves','both','2048',c['topk_route']]
                r=subprocess.run(cmd,env=env,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
                p=json.loads((out/'report.json').read_text());self.assertEqual(p['full_li_kernel_launches'],c['full_li_kernel_launches'])
                self.assertEqual(p['tail_peak_aivs'],c['tail_peak_aivs']);self.assertEqual(p['radix_parts'],c['radix_parts']);self.assertTrue(p['full_valid_history'])
                for name in ('score','li','topk'):self.assertEqual(len(p[name+'_samples_us']),2)
                self.assertEqual(sha(out/'score-keys.bin'),fixture['hashes']['oracle']);self.assertTrue(p['topk_threshold_oracle'])
                calls=(d/'calls').read_text();self.assertEqual(calls.count('S'),6);self.assertEqual(calls.count('C'),0 if c['score_parts']==1 else 3)
                self.assertEqual(calls.count('K'),6) # 3 full LI plus 3 separate tail calls
                log=d/(c['name']+'.log');log.write_text(r.stdout+r.stderr)
                evidence.append(dict(name=c['name'],round=0,case=c,exit_code=0,report_sha256=sha(out/'report.json'),log_sha256=sha(log)))
            # CPU event stubs produce synthetic samples solely to test receipt validation.
            manifest=dict(status='PASS',mode='bench',fixture=fixture,history=8192,warmup=1,iterations=2,rounds=1,rows=[])
            import shutil
            for rnd in (0,1):
                for row in evidence:
                    copy=dict(row,name=f'r{rnd}_{row["name"]}',round=rnd)
                    shutil.copytree(d/row['name'],d/copy['name']);shutil.copy2(d/(row['name']+'.log'),d/(copy['name']+'.log'));manifest['rows'].append(copy)
            (d/'manifest.json').write_text(json.dumps(manifest));compare.summarize(d)
            manifest['rows'].pop();(d/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'all three paired'):compare.summarize(d)

    def test_plan_has_one_changed_tail(self):
        serial,uniform,radix=compare.definitions()
        self.assertEqual(serial['score_parts'],1)
        self.assertEqual((uniform['score_parts'],radix['score_parts']),(16,16))
        for key in ('score_aics','score_aivs','scatter_launches'):self.assertEqual(uniform[key],radix[key])
        self.assertEqual((uniform['tail_kernel_launches'],radix['tail_kernel_launches']),(1,5))

if __name__=='__main__':unittest.main()
