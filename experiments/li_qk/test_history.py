"""History-only plans, real host under ACL stubs, and evidence pairing checks.

CPU/synthetic checks do not produce hardware timing or Model results.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import run as controls
from test_controls import ACL_STUB,LAUNCH_STUB

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARTS=(1,2,4,8,16,32,30)

class HistoryControls(unittest.TestCase):
    def test_cpp_python_balanced_coverage_and_single_wave(self):
        code='#include "plan.h"\n#include <iostream>\nint main(int argc,char** argv){unsigned h=std::stoul(argv[1]);for(unsigned t:{1,2,4,8,16,32,30}){auto p=li_qk::MakePlan(h,32,32,64,t,1);for(auto r:p.tasks)std::cout<<r.queryBegin<<","<<r.queryCount<<","<<r.begin<<","<<r.end<<","<<r.owner<<"\\n";}}'
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'plan.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True,capture_output=True)
            for h in (8192,131072):
                actual=iter(subprocess.check_output([str(d/'plan'),str(h)],text=True).splitlines())
                for t in PARTS:
                    p=controls.geometry(t,1,h);cover=np.zeros(h,np.uint8)
                    self.assertEqual(p['score_waves'],1);self.assertEqual(p['score_peak_aics'],t);self.assertEqual(p['score_peak_aivs'],2*t)
                    for task in p['task_map']:
                        self.assertEqual(list(map(int,next(actual).split(','))),[*task.values(),0])
                        cover[task['begin']:task['end']]+=1
                    self.assertTrue(np.all(cover==1));self.assertEqual(p['logical_qk_macs'],4*32*h*128)
                    self.assertLessEqual(max(p['key_tokens_per_task'])-min(p['key_tokens_per_task']),p['history_partition_quantum'])
                self.assertEqual(list(actual),[])
        p=controls.geometry(30,1);lengths=[x['end']-x['begin'] for x in p['task_map']]
        self.assertEqual(lengths.count(4352),26);self.assertEqual(lengths.count(4480),4)

    def test_actual_host_all_seven_score_scatter_topk_and_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB)
            (d/'launch.cpp').write_text(LAUNCH_STUB);(d/'batch_get_abi.h').write_text('#pragma once\n')
            subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="cpu_stub"','-I'+str(d),'-I'+str(HERE),
                '-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),
                str(HERE/'host.cpp'),str(d/'launch.cpp'),'-o',str(d/'driver')],check=True,capture_output=True)
            meta=controls.make_fixture(d/'fixture',1,8192,32)
            env=dict(os.environ,LI_QK_ORACLE=str(d/'fixture/oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'))
            for t in PARTS:
                (d/'calls').write_text('');out=d/f'T{t}'
                result=subprocess.run([str(d/'driver'),str(d/'fixture'),str(out),'8192','32','32','64',str(t),'1','1','1','0'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                report=json.loads((out/'report.json').read_text());self.assertEqual(report['score_peak_aics'],t)
                self.assertEqual(report['score_peak_aivs'],2*t);self.assertEqual(report['final_topk_aivs'],4)
                self.assertEqual(report['score_launches'],1);self.assertTrue(report['guards']);self.assertTrue(report['inputs_unchanged'])
                self.assertTrue(report['topk_threshold_oracle']);self.assertIsNone(report['score_p50_us'])
                self.assertEqual(controls.sha(out/'score-keys.bin'),meta['hashes']['oracle'])
                calls=(d/'calls').read_text();self.assertEqual(calls.count('S'),4);self.assertEqual(calls.count('K'),2)
                self.assertEqual(calls.count('C'),0 if t==1 else 2)

    def test_summary_pairs_only_complete_history_suite_and_checks_core_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);oracle=b'synthetic summary fixture; no device measurements';cases=[]
            for rnd in (0,1):
                for t in PARTS:
                    name=f'r{rnd}_T{t}_Qp1';out=d/name;out.mkdir();argv=['summary-test',name]
                    cases.append(dict(name=name,round=rnd,history_parts=t,query_parts=1,schedule='waves',argv=argv))
                    planned=controls.geometry(t,1)
                    proof=dict(status='PASS',run_mode='npu',batch=1,queries=4,history=131072,heads=32,history_parts=t,query_parts=1,
                        fetch_launches=0,warmup=4,iterations=2,guards=True,inputs_unchanged=True,topk_threshold_oracle=True,
                        score_p50_us=100,li_p50_us=200,score_samples_us=[90,110],li_samples_us=[190,210],score_waves=1,
                        score_launches=1,tasks=t,physical_query_tile=4,score_peak_aics=t,score_peak_aivs=t*2,
                        final_topk_aivs=4,history_partition_quantum=planned['history_partition_quantum'])
                    (out/'report.json').write_text(json.dumps(proof));(out/'score-keys.bin').write_bytes(oracle)
                    log=d/(name+'.log');log.write_text('synthetic timing validation only')
                    (d/(name+'.receipt.json')).write_text(json.dumps(dict(exit_code=0,argv=argv,log_sha256=controls.sha(log),report_sha256=controls.sha(out/'report.json'))))
            manifest=dict(suite='history7',cases=cases,history=131072,warmup=4,iterations=2,rounds=1,fixture=dict(hashes=dict(oracle=hashlib.sha256(oracle).hexdigest())))
            (d/'manifest.json').write_text(json.dumps(manifest));controls.write_summary(d)
            summary=json.loads((d/'summary.json').read_text());self.assertEqual([x['history_parts'] for x in summary],list(PARTS))
            self.assertEqual(summary[-1]['min_history_per_core'],4352);self.assertEqual(summary[-1]['max_history_per_core'],4480)
            path=d/'r1_T30_Qp1/report.json';proof=json.loads(path.read_text());proof['score_peak_aics']=32;path.write_text(json.dumps(proof))
            receipt=d/'r1_T30_Qp1.receipt.json';r=json.loads(receipt.read_text());r['report_sha256']=controls.sha(path);receipt.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'core allocation'):controls.write_summary(d)

    def test_repository_script_plan_only_and_budget_gate(self):
        output=json.loads(subprocess.check_output(['bash',str(HERE/'run_history_a5.sh'),'--plan-only'],text=True))
        self.assertEqual([r['history_parts'] for r in output['cases']],list(PARTS))
        self.assertTrue(all(r['query_parts']==1 for r in output['cases']))
        result=subprocess.run(['python3',str(HERE/'run.py'),'plan','--suite','history7','--aics','30','--aivs','60'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)

if __name__=='__main__':unittest.main()
