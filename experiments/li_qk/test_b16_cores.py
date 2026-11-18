"""B16 plans/packing/evidence tests; CPU stubs never measure NPU performance."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import b16_cores as controls
from run import make_fixture,sha
from test_controls import ACL_STUB,LAUNCH_STUB

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class B16CoreControls(unittest.TestCase):
    def test_cpp_python_coverage_workload_and_owner_boundaries(self):
        code='#include "plan.h"\n#include <iostream>\nint main(int argc,char** argv){auto kind=std::stoul(argv[3]);auto c=std::stoul(argv[2]);auto p=li_qk::MakePlan(std::stoul(argv[1]),32,c,2*c,2,1,16,kind!=0,std::stoul(argv[4]),false,kind==2);for(auto r:p.tasks)std::cout<<r.queryBegin<<","<<r.queryCount<<","<<r.begin<<","<<r.end<<","<<r.owner<<"\\n";}'
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'plan.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-I'+str(HERE),str(d/'plan.cpp'),'-o',str(d/'plan')],check=True,capture_output=True)
            for h in (8192,131072):
                for grain in (128,512):
                    for c in controls.configs(grain,h):
                        kind={'waves':0,'balanced':1,'balanced_owner':2}[c['schedule']]
                        actual=[list(map(int,r.split(','))) for r in subprocess.check_output([str(d/'plan'),str(h),str(c['aics']),str(kind),str(c['work_grain'])],text=True).splitlines()]
                        expected=[list(t.values()) for t in c['geometry']['task_map']];self.assertEqual(actual,expected)
                        cover=np.zeros((16,h),np.uint8)
                        for t in c['geometry']['task_map']:
                            self.assertIn(t['query_count'],(0,4));cover[t['owner'],t['begin']:t['end']]+=1
                        self.assertTrue(np.all(cover==1));self.assertEqual(c['geometry']['final_topk_aivs'],64)
                        if c['schedule']!='waves':
                            self.assertLessEqual(c['max_score_keys_per_core']-c['min_score_keys_per_core'],c['work_grain'])
        by_name={c['name']:c for c in controls.configs()}
        self.assertEqual(by_name['C30_T2_balanced_G128']['geometry']['active_segments'],60)
        self.assertEqual(by_name['C30_T2_balanced_owner_G128']['geometry']['active_segments'],44)
        self.assertEqual(by_name['C30_T2_waves_G2048']['max_score_keys_per_core'],131072)
        self.assertEqual(by_name['C30_T2_balanced_owner_G128']['max_score_keys_per_core'],70016)
        self.assertEqual(by_name['C32_T2_balanced_owner_G128']['geometry']['task_map'],by_name['C32_T2_waves_G2048']['geometry']['task_map'])

    def test_real_host_under_stubs_all_eight_cases_and_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);(d/'acl').mkdir();(d/'acl/acl.h').write_text(ACL_STUB);(d/'launch.cpp').write_text(LAUNCH_STUB);(d/'batch_get_abi.h').write_text('#pragma once\n')
            subprocess.run(['c++','-std=c++17','-O2','-DLI_QK_RUN_MODE="cpu_stub"','-I'+str(d),'-I'+str(HERE),
                '-I'+str(ROOT/'csrc/operators/lightning_indexer_with_fetch/include'),'-I'+str(ROOT/'csrc/operators/common/lifetch'),
                str(HERE/'host.cpp'),str(d/'launch.cpp'),'-o',str(d/'driver')],check=True,capture_output=True)
            fixture=d/'fixture';meta=make_fixture(fixture,16,8192,32)
            env=dict(os.environ,LI_QK_BATCH='16',LI_QK_ORACLE=str(fixture/'oracle.bin'),LI_QK_LAUNCH_LOG=str(d/'calls'))
            for c in controls.configs(history=8192):
                out=d/c['name'];(d/'calls').write_text('')
                result=subprocess.run([str(d/'driver'),str(fixture),str(out),'8192','32',str(c['aics']),str(c['aivs']),'2','1','1','1','0','16',c['schedule'],'both',str(c['work_grain'])],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr);report=json.loads((out/'report.json').read_text())
                self.assertEqual(report['schedule'],c['schedule']);self.assertEqual(report['capacity'],c['aics'])
                self.assertEqual(report['score_peak_aivs'],c['aivs']);self.assertEqual(report['final_topk_aivs'],64)
                self.assertEqual(report['score_launches'],c['geometry']['score_launches'])
                self.assertTrue(all(report[k] for k in ('guards','inputs_unchanged','topk_threshold_oracle')))
                self.assertIsNone(report['score_p50_us']);self.assertEqual(sha(out/'score-keys.bin'),meta['hashes']['oracle'])
                calls=(d/'calls').read_text();self.assertEqual(calls.count('K'),2)
                self.assertEqual(calls.count('C'),2*c['geometry']['score_slots'])
                if c['schedule']!='waves':self.assertEqual(calls.count('B'),4)

    def test_summary_pairing_core_scope_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);oracle=b'synthetic test oracle, no actual device samples';cases=[]
            for rnd in (0,1):
                for c in controls.configs():
                    name=f'r{rnd}_{c["name"]}';out=d/name;out.mkdir();argv=['summary-test',name]
                    cases.append(dict(name=name,config=c['name'],round=rnd,argv=argv))
                    value=(200 if c['aics']==30 else 100) if c['schedule']=='waves' else (120 if c['aics']==30 else 110)
                    p=c['geometry'];report=dict(status='PASS',run_mode='npu',batch=16,queries=4,history=131072,heads=32,k=2048,history_parts=2,query_parts=1,
                        schedule=c['schedule'],work_grain=c['work_grain'],phase='both',model_stage='all',capacity=c['aics'],score_peak_aics=c['aics'],score_peak_aivs=c['aivs'],
                        score_launches=p['score_launches'],scatter_launches=p['score_slots'],final_topk_aivs=64,fetch_launches=0,warmup=4,iterations=2,
                        guards=True,inputs_unchanged=True,topk_threshold_oracle=True,score_p50_us=value,li_p50_us=value*2,
                        score_samples_us=[value-1,value+1],li_samples_us=[value*2-1,value*2+1])
                    (out/'report.json').write_text(json.dumps(report));(out/'score-keys.bin').write_bytes(oracle)
                    (d/(name+'.log')).write_text('synthetic summary check')
                    (d/(name+'.receipt.json')).write_text(json.dumps(dict(argv=argv,exit_code=0,log_sha256=sha(d/(name+'.log')),report_sha256=sha(out/'report.json'))))
            import hashlib
            manifest=dict(fine_grain=128,history=131072,rounds=1,warmup=4,iterations=2,cases=cases,fixture=dict(hashes=dict(oracle=hashlib.sha256(oracle).hexdigest())))
            (d/'manifest.json').write_text(json.dumps(manifest));controls.summarize(d)
            summary=json.loads((d/'summary.json').read_text());self.assertEqual(len(summary),8)
            self.assertEqual(summary[0]['score_over_32_waves_mean'],2)
            self.assertAlmostEqual(summary[3]['score_over_own_budget_waves_mean'],0.6)
            self.assertAlmostEqual(summary[3]['score_over_matching_32_mean'],120/110)
            path=d/'r1_C30_T2_balanced_owner_G128/report.json';r=json.loads(path.read_text());r['final_topk_aivs']=60;path.write_text(json.dumps(r))
            receipt=d/'r1_C30_T2_balanced_owner_G128.receipt.json';record=json.loads(receipt.read_text());record['report_sha256']=sha(path);receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'scope/core'):controls.summarize(d)
            manifest['cases']=cases[1:];(d/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'complete eight'):controls.summarize(d)

    def test_direct_shell_and_profile_plans(self):
        for grain in (128,512):
            p=json.loads(subprocess.check_output(['bash',str(HERE/'run_b16_a5.sh'),'--plan-only','--fine-grain',str(grain)],text=True))
            self.assertEqual(len(p['cases']),8);self.assertEqual([c['aics'] for c in p['cases']],[30]*4+[32]*4)
        profile=json.loads(subprocess.check_output(['python3',str(HERE/'profile.py'),'plan','--suite','batch16_cores','--build','/tmp/unused-b16-build','--output','/tmp/unused-b16-profile'],text=True))
        self.assertEqual(len({c['name'] for c in profile}),8);self.assertEqual([c['aics'] for c in profile],[30]*4+[32]*4)

    def test_launcher_backend_profile_and_failure_propagation(self):
        # Real local Git; fake hardware/build backend verifies orchestration only.
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);repo=d/'repo';entry=repo/'experiments/li_qk';entry.mkdir(parents=True)
            for name in ('run_history_a5.sh','run_b16_a5.sh'):
                shutil.copyfile(HERE/name,entry/name)
            backend='''import json,os,sys
from pathlib import Path
args=sys.argv[1:];action=args[0];name=Path(sys.argv[0]).name
with Path(os.environ['LI_B16_TEST_CALLS']).open('a') as f:f.write(json.dumps([name,*args])+'\\n')
if action==os.environ.get('LI_B16_TEST_FAIL'):sys.exit(23)
if action=='plan':print('{}')
if action in ('build','run'):
 out=Path(args[args.index('--output')+1]);out.mkdir(parents=True)
 if name=='b16_cores.py' and action=='run':(out/'summary.csv').write_text('synthetic orchestration only\\n')
'''
            for name in ('run.py','b16_cores.py','profile.py'):(entry/name).write_text(backend)
            (entry/'test_b16_cores.py').write_text('print("stub checks; no NPU execution")\n')
            real_git=shutil.which('git');subprocess.run([real_git,'init',str(repo)],check=True,capture_output=True)
            subprocess.run([real_git,'-C',str(repo),'add','.'],check=True,capture_output=True)
            subprocess.run([real_git,'-C',str(repo),'-c','user.name=launcher-test','-c','user.email=test@example.invalid',
                '-c','commit.gpgsign=false','commit','-m','fixture'],check=True,capture_output=True)
            bin_dir=d/'bin';bin_dir.mkdir()
            for name in ('cmake','c++','npu-smi','msprof'):
                path=bin_dir/name;path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o700)
            git=bin_dir/'git';git.write_text('#!/bin/sh\ncase "$1" in rev-parse|status) exec "$LI_B16_REAL_GIT" "$@" ;; *) echo "unexpected Git mutation/network" >&2; exit 91 ;; esac\n');git.chmod(0o700)
            cann=d/'cann';(cann/'compiler/tikcpp/ascendc_kernel_cmake').mkdir(parents=True)
            (cann/'compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake').touch()
            inc=d/'inc';inc.mkdir()
            for name in ('kvaf_batch_get_device.h','batch_get_abi.h'):(inc/name).touch()
            for failing in ('','build'):
                calls=d/('calls-'+(failing or 'pass'));out=d/('out-'+(failing or 'pass'))
                env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],LD_PRELOAD='',LI_B16_REAL_GIT=real_git,
                    LI_B16_TEST_CALLS=str(calls),LI_B16_TEST_FAIL=failing)
                r=subprocess.run(['bash',str(entry/'run_b16_a5.sh'),'--profile','--fine-grain','512',
                    '--output-root',str(out),'--cann-root',str(cann),'--kvaf-include',str(inc)],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,23 if failing else 0,r.stdout+r.stderr)
                rows=[json.loads(x) for x in calls.read_text().splitlines()]
                result=next(p for p in out.glob('li_batch16_cores_*') if p.is_dir())
                self.assertEqual((result/'exit-status.txt').read_text().strip(),str(r.returncode))
                if failing:
                    self.assertFalse(any(x[1]=='run' for x in rows));self.assertNotIn('SUCCESS:',r.stdout)
                else:
                    bench=next(x for x in rows if x[:2]==['b16_cores.py','run'])
                    profile=next(x for x in rows if x[:2]==['profile.py','run'])
                    for option,value in (('--history','131072'),('--aics','32'),('--aivs','64'),('--fine-grain','512'),
                        ('--warmup','4'),('--iterations','20'),('--rounds','3')):
                        self.assertEqual(bench[bench.index(option)+1],value)
                    self.assertEqual(profile[profile.index('--suite')+1],'batch16_cores')
                    self.assertEqual(profile[profile.index('--fine-grain')+1],'512')
                    self.assertIn('SUCCESS: eight B16/core strategies completed',r.stdout)

if __name__=='__main__':unittest.main()
