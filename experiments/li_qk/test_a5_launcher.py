"""Exercise the real shell orchestration with local Git and a fake LI backend.

Does not invoke CANN kernels or measure device performance.
"""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('run_a5.sh')
BACKEND = r'''
import json, os, sys
from pathlib import Path
action = sys.argv[1]
with Path(os.environ['LI_LAUNCHER_TEST_CALLS']).open('a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\n')
if os.environ.get('LI_LAUNCHER_TEST_FAIL') == action:
    print('Intentional orchestration-test failure', flush=True)
    sys.exit(23)
if action == 'plan':
    print(json.dumps({'test_only': True, 'points': [[1,1],[2,1],[4,1],[8,1],[16,1],[1,2],[1,4]]}))
elif action in ('build', 'run'):
    out = Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir()
    (out/'TEST_ONLY.txt').write_text('Fake backend; no device result or timing\n')
elif action == 'summarize':
    Path(sys.argv[2], 'summary.csv').write_text('scope\nlauncher-test-only\n')
'''


class Launcher(unittest.TestCase):
    def setup_fixture(self, base):
        repo = base/'original checkout'; repo.mkdir()
        op = repo/'experiments/li_qk'; op.mkdir(parents=True)
        (op/'run.py').write_text(BACKEND)
        (op/'test_controls.py').write_text("print('FAKE CPU CHECK: orchestration only')\n")
        (repo/'preserve.txt').write_text('committed\n')
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()
        git('init', '-b', 'codex/split-core-unified-comparison')
        git('config', 'user.name', 'Launcher Test'); git('config', 'user.email', 'test@example.invalid')
        git('add', '.'); git('-c', 'commit.gpgsign=false', 'commit', '-m', 'Local fake LI backend')
        ref = git('rev-parse', 'HEAD')
        remote = base/'local origin.git'
        subprocess.run(['git','clone','--bare',str(repo),str(remote)], check=True, capture_output=True)
        git('remote', 'add', 'origin', str(remote))
        (repo/'preserve.txt').write_text('uncommitted work to preserve\n')
        cann = base/'fake cann'; include = base/'fake kvaf'
        cmake_file = cann/'compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake'
        cmake_file.parent.mkdir(parents=True); cmake_file.touch(); include.mkdir()
        for name in ('kvaf_batch_get_device.h', 'batch_get_abi.h'): (include/name).touch()
        bin_dir = base/'bin'; bin_dir.mkdir()
        for name in ('npu-smi', 'cmake', 'c++'):
            p = bin_dir/name; p.write_text('#!/usr/bin/env bash\necho MOCK_ENVIRONMENT_TEST_ONLY\n'); p.chmod(0o755)
        calls = base/'calls.jsonl'
        env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
                   LI_LAUNCHER_TEST_CALLS=str(calls))
        env.pop('LD_PRELOAD', None)
        cmd = ['bash',str(SCRIPT),'--repo',str(repo),'--ref',ref,
               '--cann-root',str(cann),'--kvaf-include',str(include),
               '--python',sys.executable,'--work-root',str(base/'worktrees'),
               '--output-root',str(base/'outputs'),'--aics','28','--aivs','56']
        return cmd, env, repo, calls

    def test_success_and_sdk_argument_forwarding(self):
        for sdk in (False, True):
            with self.subTest(sdk=sdk), tempfile.TemporaryDirectory() as folder:
                base = Path(folder); cmd, env, repo, calls = self.setup_fixture(base)
                if sdk:
                    # Only stub the optional import check; the shell SDK flag stays real.
                    modules = base/'sdk-stubs'; modules.mkdir()
                    for name in ('torch.py','torch_npu.py'): (modules/name).touch()
                    env['PYTHONPATH']=str(modules); cmd.append('--sdk-reference')
                r = subprocess.run(cmd,env=env,text=True,capture_output=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                actions = [json.loads(s) for s in calls.read_text().splitlines()]
                self.assertEqual([x[0] for x in actions], ['plan','build','run','summarize'])
                run = actions[2]
                for key, value in (('--history','131072'),('--aics','28'),('--aivs','56'),
                                   ('--warmup','4'),('--iterations','20'),('--rounds','3')):
                    self.assertEqual(run[run.index(key)+1],value)
                self.assertEqual('--sdk-reference' in run,sdk)
                outputs = [p for p in (base/'outputs').iterdir() if p.is_dir()]
                self.assertEqual(len(outputs),1)
                out = outputs[0]
                self.assertEqual((out/'exit-status.txt').read_text().strip(),'0')
                self.assertTrue((out/'results/summary.csv').is_file())
                with tarfile.open(str(out)+'.tar.gz') as tar:
                    self.assertIn(out.name+'/results/summary.csv',tar.getnames())
                self.assertEqual((repo/'preserve.txt').read_text(),'uncommitted work to preserve\n')

    def test_failures_stop_and_preserve_exit_code_and_logs(self):
        for fail in ('build','run'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                base = Path(folder); cmd, env, repo, calls = self.setup_fixture(base)
                env['LI_LAUNCHER_TEST_FAIL']=fail
                r = subprocess.run(cmd,env=env,text=True,capture_output=True)
                self.assertEqual(r.returncode,23,r.stdout+r.stderr)
                actions = [json.loads(s)[0] for s in calls.read_text().splitlines()]
                self.assertEqual(actions, ['plan','build'] if fail=='build' else ['plan','build','run'])
                out = next(p for p in (base/'outputs').iterdir() if p.is_dir())
                self.assertEqual((out/'exit-status.txt').read_text().strip(),'23')
                self.assertTrue(Path(str(out)+'.tar.gz').is_file())
                self.assertFalse((out/'results/summary.csv').exists())
                self.assertEqual((repo/'preserve.txt').read_text(),'uncommitted work to preserve\n')

    def test_help_and_invalid_budget_without_side_effects(self):
        r = subprocess.run(['bash',str(SCRIPT),'--help'],text=True,capture_output=True)
        self.assertEqual(r.returncode,0); self.assertIn('seven orthogonal',r.stdout)
        r = subprocess.run(['bash',str(SCRIPT),'--aics','0'],text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0); self.assertIn('Supported budget',r.stderr)

    def test_offline_commit_and_bundle_with_unreachable_origin(self):
        for mode in ('local', 'bundle'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                base=Path(folder); cmd,env,repo,calls=self.setup_fixture(base)
                def git(*args):
                    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.STDOUT).strip()
                if mode=='bundle':
                    bundle=base/'offline code.bundle'
                    ref='refs/remotes/origin/codex/split-core-unified-comparison'
                    git('update-ref',ref,git('rev-parse','HEAD'))
                    git('bundle','create',str(bundle),ref)
                    cmd+=['--git-bundle',str(bundle)]
                else: cmd.append('--skip-fetch')
                git('remote','set-url','origin','https://offline.invalid/KVIP.git')
                r=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                self.assertEqual([json.loads(s)[0] for s in calls.read_text().splitlines()],['plan','build','run','summarize'])
                out=next(p for p in (base/'outputs').iterdir() if p.is_dir())
                self.assertIn('fetch_mode='+('offline_bundle' if mode=='bundle' else 'local_commit'),(out/'launcher-config.txt').read_text())
                self.assertTrue((out/'results/summary.csv').is_file())

    def test_direct_fetch_clears_only_process_proxy_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); cmd,env,repo,calls=self.setup_fixture(base)
            real_git=shutil.which('git'); proof=base/'proxy-proof.json'
            wrapper=base/'bin/git'
            wrapper.write_text('#!/usr/bin/env python3\n'+
                'import json, os, sys\nfrom pathlib import Path\n'+
                'if "fetch" in sys.argv and "origin" in sys.argv:\n'+
                '    keys=("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy")\n'+
                '    ok=not any(k in os.environ for k in keys) and "http.proxy=" in sys.argv and "remote.origin.proxy=" in sys.argv\n'+
                '    Path('+repr(str(proof))+').write_text(json.dumps({"proxy_cleared":ok}))\n'+
                '    if not ok: sys.exit(91)\n'+
                'os.execv('+repr(real_git)+', ['+repr(real_git)+']+sys.argv[1:])\n')
            wrapper.chmod(0o755)
            for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env[key]='http://test.invalid:1'
            subprocess.run([real_git,'-C',str(repo),'config','remote.origin.proxy','http://test.invalid:1'],check=True)
            cmd.append('--no-proxy')
            r=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertTrue(json.loads(proof.read_text())['proxy_cleared'])
            saved=subprocess.check_output([real_git,'-C',str(repo),'config','remote.origin.proxy'],text=True).strip()
            self.assertEqual(saved,'http://test.invalid:1')

    def test_skip_fetch_rejects_absent_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); cmd,env,repo,calls=self.setup_fixture(base)
            cmd[cmd.index('--ref')+1]='0000000000000000000000000000000000000001'
            cmd.append('--skip-fetch')
            r=subprocess.run(cmd,env=env,text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('absent locally',r.stderr)
            self.assertFalse(calls.exists())


if __name__ == '__main__': unittest.main(verbosity=2)
