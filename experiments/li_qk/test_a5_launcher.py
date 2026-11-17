"""Exercise the real shell orchestration with local Git and a fake LI backend.

Does not invoke CANN kernels or measure device performance.
"""
import json
import os
from pathlib import Path
import subprocess
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


if __name__ == '__main__': unittest.main(verbosity=2)
