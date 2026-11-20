# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the sequential history/fusion launcher."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_validation_a5 as launch


class ValidationLauncher(unittest.TestCase):
    def test_default_chain_covers_all_requested_batches_and_misses(self):
        a = launch.parser().parse_args(['--kvaf-root', '/pinned'])
        self.assertEqual(a.batches, '1,2,4,8,16')
        self.assertEqual(a.miss_percent, '5,10,15,20,25,30')
        self.assertEqual(a.kvaf_server, '127.0.0.1:28985')
        self.assertEqual((a.rounds, a.warmup, a.iterations), (3, 4, 20))
        launch.validate(a, launch.parser())
        commands = launch.planned_commands(a, Path('/results'), Path('/mr14'), Path('/pinned'),
            dict(cann_root='/cann', soc='actual_soc', kvaf_include='/headers'))
        self.assertEqual([c['stage'] for c in commands], ['history-checks', 'history-build', 'history-run', 'chain-run'])
        history, chain = commands[2]['argv'], commands[-1]['argv']
        self.assertIn('history7', history)
        self.assertNotIn('--miss-percent', history)
        self.assertIn('5,10,15,20,25,30', chain)
        self.assertNotIn('--compare-mr14-serial', chain)
        self.assertNotIn('--mr14-source', chain)
        self.assertIn('/results/history7/results', history)
        self.assertIn('/results/chain', chain)

    def test_sampler_override_is_forwarded_to_both_benchmarks(self):
        a = launch.parser().parse_args(['--kvaf-root', '/pinned', '--rounds', '2', '--warmup', '7', '--iterations', '31'])
        commands = launch.planned_commands(a, Path('/results'), Path('/mr14'), Path('/pinned'),
            dict(cann_root='/cann', soc='actual_soc', kvaf_include='/headers'))
        for command in (commands[2]['argv'], commands[-1]['argv']):
            for option, expected in (('--rounds', '2'), ('--warmup', '7'), ('--iterations', '31')):
                self.assertEqual(command[command.index(option)+1], expected)

    def test_exact_source_export_has_its_own_pinned_git_head(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            source = launch.mr14_source(output)
            self.assertEqual(launch.git(source, 'rev-parse', 'HEAD'), launch.MR14)
            self.assertEqual(launch.git(source, 'status', '--porcelain', '--untracked-files=no'), '')
            plan = json.loads(subprocess.check_output([os.sys.executable,
                str(source/'experiments/li_qk/run.py'), 'plan', '--suite', 'history7', '--details'], text=True))
            rows = plan['cases']
            self.assertEqual([r['history_parts'] for r in rows], list(launch.PARTS))
            self.assertTrue(all(r['score_launches']==1 and r['final_topk_aivs']==4 for r in rows))
            p30 = [task['end']-task['begin'] for task in rows[-1]['task_map']]
            self.assertEqual((p30.count(4352), p30.count(4480)), (26, 4))
            evidence = json.loads((output/'mr14-source.json').read_text())
            self.assertEqual(evidence['commit'], launch.MR14)
            help_text = subprocess.check_output([os.sys.executable,
                str(source/'experiments/li_qk/run.py'), 'build', '--help'], text=True)
            self.assertIn('--jobs JOBS', help_text)

    def test_failure_preserves_manifest_and_never_starts_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results'
            stages = []
            def execute(item, env):
                stages.append(item['stage'])
                if item['stage'] == 'history-run':
                    raise RuntimeError('simulated history failure')
            tools = dict(cann_root='/cann', soc='actual_soc', kvaf_include='/headers')
            original_git = launch.git
            def git(root, *args):
                return launch.KVAF if str(root)=='/pinned' else original_git(root, *args)
            with patch.object(launch, 'toolchain', return_value=tools), patch.object(launch, 'git', side_effect=git), \
                    patch.object(launch, 'activated_environment', return_value=dict(os.environ)), \
                    patch.object(launch, 'physical_info'), \
                    patch.object(launch.shutil, 'which', return_value='/mock/bin'), \
                    patch.object(launch, 'run_logged', side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulated history failure'):
                    launch.main(['--kvaf-root', '/pinned', '--output', str(output)])
            self.assertEqual(stages, ['history-checks', 'history-build', 'history-run'])
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'FAILED')
            self.assertEqual(manifest['completed_stages'], ['history-checks', 'history-build'])
            self.assertFalse((output/'chain').exists())

    def test_dry_history_plan_does_not_build_or_probe_hardware(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'results'
            with patch.object(launch, 'run_logged', side_effect=AssertionError('execution in dry history')), \
                    patch.object(launch, 'activated_environment', side_effect=AssertionError('hardware environment in dry run')), \
                    contextlib.redirect_stdout(io.StringIO()):
                launch.main(['--only', 'history7', '--kvaf-root', '/not-built', '--output', str(output), '--dry-run'])
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'NOT_RUN')
            self.assertEqual(manifest['history_case_count_including_r0'], 28)
            self.assertFalse((output/'history7/build').exists())

    def test_aligned_prefetch_and_invalid_high_miss_are_rejected(self):
        for extra in (['--prefetch', '1024'], ['--miss-percent', '101'], ['--miss-percent', 'nan'], ['--batches', '1,1']):
            p = launch.parser()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.validate(p.parse_args(['--kvaf-root', '/pinned', *extra]), p)


if __name__ == '__main__':
    unittest.main()
