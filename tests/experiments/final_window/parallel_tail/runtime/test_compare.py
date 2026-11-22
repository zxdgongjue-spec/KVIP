import ast
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('terminal_compare', HERE / 'compare.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RunnerTests(unittest.TestCase):
    def test_both_runners_are_executable_python_and_distinct(self):
        base = (HERE / 'snapshot_runner_base.py').read_text()
        for candidate in (False, True):
            text = module.runner_source(base, candidate, Path('/tmp/test-terminal'))
            ast.parse(text)
            self.assertEqual('forward = ParallelTerminalForward' in text, candidate)
            self.assertIn('selected-{iteration-args.warmup}.pt', text)
            self.assertIn('streaming_resolve_update|parallel_terminal', text)

    def test_profile_requires_every_stage_and_sample(self):
        for candidate in (False, True):
            names = ['v9_li_kernel_0', 'v9_li_kernel_1_1', 'v9_li_kernel_2_2']
            names += (['v9_parallel_terminal_0'] if candidate else
                      ['v9_li_kernel_3_3', 'v9_streaming_resolve_update_5'])
            values = {name: [1.0, 2.0] for name in names}
            module.validate_kernel_chain(values, candidate, 2)
            for name in names:
                with self.subTest(candidate=candidate, missing=name):
                    incomplete = dict(values)
                    del incomplete[name]
                    with self.assertRaises(ValueError):
                        module.validate_kernel_chain(incomplete, candidate, 2)
            with self.assertRaises(ValueError):
                module.validate_kernel_chain(values, candidate, 3)
            with self.assertRaises(ValueError):
                module.validate_kernel_chain({**values, 'v9_li_kernel_0_9': [1, 2]}, candidate, 2)

    def test_changed_template_rejected(self):
        with self.assertRaises(ValueError):
            module.runner_source('', True, Path('/tmp/test-terminal'))


if __name__ == '__main__':
    unittest.main()
