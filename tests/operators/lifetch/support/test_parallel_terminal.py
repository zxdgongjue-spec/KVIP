"""CPU regression checks for formal bench integration."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics import account
from performance import split_graph_invocations
from parallel_terminal import TerminalFactory, load_runtime_module, prepare_terminal
from workloads import configuration


class ParallelBenchTests(unittest.TestCase):
    def test_both_input_sources(self):
        for source, extra in [('synthetic', []), ('real', ['--trace-root', '/tmp/trace', '--group', '2', '--round-range', '1:1'])]:
            args = configuration(['bench', '--terminal', 'parallel', '--input', source] + extra)
            self.assertEqual(args.terminal, 'parallel')
        self.assertEqual(configuration(['bench']).terminal, 'legacy')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            configuration(['bench', '--terminal', 'parallel', '--transport', 'ubmem'])

    def test_runtime_imports(self):
        self.assertTrue(callable(load_runtime_module('build').build))
        self.assertTrue(callable(load_runtime_module('native_forward').ParallelTerminalForward))

    def test_shared_build_independent_wrappers(self):
        config = SimpleNamespace(terminal='parallel', mode='bench', transport='urma')
        pool = SimpleNamespace()
        with patch('parallel_terminal.TerminalFactory') as factory:
            a = prepare_terminal(config, Path('/tmp/a'), '/headers', '/cann', 'soc', pool)
            b = prepare_terminal(config, Path('/tmp/b'), '/headers', '/cann', 'soc', pool)
            self.assertIs(a, b)
            factory.assert_called_once()
        factory = TerminalFactory.__new__(TerminalFactory)
        factory.prefix, factory.terminal = object(), object()
        factory.wrapper = lambda forward, *libs: SimpleNamespace(original=forward)
        states = [SimpleNamespace(forward=object()) for _ in range(20)]
        for state in states:
            factory.attach(state)
        self.assertEqual(len({id(s.forward) for s in states}), 20)
        self.assertEqual(len({id(s.forward.original) for s in states}), 20)

    def test_same_graph_accounting(self):
        events = [dict(name='scope', ph='X', ts=0, dur=1000)]
        for offset in (10, 110):
            for name, start, duration in [('v9_li_kernel_0', 0, 10), ('v9_li_kernel_1_1', 10, 10),
                                         ('v9_li_kernel_2_2', 20, 30), ('v9_parallel_terminal_0', 50, 30)]:
                events.append(dict(name=name, ph='X', cat='kernel', ts=offset+start, dur=duration))
        result = split_graph_invocations(account(events, 'scope', 200, full=True, fused_final=True), 2)
        self.assertEqual(result['timing']['profiler_mean_us'], 80)
        self.assertEqual(result['timing']['event_mean_us'], 100)
        self.assertEqual(result['timing']['stage_mean_us']['terminal_topk_update_fetch'], 30)
        mixed = events + [dict(name='v9_streaming_resolve_update_5', ph='X', cat='kernel', ts=190, dur=1)]
        self.assertIn('Mixed', account(mixed, 'scope', 200, full=True)['error'])
        events.pop(1)
        with self.assertRaisesRegex(ValueError, 'three prefix'):
            split_graph_invocations(account(events, 'scope', 200, full=True, fused_final=True), 2)


if __name__ == '__main__':
    unittest.main()
