"""Opt-in experimental terminal for both formal bench input sources."""
import ctypes
import importlib.util
from pathlib import Path
import sys


def load_runtime_module(name):
    directory = Path(__file__).resolve().parents[3] / 'experiments/final_window/parallel_tail/runtime'
    spec = importlib.util.spec_from_file_location('lifetch_terminal_' + name, directory / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    # Generators use sibling imports. Restore the search path after startup.
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


class TerminalFactory:
    def __init__(self, directory, headers, cann, soc):
        products = load_runtime_module('build').build(directory, headers, cann, soc)
        self.metadata = dict(implementation='parallel', libraries=products)
        self.prefix = ctypes.CDLL(products['prefix']['path'])
        self.terminal = ctypes.CDLL(products['terminal']['path'])
        self.wrapper = load_runtime_module('native_forward').ParallelTerminalForward
        self.save_failure = load_runtime_module('failure_report').save_failure

    def attach(self, state):
        state.forward = self.wrapper(state.forward, self.prefix, self.terminal)
        state.terminal_factory = self


def prepare_terminal(config, directory, headers, cann, soc, pool=None):
    if getattr(config, 'terminal', 'legacy') != 'parallel':
        return None
    if config.mode != 'bench' or config.transport != 'urma':
        raise ValueError('Parallel terminal is supported by URMA bench only')
    key = (str(Path(headers).resolve()), str(Path(cann).resolve()), soc)
    cached = getattr(pool, 'parallel_terminal_factory', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    print('BENCH terminal=parallel: building isolated prefix + four-core terminal', flush=True)
    factory = TerminalFactory(directory / 'parallel-native', headers, cann, soc)
    if pool is not None:
        pool.parallel_terminal_factory = (key, factory)
    return factory
