"""Opt-in experimental terminal for both formal bench input sources."""
from contextlib import contextmanager
import ctypes
import importlib.util
from pathlib import Path
import sys


@contextmanager
def runtime_import_path():
    directory = Path(__file__).resolve().parents[3] / 'experiments/final_window/parallel_tail/runtime'
    previous = sys.path[:]
    sys.path.insert(0, str(directory))
    try:
        yield directory
    finally:
        sys.path[:] = previous


def load_runtime_module(name):
    with runtime_import_path() as directory:
        spec = importlib.util.spec_from_file_location('lifetch_terminal_' + name, directory / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def build_runtime(directory, headers, cann, soc):
    # generate() also imports sibling code generators lazily. Keep the path
    # available through the entire build, not only module initialization.
    with runtime_import_path():
        return load_runtime_module('build').build(directory, headers, cann, soc)


class TerminalFactory:
    def __init__(self, directory, headers, cann, soc):
        products = build_runtime(directory, headers, cann, soc)
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
