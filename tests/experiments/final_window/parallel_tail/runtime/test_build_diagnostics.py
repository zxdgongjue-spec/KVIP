"""Build failures retain the real compiler diagnostic in the worker traceback."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location('terminal_build_test', HERE / 'build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


class BuildDiagnosticsTest(unittest.TestCase):
    def test_compiler_error_survives_long_make_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'build.log'
            with path.open('w') as log:
                with self.assertRaises(RuntimeError) as caught:
                    module.run_logged([sys.executable, '-c',
                        "print('ld.lld: error: undefined symbol: CoveredAddress'); "
                        "print('referenced by kernel.cpp:73'); "
                        "print('make output\\n'*200); raise SystemExit(2)"], log)
            text = str(caught.exception)
            self.assertIn('undefined symbol: CoveredAddress', text)
            self.assertIn('referenced by kernel.cpp:73', text)
            self.assertIn(str(path), text)
            self.assertIn('make output', path.read_text())

    def test_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (Path(tmp) / 'build.log').open('w') as log:
                module.run_logged([sys.executable, '-c', "print('built')"], log)
