"""Ship device sources in wheels; compilation is explicit and shape-specific."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "operators" / "v9"
        target = Path(self.build_lib) / "kvop" / "v9" / "_native"
        shutil.copytree(
            source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "build*", "*.pyc")
        )


setup(cmdclass={"build_py": BuildPy})
