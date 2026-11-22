"""Prepare an isolated output or Index simulator module; no production edits."""
from pathlib import Path
import shutil
import subprocess
import sys
here = Path(__file__).resolve().parent
kind, destination = sys.argv[1:]
if kind not in ('output', 'index'):
    raise ValueError(kind)
out = Path(destination)
subprocess.run([sys.executable, str(here / 'prepare.py'), str(out)], check=True)
for source in (here / kind).iterdir():
    shutil.copy2(source, out / source.name)
