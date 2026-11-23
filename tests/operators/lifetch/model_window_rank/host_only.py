"""Link an incremental host runner to an existing compiled probe device archive."""

import argparse
import json
import shlex
import subprocess
from hashlib import sha256
from pathlib import Path


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(previous, dest):
    base = previous.resolve()
    src = Path(__file__).resolve().parent
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    kernel = base / "lib/libwindow_rank.a"
    flags = (base / "CMakeFiles/probe.dir/flags.make").read_text()
    includes = next(
        line.partition("=")[2] for line in flags.splitlines() if line.startswith("CXX_INCLUDES =")
    )
    compile_flags = next(
        line.partition("=")[2] for line in flags.splitlines() if line.startswith("CXX_FLAGS =")
    )
    link = shlex.split((base / "CMakeFiles/probe.dir/link.txt").read_text())
    obj = dest / "main.cpp.o"
    exe = dest / "probe"
    before = digest(kernel)
    subprocess.run(
        [
            link[0],
            *shlex.split(includes),
            *shlex.split(compile_flags),
            "-std=c++17",
            "-c",
            str(src / "main.cpp"),
            "-o",
            str(obj),
        ],
        cwd=base,
        check=True,
    )
    pos = link.index("-o")
    link[pos + 1] = str(exe)
    link[link.index("CMakeFiles/probe.dir/main.cpp.o")] = str(obj)
    subprocess.run(link, cwd=base, check=True)
    if digest(kernel) != before:
        raise ValueError("Device library changed during link")
    proof = dict(
        device_archive=str(kernel),
        device_sha256=before,
        host_source_sha256=digest(src / "main.cpp"),
        previous_executable_sha256=digest(base / "probe"),
        runner_sha256=digest(exe),
    )
    (dest / "kernel.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_build", type=Path)
    parser.add_argument("new_host_dir", type=Path)
    args = parser.parse_args()
    build(args.previous_build, args.new_host_dir)
