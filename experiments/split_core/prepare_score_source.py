"""Copy the current operator into an isolated score validation build."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OP = Path("csrc/operators/lightning_indexer_with_fetch")


def prepare(output, version="current"):
    if version != "current":
        raise ValueError("Only current operator sources are included; historical sources remain in Git history")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "csrc/operators", output / "csrc/operators")
    evidence = dict(
        version=version,
        base_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
        ),
        source_files={
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((output / "csrc/operators").rglob("*"))
            if p.is_file()
        },
        scope="current score wrapper; not native transport validation",
    )
    (output / "score-source.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", choices=("current",), default="current")
    args = parser.parse_args()
    evidence = prepare(args.output, args.version)
    print(json.dumps(dict(version=evidence["version"], files=len(evidence["source_files"]))))
