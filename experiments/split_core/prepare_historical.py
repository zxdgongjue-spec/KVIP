"""Recreate the pinned experiment sources and synthetic fixture from Git, not .a files."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = Path(__file__).resolve().parent / "historical"
PIN = "81b1472809b7a5f3469e26467feee0ca1c61d71a"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((SNAP / "source_manifest.json").read_text())
    assert manifest["base"] == PIN
    for relative, expected in manifest["files"].items():
        assert hashlib.sha256((SNAP / relative).read_bytes()).hexdigest() == expected
    raw = subprocess.check_output(["git", "-C", str(ROOT), "archive", PIN])
    base = a.output / "full-source"
    base.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(base, filter="data")
    for variant, destination in (
        ("base", base),
        ("pooled", a.output / "pooled-source"),
        ("history", a.output / "history-source"),
    ):
        if variant != "base":
            shutil.copytree(base, destination)
        shutil.copytree(SNAP / "overlay" / variant, destination, dirs_exist_ok=True)
    sys.path.insert(0, str(base / "tests/operators/lifetch/support"))
    from real_fetch_inputs import prepare_full_chain_fixture

    fixture = a.output / "standard-fixture"
    prepare_full_chain_fixture(fixture, 131072, 16384, 10, index_heads=32)
    weights = np.fromfile(fixture / "arg2.bin", np.float32)
    (weights.view(np.uint32) >> 16).astype(np.uint16).tofile(fixture / "weights-bf16.bin")
    bits = np.load(fixture / "logical_scores.npy")[:, np.load(fixture / "physical_to_logical.npy")]
    thresholds = np.partition(bits, -2048, axis=1)[:, -2048][:, None]
    (bits >= thresholds).astype(np.uint8).tofile(fixture / "expected-selected.bin")
    shutil.copy2(SNAP / "lru-seed.bin", fixture / "lru.bin")
    (a.output / "prepared.json").write_text(
        json.dumps(
            dict(
                base=PIN,
                fixture_files={
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in fixture.iterdir() if f.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(a.output.resolve())


if __name__ == "__main__":
    main()
