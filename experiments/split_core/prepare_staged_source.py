"""Prepare current score and tail sources for isolated Model validation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from prepare_score_source import ROOT, OP, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--version", choices=("current",), default="current")
    p.add_argument("--model", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    prepare(a.output / "producer", a.version)
    shutil.copytree(ROOT / "csrc/operators", a.output / "tail/csrc/operators")
    if a.model:
        path = a.output / "tail" / OP / "op_kernel/streaming.cpp"
        s = path.read_text()
        anchor = '#include "staged_fetch_worker.h"'
        assert s.count(anchor) == 1
        s = s.replace(anchor, anchor + '\n#include "fusion3_transport_mock.h"')
        assert s.count("kvaf_device::KvafBatchGetDevice(args,0,0,size)") == 1
        s = s.replace("kvaf_device::KvafBatchGetDevice(args,0,0,size)", "fusion3_test::DeviceGet(args,0,size)")
        path.write_text(s)
    files = {
        str(p.relative_to(a.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(a.output.rglob("*"))
        if p.is_file()
    }
    (a.output / "staged-source.json").write_text(
        json.dumps(
            dict(
                producer=a.version,
                tail="current streaming",
                model_transport="synchronous GM adapter" if a.model else None,
                model_prefetch_budget=0 if a.model else None,
                files=files,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
