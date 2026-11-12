"""Check rebuilt history replay data against the same-binary original control."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "historical"))
from history_oracle import verify_local_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    a = p.parse_args()
    for directory in (a.baseline, a.candidate):
        oracle = json.loads((directory / "correctness.json").read_text())
        assert (
            oracle["status"] == "pass"
            and oracle["all_index_fields_checked"]
            and oracle["outputs_checked"]
            and oracle["guards_checked"]
        )
    raw = (a.baseline / "score-keys.bin").read_bytes()
    assert (a.candidate / "score-keys.bin").read_bytes() == raw
    geometry = json.loads((a.candidate / "history-geometry.json").read_text())
    local = verify_local_candidates(a.candidate, geometry, raw)
    h = geometry["history"]
    queries = geometry["real_batch"] * 4
    assert queries > 0 and len(raw) == queries * h * 2
    keys = np.frombuffer(raw, np.uint16).reshape(queries, h)
    indices = np.fromfile(a.candidate / "topk.bin", np.int32)
    assert indices.size == queries * 2048 + 128
    assert (indices[-128:] == 0x5A5A5A5A).all()
    for row, ids in zip(keys, indices[:-128].reshape(-1, 2048)):
        assert len(set(ids.tolist())) == 2048 and ((ids >= 0) & (ids < h)).all()
        threshold = np.partition(row, -2048)[-2048]
        assert (row[ids] >= threshold).all()
        assert set(np.flatnonzero(row > threshold)).issubset(ids.tolist())
    print(
        json.dumps(
            dict(
                status="passed",
                score_keys_sha256=hashlib.sha256(raw).hexdigest(),
                score_keys_equal=True,
                local_candidates=local,
                global_topk=True,
            )
        )
    )


if __name__ == "__main__":
    main()
