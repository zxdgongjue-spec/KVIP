# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import large_batch_reference as ref


def test_only_verified_predecessor_serial_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(ref.subprocess, "check_output", lambda *a, **k: ref.SOURCE_REV)
    current = dict(
        native="fixed",
        group=10,
        batch=20,
        physical_npu=4,
        sequence_sha256=ref.G10_BS20_SHA,
        calls=10,
        warmup=4,
        kvaf_get_kernel_sha256="same-sdk",
    )
    old, new = (tmp_path / n for n in ("old", "new"))
    src = old / "g10/bs20/serial"
    staged = new / "g10/bs20/serial-import"
    (src / "path").mkdir(parents=True)
    staged.mkdir(parents=True)
    for root in (old, new):
        for name, value in (("host.txt", "localhost"), ("machine_id.txt", "machine20"), ("physical_npu.txt", "4")):
            (root / name).write_text(value)
    (old / "native.txt").write_text(ref.SOURCE_REV)
    np.save(src / "selected.npy", np.zeros((10, 20, 4, 2048), dtype=np.int32))
    (src / "thp.json").write_text(
        json.dumps(
            dict(
                numa_pages_overlapping_vmas={"3": 50},
                status="measured",
                page_mode="thp",
                ub_device="udma",
                coverage_min_pct=99.9,
            )
        )
    )
    (src / "path/path_result.json").write_text(
        json.dumps(
            dict(
                correctness="PASS",
                failures=[],
                path="serial",
                geometry=dict(batch=20),
                runtime_workers=2,
                transport="urma",
            )
        )
    )
    sample = dict(graph=3.0, LI=1.0, tail=1.5)
    summary = dict(
        identity=dict(current, native=ref.SOURCE_REV),
        arm="serial",
        get_workers=2,
        checked_queries=800,
        checked_rows=200,
        full_KV_Index_checked_calls=1,
        full_KV_Index_checked_rows=20,
        samples=[sample] * 10,
        mean_p50_us={k: [v, v] for k, v in sample.items()},
        selection_sha256=hashlib.sha256((src / "selected.npy").read_bytes()).hexdigest(),
    )
    (src / "ARM_RESULT.json").write_text(json.dumps(summary))
    ref.stage_serial(old, staged, current, tmp_path)
    identity, marker = ref.reference_identity(staged, current, tmp_path)
    assert identity == summary["identity"] and marker["source"] == str(src)
    assert (staged / "ARM_RESULT.json").read_bytes() == (src / "ARM_RESULT.json").read_bytes()
    assert (staged / "selected.npy").read_bytes() == (src / "selected.npy").read_bytes()
    with pytest.raises(ValueError):
        ref.stage_serial(old, staged, dict(current, kvaf_get_kernel_sha256="changed"), tmp_path)
    (new / "machine_id.txt").write_text("another")
    with pytest.raises(ValueError):
        ref.stage_serial(old, staged, current, tmp_path)
    (new / "machine_id.txt").write_text("machine20")
    with (src / "selected.npy").open("ab") as f:
        f.write(b"mismatch")
    with pytest.raises(ValueError, match="checksum"):
        ref.stage_serial(old, staged, current, tmp_path)
    monkeypatch.setattr(ref.subprocess, "check_output", lambda *a, **k: "unknown")
    with pytest.raises(ValueError):
        ref.reference_identity(staged, current, tmp_path)
