# SPDX-License-Identifier: Apache-2.0
"""An archived oracle can provide IDs; a single-call time can never replace a packed event."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "support"))
from fixed_q4_packed_archive import (
    IDENTITY_KEYS, NATIVE, PACKED, SINGLE,
    load_archived_oracle, oracle_dir,
)


class ArchiveTest(unittest.TestCase):
    def test_oracle_only_and_changed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "path").mkdir()
            batch = 1
            identity = {key: "test" for key in IDENTITY_KEYS}
            identity.update(batch=batch, calls=10, warmup=4, protocol=PACKED, native=NATIVE)
            ids = np.broadcast_to(np.arange(2048, dtype=np.int32), (10, batch, 4, 2048))
            np.save(directory / "selected.npy", ids, allow_pickle=False)
            receipt = dict(identity=dict(identity, protocol=SINGLE), arm="serial", get_workers=2,
                           checked_queries=40, checked_rows=10, full_KV_Index_checked_calls=1, full_KV_Index_checked_rows=1,
                           selection_sha256=hashlib.sha256((directory / "selected.npy").read_bytes()).hexdigest())
            (directory / "path/path_result.json").write_text(json.dumps(dict(path="serial", transport="urma",
                runtime_workers=2, correctness="PASS", failures=[], geometry={"batch": 1})))
            (directory / "ARM_RESULT.json").write_text(json.dumps(receipt))
            repo = Path(__file__).resolve().parents[3]
            np.testing.assert_array_equal(load_archived_oracle(directory, identity, repo), ids)
            with self.assertRaisesRegex(ValueError, "sequence_sha256"):
                load_archived_oracle(directory, dict(identity, sequence_sha256="other"), repo)
            receipt["selection_sha256"] = "1" * 64
            (directory / "ARM_RESULT.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_archived_oracle(directory, identity, repo)

    def test_unpacked_native_oracle_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for batch, cohort, destination in [(1, 7, "g2/bs1/cohort7"), (32, 0, "g2/bs32")]:
                pair = root / destination
                (pair / "fused").mkdir(parents=True)
                row = dict(arm="fused", identity={"group": 2, "batch": batch})
                if batch > 16:
                    row["comparison_archive"] = str(root / "older/g2/bs32")
                (pair / "fused/ARM_RESULT.json").write_text(json.dumps(row))
                self.assertEqual(oracle_dir(root, 2, batch, cohort), Path(row.get("comparison_archive", pair)) / "serial")


if __name__ == "__main__":
    unittest.main()
