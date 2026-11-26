# SPDX-License-Identifier: Apache-2.0
"""Keep B20 published numbers out of the B24 per-card acceptance decision."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import fixed_bs16_report as report


class PackedOnCard3(TestCase):
    def test_local_reference_with_reversed_blocks(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            archive = out / "b20-archive"
            archive.mkdir()
            (out / "native.txt").write_text("candidate-sha")
            (out / "physical_npu.txt").write_text("3")
            for block in (1, 2):
                for arm in ("candidate", "pr30_reference"):
                    path = out / f"pr30/block{block}/g2/{arm}"
                    path.mkdir(parents=True)
                    (path / "oracle.txt").write_text(str(out / "npu3-oracle"))

            def packed(root, group, arm, revision, owner, verify_library):
                self.assertEqual(group, 2)
                self.assertEqual(owner, out / "npu3-oracle")
                self.assertTrue(verify_library)
                self.assertEqual(revision, "candidate-sha" if arm == "candidate" else report.PACED)
                index = int(root.name[-1]) - 1
                times = (301.0, 302.0) if arm == "candidate" else (306.0, 305.0)
                return times[index], "same_numa", "same_operands", "same_capture"

            archive_check = Mock(side_effect=AssertionError("NPU3 must measure a local reference"))
            placement = Mock(return_value="matched")
            modules = {"feature_host_reference": SimpleNamespace(same_local_device=archive_check),
                       "peer_delta": SimpleNamespace(pair_placement=placement)}
            with patch.dict(sys.modules, modules), patch.object(report, "packed_block", side_effect=packed):
                result = report.evaluate_packed(out, 2, archive)
                self.assertEqual(result["reference_kind"], "same_card_live")
                self.assertNotIn("published_packed_us", result)
                self.assertTrue(result["passed"])
                self.assertEqual(result["saved_us"], 4.0)
                placement.return_value = "different"
                self.assertFalse(report.evaluate_packed(out, 2, archive)["passed"])
                self.assertEqual(placement.call_count, 4)
                archive_check.assert_not_called()
