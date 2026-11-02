"""Standard-library regression tests for portable source delivery."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/package_delivery.py"
SPEC = importlib.util.spec_from_file_location("package_delivery", MODULE_PATH)
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.package = self.base / "source"
        self.package.mkdir()
        names = ["v0_base", "v2_low"]
        (self.package / "variants.json").write_text(
            json.dumps(
                {"schema_version": 1, "variants": [{"name": name, "path": f"variants/{name}"} for name in names]}
            )
        )
        for name in names:
            for relative in delivery.REQUIRED_VARIANT_FILES:
                path = self.package / "variants" / name / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture {relative}\n")
        (self.package / "README.md").write_text("portable delivery\n")
        delivery.refresh_manifest(self.package)
        self.kvip = self.base / "KVIP"
        native = self.kvip / "operators/native_lidu/op_kernel"
        native.parent.mkdir(parents=True)
        shutil.copytree(self.package / "variants/v0_base/op_kernel", native)
        (self.kvip / "operators/private.bin").write_bytes(b"protected\0payload")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, root, *args):
        self.assertEqual(root, self.kvip.resolve())
        if args == ("rev-parse", "--show-toplevel"):
            return str(self.kvip)
        if args == ("remote",):
            return "origin"
        if args == ("remote", "get-url", "origin"):
            return "https://user:secret@gitcode.com/CacheOS/KVIP.git?credential=secret"
        if args[:1] == ("rev-parse",):
            return "a" * 40
        raise AssertionError(args)

    def test_manifest_detects_tampering_and_unlisted_sources(self):
        delivery.checked_files(self.package)
        (self.package / "README.md").write_text("tampered\n")
        with self.assertRaisesRegex(delivery.DeliveryError, "SHA256 mismatch"):
            delivery.checked_files(self.package)
        delivery.refresh_manifest(self.package)
        (self.package / "unlisted.txt").write_text("new\n")
        with self.assertRaisesRegex(delivery.DeliveryError, "file set changed"):
            delivery.checked_files(self.package)

    def test_variant_requires_all_eleven_sources(self):
        self.assertEqual(len(delivery.REQUIRED_VARIANT_FILES), 11)
        (self.package / "variants/v2_low/ops.json").unlink()
        with self.assertRaisesRegex(delivery.DeliveryError, "missing required variant"):
            delivery.refresh_manifest(self.package)

    def test_variant_manifest_cannot_redirect_source_paths(self):
        manifest = self.package / "variants.json"
        value = json.loads(manifest.read_text())
        value["variants"][0]["path"] = "../../outside"
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(delivery.DeliveryError, "variant path must be"):
            delivery.refresh_manifest(self.package)

    def test_archive_is_deterministic_and_excludes_artifacts(self):
        for directory in (".artifacts", "artifacts", "dist", "__pycache__"):
            target = self.package / directory
            target.mkdir()
            (target / "large_trace.json").write_text("generated\n")
        (self.package / "temp.pyc").write_bytes(b"compiled")
        first = self.base / "one.tar.gz"
        second = self.base / "two.tar.gz"
        a = delivery.archive_package(first, self.package)
        b = delivery.archive_package(second, self.package)
        self.assertEqual(a["sha256"], b["sha256"])
        with tarfile.open(first) as archive:
            names = archive.getnames()
            self.assertIn("mtp_prepare/package_files.json", names)
            self.assertTrue(all(name.startswith("mtp_prepare/") for name in names))
            self.assertFalse(any("large_trace" in name or name.endswith(".pyc") for name in names))
            self.assertTrue(all(item.uid == item.gid == item.mtime == 0 for item in archive))
        self.assertTrue(first.with_name(first.name + ".sha256").is_file())

    def test_refresh_does_not_capture_local_builds_or_binary_outputs(self):
        before = delivery.checked_files(self.package)
        variant = self.package / "variants/v2_low"
        for name in ("build_release", "build_aiv", "OPPROF_20261102", "dump", "traces"):
            generated = variant / name
            generated.mkdir()
            (generated / "metadata.json").write_text("generated\n")
        for suffix in (".so", ".dylib", ".o", ".a", ".bin", ".whl"):
            (variant / ("compiled" + suffix)).write_bytes(b"generated")
        payload = delivery.refresh_manifest(self.package)
        self.assertEqual(set(payload["files"]), set(before))
        self.assertEqual(set(delivery.checked_files(self.package)), set(before))
        result = self.base / "clean.tar.gz"
        delivery.archive_package(result, self.package)
        with tarfile.open(result) as archive:
            self.assertFalse(any("generated" in name or "compiled" in name for name in archive.getnames()))

    def test_symlink_dependency_and_manifest_are_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("external\n")
        link = self.package / "external.txt"
        link.symlink_to(outside)
        with self.assertRaisesRegex(delivery.DeliveryError, "regular file"):
            delivery.checked_files(self.package)
        link.unlink()
        manifest = self.package / delivery.MANIFEST_NAME
        manifest.unlink()
        manifest.symlink_to(outside)
        with self.assertRaisesRegex(delivery.DeliveryError, "cannot be a symlink"):
            delivery.refresh_manifest(self.package)
        self.assertEqual(outside.read_text(), "external\n")

    def test_plan_is_read_only_and_redacts_remote_credentials(self):
        with patch.object(delivery, "git", self.git):
            result = delivery.kvip_plan(self.kvip, self.package)
        self.assertEqual(result["action"], "plan")
        self.assertEqual(result["remotes"]["origin"], "https://gitcode.com/CacheOS/KVIP.git")
        self.assertFalse((self.kvip / "experiments").exists())

    def test_stage_protects_all_operators_and_refuses_existing_target(self):
        before = delivery.operator_hashes(self.kvip / "operators")
        with patch.object(delivery, "git", self.git):
            result = delivery.stage_package(self.kvip, self.package)
            with self.assertRaisesRegex(delivery.DeliveryError, "already exists"):
                delivery.stage_package(self.kvip, self.package)
        self.assertEqual(result["action"], "staged")
        self.assertEqual(delivery.operator_hashes(self.kvip / "operators"), before)
        delivery.checked_files(self.kvip / "experiments/mtp_prepare")

    def test_stage_rejects_native_baseline_change_and_escaping_path(self):
        kernel = self.kvip / "operators/native_lidu/op_kernel" / delivery.KERNEL_NAMES[0]
        original = kernel.read_bytes()
        kernel.write_bytes(b"different baseline")
        with patch.object(delivery, "git", self.git):
            with self.assertRaisesRegex(delivery.DeliveryError, "baseline differs"):
                delivery.stage_package(self.kvip, self.package)
            kernel.write_bytes(original)
            outside = self.base / "escape"
            outside.mkdir()
            (self.kvip / "experiments").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(delivery.DeliveryError, "inside KVIP"):
                delivery.stage_package(self.kvip, self.package)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
