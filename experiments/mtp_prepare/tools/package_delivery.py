#!/usr/bin/env python3
"""Verify, archive, and stage the self-contained MTP kernel research package.

This tool never adds, commits, or pushes Git content. Staging adds an experiment
directory and verifies that every production operator byte remains unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "package_files.json"
EXCLUDED_DIRS = frozenset(
    {
        ".artifacts",
        "artifacts",
        "dist",
        "__pycache__",
        ".git",
        "build",
        "_custom_opp",
        ".pytest_cache",
        ".ruff_cache",
        "dump",
        "traces",
    }
)
EXCLUDED_DIR_PREFIXES = ("build_", "OPPROF_")
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo", ".so", ".dylib", ".o", ".a", ".bin", ".whl")
KERNEL_NAMES = (
    "a5_lidu_fused_batch_get_prepare_mtp.cpp",
    "a5_lidu_fused_batch_get_prepare_mtp_tiling.h",
    "a5_lidu_fused_batch_get_prepare_mtp_vector.h",
)
REQUIRED_VARIANT_FILES = (
    "README.md",
    "build_lidu_fused_batch_get_prepare_mtp.sh",
    "ops.json",
    "op_host/a5_lidu_fused_batch_get_prepare_mtp.cpp",
    *(f"op_kernel/{name}" for name in KERNEL_NAMES),
    "torch_extension/setup.py",
    "torch_extension/csrc/lidu_fused_batch_get_prepare_mtp_op_api.h",
    "torch_extension/csrc/lidu_fused_batch_get_prepare_mtp_ops.cpp",
    "torch_extension/nanovllm_dsa_lidu_fused_batch_get_prepare_mtp/__init__.py",
)


class DeliveryError(ValueError):
    """The source bundle or staging target is not suitable for delivery."""


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_files(root: Path) -> dict[str, Path]:
    """Inventory included regular files, refusing symlink dependencies."""
    root = root.resolve()
    found = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(
            name for name in dirs if name not in EXCLUDED_DIRS and not name.startswith(EXCLUDED_DIR_PREFIXES)
        )
        for name in dirs:
            if (base / name).is_symlink():
                raise DeliveryError(f"source directory is a symlink: {base / name}")
        for name in sorted(names):
            if name == ".DS_Store" or name.endswith(EXCLUDED_FILE_SUFFIXES):
                continue
            path = base / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                raise DeliveryError(f"source must be a regular file: {relative}")
            if not path.resolve().is_relative_to(root):
                raise DeliveryError(f"source escapes package: {relative}")
            found[relative] = path
    return dict(sorted(found.items()))


def variant_names(root: Path) -> list[str]:
    """Require one explicit manifest entry for each complete variant tree."""
    path = root / "variants.json"
    if not path.is_file() or path.is_symlink():
        raise DeliveryError("missing regular variants.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("variants", payload) if isinstance(payload, dict) else payload
    names = []
    if isinstance(entries, dict):
        names = list(entries)
    elif isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                name = entry.get("id", entry.get("name"))
                if not name:
                    raise DeliveryError("variant entry needs id or name")
                if entry.get("path", f"variants/{name}") != f"variants/{name}":
                    raise DeliveryError(f"variant path must be variants/{name}")
                names.append(name)
            else:
                raise DeliveryError("invalid variant entry")
    else:
        raise DeliveryError("variants.json must contain a list or mapping of variants")
    if not names or len(names) != len(set(names)):
        raise DeliveryError("variant names must be nonempty and unique")
    for name in names:
        if not isinstance(name, str) or PurePosixPath(name).parts != (name,):
            raise DeliveryError(f"unsafe variant name: {name!r}")
        base = root / "variants" / name
        for relative in REQUIRED_VARIANT_FILES:
            candidate = base / relative
            if not candidate.is_file() or candidate.is_symlink():
                raise DeliveryError(f"missing required variant source: variants/{name}/{relative}")
    variants = root / "variants"
    actual = {path.name for path in variants.iterdir() if path.is_dir()}
    if actual != set(names):
        raise DeliveryError(f"variant manifest/tree mismatch: {sorted(actual ^ set(names))}")
    return sorted(names)


def refresh_manifest(root: Path = PACKAGE_ROOT) -> dict:
    variant_names(root)
    if (root / MANIFEST_NAME).is_symlink():
        raise DeliveryError("package manifest cannot be a symlink")
    files = source_files(root)
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "files": {name: digest_file(path) for name, path in files.items()},
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def checked_files(root: Path = PACKAGE_ROOT) -> dict[str, Path]:
    variant_names(root)
    manifest = root / MANIFEST_NAME
    if not manifest.is_file() or manifest.is_symlink():
        raise DeliveryError("missing package_files.json; run --refresh-manifest first")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
        raise DeliveryError("unsupported package manifest schema")
    expected = payload.get("files")
    if not isinstance(expected, dict) or not expected:
        raise DeliveryError("package manifest files must be a nonempty mapping")
    current = source_files(root)
    if current.keys() != expected.keys():
        raise DeliveryError(f"package file set changed: {sorted(current.keys() ^ expected.keys())}")
    changed = [name for name, path in current.items() if digest_file(path) != expected[name]]
    if changed:
        raise DeliveryError(f"package SHA256 mismatch: {', '.join(changed)}")
    return current


def archive_package(destination: Path, root: Path = PACKAGE_ROOT) -> dict:
    files = checked_files(root)
    destination = destination.absolute()
    checksum = destination.with_name(destination.name + ".sha256")
    if any(path.exists() or path.is_symlink() for path in (destination, checksum)):
        raise DeliveryError(f"archive/checksum already exists: {destination}")
    if destination.is_relative_to(root.resolve()) and not any(
        part in EXCLUDED_DIRS for part in destination.relative_to(root.resolve()).parts[:-1]
    ):
        raise DeliveryError("archive inside the package must be under excluded dist/ or .artifacts/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            entries = dict(files, **{MANIFEST_NAME: root / MANIFEST_NAME})
            for name, path in sorted(entries.items()):
                data = path.read_bytes()
                info = tarfile.TarInfo("mtp_prepare/" + name)
                info.size = len(data)
                info.mode = 0o755 if path.suffix == ".sh" else 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
        temporary.rename(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = digest_file(destination)
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return {"archive": str(destination), "sha256": digest, "files": len(files) + 1}


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], check=False, text=True, capture_output=True)
    if result.returncode:
        raise DeliveryError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def safe_remote(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    return url


def operator_hashes(directory: Path) -> dict[str, str]:
    """Protect all production operator files, including ignored files and links."""
    if not directory.is_dir() or directory.is_symlink():
        raise DeliveryError(f"missing regular operators directory: {directory}")
    result = {}
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(base) / name
            key = path.relative_to(directory).as_posix()
            if path.is_symlink():
                result[key] = "symlink:" + os.readlink(path)
            elif path.is_file():
                result[key] = digest_file(path)
    return result


def kvip_plan(kvip_root: Path, root: Path = PACKAGE_ROOT) -> dict:
    checked_files(root)
    kvip_root = kvip_root.resolve()
    if Path(git(kvip_root, "rev-parse", "--show-toplevel")).resolve() != kvip_root:
        raise DeliveryError("--kvip-root must be its own Git checkout root")
    native = kvip_root / "operators/native_lidu/op_kernel"
    baseline = root / "variants/v0_base/op_kernel"
    operator_hashes(kvip_root / "operators")
    for name in KERNEL_NAMES:
        if not (native / name).is_file() or (native / name).is_symlink():
            raise DeliveryError(f"KVIP native kernel is missing or linked: {name}")
        if digest_file(native / name) != digest_file(baseline / name):
            raise DeliveryError(f"KVIP native baseline differs from v0_base: {name}")
    remotes = {
        name: safe_remote(git(kvip_root, "remote", "get-url", name)) for name in git(kvip_root, "remote").splitlines()
    }
    main = git(kvip_root, "rev-parse", "--verify", "refs/heads/main")
    return {
        "action": "plan",
        "kvip_root": str(kvip_root),
        "head": git(kvip_root, "rev-parse", "HEAD"),
        "local_main": main,
        "remotes": remotes,
        "native_baseline": "three_kernel_sources_match_v0_base",
        "destination": str(kvip_root / "experiments/mtp_prepare"),
        "production_operators": "will_remain_byte_identical",
        "write_permission": "not_checked_by_read_only_plan",
        "next_step": "review fork remote; signed commit on codex/ branch; open MR",
    }


def stage_package(kvip_root: Path, root: Path = PACKAGE_ROOT) -> dict:
    plan = kvip_plan(kvip_root, root)
    kvip_root = Path(plan["kvip_root"])
    destination = Path(plan["destination"])
    experiments = destination.parent
    if destination.exists() or destination.is_symlink():
        raise DeliveryError(f"staging destination already exists: {destination}")
    if experiments.is_symlink() or not experiments.resolve().is_relative_to(kvip_root):
        raise DeliveryError("experiments path must remain inside KVIP and cannot be a symlink")
    before = operator_hashes(kvip_root / "operators")
    files = checked_files(root)
    experiments.mkdir(exist_ok=True)
    destination.mkdir()
    try:
        for relative, source in dict(files, **{MANIFEST_NAME: root / MANIFEST_NAME}).items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        checked_files(destination)
    finally:
        after = operator_hashes(kvip_root / "operators")
        if after != before:
            raise DeliveryError("production operators changed during staging; inspect checkout")
    plan.update(action="staged", production_operators="all_files_byte_identical")
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--kvip-root", type=Path)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args(argv)
    if args.stage and args.kvip_root is None:
        parser.error("--stage requires --kvip-root")
    if not any((args.refresh_manifest, args.check, args.archive, args.kvip_root)):
        parser.error("select --refresh-manifest, --check, --archive, or --kvip-root")
    try:
        if args.refresh_manifest:
            payload = refresh_manifest()
            print(json.dumps({"action": "manifest_refreshed", "files": len(payload["files"])}))
        if args.check:
            print(json.dumps({"action": "checked", "files": len(checked_files())}))
        if args.archive:
            print(json.dumps(archive_package(args.archive), indent=2))
        if args.kvip_root:
            result = stage_package(args.kvip_root) if args.stage else kvip_plan(args.kvip_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (DeliveryError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"package delivery failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
