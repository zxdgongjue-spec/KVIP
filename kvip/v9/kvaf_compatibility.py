"""Read-only KVAF source/runtime surface evidence, never SDK certification.

No framework, Torch, KVAF extension, compiler, or device is imported or started.
Raw pinned source and a consumer bridge overlay are distinct provenance layers.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

PROFILES = ("legacy", "v9-eager", "v9-frame-graph")
_INCLUDE = "access_operator/demo/aiv_batch_get/include/"
_BASE_HEADERS = ("batch_get_abi.h", "kvaf_aiv_batch_get.h")
_FUSED_HEADERS = ("kvaf_batch_get_device.h", "kvaf_batch_get_device_policy.h")
_BINDING = "bindings/src/npu_python_runtime.cc"
_LEGACY_EXPORTS = (
    "link_to",
    "initialize",
    "shutdown",
    "batch_put",
    "prepare_access_table",
    "access_table_view",
    "release_access_table",
    "batch_free",
    "batch_get",
)
_EAGER_EXPORTS = ("acquire_fused_batch_get_context", "release_fused_batch_get_context")
_GRAPH_EXPORTS = (
    "fused_graph_abi",
    "register_fused_graph_output",
    "begin_fused_graph",
    "end_fused_graph",
)
_COMMENTS = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', re.DOTALL)
_EXPORT = re.compile(r'\.def(?:_static)?\s*\(\s*"([A-Za-z_][A-Za-z_0-9]*)"')


class KvafCompatibilityError(ValueError):
    """Fail-closed evidence failure; evidence is safe to serialize as JSON."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = dict(evidence or {}, status="fail", error=message)


def _requirements(profile):
    if profile not in PROFILES:
        raise KvafCompatibilityError(f"unsupported KVAF profile: {profile!r}", {"profile": profile})
    exports = _LEGACY_EXPORTS
    headers = _BASE_HEADERS
    if profile != "legacy":
        exports += _EAGER_EXPORTS
        headers += _FUSED_HEADERS
    if profile == "v9-frame-graph":
        exports += _GRAPH_EXPORTS
    return tuple(_INCLUDE + name for name in headers), exports


def _git(root, *arguments):
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise KvafCompatibilityError(f"cannot inspect KVAF Git checkout: {error}") from error
    return result.stdout.strip()


def source_evidence(root, expected_revision=None, profile="v9-eager"):
    """Inspect a real checkout (including submodule/worktree gitfiles).

    A passing result proves required files and literal pybind exports exist in
    this working tree. It does not load the built extension, execute C++, check
    the device ABI, validate SDK/driver compatibility, or certify hardware.
    Dirty trees are reported with current-file hashes; their HEAD alone does
    not identify those bytes. No fetch, checkout, build, or patch is performed.
    """
    headers, exports = _requirements(profile)
    evidence = dict(
        kind="source-only",
        profile=profile,
        expected_revision=expected_revision,
        runtime_validated=False,
        hardware_certified=False,
        required_headers=list(headers),
        required_exports=list(exports),
        limitations=[
            "Source evidence is not built-runtime, SDK, device ABI, or hardware certification.",
            "A consumer Graph bridge overlay needs its own patch/build provenance.",
        ],
    )
    if expected_revision is not None and (
        not isinstance(expected_revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", expected_revision) is None
    ):
        raise KvafCompatibilityError("expected_revision must be a complete 40-digit Git SHA", evidence)
    try:
        root = Path(root).expanduser().resolve(strict=True)
        evidence["root"] = str(root)
        if not root.is_dir():
            raise KvafCompatibilityError("KVAF root must be a directory", evidence)
        actual_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
        if actual_root != root:
            raise KvafCompatibilityError("KVAF root must be the actual Git checkout root", evidence)
        revision = _git(root, "rev-parse", "--verify", "HEAD")
        evidence["revision"] = revision
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise KvafCompatibilityError("KVAF HEAD is not a complete SHA-1 revision", evidence)
        if expected_revision is not None and revision != expected_revision.lower():
            raise KvafCompatibilityError("KVAF revision differs from the required pinned commit", evidence)
        evidence["dirty"] = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal"))
        evidence["tracked_changes"] = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=no"))
        evidence["source_tree_matches_revision"] = not evidence["tracked_changes"]
        required = ("CMakeLists.txt", "build.sh", _BINDING, *headers)
        missing_files = [name for name in required if not (root / name).is_file()]
        binding = "" if _BINDING in missing_files else (root / _BINDING).read_text(encoding="utf-8")
        # Ignore commented-out .def declarations; this remains a source surface
        # check, deliberately not a C++ parser or proof of a compiled binding.
        binding = _COMMENTS.sub(lambda match: match.group(1) or "", binding)
        declared = set(_EXPORT.findall(binding))
        missing_exports = [name for name in exports if name not in declared]
        evidence.update(
            missing_files=missing_files,
            missing_exports=missing_exports,
            graph_bridge_required=(profile == "v9-frame-graph" and bool(set(missing_exports) & set(_GRAPH_EXPORTS))),
            file_sha256={
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in required
                if name not in missing_files
            },
        )
        if missing_files or missing_exports:
            details = []
            if missing_files:
                details.append("missing files: " + ", ".join(missing_files))
            if missing_exports:
                details.append("missing pybind exports: " + ", ".join(missing_exports))
            raise KvafCompatibilityError("KVAF source profile unavailable; " + "; ".join(details), evidence)
    except KvafCompatibilityError as error:
        if error.evidence.get("kind") != "source-only":
            raise KvafCompatibilityError(str(error), evidence) from error
        raise
    except (OSError, TypeError, UnicodeError) as error:
        raise KvafCompatibilityError(f"cannot read KVAF source: {error}", evidence) from error
    return dict(evidence, status="pass", source_compatible=True)


def validate_runtime_surface(runtime_type, profile="v9-eager"):
    """Check an already-loaded NpuRuntime class/instance without acquiring it.

    Pass an instance for instance-bound fused_graph_abi. This helper never
    constructs a runtime or calls initialize/acquire/register/begin/end/close.
    """
    _, exports = _requirements(profile)
    evidence = dict(
        kind="runtime-surface-only",
        profile=profile,
        required_exports=list(exports),
        hardware_certified=False,
        sdk_compatibility_verified=False,
        runtime_validated=False,
    )
    try:
        missing = [name for name in exports if not callable(getattr(runtime_type, name, None))]
        evidence["missing_exports"] = missing
        if missing:
            raise KvafCompatibilityError("KVAF runtime is missing callable exports: " + ", ".join(missing), evidence)
        if profile == "v9-frame-graph":
            try:
                abi = runtime_type.fused_graph_abi()
            except Exception as error:
                raise KvafCompatibilityError(
                    "cannot query KVAF Graph ABI; pass an existing instance for an instance-bound method: "
                    + str(error),
                    evidence,
                ) from error
            if type(abi) is not int or abi != 1:
                raise KvafCompatibilityError("KVAF fused Graph ABI must be integer 1", evidence)
            evidence["graph_transport_abi"] = abi
    except KvafCompatibilityError:
        raise
    except Exception as error:
        raise KvafCompatibilityError(f"cannot inspect KVAF runtime surface: {error}", evidence) from error
    return dict(evidence, status="pass", surface_compatible=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", default="v9-eager")
    parser.add_argument("--expected-revision")
    args = parser.parse_args(argv)
    try:
        evidence = source_evidence(args.root, args.expected_revision, args.profile)
    except KvafCompatibilityError as error:
        print(json.dumps(error.evidence, indent=2, sort_keys=True))
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
