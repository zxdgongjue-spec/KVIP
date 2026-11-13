# SPDX-License-Identifier: Apache-2.0
"""Isolate fixed upstream production code from the feature's diagnostic readers.

Activation happens once in a fresh replay process, before any production KVIP
module is imported. The diagnostic package remains feature-owned; every other
KVIP module, including the package initializer and native builders, is upstream.
"""

import importlib.util
import hashlib
from pathlib import Path
import re
import subprocess
import sys


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True, stderr=subprocess.PIPE).strip()


def validate_checkout(root, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Upstream revision must be an explicit full 40-character commit SHA")
    root = Path(root).resolve()
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError(f"Upstream reference must be exactly {revision}: {root}")
    if _git(root, "status", "--porcelain", "--untracked-files=all", "--", "python/kvip", "csrc"):
        raise ValueError(f"Upstream production sources are modified or untracked: {root}")
    for relative in ("python/kvip/__init__.py", "python/kvip/runtime/build.py", "csrc/operators"):
        if not (root / relative).exists():
            raise ValueError(f"Missing upstream production source: {root / relative}")
    return root


def verify_lidu_sources(root, feature_root):
    """An existing extension can be reused only with identical tracked sources."""
    signatures = []
    for checkout in (root, feature_root):
        files = _git(checkout, "ls-files", "--", "csrc/operators/lidu").splitlines()
        if not files:
            raise ValueError(f"No tracked LIDU sources: {checkout}")
        digest = hashlib.sha256()
        for relative in files:
            digest.update(relative.encode() + b"\0")
            digest.update((Path(checkout) / relative).read_bytes())
        signatures.append(digest.hexdigest())
    if signatures[0] != signatures[1]:
        raise ValueError("Existing LIDU extension cannot be reused: feature and upstream LIDU sources differ")
    return dict(
        sources_identical=True,
        source_sha256=signatures[0],
        source_roots=[str(Path(root).resolve()), str(Path(feature_root).resolve())],
        artifact_policy="Reuse the installed extension; its loaded paths/hashes are recorded, not rebuilt upstream",
    )


def _production(name):
    return name == "kvip" or (
        name.startswith("kvip.") and name != "kvip.diagnostics" and not name.startswith("kvip.diagnostics.")
    )


def activate(root, revision):
    """Load the actual upstream package, with no production feature fallback."""
    root = validate_checkout(root, revision)
    already_loaded = [name for name in sys.modules if _production(name) and name not in ("kvip", "kvip._compat")]
    if already_loaded:
        raise RuntimeError(f"Upstream must be selected before production imports: {already_loaded}")
    diagnostics = sys.modules.get("kvip.diagnostics")
    if diagnostics is None:
        raise RuntimeError("The feature diagnostic package must be loaded before upstream activation")
    # The original initializer installs a lazy alias finder. Replace it too,
    # rather than retaining feature-owned import behavior in the reference.
    sys.meta_path[:] = [finder for finder in sys.meta_path if type(finder).__module__ != "kvip._compat"]
    sys.modules.pop("kvip._compat", None)
    package = root / "python/kvip"
    spec = importlib.util.spec_from_file_location(
        "kvip", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kvip"] = module
    spec.loader.exec_module(module)
    module.diagnostics = diagnostics
    return provenance(root, revision)


def provenance(root, revision):
    """Verify actual loaded module origins; native source roots follow builders."""
    root = Path(root).resolve()
    package = root / "python/kvip"
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if not _production(name):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            raise RuntimeError(f"Production module has no verifiable source: {name}")
        path = Path(source).resolve()
        if not path.is_relative_to(package):
            raise RuntimeError(f"Mixed production imports: {name} came from {path}, expected {package}")
        modules[name] = str(path)
    return dict(
        kind="exact_upstream_checkout",
        root=str(root),
        commit=revision,
        python_modules=modules,
        native_source=str(root / "csrc/operators"),
        diagnostics_source="Feature-branch readers/restoration/oracles/timer; production modules are upstream",
        kvaf_runtime="Shared explicitly supplied current build; separately allocated process arena",
    )


def loaded_lidu_artifacts():
    """Record the extension actually imported and its loaded vendor dependencies."""
    paths = set()
    for name, module in tuple(sys.modules.items()):
        if name.startswith("nanovllm_dsa_lidu_fused_batch_get_prepare_mtp") and getattr(module, "__file__", None):
            paths.add(Path(module.__file__).resolve())
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            value = line.split(maxsplit=5)
            if len(value) == 6 and value[5].startswith("/"):
                path = Path(value[5])
                if path.name == "libcust_opapi.so" or "lidu" in path.name.lower():
                    paths.add(path.resolve())
    return [
        dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(paths)
        if path.is_file()
    ]
