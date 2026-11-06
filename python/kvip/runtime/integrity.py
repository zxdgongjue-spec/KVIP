import hashlib
import json
from pathlib import Path


def verify_sources():
    """Audit the historical migration snapshot when explicitly requested.

    Development builds and tests validate the current sources through CMake,
    ABI checks and functional tests. They must accept intentional kernel edits.
    """
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "scripts/migration_sources.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source integrity mismatch: {name}")
    return len(manifest["files"])
