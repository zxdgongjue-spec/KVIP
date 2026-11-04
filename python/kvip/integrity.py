import hashlib
import json
from pathlib import Path


def verify_sources():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "scripts/migration_sources.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source integrity mismatch: {name}")
    return len(manifest["files"])
