"""Shared-library loading for LIFetch and its Index update dependency."""

import ctypes
from pathlib import Path


def load_lifetch_library(path):
    """Load sibling dependency first; independent of CANN's RPATH policy."""
    path = Path(path).resolve(strict=True)
    dependency = path.with_name("libkvop_cache_update.so").resolve(strict=True)
    update = ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
    bridge = ctypes.CDLL(str(path))
    bridge._kvip_update_dependency = update
    return bridge
