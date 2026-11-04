"""Shared-library loading for LIFetch and its Index update dependency."""

import ctypes
import importlib
from pathlib import Path


def load_lifetch_library(path):
    """Load sibling dependency first; independent of CANN's RPATH policy."""
    path = Path(path).resolve(strict=True)
    dependency = path.with_name("libkvop_cache_update.so").resolve(strict=True)
    update = ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
    bridge = ctypes.CDLL(str(path))
    bridge._kvip_update_dependency = update
    return bridge


def load_native_get(runtime):
    """Resolve the launcher from this Runtime's loaded SDK, not another build.

    The SDK shared object's dependency scope contains the exact host launcher
    whose kernel handle was initialized by NpuRuntime.initialize().
    """
    module = importlib.import_module(type(runtime).__module__)
    library = ctypes.CDLL(module.__file__)
    launch = library.KvafExecuteBatchGetLaunch
    launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    launch.restype = ctypes.c_int32
    return library, launch
