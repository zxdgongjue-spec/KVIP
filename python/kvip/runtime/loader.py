"""Shared-library loading for LIFetch and native KVAF."""

import ctypes
import importlib
from pathlib import Path


def load_lifetch_library(path):
    """Load the self-contained LIFetch bridge."""
    path = Path(path).resolve(strict=True)
    return ctypes.CDLL(str(path))


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
