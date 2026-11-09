"""Lazy legacy kvip import aliases. No kvop namespace and no duplicate implementation."""

from importlib import import_module
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import sys

ALIASES = (
    # The vLLM UBmem integration still imports these pre-merge module paths.
    ("adapters.ubmem", "adapters.kvaf"),
    ("adapters.vllm.ubmem_binding", "adapters.vllm.binding"),
    ("fused_inputs", "adapters.vllm.inputs"),
    ("fused_selector", "adapters.vllm.selector"),
    ("fused_service", "adapters.vllm.binding"),
    ("fused_workspace", "operators.lifetch.workspace"),
    ("index_storage", "adapters.vllm.index_import"),
    ("fused_build", "runtime.build"),
    ("integrity", "runtime.integrity"),
    ("pipeline", "operators.lifetch.forward"),
    ("index_update", "operators.lifetch.index_update"),
    ("transport", "adapters.kvaf"),
    ("lru", "cache.lru"),
    ("lru_core", "cache._lru_kernel"),
    ("abc_update", "cache.abc"),
    ("abc_update_kernel", "cache._abc_kernel"),
    ("_lifetch.launch", "operators.lifetch.binding"),
    ("_lifetch.geometry", "operators.lifetch.geometry"),
    ("_lifetch.runtime", "operators.lifetch.tiling"),
    ("_lifetch.device_resources", "runtime.device"),
    ("_lifetch", "operators.lifetch"),
)


class LegacyImports(MetaPathFinder, Loader):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in ALIASES:
            if fullname == "kvip." + old:
                return spec_from_loader(fullname, self, is_package=old == "_lifetch")
        return None

    def create_module(self, spec):
        target = dict(ALIASES)[spec.name.removeprefix("kvip.")]
        module = import_module("kvip." + target)
        for old, new in (
            ("FusedQueryInputs", "QueryInputs"),
            ("FusedWorkspace", "LIFetchWorkspace"),
            ("FusedIndexSelector", "VllmIndexSelector"),
            ("RotatingLaunch", "LIFetchBinding"),
            ("FusedForward", "LIFetchForward"),
            ("FusedGetLease", "KvafGetLease"),
            ("FusedQueueView", "KvafQueueView"),
        ):
            if hasattr(module, new):
                setattr(module, old, getattr(module, new))
        return module

    def exec_module(self, module):
        pass


def install_legacy_imports():
    if not any(isinstance(finder, LegacyImports) for finder in sys.meta_path):
        sys.meta_path.append(LegacyImports())
