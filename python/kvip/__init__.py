# SPDX-License-Identifier: Apache-2.0
"""Public Python API for KVIP operators and cache state; imports are lazy."""

from importlib import import_module
from ._compat import install_legacy_imports

__all__ = [
    "QueryInputs",
    "LIFetchWorkspace",
    "IndexCacheStorage",
    "LIFetch",
    "SlotLRU",
    "bind_selectors",
    "build_libraries",
    "lidu",
    "update_abc",
]
_EXPORTS = {
    "lidu": ("operators.lidu", "lidu"),
    "QueryInputs": ("adapters.vllm.inputs", "QueryInputs"),
    "LIFetchWorkspace": ("operators.lifetch.workspace", "LIFetchWorkspace"),
    "IndexCacheStorage": ("adapters.vllm.index_import", "VllmIndexCacheStorage"),
    "LIFetch": ("operators.lifetch.forward", "LIFetchForward"),
    "SlotLRU": ("cache.lru", "SlotLRU"),
    "update_abc": ("cache.abc", "update_abc"),
    "bind_selectors": ("adapters.vllm.binding", "bind_selectors"),
    "build_libraries": ("runtime.build", "build_libraries"),
}

_EXPORTS.update(
    {
        "FusedQueryInputs": ("adapters.vllm.inputs", "QueryInputs"),
        "FusedWorkspace": ("operators.lifetch.workspace", "LIFetchWorkspace"),
    }
)
install_legacy_imports()


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    return getattr(import_module(f"{__name__}.{module}"), symbol)
