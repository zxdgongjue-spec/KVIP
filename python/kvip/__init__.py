# SPDX-License-Identifier: Apache-2.0
"""Public Python API for KVIP operators and cache state; imports are lazy."""

from importlib import import_module

__all__ = [
    "FusedQueryInputs",
    "FusedWorkspace",
    "IndexCacheStorage",
    "LIFetch",
    "SlotLRU",
    "bind_selectors",
    "build_libraries",
    "lidu",
    "update_abc",
]
_EXPORTS = {
    "FusedQueryInputs": ("fused_inputs", "FusedQueryInputs"),
    "FusedWorkspace": ("fused_workspace", "FusedWorkspace"),
    "IndexCacheStorage": ("index_storage", "IndexCacheStorage"),
    "LIFetch": ("pipeline", "FusedForward"),
    "SlotLRU": ("lru", "SlotLRU"),
    "update_abc": ("abc_update", "update_abc"),
    "bind_selectors": ("fused_service", "bind_selectors"),
    "build_libraries": ("fused_build", "build_libraries"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    return getattr(import_module(f"{__name__}.{module}"), symbol)


def lidu(*args, **kwargs):
    """Call the original out-parameter LIDU ABI on Torch's current NPU stream."""
    import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp

    return nanovllm_dsa_lidu_fused_batch_get_prepare_mtp.lidu_fused_batch_get_prepare_mtp_out(*args, **kwargs)
