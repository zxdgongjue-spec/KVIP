"""Print the public contract without loading Torch, NPU, or the KVAF runtime."""

import json
from dataclasses import asdict

from .access import FIELD_ACCESS
from .api import query_capabilities
from .contract import GEOMETRY_FIELDS, POINTER_FIELDS, TENSOR_FIELDS


def describe_contract():
    from . import __all__

    info = asdict(query_capabilities())
    info["features"] = sorted(info["features"])
    info["field_access"] = {
        name: FIELD_ACCESS.get(name, "read") for name in TENSOR_FIELDS
    }
    info["device_pointer_fields"] = list(POINTER_FIELDS)
    info["device_geometry_fields"] = list(GEOMETRY_FIELDS)
    info["index_cache_pointees"] = "read_write; full ranges checked during preparation"
    info["public_symbols"] = sorted(__all__)
    info["owner_protocol"] = {
        "lease_active": "bool; true only after successful acquisition",
        "lease_generation": "positive host integer; increments on reacquisition",
        "lease_stream": "explicit acquired stream",
        "closed": "bool",
        "session_binding": "one active PreparedSession or None",
        "session_lock": "per-owner reentrant context-manager lock",
    }
    info["page_refresh"] = (
        "append-only host page_ids; replacement requires checked rebind"
    )
    return info


if __name__ == "__main__":
    print(json.dumps(describe_contract(), indent=2, sort_keys=True))
