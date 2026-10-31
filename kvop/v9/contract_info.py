"""Print the public contract without loading Torch, NPU, or the KVAF runtime."""

import json
from dataclasses import asdict

from .api import query_capabilities
from .contract import GEOMETRY_FIELDS, POINTER_FIELDS, TENSOR_FIELDS
from .access import FIELD_ACCESS


def describe_contract():
    info = asdict(query_capabilities())
    info["features"] = sorted(info["features"])
    info["field_access"] = {
        name: FIELD_ACCESS.get(name, "read") for name in TENSOR_FIELDS
    }
    info["device_pointer_fields"] = list(POINTER_FIELDS)
    info["device_geometry_fields"] = list(GEOMETRY_FIELDS)
    info["index_cache_pointees"] = "read_write; full ranges checked during preparation"
    return info


if __name__ == "__main__":
    print(json.dumps(describe_contract(), indent=2, sort_keys=True))
