"""Query the active logical NPU through the installed ACL runtime."""

import ctypes
from pathlib import Path


def query_device_resources(cann, device, requested=0):
    # Call after torch.npu.set_device, which initializes ACL. No guessed 32-core fallback.
    candidates = [Path(cann) / "lib64/libascendcl.so", Path(cann) / "aarch64-linux/lib64/libascendcl.so"]
    runtime = ctypes.CDLL(str(next((p for p in candidates if p.exists()), "libascendcl.so")))
    query = runtime.aclrtGetDeviceInfo
    query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    query.restype = ctypes.c_int
    values = {}
    # CANN 9.1 acl_rt.h: CUBE_CORE_NUM=102, VECTOR_CORE_NUM=201.
    for name, attribute in (("aic", 102), ("aiv", 201)):
        value = ctypes.c_int64()
        error = query(device, attribute, ctypes.byref(value))
        if error or value.value <= 0:
            raise RuntimeError(f"aclrtGetDeviceInfo({name}) failed: {error}, value={value.value}")
        values[name] = value.value
    available = min(values["aic"], values["aiv"] // 2, 32)
    if requested < 0 or requested > available:
        raise ValueError(f"Requested AIC budget {requested} exceeds available {available}")
    values["effective_aic"] = requested or available
    values["source"] = "aclrtGetDeviceInfo_on_logical_device"
    return values
