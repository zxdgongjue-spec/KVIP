"""Real KVAF preparation and ownership for the V8 staged AIV1 benchmark.

Everything here is setup/verification, never part of an operator timing scope.
The proven V5 service session is reused ONLY for transport setup, not its
operator, plan, profiler or benchmark. No synthetic Fetch fallback exists.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

MAGIC = 0x514C495638535447
VERSION = 1
TOKENS = 131075
PADDED_TOKENS = 131200
TOPK = 2048
TOKEN_BYTES = 656
BLOCK_TOKENS = 128
CONTEXT_WORDS = 64
DEFAULT_SERVER = "127.0.0.1:28999"


def unsigned_address_view(addresses, torch):
    """Reinterpret KVAF's address bits for ACL_UINT64; never cast or copy."""
    if addresses.dtype not in (torch.int64, torch.uint64) or not addresses.is_contiguous():
        raise ValueError(f"kvaf_address_storage_invalid:{addresses.dtype}")
    result = addresses if addresses.dtype == torch.uint64 else addresses.view(torch.uint64)
    if result.data_ptr() != addresses.data_ptr() or result.shape != addresses.shape:
        raise ValueError("kvaf_address_view_must_alias_original_storage")
    return result


def discover(
    *,
    server: str | None = None,
    kvaf_python_dir: Path | None = None,
    get_kernel: Path | None = None,
    put_kernel: Path | None = None,
    kvaf_root: Path | None = None,
) -> dict[str, Any]:
    """Read existing build artifacts/config; never build or contact a service."""
    candidates = []
    if kvaf_root is not None:
        candidates = [Path(kvaf_root) / "build-npu", Path(kvaf_root) / "build"]
    values = {
        "kvaf_python_dir": kvaf_python_dir,
        "get_kernel": get_kernel,
        "put_kernel": put_kernel,
    }
    suffixes = {
        "kvaf_python_dir": "python",
        "get_kernel": "bin/kvaf_execute_batch_get_kernels.o",
        "put_kernel": "bin/kvaf_execute_batch_put_kernels.o",
    }

    def ready(name: str, path: Path) -> bool:
        if name != "kvaf_python_dir":
            return path.is_file()
        package = path / "kvaf"
        return (package / "__init__.py").is_file() and any(package.glob("_kvaf*.so"))

    bundle = next(
        (
            base
            for base in candidates
            if all(values[name] is not None or ready(name, base / suffix) for name, suffix in suffixes.items())
        ),
        None,
    )
    for name, suffix in suffixes.items():
        if values[name] is None:
            values[name] = bundle / suffix if bundle is not None else None
        if values[name] is None or not ready(name, Path(values[name])):
            raise ValueError(
                f"v8_fetch_dependency_missing:{name}: expected a matching KVAF "
                "build-npu bundle; pass --kvaf-root to its source checkout"
            )
        values[name] = Path(values[name]).resolve()
    values["server"] = server or DEFAULT_SERVER
    return values


def quarantine_all(resources: Any, reason: str, error: BaseException | str) -> None:
    """Do not unwind Python/NPU destructors after any uncertain stage3 launch."""
    # Keep the argument referenced until process teardown; os._exit is the
    # existing KVAF safety boundary, not an exception that finally may release.
    print(
        f"[QLI_V8_NATIVE] status=failed stage=fetch_quarantine reason={reason} "
        f"detail={str(error)[:240]!r} resources_retained={resources is not None}",
        flush=True,
    )
    os._exit(70)


def _session_class() -> Any:
    path = Path(__file__).resolve().parent / "native_batchget_support.py"
    spec = importlib.util.spec_from_file_location("qli_v8_verified_service_setup", path)
    if spec is None or spec.loader is None:
        raise ValueError("v8_verified_service_setup_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Session(module.NativeFetchSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # 128 blocks = 16384 KV tokens, ~10.25 MiB per PUT. All setup,
            # including completion fences, is outside benchmark timing.
            kwargs["put_chunk_blocks"] = 128
            super().__init__(*args, **kwargs)

        def quarantine(self, reason: str, error: BaseException) -> None:
            quarantine_all(self, reason, error)

    return Session
