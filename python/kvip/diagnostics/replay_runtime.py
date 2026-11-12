# SPDX-License-Identifier: Apache-2.0
"""Select a native Runtime bundle without allocating DDR or connecting a peer."""

from pathlib import Path

from kvip.operators.lifetch.geometry import final_aivs


def validate_urma_serial_max_inflight(value, workers):
    if value is not None and (type(value) is not int or not 0 <= value <= 0xFFFFFFFF):
        raise ValueError("URMA serial max inflight must be a nonnegative uint32")
    if value is not None and value != 0 and value < 2 * workers:
        raise ValueError("Nonzero URMA serial max inflight must be at least twice the worker count")


def runtime_worker_config(args, archive, aics):
    """Report explicit CLI overrides separately from serving/backend defaults."""
    if args.transport not in ("ubmem", "urma") or args.mode not in ("serial", "fused"):
        raise ValueError("Replay requires ubmem/urma transport and serial/fused mode")
    override = getattr(args, "serial_workers", None)
    if override is not None and (type(override) is not int or not 1 <= override <= 48):
        raise ValueError("serial_workers must be an integer in 1..48")
    inflight = getattr(args, "urma_serial_max_inflight", None)
    dispatch = getattr(args, "ubmem_serial_dispatch", "scalar")
    if dispatch not in ("scalar", "tiled"):
        raise ValueError("UBmem serial dispatch must be scalar or tiled")
    if dispatch != "scalar" and (args.transport, args.mode) != ("ubmem", "serial"):
        raise ValueError("UBmem tiled dispatch applies only to UBmem serial replay")
    post_size = getattr(args, "urma_serial_post_size", 128)
    if type(post_size) is not int or post_size not in (32, 128):
        raise ValueError("URMA serial post size must be 32 or 128")
    if post_size != 128 and (args.transport, args.mode) != ("urma", "serial"):
        raise ValueError("URMA serial post size applies only to URMA serial replay")
    urma_dispatch = getattr(args, "urma_serial_dispatch", "scalar")
    if urma_dispatch not in ("scalar", "bulk"):
        raise ValueError("URMA serial dispatch must be scalar or bulk")
    if urma_dispatch != "scalar" and (args.transport, args.mode) != ("urma", "serial"):
        raise ValueError("URMA bulk dispatch applies only to URMA serial replay")
    if urma_dispatch == "bulk" and (post_size != 32 or inflight != 0):
        raise ValueError("URMA bulk dispatch requires post size 32 and max inflight 0")
    if inflight is not None and (args.transport, args.mode) != ("urma", "serial"):
        raise ValueError("URMA serial max inflight applies only to URMA serial replay")
    if args.mode == "serial":
        config = {
            "runtime_workers": override if override is not None else (32 if args.transport == "ubmem" else 2),
            "runtime_workers_source": "cli" if override is not None else "service_default",
        }
        if args.transport == "urma":
            validate_urma_serial_max_inflight(inflight, config["runtime_workers"])
            config.update(
                batch_get_max_inflight_total=2048 if inflight is None else inflight,
                batch_get_max_inflight_total_source="native_default" if inflight is None else "cli",
                jettys_per_core=2,
                batch_get_post_size=post_size,
                urma_batch_get_dispatch=urma_dispatch,
            )
        else:
            config["ubmem_batch_get_dispatch"] = dispatch
        return config
    if override is not None:
        raise ValueError("serial_workers applies only to serial replay")
    if args.transport == "ubmem":
        workers = 2
    else:
        geometry = archive.manifest["geometry"]
        batch, captured_aics = geometry["batch"], geometry["aic_count"]
        if aics < captured_aics:
            raise ValueError("Replay device has fewer AICs than the captured fusion")
        final = final_aivs(batch, "auto", captured_aics)
        workers = (batch * (final + 1) + 1) // 2
    return {"runtime_workers": workers, "runtime_workers_source": "fused_backend_geometry"}


def get_kernel_path(args):
    if args.transport == "ubmem":
        name = (
            "kvaf_ubmem_batch_get_tiled.o"
            if getattr(args, "ubmem_serial_dispatch", "scalar") == "tiled"
            else "kvaf_ubmem_batch_get.o"
        )
    elif args.mode == "serial" and getattr(args, "urma_serial_dispatch", "scalar") == "bulk":
        name = "kvaf_execute_batch_get_bulk.o"
    elif args.mode == "serial" and getattr(args, "urma_serial_post_size", 128) == 32:
        name = "kvaf_execute_batch_get_post32.o"
    else:
        name = "kvaf_execute_batch_get_kernels.o"
    return Path(args.build_dir).resolve() / "bin" / name


def create_runtime(args, archive, kvaf, torch, device, stream, cann, aics):
    """Construct only; caller retains the result before link_to/initialize.

    An URMA Runtime uses the original constructor ABI, with no transport
    keyword or UBmem options. The caller owns all device/stream setup and
    must select a separate process for each native extension bundle.
    """
    config = runtime_worker_config(args, archive, aics)
    geometry, ddr = archive.manifest["geometry"], archive.manifest["ddr"]
    if geometry["token_bytes"] != ddr["token_bytes"]:
        raise ValueError("Snapshot operator and DDR token geometry differ")
    if args.transport == "urma" and not getattr(args, "kvaf_server", None):
        raise ValueError("URMA replay requires --kvaf-server")
    build = Path(args.build_dir).resolve()
    pattern = "kvaf_ubmem_batch_{}.o" if args.transport == "ubmem" else "kvaf_execute_batch_{}_kernels.o"
    kernels = {kind: build / "bin" / pattern.format(kind) for kind in ("get", "put")}
    kernels["get"] = get_kernel_path(args)
    for path in kernels.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {args.transport} Runtime kernel: {path}")
    options = dict(
        device_id=args.device,
        npu_core_count=config["runtime_workers"],
        batch_put_kernel_object_path=str(kernels["put"]),
    )
    if args.transport == "ubmem":
        options.update(transport="ubmem", ubmem_page_mode=ddr["page_mode"], ubmem_numa_node=ddr["numa_node"])
    else:
        options["jettys_per_core"] = 2
        if args.mode == "serial" and getattr(args, "urma_serial_max_inflight", None) is not None:
            options["batch_get_max_inflight_total"] = config["batch_get_max_inflight_total"]
    return kvaf.NpuRuntime(str(kernels["get"]), ddr["token_bytes"], ddr["tokens_per_block"], **options)
