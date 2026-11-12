# SPDX-License-Identifier: Apache-2.0
"""Use the current dual-transport KVAF ABI; never silently change a baseline."""

from pathlib import Path

from kvip.operators.lifetch.geometry import final_aivs


def runtime_worker_config(args, archive, aics):
    if args.transport not in ("ubmem", "urma") or args.mode not in ("serial", "fused"):
        raise ValueError("Replay requires ubmem/urma and serial/fused")
    if args.mode == "serial":
        workers = args.serial_workers if args.serial_workers is not None else 32
        if type(workers) is not int or not 1 <= workers <= 48:
            raise ValueError("serial-workers must be in 1..48")
        result = dict(runtime_workers=workers, runtime_workers_source="explicit replay serial baseline")
        if args.transport == "urma":
            inflight = 2048
            result.update(batch_get_max_inflight_total=inflight, batch_get_post_size=128, jettys_per_core=2)
        return result
    if args.serial_workers is not None:
        raise ValueError("serial-workers applies only to serial")
    g = archive.manifest["geometry"]
    if aics < g["aic_count"]:
        raise ValueError("Replay NPU has fewer AICs than captured geometry")
    workers = (
        2
        if args.transport == "ubmem"
        else (max(g["batch"] * (final_aivs(g["batch"], "auto", aics) + 1), 2 * g["batch"] + 16) + 1) // 2
    )
    return dict(runtime_workers=workers, runtime_workers_source="fused queue geometry")


def get_kernel_path(args):
    return Path(args.build_dir).resolve() / "bin/kvaf_execute_batch_get_kernels.o"


def create_runtime(args, archive, kvaf, torch, device, stream, cann, aics):
    config = runtime_worker_config(args, archive, aics)
    geometry, ddr = archive.manifest["geometry"], archive.manifest["ddr"]
    if geometry["token_bytes"] != ddr["token_bytes"]:
        raise ValueError("Snapshot operator and DDR token geometry differ")
    get = get_kernel_path(args)
    put = Path(args.build_dir).resolve() / "bin/kvaf_execute_batch_put_kernels.o"
    for path in (get, put):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {args.transport} Runtime kernel: {path}")
    options = dict(
        device_id=args.device,
        npu_core_count=config["runtime_workers"],
        batch_put_kernel_object_path=str(put),
        jettys_per_core=2,
    )
    if args.transport == "ubmem":
        options["transport"] = "ubmem"
    if args.mode == "serial" and args.transport == "urma":
        options["batch_get_max_inflight_total"] = config["batch_get_max_inflight_total"]
    return kvaf.NpuRuntime(str(get), ddr["token_bytes"], ddr["tokens_per_block"], **options)
