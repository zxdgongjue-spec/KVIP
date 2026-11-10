# SPDX-License-Identifier: Apache-2.0
"""Bind one frozen service input to an unchanged production fusion backend.

Stage callables enqueue the same native launches in the same order as forward.
Events or separate stage graphs perturb launch/overlap timing: use them only for
diagnosis, and retain the uninstrumented production graph as the total baseline.
"""

import ctypes
from pathlib import Path
from types import SimpleNamespace

from kvip.cache.index_storage import IndexCacheStorage
from kvip.operators.lifetch.geometry import final_aivs, li_geometry, prefetch_budget


def _stage(library, symbol, argtypes, arguments, lease):
    function = getattr(library, symbol)
    function.argtypes, function.restype = argtypes, ctypes.c_uint32

    def launch():
        stream = lease.check_stream()
        result = function(*arguments(stream.npu_stream))
        if result:
            raise RuntimeError(f"Fusion diagnostic stage {symbol} failed: {result}")

    return launch


def fused_stages(transport, rotating, update, lease, batch, *, split_index_stages=False):
    """Actual backend boundaries; URMA Index work overlaps outstanding Get DMA."""
    pointer, u32 = ctypes.c_void_p, ctypes.c_uint32
    library = rotating.library
    if transport == "ubmem":
        result = [
            (
                name,
                _stage(
                    library, symbol, [pointer, pointer], lambda stream: (rotating.bindings.data_ptr(), stream), lease
                ),
            )
            for name, symbol in (
                ("init", "v9_init"),
                ("li_with_prefetch", "v9_li"),
                ("resolve_final_fetch", "v9_resolve"),
                ("complete", "v9_fetch"),
            )
        ]
        updates = (
            (
                ("index_plan", "kvip_ubmem_update_plan"),
                ("index_outputs", "kvip_ubmem_update_outputs"),
                ("index_apply", "kvip_ubmem_update_apply"),
            )
            if split_index_stages
            else (("index_update", "kvip_ubmem_update"),)
        )
        result.extend(
            (
                name,
                _stage(
                    library,
                    symbol,
                    [pointer, pointer],
                    lambda stream: (update.bindings.data_ptr(), stream),
                    lease,
                ),
            )
            for name, symbol in updates
        )
        return tuple(result)
    if transport != "urma":
        raise ValueError("Fusion replay transport must be ubmem or urma")
    result = [
        (
            name,
            _stage(
                library,
                symbol,
                [pointer, pointer, pointer],
                lambda stream: (rotating.bindings.data_ptr(), stream, rotating.shape_pointer),
                lease,
            ),
        )
        for name, symbol in (("init", "v9_init"), ("li_with_prefetch", "v9_li"), ("resolve_final_submit", "v9_resolve"))
    ]
    result.extend(
        (
            (
                "index_update_while_fetch",
                _stage(
                    update.library,
                    "kvop_direct_update_outputs_early",
                    [u32, pointer, pointer],
                    lambda stream: (batch, update.bindings.data_ptr(), stream),
                    lease,
                ),
            ),
            (
                "fetch_complete",
                _stage(
                    library,
                    "aclrtlaunch_v9_fetch_update_kernel",
                    [u32, pointer, pointer, pointer],
                    lambda stream: (batch, stream, rotating.bindings.data_ptr(), update.bindings.data_ptr()),
                    lease,
                ),
            ),
        )
    )
    return tuple(result)


def _check_geometry(geometry, aics):
    if aics < geometry["aic_count"]:
        raise ValueError("Replay device has fewer AICs than the captured fusion")
    shape = li_geometry(geometry["batch"], "auto", geometry["requested_windows"], geometry["aic_count"])
    expected = dict(
        rounds=shape["intermediate_fetches"],
        service_lanes=shape["service_aivs_per_window"],
        final_workers=final_aivs(geometry["batch"], "auto", geometry["aic_count"]),
    )
    if any(geometry[name] != value for name, value in expected.items()):
        raise ValueError("Captured fusion scheduling differs from this replay implementation")


def fusion_tuning(geometry, transport, overrides=None):
    """Resolve opt-in UBmem experiments without mutating the captured geometry."""
    values = dict(
        parallel_init=False,
        balanced_index=False,
        prefetch_limit=geometry["prefetch_limit"],
        window_limit=geometry["window_limit"],
    )
    if overrides:
        if transport != "ubmem" or set(overrides) - set(values):
            raise ValueError("Fusion tuning only accepts known UBmem options")
        values.update(overrides)
    if any(type(values[name]) is not bool for name in ("parallel_init", "balanced_index")):
        raise ValueError("UBmem optimization switches must be booleans")
    if any(type(values[name]) is not int for name in ("prefetch_limit", "window_limit")):
        raise ValueError("UBmem prefetch budgets must be integers")
    prefetch_budget(values["prefetch_limit"], values["window_limit"], geometry["requested_windows"])
    return values


def create_fused(
    torch,
    runtime,
    archive,
    restored,
    device_table,
    build,
    soc,
    device,
    stream,
    cann,
    aics,
    *,
    transport,
    tuning=None,
    split_index_stages=False,
):
    """Build/bind before capture; the caller restores every archived pre tensor.

    Runtime allocation, page import, seed copies and completion/oracle checks
    remain owned by the replay controller. URMA callers must provision enough
    KVAF workers for B*(final_workers+1) exclusive queues, two per worker.
    """
    if transport not in ("ubmem", "urma") or getattr(runtime, "transport", transport) != transport:
        raise ValueError("Fusion replay transport differs from its Runtime")
    if restored.get("access_table") is None:
        raise ValueError("Fusion replay requires the imported published DDR pages")
    geometry = archive.manifest["geometry"]
    _check_geometry(geometry, aics)
    effective = fusion_tuning(geometry, transport, tuning)
    build, cann = Path(build), Path(cann)
    b, h, r = (geometry[name] for name in ("batch", "history", "resident"))
    common_workspace = dict(
        batch=b,
        history=h,
        aics=geometry["aic_count"],
        windows=geometry["requested_windows"],
        device=device,
        index_heads=geometry["heads"],
    )
    if transport == "ubmem":
        from kvip.adapters.ubmem import UbmemLease
        from kvip.operators.lifetch_ubmem.binding import UbmemBinding
        from kvip.operators.lifetch_ubmem.forward import UbmemForward
        from kvip.operators.lifetch_ubmem.index_update import UbmemIndexUpdate
        from kvip.operators.lifetch_ubmem.specialization import Specialization, build_specialization
        from kvip.operators.lifetch_ubmem.workspace import UbmemWorkspace

        shape = Specialization(
            b,
            h,
            r,
            geometry["heads"],
            geometry["token_bytes"],
            geometry["aic_count"],
            geometry["requested_windows"],
            soc,
            parallel_init=effective["parallel_init"],
            balanced_index=effective["balanced_index"],
        )
        library = build_specialization(cache_dir=build / "lifetch_ubmem", cann=cann, shape=shape)
        workspace = UbmemWorkspace(torch, resident=r, **common_workspace)
    else:
        from kvip.adapters.kvaf import KvafGetLease, KvafQueueView
        from kvip.operators.lifetch.binding import LIFetchBinding
        from kvip.operators.lifetch.forward import LIFetchForward
        from kvip.operators.lifetch.index_update import DirectIndexUpdate
        from kvip.operators.lifetch.workspace import LIFetchWorkspace
        from kvip.runtime.build import build_libraries, kvaf_build_paths

        runtime_cann, headers = kvaf_build_paths(build)
        if runtime_cann.resolve() != cann.resolve():
            raise ValueError("URMA Runtime and fusion must use the same CANN build")
        # Reuse install_kvip.sh's shape-independent build cache. Its hash also
        # includes the selected KVAF headers/toolchain, so bundles stay isolated.
        cache = Path(__file__).resolve().parents[3] / "build/lifetch"
        library, update_library = build_libraries(cache_dir=cache, cann=cann, headers=headers, soc=soc)
        workspace = LIFetchWorkspace(torch, **common_workspace)
    storage = IndexCacheStorage(torch, geometry["owners"], h, r, device)
    output = archive.tensor("pre.output", torch, device)
    common = dict(
        query_ends=workspace.inputs.query_ends.view(b, 4),
        hot_valid=storage.hot_valid,
        row_owners=workspace.inputs.row_owners,
        token_addresses=storage.addresses,
        tail_bounds=workspace.tail_bounds,
        output_stride=geometry["output_stride"],
        prefetch_limit=effective["prefetch_limit"],
        window_limit=effective["window_limit"],
    )
    if transport == "ubmem":
        lease = UbmemLease(torch, runtime, output, stream)
        lease.register_table(device_table)
        rotating = UbmemBinding(
            torch,
            library,
            workspace.arguments(storage),
            lease,
            output,
            shape=shape,
            scratch=workspace.scratch,
            **common,
        )
        update_class, update_args = UbmemIndexUpdate, ()
        update_workspace = dict(plan_buffer=workspace.index_plan, extended_work=workspace.index_work)
        forward_class = UbmemForward
    else:
        # Native Get registers row-shaped output; preserve the archive's exact
        # tensor view for restore/oracle without changing the backing address.
        output_rows = output.view(-1, geometry["token_bytes"])
        lease = KvafGetLease(torch, runtime, library, output_rows, stream)
        owner = KvafQueueView(lease, output_rows, device_table.addresses, device_table.valid)
        workspace.allocate_rotation(
            torch,
            batch=b,
            history=h,
            resident=r,
            aics=geometry["aic_count"],
            windows=geometry["requested_windows"],
            status_bytes=lease.sizes["status"],
        )
        rotating = LIFetchBinding(
            torch,
            library,
            workspace.arguments(storage),
            owner,
            batch=b,
            history=h,
            resident=r,
            aic_count=geometry["aic_count"],
            intermediate_fetches=geometry["requested_windows"],
            token_bytes=geometry["token_bytes"],
            scratch=workspace.scratch,
            **common,
        )
        rotating.output = output
        update_class, update_args, update_workspace = DirectIndexUpdate, (update_library,), {}
        forward_class = LIFetchForward
    update = update_class(
        torch,
        *update_args,
        keys=storage.keys,
        scales=storage.scales,
        addresses=storage.addresses,
        logical_ids=storage.logical_ids,
        row_owners=workspace.inputs.row_owners,
        rotating=rotating,
        status=torch.zeros(b, dtype=torch.int32, device=device),
        query_inputs=workspace.inputs,
        outputs=torch.empty((2, b * 4, 1, 2048), dtype=torch.int32, device=device),
        **update_workspace,
    )
    forward = forward_class(rotating, update)
    selector = SimpleNamespace(
        storage=storage, inputs=workspace.inputs, rotating=rotating, update=update, topk=workspace.topk
    )
    return SimpleNamespace(
        selector=selector,
        rotating=rotating,
        update=update,
        forward=forward,
        lease=lease,
        library=library,
        libraries=(library,) if transport == "ubmem" else (library, update_library),
        workspace=workspace,
        storage=storage,
        output=output,
        restored=restored,
        device_table=device_table,
        stage_calls=fused_stages(transport, rotating, update, lease, b, split_index_stages=split_index_stages),
        tuning=effective,
    )
