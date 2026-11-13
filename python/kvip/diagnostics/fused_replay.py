# SPDX-License-Identifier: Apache-2.0
"""Replay archived inputs through the current production streaming forward."""

from pathlib import Path
from types import SimpleNamespace
import ctypes

from kvip.cache.index_storage import IndexCacheStorage


def validate_fused_tuning(value, transport):
    if transport not in ("ubmem", "urma") or not isinstance(value, dict):
        raise ValueError("Expected a tuning object and ubmem/urma transport")
    if set(value) - {
        "recent_threshold",
        "recent_pipeline",
        "recent_unified_topk",
        "recent_skip",
        "recent_threshold_only",
        "recent_seed_ub",
        "recent_profile",
        "recent_local_victims",
        "recent_register_kth",
        "recent_overlap_victims",
    }:
        raise ValueError("Only recent-threshold seed options are changed in this production A/B comparison")
    if any(type(flag) is not bool for flag in value.values()):
        raise ValueError("Recent-threshold seed options must be boolean")
    if transport != "urma" and value.get("recent_threshold_only"):
        raise ValueError("recent_threshold_only is supported only by URMA")
    if transport != "urma" and (value.get("recent_seed_ub") or value.get("recent_profile")):
        raise ValueError("recent_seed_ub/recent_profile are supported only by URMA")
    if transport != "urma" and value.get("recent_local_victims"):
        raise ValueError("recent_local_victims is supported only by URMA")
    if value.get("recent_overlap_victims"):
        if transport != "urma":
            raise ValueError("recent_overlap_victims is supported only by URMA")
        if not all(value.get(key) for key in ("recent_register_kth", "recent_local_victims", "recent_threshold_only")):
            raise ValueError(
                "recent_overlap_victims requires recent_register_kth, recent_local_victims and recent_threshold_only"
            )
    if transport != "urma" and value.get("recent_register_kth"):
        raise ValueError("recent_register_kth is supported only by URMA")
    if value.get("recent_register_kth") and not value.get("recent_threshold_only"):
        raise ValueError("recent_register_kth requires recent_threshold_only")
    if value.get("recent_local_victims") and not value.get("recent_threshold_only"):
        raise ValueError("recent_local_victims requires recent_threshold_only")
    if value.get("recent_seed_ub") and not value.get("recent_threshold_only"):
        raise ValueError("recent_seed_ub requires recent_threshold_only")
    return value


def verify_build_options(library, options):
    """Reject a stale or mismatched binary before capturing its graph."""
    for key, expected in options.items():
        function = getattr(library, "kvip_" + key)
        function.argtypes, function.restype = [], ctypes.c_uint32
        if function() != int(expected):
            raise RuntimeError(f"Loaded kernel does not match requested {key}={expected}")


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
    upstream=False,
    pipeline_options=None,
    q_workers=2,
    topk_probe=False,
    admission_controls=None,
):
    from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, UbmemLease
    from kvip.operators.lifetch.binding import LIFetchBinding
    from kvip.operators.lifetch.forward import LIFetchForward
    from kvip.operators.lifetch.index_update import DirectIndexUpdate, UbmemIndexUpdate
    from kvip.operators.lifetch.workspace import LIFetchWorkspace, UbmemWorkspace
    from kvip.operators.lifetch_ubmem.binding import UbmemBinding
    from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    if split_index_stages:
        raise ValueError("Streaming stages overlap; use the complete production graph timing")
    if type(q_workers) is not int or q_workers not in (1, 2, 4):
        raise ValueError("q_workers must be 1, 2 or 4")
    if q_workers != 2 and (not upstream or pipeline_options is None or transport != "urma"):
        raise ValueError("Nondefault q_workers requires an exact URMA pipeline candidate")
    tuning = validate_fused_tuning(tuning or {}, transport)
    g = archive.manifest["geometry"]
    b, h, r = (g[n] for n in ("batch", "history", "resident"))
    if aics < g["aic_count"]:
        raise ValueError("Replay device has fewer AICs than the recorded input")
    if g["requested_windows"] != 8:
        raise ValueError("This comparison uses current serving defaults: 8 windows, 512 prefetch tokens")
    common = dict(
        batch=b, history=h, aics=g["aic_count"], windows=g["requested_windows"], device=device, index_heads=g["heads"]
    )
    effective = dict(
        recent_threshold=tuning.get("recent_threshold", False),
        recent_pipeline=tuning.get("recent_pipeline", True),
        recent_unified_topk=tuning.get("recent_unified_topk", True),
        recent_skip=tuning.get("recent_skip", transport == "ubmem"),
        recent_threshold_only=tuning.get("recent_threshold_only", False),
        recent_seed_ub=tuning.get("recent_seed_ub", False),
        recent_profile=tuning.get("recent_profile", transport == "urma"),
        recent_local_victims=tuning.get("recent_local_victims", False),
        recent_register_kth=tuning.get("recent_register_kth", False),
        recent_overlap_victims=tuning.get("recent_overlap_victims", False),
        prefetch_limit=512,
        window_limit=64,
    )
    build_options = {key: effective[key] for key in ("recent_pipeline", "recent_unified_topk", "recent_skip")}
    if transport == "urma":
        build_options.update(
            {
                key: effective[key]
                for key in (
                    "recent_threshold_only",
                    "recent_seed_ub",
                    "recent_profile",
                    "recent_local_victims",
                    "recent_register_kth",
                    "recent_overlap_victims",
                )
            }
        )
    if upstream:
        if set(tuning) - {"recent_threshold"}:
            raise ValueError("An exact production checkout cannot accept legacy experiment options")
        effective = dict(
            recent_threshold=tuning.get("recent_threshold", False),
            prefetch_limit=512,
            window_limit=64,
            q_workers=q_workers,
        )
        build_options = {}  # Production owns its implementation; no legacy experiment flags.
        if pipeline_options is not None:
            from kvip.diagnostics.pipeline_options import validate_options

            if transport != "urma" or not effective["recent_threshold"]:
                raise ValueError("Pipeline options require the exact URMA recent production checkout")
            build_options = validate_options(pipeline_options)
            effective.update(build_options)
            if topk_probe:
                build_options["topk_prune_diagnostics"] = True
                effective["topk_prune_diagnostics"] = True
    elif pipeline_options is not None:
        raise ValueError("Pipeline variants cannot use legacy experiment production code")
    if topk_probe and (not upstream or pipeline_options is None or not build_options.get("final_topk_prune")):
        raise ValueError("TopK probe requires the exact production pruning candidate")
    threshold_bounds = None
    if effective["recent_threshold"]:
        # Immutable per-replay bounds; all scoring/merging occurs inside forward.
        # Never substitute the current ABC start: that would omit retired pages.
        threshold_bounds = archive.tensor("pre.threshold_bounds", torch, device)
    if transport == "ubmem":
        shape = UbmemGeometry(b, h, r, g["heads"], g["token_bytes"], g["aic_count"], g["requested_windows"], soc)
        library = build_ubmem_library(cache_dir=Path(build) / "lifetch_ubmem", cann=cann, shape=shape, **build_options)
        workspace = UbmemWorkspace(torch, resident=r, **common)
    else:
        runtime_cann, headers = kvaf_build_paths(build)
        if runtime_cann.resolve() != Path(cann).resolve():
            raise ValueError("KVAF and fusion must use the same CANN build")
        (library,) = build_libraries(
            cache_dir=Path(build) / "lifetch", cann=cann, headers=headers, soc=soc, **build_options
        )
        workspace = LIFetchWorkspace(torch, **common)
    storage = IndexCacheStorage(torch, g["owners"], h, r, device)
    output = archive.tensor("pre.output", torch, device)
    output_rows = output.view(-1, g["token_bytes"])
    kwargs = dict(
        query_ends=workspace.inputs.query_ends.view(b, 4),
        hot_valid=storage.hot_valid,
        row_owners=workspace.inputs.row_owners,
        token_addresses=storage.addresses,
        tail_bounds=workspace.tail_bounds,
        output_stride=g["output_stride"],
        prefetch_limit=effective["prefetch_limit"],
        window_limit=effective["window_limit"],
    )
    if not upstream or effective["recent_threshold"]:
        kwargs["threshold_bounds"] = threshold_bounds
    if transport == "ubmem":
        lease = UbmemLease(torch, runtime, output_rows, stream)
        lease.register_table(device_table)
        rotating = UbmemBinding(
            torch,
            library,
            workspace.arguments(storage),
            lease,
            output_rows,
            shape=shape,
            scratch=workspace.scratch,
            **kwargs,
        )
        update_type = UbmemIndexUpdate
    else:
        lease = KvafGetLease(torch, runtime, library, output_rows, stream)
        owner = KvafQueueView(lease, output_rows, device_table.addresses, device_table.valid)
        workspace.allocate_rotation(
            torch,
            batch=b,
            history=h,
            resident=r,
            aics=g["aic_count"],
            windows=g["requested_windows"],
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
            aic_count=g["aic_count"],
            intermediate_fetches=g["requested_windows"],
            token_bytes=g["token_bytes"],
            scratch=workspace.scratch,
            **kwargs,
        )
        update_type = DirectIndexUpdate
    verify_build_options(rotating.library, build_options)
    if admission_controls is not None:
        if not upstream or transport != "urma":
            raise ValueError("Admission controls require an exact current URMA checkout")
        rotating.set_admission_controls(
            torch.tensor([admission_controls] * b, dtype=torch.int32, device="cpu")
        )
        effective["admission_controls"] = list(admission_controls)
    if effective.get("recent_seed_ub", False):
        from kvip.diagnostics.seed_eligibility import predict_seed_eligibility

        if not predict_seed_eligibility(archive, rotating.runtime_shape)["predicted_eligible_queries"]:
            raise ValueError("Seed-UB has no eligible queries; every query would use the fallback")
    # Keep the archived output shape for generic state reset/oracle.
    rotating.output = output
    update = update_type(
        torch,
        keys=storage.keys,
        scales=storage.scales,
        addresses=storage.addresses,
        logical_ids=storage.logical_ids,
        row_owners=workspace.inputs.row_owners,
        rotating=rotating,
        status=torch.zeros(b, dtype=torch.int32, device=device),
        query_inputs=workspace.inputs,
        outputs=torch.empty((2, b * 4, 1, 2048), dtype=torch.int32, device=device),
    )
    forward = LIFetchForward(rotating, update, **({"q_workers": q_workers} if q_workers != 2 else {}))
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
        libraries=(library,),
        workspace=workspace,
        storage=storage,
        output=output,
        restored=restored,
        device_table=device_table,
        stage_calls=(),
        tuning=effective,
    )


def _check_geometry(geometry, aics):
    from kvip.operators.lifetch.geometry import final_aivs, li_geometry

    if aics < geometry["aic_count"]:
        raise ValueError("Replay device has fewer AICs than captured geometry")
    shape = li_geometry(geometry["batch"], "auto", geometry["requested_windows"], geometry["aic_count"])
    expected = dict(
        rounds=shape["intermediate_fetches"],
        service_lanes=shape["service_aivs_per_window"],
        final_workers=final_aivs(geometry["batch"], "auto", geometry["aic_count"]),
    )
    if any(geometry[name] != value for name, value in expected.items()):
        raise ValueError("Unsupported recorded Q4 scheduling geometry")
