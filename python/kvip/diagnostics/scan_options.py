"""Independent URMA scan-order/admission experiment; no graph-route changes."""

OPTIONS = ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_batched", "rolling_windowed", "rolling_compact", "rolling_first_window", "refilter_safe", "refilter_refill", "compact_out_of_place", "rolling_reuse_seed", "rolling_fast_path", "rolling_sparse", "rolling_boundary", "incremental_topk", "incremental_window", "incremental_register", "incremental_prefilter", "incremental_direct", "incremental_ties", "incremental_light", "incremental_inplace", "incremental_progress", "incremental_segments")
EXPERIMENT_OPTIONS = ("consumed_prefix", "window_probe", "seed_mask", "seed_parallel")
DEFAULT_VARIANTS = ("forward", "reverse", "forward-wide", "reverse-wide", "forward-end")
VARIANTS = {name: ("reverse_cold",) if name.startswith("reverse") else () for name in DEFAULT_VARIANTS}
ROLLING_VARIANTS = ("forward", "reverse", "reverse-roll", "reverse-refilter", "forward-refilter", "forward-end")
VARIANTS.update({
    "reverse-roll": ("reverse_cold", "rolling_threshold"),
    "reverse-refilter": ("reverse_cold", "rolling_threshold", "window_refilter"),
    "forward-refilter": ("rolling_threshold", "window_refilter"),
})

BATCHED_VARIANTS = ("forward", "reverse", "reverse-refilter", "reverse-batch",
                    "reverse-batch-ref", "forward-batch-ref", "forward-end")
VARIANTS.update({
    "reverse-batch": ("reverse_cold", "rolling_threshold", "rolling_batched"),
    "reverse-batch-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_batched"),
    "forward-batch-ref": ("rolling_threshold", "window_refilter", "rolling_batched"),
})

WINDOWED_VARIANTS = ("forward", "reverse-refilter", "reverse-batch-ref", "reverse-window",
                    "reverse-window-ref", "forward-window-ref", "forward-end")
VARIANTS.update({
    "reverse-window": ("reverse_cold", "rolling_threshold", "rolling_windowed"),
    "reverse-window-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed"),
    "forward-window-ref": ("rolling_threshold", "window_refilter", "rolling_windowed"),
})


LEAN_VARIANTS = ("forward", "reverse", "reverse-window-ref", "reverse-compact-ref",
                 "reverse-first-ref", "forward-end")
VARIANTS.update({
    "reverse-compact-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "rolling_compact"),
    "reverse-first-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "rolling_compact", "rolling_first_window"),
})


SAFE_VARIANTS = ("forward", "reverse", "reverse-window-ref", "reverse-safe-ref",
                 "reverse-refill-ref", "forward-end")
VARIANTS.update({
    "reverse-safe-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "refilter_safe"),
    "reverse-refill-ref": ("reverse_cold", "rolling_threshold", "window_refilter", "rolling_windowed", "refilter_safe", "refilter_refill"),
})


ECONOMY_VARIANTS = ("forward", "reverse", "reverse-out", "reverse-compact-safe",
                    "reverse-first", "reverse-first-safe", "reverse-end", "forward-end")
VARIANTS.update({
    "reverse-out": ("reverse_cold", "compact_out_of_place"),
    "reverse-compact-safe": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "window_refilter", "refilter_safe"),
    "reverse-first": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "rolling_first_window"),
    "reverse-first-safe": ("reverse_cold", "compact_out_of_place", "rolling_threshold", "rolling_windowed", "rolling_compact", "rolling_first_window", "window_refilter", "refilter_safe"),
    "reverse-end": ("reverse_cold",),
})


SEED_VARIANTS = ("forward", "reverse", "reverse-first-in", "reverse-first-reuse",
                 "reverse-window-in", "reverse-window-reuse", "reverse-end", "forward-end")
for _name in SEED_VARIANTS[2:6]:
    _flags = ("reverse_cold", "rolling_threshold", "rolling_windowed", "rolling_compact",
              "window_refilter", "refilter_safe")
    if "first" in _name:
        _flags += ("rolling_first_window",)
    if _name.endswith("reuse"):
        _flags += ("rolling_reuse_seed",)
    VARIANTS[_name] = _flags

FAST_VARIANTS = ("forward", "reverse", "reverse-window-in", "reverse-window-reuse",
                 "reverse-window-fast", "reverse-end", "forward-end")
VARIANTS["reverse-window-fast"] = VARIANTS["reverse-window-reuse"] + ("rolling_fast_path",)


SPARSE_VARIANTS = ("forward", "reverse", "reverse-window-fast", "reverse-sparse",
                   "reverse-sparse-ref", "reverse-end", "forward-end")
VARIANTS["reverse-sparse-ref"] = VARIANTS["reverse-window-fast"] + ("rolling_sparse",)
VARIANTS["reverse-sparse"] = tuple(
    flag for flag in VARIANTS["reverse-sparse-ref"]
    if flag not in ("window_refilter", "refilter_safe")
)


BOUNDARY_VARIANTS = ("forward", "reverse", "reverse-window-fast", "reverse-boundary",
                     "reverse-boundary-ref", "reverse-end", "forward-end")
VARIANTS["reverse-boundary-ref"] = VARIANTS["reverse-window-fast"] + ("rolling_boundary",)
VARIANTS["reverse-boundary"] = tuple(
    flag for flag in VARIANTS["reverse-boundary-ref"]
    if flag not in ("window_refilter", "refilter_safe")
)


INCREMENTAL_VARIANTS = ("forward", "reverse", "reverse-incremental", "reverse-end", "forward-end")
VARIANTS["reverse-incremental"] = ("reverse_cold", "incremental_topk")


INCREMENTAL_WINDOW_VARIANTS = (
    "forward", "reverse", "reverse-incremental", "reverse-inc-window",
    "reverse-inc-register", "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-window"] = ("reverse_cold", "incremental_topk", "incremental_window")
VARIANTS["reverse-inc-register"] = VARIANTS["reverse-inc-window"] + ("incremental_register",)


INCREMENTAL_PREFILTER_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-prefilter",
    "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-prefilter"] = VARIANTS["reverse-inc-register"] + ("incremental_prefilter",)


INCREMENTAL_STATE_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-direct",
    "reverse-inc-ties", "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-direct"] = VARIANTS["reverse-inc-prefilter"] + ("incremental_direct",)
VARIANTS["reverse-inc-ties"] = VARIANTS["reverse-inc-direct"] + ("incremental_ties",)


INCREMENTAL_LIGHT_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-light",
    "reverse-inc-light-ties", "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-light"] = VARIANTS["reverse-inc-direct"] + ("incremental_light",)
VARIANTS["reverse-inc-light-ties"] = VARIANTS["reverse-inc-ties"] + ("incremental_light",)
INCREMENTAL_INPLACE_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-light-ties",
    "reverse-inc-inplace", "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-inplace"] = VARIANTS["reverse-inc-light-ties"] + ("incremental_inplace",)


INCREMENTAL_PROGRESS_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-progress",
    "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-progress"] = VARIANTS["reverse-inc-register"] + ("incremental_progress",)


SEGMENT_VARIANTS = (
    "forward", "reverse", "reverse-inc-register", "reverse-inc-light-ties", "reverse-inc-segments",
    "reverse-end", "forward-end",
)
VARIANTS["reverse-inc-segments"] = VARIANTS["reverse-inc-light-ties"] + ("incremental_segments",)


VARIANTS["reverse-consumed"] = VARIANTS["reverse-inc-segments"] + ("consumed_prefix",)
for _base in ("reverse", "reverse-inc-segments", "reverse-consumed"):
    for _cap in (128, 480):
        VARIANTS[f"{_base}-cap{_cap}"] = VARIANTS[_base]
for _base in ("reverse", "reverse-inc-register", "reverse-inc-segments", "reverse-consumed"):
    VARIANTS[_base + "-probe"] = VARIANTS[_base] + ("window_probe",)


VARIANTS["reverse-seed-mask"] = VARIANTS["reverse-consumed"] + ("seed_mask",)
VARIANTS["reverse-seed-mask-probe"] = VARIANTS["reverse-seed-mask"] + ("window_probe",)
VARIANTS["reverse-seed-parallel"] = VARIANTS["reverse-seed-mask"] + ("seed_parallel",)
VARIANTS["reverse-seed-parallel-probe"] = VARIANTS["reverse-seed-parallel"] + ("window_probe",)
for _base in ("reverse-consumed", "reverse-seed-mask", "reverse-seed-parallel", "reverse-cap128"):
    VARIANTS[_base + "-end"] = VARIANTS[_base]


def options_for(name):
    options = {key: key in VARIANTS[name] for key in OPTIONS}
    options.update({key: True for key in EXPERIMENT_OPTIONS if key in VARIANTS[name]})
    return options


def budgets_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown scan variant: {name}")
    if "-cap" in name:
        return 512, int(name.removesuffix("-end").rsplit("-cap", 1)[1])
    return (1024, 128) if name.endswith("-wide") else (512, 64)


def q_workers_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown scan variant: {name}")
    return 2


def validate_options(options):
    if not isinstance(options, dict) or set(options).difference(EXPERIMENT_OPTIONS) != set(OPTIONS):
        raise ValueError("Scan experiment options do not match the selected native interface")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Scan build options must be bool")
    if options.get("seed_parallel") and not options.get("seed_mask"):
        raise ValueError("Parallel seed preparation requires the seed mask")
    if options.get("seed_mask") and not options["incremental_segments"]:
        raise ValueError("Seed mask requires segment lists")
    if options.get("consumed_prefix") and not options["incremental_segments"]:
        raise ValueError("Consumed prefix requires segment lists")
    if options["incremental_segments"] and (
        not options["incremental_light"] or not options["incremental_ties"]
        or options["incremental_progress"] or options["incremental_inplace"]
    ):
        raise ValueError("Segment lists require light tie selection and its private staging bank")
    if options["incremental_progress"] and not options["incremental_window"]:
        raise ValueError("Independent progress requires fixed-window incremental staging")
    if options["incremental_inplace"] and not (options["incremental_light"] and options["incremental_ties"]):
        raise ValueError("In-place boundary updates require light state and exact tie updates")
    if options["incremental_light"] and not options["incremental_direct"]:
        raise ValueError("Light incremental state requires direct selection")
    if options["incremental_ties"] and not options["incremental_direct"]:
        raise ValueError("Boundary tie updates require direct incremental state")
    if options["incremental_direct"] and not options["incremental_prefilter"]:
        raise ValueError("Direct incremental state requires window prefilter")
    if options["incremental_prefilter"] and not options["incremental_register"]:
        raise ValueError("Window prefilter requires register window selection")
    if options["incremental_window"] and not options["incremental_topk"]:
        raise ValueError("Window staging requires incremental TopK")
    if options["incremental_register"] and not options["incremental_window"]:
        raise ValueError("Register selection requires incremental window staging")
    if options["incremental_topk"] and (options["rolling_threshold"] or options["window_refilter"]):
        raise ValueError("Incremental TopK replaces rolling/refilter")
    if (options["window_refilter"] or options["rolling_batched"] or options["rolling_windowed"]) and not options["rolling_threshold"]:
        raise ValueError("Window refilter requires rolling threshold")
    if options["rolling_batched"] and options["rolling_windowed"]:
        raise ValueError("Select either batch or window accumulation")
    if (options["rolling_compact"] or options["rolling_first_window"]) and not options["rolling_windowed"]:
        raise ValueError("Compact/first-window options require window accumulation")
    if options["rolling_first_window"] and not options["rolling_compact"]:
        raise ValueError("First-window correction requires compact admission")
    if options["rolling_reuse_seed"] and not options["rolling_compact"]:
        raise ValueError("Seed reuse requires compact window accumulation")
    if options["rolling_boundary"] and (not options["rolling_fast_path"] or
            options["rolling_sparse"] or options["rolling_first_window"] or options["rolling_batched"]):
        raise ValueError("Boundary readback requires dense fast window counts and every window")
    if options["rolling_sparse"] and not options["rolling_fast_path"]:
        raise ValueError("Sparse window counts require packed seed reuse")
    if options["rolling_fast_path"] and not options["rolling_reuse_seed"]:
        raise ValueError("Packed byte accumulation requires seed histogram reuse")
    if options["refilter_safe"] and not options["window_refilter"]:
        raise ValueError("Visibility-safe admission requires window refilter")
    if options["refilter_refill"] and not options["refilter_safe"]:
        raise ValueError("Same-window refill requires visibility-safe admission")
    return dict(options)


def verify_library(path, options):
    """Check the real host ABI in a child so registrations cannot accumulate."""
    import json
    import subprocess
    import sys

    check = (
        "import ctypes,json,sys; "
        "from kvip.diagnostics.fused_replay import verify_build_options; "
        "verify_build_options(ctypes.CDLL(sys.argv[1]),json.loads(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", check, str(path), json.dumps(options)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout)[-4000:]
        raise RuntimeError(
            f"LIFetch ABI preflight failed (exit={result.returncode}): {path}\n{detail}"
        )


def prebuild(args, retained):
    """Use the production builder before allocating/restoring any NPU arena."""
    from pathlib import Path

    from kvip.runtime.build import build_libraries, kvaf_build_paths
    from kvip.diagnostics.console import emit

    cann, headers = kvaf_build_paths(args.build_dir)
    seen = set()
    for name in args.scan_variants:
        options = options_for(name)
        key = tuple(sorted(options.items()))
        if key in seen:
            continue
        retained.setdefault("progress", {}).update(variant=name, phase="compile_preflight", step=None)
        library, = build_libraries(cache_dir=Path(args.build_dir) / "lifetch", cann=cann,
                                   headers=headers, soc=args.soc, **options)
        retained["progress"].update(phase="abi_preflight")
        verify_library(library, options)
        seen.add(key)
    emit(f"BUILD_VARIANTS_READY unique={len(seen)} abi=verified")
