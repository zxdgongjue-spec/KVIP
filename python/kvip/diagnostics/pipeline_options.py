"""Independent, opt-in variants on top of the consolidated recent-index PR."""

OPTIONS = ("final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init", "ready_batch_fetch")
OPTIONAL_OPTIONS = ("ready_direct_ranges", "adaptive_admission", "prefetch_gate",
                    "skip_exhausted_prefetch", "early_prefetch_drain", "inverse_mapping")
DEFAULT_VARIANTS = ("control", "metadata", "ready-batch", "workers4", "ready-batch4", "metadata-end", "control-end")
VARIANTS = {
    "control": (),
    "topk": ("final_topk_prune",),
    "metadata": ("selected_metadata",),
    "ready": ("early_fetch_ready",),
    "init": ("compact_init",),
    "combined": OPTIONS[:4],
    "ready-batch": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "workers4": ("selected_metadata",),
    "ready-batch4": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "control-end": (),
    "metadata-end": ("selected_metadata",),
    "carry": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "window": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "capacity": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "tight": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "direct": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "ready_direct_ranges"),
    "direct-tight": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "ready_direct_ranges"),
    "ready-batch4-end": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "saturated-skip": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "skip_exhausted_prefetch"),
    "early-drain": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "early_prefetch_drain"),
    "skip-drain": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "skip_exhausted_prefetch", "early_prefetch_drain"),
    "capacity-skip-drain": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "skip_exhausted_prefetch", "early_prefetch_drain"),
    "inverse-map": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "inverse_mapping"),
    "inverse-map-skip": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "inverse_mapping", "skip_exhausted_prefetch"),
    "capacity-inverse-map": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "inverse_mapping"),
}

# Fixed policies are controlled experiments, not a claim that one is best for
# every layer. Only prefetch admission changes; exact selection remains intact.
ADMISSION = {
    "carry": (512, 64, 0, 1),
    "window": (512, 128, 0, 1),
    "capacity": (768, 128, 0, 1),
    "tight": (768, 128, 1, 1),
    "direct": (768, 128, 0, 1),
    "direct-tight": (768, 128, 1, 1),
    "capacity-skip-drain": (768, 128, 0, 1),
    "capacity-inverse-map": (768, 128, 0, 1),
}


def admission_for(name):
    return ADMISSION.get(name)


def options_for(name):
    values = {key: key in VARIANTS[name] for key in OPTIONS}
    values.update({key: True for key in OPTIONAL_OPTIONS if key in VARIANTS[name]})
    if name in ADMISSION:
        values["adaptive_admission"] = True
        if ADMISSION[name][2]:
            values["prefetch_gate"] = True
    return values


def q_workers_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown pipeline variant: {name}")
    return 4 if name in ("workers4", "ready-batch4", "ready-batch4-end",
                        "saturated-skip", "early-drain", "skip-drain",
                        "inverse-map", "inverse-map-skip", *ADMISSION) else 2


def validate_options(options):
    if not isinstance(options, dict) or not set(OPTIONS) <= set(options) <= set(OPTIONS + OPTIONAL_OPTIONS):
        raise ValueError("Pipeline variants require the five base options and known optional options")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Pipeline options must be bool")
    if options["ready_batch_fetch"] and not (options["selected_metadata"] and options["early_fetch_ready"]):
        raise ValueError("ready_batch_fetch requires selected_metadata and early_fetch_ready")
    if options.get("ready_direct_ranges") and not options["ready_batch_fetch"]:
        raise ValueError("ready_direct_ranges requires ready_batch_fetch")
    if options.get("prefetch_gate") and not options.get("adaptive_admission"):
        raise ValueError("prefetch_gate requires adaptive_admission")
    return dict(options)
