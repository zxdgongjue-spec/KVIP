"""Independent, opt-in variants on top of the consolidated recent-index PR."""

OPTIONS = ("final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init", "ready_batch_fetch")
OPTIONAL_OPTIONS = ("ready_direct_ranges", "adaptive_admission", "prefetch_gate",
                    "skip_exhausted_prefetch", "early_prefetch_drain", "inverse_mapping", "low_miss_skip", "auto_route", "low_miss_tail")
NUMERIC_OPTIONS = ("prefetch_rank",)
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
    "quota-zero": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch"),
    "light-low": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip"),
    "light-high": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip"),
    "light-high-end": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip"),
    "auto": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "auto_route"),
    "auto-end": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "auto_route"),
    "light-low-tail": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "low_miss_tail"),
    "light-high-tail": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "low_miss_tail"),
    "light-high-tail-end": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "low_miss_tail"),
    "auto-tail": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "auto_route", "low_miss_tail"),
    "auto-tail-end": ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "low_miss_skip", "auto_route", "low_miss_tail"),
}

# The exact selection remains Top2048. These ranks only admit speculative
# cold transfers, with an explicit K2048 control for the enabled gate build.
RANKS = {"rank2048": 2048, "rank1792": 1792, "rank1536": 1536,
         "capacity-rank2048": 2048, "capacity-rank1792": 1792}
for _name in RANKS:
    VARIANTS[_name] = ("selected_metadata", "early_fetch_ready", "ready_batch_fetch", "prefetch_gate")

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
    "quota-zero": (0, 64, 0, 0),
    "light-low": (0, 64, 0, 0),
    "light-high": (512, 64, 0, 0),
    "light-high-end": (512, 64, 0, 0),
    "auto": (512, 64, 0, 0),
    "auto-end": (512, 64, 0, 0),
    "light-low-tail": (0, 64, 0, 0),
    "light-high-tail": (512, 64, 0, 0),
    "light-high-tail-end": (512, 64, 0, 0),
    "auto-tail": (512, 64, 0, 0),
    "auto-tail-end": (512, 64, 0, 0),
}
for _name in RANKS:
    ADMISSION[_name] = (768, 128, 0, 1) if _name.startswith("capacity-") else (512, 64, 0, 0)


def admission_for(name):
    return ADMISSION.get(name)


def options_for(name):
    values = {key: key in VARIANTS[name] for key in OPTIONS}
    values.update({key: True for key in OPTIONAL_OPTIONS if key in VARIANTS[name]})
    if name in ADMISSION:
        values["adaptive_admission"] = True
        if ADMISSION[name][2]:
            values["prefetch_gate"] = True
    if name in RANKS:
        values["prefetch_rank"] = RANKS[name]
    return values


def q_workers_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown pipeline variant: {name}")
    return 4 if name in ("workers4", "ready-batch4", "ready-batch4-end",
                        "saturated-skip", "early-drain", "skip-drain",
                        "inverse-map", "inverse-map-skip", *ADMISSION) else 2


def validate_options(options):
    if not isinstance(options, dict) or not set(OPTIONS) <= set(options) <= set(OPTIONS + OPTIONAL_OPTIONS + NUMERIC_OPTIONS):
        raise ValueError("Pipeline variants require the five base options and known optional options")
    if any(type(value) is not bool for key, value in options.items() if key not in NUMERIC_OPTIONS):
        raise ValueError("Pipeline options must be bool")
    if "prefetch_rank" in options:
        rank = options["prefetch_rank"]
        if type(rank) is not int or not 1 <= rank <= 2048 or not options.get("prefetch_gate"):
            raise ValueError("prefetch_rank requires an integer in 1..2048 and prefetch_gate")
    if options["ready_batch_fetch"] and not (options["selected_metadata"] and options["early_fetch_ready"]):
        raise ValueError("ready_batch_fetch requires selected_metadata and early_fetch_ready")
    if options.get("ready_direct_ranges") and not options["ready_batch_fetch"]:
        raise ValueError("ready_direct_ranges requires ready_batch_fetch")
    if options.get("prefetch_gate") and not options.get("adaptive_admission"):
        raise ValueError("prefetch_gate requires adaptive_admission")
    if options.get("low_miss_skip") and not options.get("adaptive_admission"):
        raise ValueError("low_miss_skip requires adaptive_admission")
    if options.get("auto_route") and not all(options.get(key) for key in (
            "low_miss_skip", "adaptive_admission", "ready_batch_fetch", "early_fetch_ready")):
        raise ValueError("auto_route requires low_miss_skip, adaptive_admission, ready_batch_fetch and early_fetch_ready")
    if options.get("low_miss_tail") and not all(options.get(key) for key in (
            "low_miss_skip", "selected_metadata", "ready_batch_fetch", "early_fetch_ready")):
        raise ValueError("low_miss_tail requires low_miss_skip, selected_metadata, ready_batch_fetch and early_fetch_ready")
    return dict(options)


def is_auto_variant(name):
    return "auto_route" in VARIANTS[name]


def early_threshold_skipped(tuning):
    """Replay supplies the same validated controls to every active row."""
    tuning = tuning or {}
    if tuning.get("auto_route"):
        return bool(tuning.get("low_miss_skip") and tuning.get("auto_low_active", False))
    controls = tuning.get("admission_controls")
    return bool(tuning.get("low_miss_skip") and tuning.get("adaptive_admission")
                and controls is not None and len(controls) == 4 and controls[0] == 0)
