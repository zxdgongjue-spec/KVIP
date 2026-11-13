"""Independent, opt-in variants on top of the consolidated recent-index PR."""

OPTIONS = ("final_topk_prune", "selected_metadata", "early_fetch_ready", "compact_init", "ready_batch_fetch")
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
}


def options_for(name):
    return {key: key in VARIANTS[name] for key in OPTIONS}


def q_workers_for(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown pipeline variant: {name}")
    return 4 if name in ("workers4", "ready-batch4") else 2


def validate_options(options):
    if not isinstance(options, dict) or set(options) != set(OPTIONS):
        raise ValueError("Pipeline variants must specify exactly the five build options")
    if any(type(value) is not bool for value in options.values()):
        raise ValueError("Pipeline options must be bool")
    if options["ready_batch_fetch"] and not (options["selected_metadata"] and options["early_fetch_ready"]):
        raise ValueError("ready_batch_fetch requires selected_metadata and early_fetch_ready")
    return dict(options)
