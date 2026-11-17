"""Verify CANN's generated kernel types against the staged physical budget."""

import argparse
import hashlib
import json
from pathlib import Path
import re

AIV_KERNELS = ("prepare", "prepare_all", "scatter", "scatter_predicate",
               "progress", "service", "service_ready", "topk")
SCORE_KERNELS = ("score", "score_direct")


def verify(folder):
    paths = {name: folder / name for name in ("aic_config.cmake", "aiv_config.cmake")}
    configs = {name: path.read_text() for name, path in paths.items()}
    mappings = {
        name: dict(re.findall(r"auto_gen_(v9_staged_\w+)_kernel=([\w]+)", value)) for name, value in configs.items()
    }
    for suffix in AIV_KERNELS:
        kernel = "v9_staged_" + suffix
        if kernel in mappings["aic_config.cmake"]:
            raise ValueError(f"{kernel} unexpectedly occupies an AIC")
        symbol = mappings["aiv_config.cmake"].get(kernel, "")
        if not symbol or "_mix_" in symbol:
            raise ValueError(f"{kernel} lacks an AIV-only generated entry")
    for kernel in SCORE_KERNELS:
        for name, suffix in (("aic_config.cmake", "_mix_aic"), ("aiv_config.cmake", "_mix_aiv")):
            if not mappings[name].get("v9_staged_" + kernel, "").endswith(suffix):
                raise ValueError(f"{kernel} must retain its MIX AIC 1:2 entries")
    generated = folder / "auto_gen_indexer.cpp"
    text = generated.read_text()
    for kernel in SCORE_KERNELS:
        if not re.search(r"v9_staged_" + kernel + r"_mix_aic_section[^;]+K_TYPE_MIX_AIC_MAIN[^;]+1, 2", text):
            raise ValueError(f"{kernel} generated metadata is not MIX 1:2")
    paths[generated.name] = generated
    vector = folder / "auto_gen_staged_vector.cpp"
    paths[vector.name] = vector
    vector_text = vector.read_text()
    if "_mix_aic_section" in vector_text or "_mix_aiv_section" in vector_text:
        raise ValueError("staged vector translation unit still has MIX metadata")
    return {
        "status": "PASSED",
        "aiv_only": ["v9_staged_" + name for name in AIV_KERNELS],
        "score": "MIX AIC 1:2",
        "generated_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-gen", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.auto_gen)
    except (ValueError, OSError) as error:
        parser.exit(1, f"STAGED_METADATA_FAILED: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
