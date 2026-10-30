"""Minimum native baseline dependencies. Never import vLLM registration."""

import subprocess
from pathlib import Path


def command(argv, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        subprocess.run([str(arg) for arg in argv], stdout=output, stderr=subprocess.STDOUT, check=True)


def find_headers(root):
    if root is None:
        raise ValueError("--kvaf-root is required for the real KVAF backend")
    path = Path(root) / "access_operator/demo/aiv_batch_get/include"
    if not (path / "kvaf_batch_get_device.h").is_file():
        raise ValueError(f"missing KVAF device headers: {path}")
    return path


class Registration:
    @staticmethod
    def require_a5_dsa_registration(names):
        # Load only on explicit request. Fused-only tests need no LIDU build.
        import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp  # noqa: F401
        import torch

        for name in names:
            if not callable(getattr(torch.ops.nanovllm_dsa, name, None)):
                raise RuntimeError(f"native LIDU binding missing: {name}; bash build.sh native-lidu")


def _load_native_operators(mode):
    import torch
    import torch_npu  # noqa: F401

    return torch, Registration, None
