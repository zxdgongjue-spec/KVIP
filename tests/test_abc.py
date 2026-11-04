# SPDX-License-Identifier: Apache-2.0
"""Compare the actual batched NPU update with its CPU reference; no CPU service needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
from kvip.abc_update import update_abc


def run_npu(device=0):
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    shape = (3, 5, 2048)
    descriptors = torch.tensor(
        [[[2, 4, 350, 734], [2, 1, 768, 768]], [[0, 3, 383, 390], [0, 2, 128, 140]]], dtype=torch.int32
    )
    cpu_t = torch.full(shape, 99, dtype=torch.int32)
    cpu_s = cpu_t.clone()
    npu_t, npu_s = cpu_t.npu(), cpu_s.npu()
    device_desc = descriptors.npu()
    for shift in (0, 128):
        descriptors[:, :, 2:] += shift
        device_desc.copy_(descriptors)
        update_abc(cpu_t, cpu_s, descriptors, 2, 1024, 384)
        update_abc(npu_t, npu_s, device_desc, 2, 1024, 384)
        torch.testing.assert_close(npu_t.cpu(), cpu_t)
        torch.testing.assert_close(npu_s.cpu(), cpu_s)
    print("PASS: batched ABC forward/reverse maps, wrap, empty tail, and untouched history")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    run_npu(parser.parse_args().device)


def test_cpu_abc_ownership():
    mapping = torch.full((2, 3, 2048), -1, dtype=torch.int32)
    reverse = mapping.clone()
    desc = torch.tensor([[[1, 2, 383, 390]]], dtype=torch.int32)
    update_abc(mapping, reverse, desc, 1, 1024, 384)
    for token in range(383, 390):
        slot = 1024 + token % 384
        assert mapping[1, 2, token] == slot
        assert reverse[1, 2, slot] == token
    assert (mapping[0] == -1).all()
