"""Original LIDU: all-hit Q4 and empty graph-padding row, no DDR service."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests/operators/lifetch/support")]


def run(device):
    import torch
    import torch_npu  # noqa: F401
    from native_mtp import NativeMtp

    torch.npu.set_device(device)
    device = torch.device(f"npu:{device}")
    batch, history, hot = 2, 4096, 4096
    owner = SimpleNamespace(
        batch=batch,
        resident_budget=hot,
        lease=None,
        cache_map=torch.arange(history, dtype=torch.int32, device=device).repeat(batch, 1),
        session=SimpleNamespace(cpu_payload=torch.zeros((batch, history, 656), dtype=torch.uint8)),
    )
    fixture = {
        "actual_query": torch.tensor([4, 4], dtype=torch.int32, device=device),
        "actual_key": torch.tensor([history, 0], dtype=torch.int32, device=device),
    }
    native = NativeMtp(torch, device, owner, fixture, source_tokens=history, padded_tokens=history)
    native.bitmap = torch.zeros((batch, 4176), dtype=torch.int32, device=device)
    native.bitmap[:, :64] = -1
    topk = torch.arange(2048, dtype=torch.int32, device=device).repeat(4, 1).reshape(4, 1, 2048)
    # NativeMtp.prepare uses the public kvip.lidu entry point.
    native.prepare(topk)
    torch.npu.synchronize()
    assert native.counts.cpu().tolist() == [0, 0, 0]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        native.prepare(topk)
    for shift in (0, 256, 512):
        topk.copy_((torch.arange(2048, dtype=torch.int32, device=device) + shift).repeat(4, 1).reshape_as(topk))
        graph.replay()
        torch.npu.synchronize()
        assert native.counts.cpu().tolist() == [0, 0, 0]
        expected = topk.cpu().reshape(4, 2048).sort().values
        actual = native.destinations[:4].cpu().reshape(4, 2048).sort().values
        torch.testing.assert_close(actual, expected)
    print("PASS: kvip.lidu eager/graph, Q4 all-hit, empty padding row, repeated replay")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=0)
    run(p.parse_args().device)
