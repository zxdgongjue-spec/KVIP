"""Preparation-time consumer checks for the external KVAF lease, not an SDK clone."""

from numbers import Integral

from .api import GRAPH_TRANSPORT_ABI, LayoutSpec


def validate_lease(lease, output, *, layout=LayoutSpec()):
    expected = dict(
        output_address=output.data_ptr(),
        output_bytes=output.numel() * output.element_size(),
        token_bytes=layout.token_bytes,
        tokens_per_block=layout.block_tokens,
        jettys_per_core=2,
    )
    if not isinstance(lease, dict) or any(
        lease.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("KVAF lease geometry differs from the public layout/output")
    for name in (
        "worker_count",
        "device_channel_entity",
        "device_queue_state",
        "device_core_status",
    ):
        value = lease.get(name)
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid KVAF lease {name}")
    return lease


def validate_graph_transport(runtime):
    for name in (
        "fused_graph_abi",
        "register_fused_graph_output",
        "begin_fused_graph",
        "end_fused_graph",
    ):
        if not callable(getattr(runtime, name, None)):
            raise RuntimeError(f"KVAF lacks {name}")
    if runtime.fused_graph_abi() != GRAPH_TRANSPORT_ABI:
        raise RuntimeError("unsupported KVAF Graph transport ABI")
