from types import SimpleNamespace

import pytest

from kvop.v9.transport import validate_graph_transport, validate_lease


def test_layout_uses_output_bytes_and_checks_actual_lease():
    output = SimpleNamespace(
        data_ptr=lambda: 1024, numel=lambda: 100, element_size=lambda: 2
    )
    lease = dict(
        output_address=1024,
        output_bytes=200,
        token_bytes=656,
        tokens_per_block=128,
        jettys_per_core=2,
        worker_count=1,
        device_channel_entity=8,
        device_queue_state=16,
        device_core_status=24,
    )
    assert validate_lease(lease, output) is lease
    for changes in (
        {"output_bytes": 100},
        {"token_bytes": 657},
        {"worker_count": True},
        {"device_queue_state": 0},
    ):
        with pytest.raises(ValueError):
            validate_lease(lease | changes, output)


def test_graph_transport_checks_version_and_required_methods_before_registration():
    runtime = SimpleNamespace(
        fused_graph_abi=lambda: 1,
        register_fused_graph_output=lambda: None,
        begin_fused_graph=lambda: None,
        end_fused_graph=lambda: None,
    )
    validate_graph_transport(runtime)
    runtime.fused_graph_abi = lambda: 2
    with pytest.raises(RuntimeError, match="ABI"):
        validate_graph_transport(runtime)
    del runtime.end_fused_graph
    with pytest.raises(RuntimeError, match="end_fused_graph"):
        validate_graph_transport(runtime)
