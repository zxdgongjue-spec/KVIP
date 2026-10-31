import numpy as np
import pytest
from benchmark_contract import parse_args
from contract_fixture import make_fixture
from contract_simulator import Simulator, seed_output
from contract_verify import snapshot
from host_transport import HostTransport
from kvip.v9 import Status, native_source
from memory_build import prepare_source


def test_deferred_copy_and_lifetime():
    source = np.arange(4 * 656, dtype=np.int32).astype(np.uint8).reshape(4, 656)
    output = np.full_like(source, 0xA5)
    transport = HostTransport(source, output)
    transport.submit(1, 3)
    assert np.all(output == 0xA5)
    with pytest.raises(RuntimeError, match="release"):
        transport.close()
    transport.complete()
    np.testing.assert_array_equal(output[3], source[1])
    transport.close()
    transport.close()
    with pytest.raises(RuntimeError, match="unavailable"):
        transport.submit(0, 0)


def test_failed_completion_retains_storage():
    source = np.zeros((1, 656), np.uint8)
    output = np.ones_like(source)
    transport = HostTransport(source, output)
    transport.submit(0, 0)
    with pytest.raises(RuntimeError, match="unknown"):
        transport.complete(fail=True)
    assert transport.source is source and transport.output is output and transport.pending
    with pytest.raises(RuntimeError, match="release"):
        transport.close()


@pytest.mark.parametrize("failure", ["prefetch_cq", "final_cq"])
def test_failed_get_never_publishes_maps(failure):
    g, arrays, scores = make_fixture()
    seed_output(arrays, g)
    before = snapshot(arrays)
    model = Simulator(g, arrays)
    model.launch(scores, rounds=4, failure=failure)
    assert arrays["status"][0] == Status.COMMUNICATION
    for name in ("forward", "reverse", "cursors"):
        np.testing.assert_array_equal(arrays[name], before[name])
    model.launch(scores)
    assert arrays["status"][0] == Status.POISONED


def test_memory_build_does_not_modify_production(tmp_path):
    source = native_source()
    original = (source / "lifecycle.cpp").read_bytes()
    target = prepare_source(source, tmp_path / "test-source")
    assert (source / "lifecycle.cpp").read_bytes() == original
    assert "v9_test_transport" in (target / "bridge.cpp").read_text()
    assert "return TestDrainQueue" in (target / "lifecycle.cpp").read_text()
    assert not any("st_dev(" in p.read_text() for p in target.glob("*.h"))


def test_mock_cannot_claim_network_performance():
    with pytest.raises(SystemExit):
        parse_args(["--transport", "npu-memory", "--case-suite", "performance", "--native-baseline", "off"])
    a = parse_args(["--transport", "npu-memory", "--native-baseline", "off"])
    assert a.kvaf_server is None


def test_transport_marker_rejects_cross_loading():
    from types import SimpleNamespace

    from kvip.v9.launch import validate_transport_library

    production = SimpleNamespace()
    test = SimpleNamespace(transport="npu-memory-test")
    real_library = SimpleNamespace()
    mock_library = SimpleNamespace(v9_test_transport=lambda: 1)
    validate_transport_library(real_library, production)
    validate_transport_library(mock_library, test)
    with pytest.raises(ValueError, match="must match"):
        validate_transport_library(mock_library, production)
    with pytest.raises(ValueError, match="must match"):
        validate_transport_library(real_library, test)
