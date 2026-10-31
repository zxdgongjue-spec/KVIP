"""Versioned, immutable public contract; no Torch or NPU initialization."""

from dataclasses import dataclass

from kvop import __version__

from .contract import ABI_VERSION, BLOCK_TOKENS, TOKEN_BYTES, TOPK

CONTRACT_VERSION = "3.0.0"
CONTRACT_MAJOR = 3
CONTRACT_MINOR = 0
GRAPH_TRANSPORT_ABI = 1
INDEX_WRITER_ABI = 2
PROFILE = "v9-fused-selection"


class ContractMismatchError(RuntimeError):
    """The provider cannot satisfy the consumer before any device work."""


@dataclass(frozen=True)
class LayoutSpec:
    name: str = "dsa-c8-fp8-bf16-scales-v1"
    token_bytes: int = TOKEN_BYTES
    block_tokens: int = BLOCK_TOKENS
    topk: int = TOPK
    index_dim: int = 128
    kernel_heads: int = 64
    byte_order: str = "little"


@dataclass(frozen=True)
class Capabilities:
    package_version: str = __version__
    contract_version: str = CONTRACT_VERSION
    contract_major: int = CONTRACT_MAJOR
    contract_minor: int = CONTRACT_MINOR
    device_abi: int = ABI_VERSION
    graph_transport_abi: int = GRAPH_TRANSPORT_ABI
    index_writer_abi: int = INDEX_WRITER_ABI
    profile: str = PROFILE
    layout: LayoutSpec = LayoutSpec()
    model_heads: tuple[int, ...] = (32, 64)
    queries: tuple[int, ...] = (4,)
    max_batch: int = 32
    max_candidates: int = 262144
    max_resident: int = 16384
    equal_candidate_lengths: bool = True
    equal_logical_lengths: bool = True
    equal_budgets: bool = True
    features: frozenset[str] = frozenset(
        {
            "eager",
            "fixed-q4-graph",
            "noncontiguous-pages",
            "owned-index",
            "extra-feedback",
            "checked-completion",
            "exclusive-session-lease",
            "append-only-pages",
            "checked-page-rebind",
        }
    )
    transport: str = "kvaf"
    # Implementation availability is distinct from a hardware certification.
    hardware_certified: bool = False
    sdk_compatibility: str = (
        "require matching runtime/device SDK and validate built library"
    )


@dataclass(frozen=True)
class ContractRequirements:
    contract_major: int = CONTRACT_MAJOR
    minimum_minor: int = 0
    device_abi: int = ABI_VERSION
    profile: str = PROFILE
    layout: LayoutSpec = LayoutSpec()
    features: frozenset[str] = frozenset()
    require_hardware_certified: bool = False
    graph_transport_abi: int = GRAPH_TRANSPORT_ABI
    index_writer_abi: int = INDEX_WRITER_ABI


def query_capabilities() -> Capabilities:
    return Capabilities()


def negotiate_contract(requirements=None, *, provider=None) -> Capabilities:
    """Validate semantic compatibility; binary/lease validation remains mandatory."""
    requirements = ContractRequirements() if requirements is None else requirements
    provider = query_capabilities() if provider is None else provider
    checks = (
        (provider.contract_major == requirements.contract_major, "contract major"),
        (provider.contract_minor >= requirements.minimum_minor, "contract minor"),
        (provider.device_abi == requirements.device_abi, "device ABI"),
        (provider.profile == requirements.profile, "profile"),
        (provider.layout == requirements.layout, "layout"),
        (requirements.features <= provider.features, "required capabilities"),
        (
            "fixed-q4-graph" not in requirements.features
            or provider.graph_transport_abi == requirements.graph_transport_abi,
            "Graph transport ABI",
        ),
        (
            "owned-index" not in requirements.features
            or provider.index_writer_abi == requirements.index_writer_abi,
            "Index writer ABI",
        ),
        (
            not requirements.require_hardware_certified or provider.hardware_certified,
            "hardware certification",
        ),
        (provider.transport == "kvaf", "production transport"),
    )
    for supported, field in checks:
        if not supported:
            raise ContractMismatchError(f"KVOP incompatible {field}")
    return provider
