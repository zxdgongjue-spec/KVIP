"""Host-only admission before ownership changes; device preflight is retained."""

from dataclasses import dataclass
from numbers import Integral

from .contract import BLOCK_TOKENS, TOPK


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str
    message: str


def _integers(values):
    return all(isinstance(v, Integral) and not isinstance(v, bool) for v in values)


def admit_request(
    *,
    queries_per_request,
    candidate_lengths,
    resident_budgets,
    logical_lengths=None,
    pool_rows=None,
    geometry=None,
    aic_count=None,
):
    """Use existing host metadata, never device scalar readback.

    Optional geometry/logical/pool/resource arguments allow callers to complete
    validation before owned-index import. With only the original three keyword
    arguments, this remains a structural check, not device-value certification.
    """

    def reject(reason, message):
        return SupportResult(False, reason, message)

    if not _integers((queries_per_request,)) or queries_per_request != 4:
        return reject(
            "query_outside_v9", "V9 requires uniform MTP3/Q4; MTP1/2 are not supported"
        )
    try:
        lengths, budgets = tuple(candidate_lengths), tuple(resident_budgets)
        logical = lengths if logical_lengths is None else tuple(logical_lengths)
        pools = None if pool_rows is None else tuple(pool_rows)
    except TypeError:
        return reject(
            "invalid_metadata", "host metadata must contain integer sequences"
        )
    batch = len(lengths)
    if not 1 <= batch <= 32:
        return reject("batch_outside_v9", "V9 requires B1..32")
    if (
        len(budgets) != batch
        or len(logical) != batch
        or (pools is not None and len(pools) != batch)
    ):
        return reject(
            "invalid_metadata", "one length/budget/pool entry per request is required"
        )
    if not _integers(lengths + budgets + logical + (() if pools is None else pools)):
        return reject(
            "invalid_metadata",
            "lengths, budgets and pool rows must be integers, not bool",
        )
    # Convert NumPy host scalars before arithmetic to avoid fixed-width overflow.
    lengths = tuple(map(int, lengths))
    budgets = tuple(map(int, budgets))
    logical = tuple(map(int, logical))
    pools = None if pools is None else tuple(map(int, pools))
    if any(
        not 0 <= value <= 0x7FFFFFFF
        for value in logical + (() if pools is None else pools)
    ):
        return reject(
            "invalid_metadata", "logical lengths and pool rows must fit int32"
        )
    if any(len(set(values)) != 1 for values in (lengths, budgets, logical)):
        return reject(
            "nonuniform",
            "V9 requires equal candidate/logical lengths and resident budgets",
        )
    if any(not 1 <= v <= 262144 for v in lengths) or any(
        not TOPK <= v <= 16384 for v in budgets
    ):
        return reject("invalid_metadata", "invalid candidate length or resident budget")
    if any(total < candidate for total, candidate in zip(logical, lengths)):
        return reject(
            "invalid_metadata", "logical length cannot precede the candidate end"
        )
    if pools is not None and (len(set(pools)) != batch or any(p < 0 for p in pools)):
        return reject("invalid_metadata", "pool rows must be distinct and nonnegative")
    if aic_count is not None:
        if (
            not _integers((aic_count,))
            or not 1 <= aic_count <= 32
            or batch > 2 * aic_count
        ):
            return reject(
                "insufficient_aiv", "batch exceeds the available AIC/AIV resources"
            )
    if geometry is not None:
        try:
            geometry.validate()
        except (ValueError, TypeError) as error:
            return reject("invalid_geometry", str(error))
        if geometry.batch != batch or geometry.queries != queries_per_request:
            return reject(
                "invalid_geometry",
                "request batch/query count differs from prepared geometry",
            )
        if (
            lengths[0] > geometry.candidate_capacity
            or budgets[0] > geometry.resident_capacity
        ):
            return reject(
                "capacity", "request exceeds prepared candidate/resident capacity"
            )
        if (
            budgets[0] + logical[0] - lengths[0]
            > geometry.resident_columns * BLOCK_TOKENS
        ):
            return reject("capacity", "resident page table does not cover the tail")
        if pools is not None and any(p >= geometry.max_requests for p in pools):
            return reject(
                "invalid_metadata", "pool row exceeds the real request capacity"
            )
    return SupportResult(True, "v9", "supported V9 ABI13 request")


def validate_request(**kwargs):
    """Backward-compatible raising wrapper around structured host admission."""
    result = admit_request(**kwargs)
    if not result.supported:
        raise ValueError(result.message)
