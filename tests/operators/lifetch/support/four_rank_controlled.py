# SPDX-License-Identifier: Apache-2.0
"""Test-only ordered prefetch control and evidence from the ten measured device states."""
import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np


GUARD = Path(__file__).with_name("four_rank_guarded_jobs.inc")
SENTINEL = -1


def controlled_build(build):
    def isolated(source, normal_library, sdk, soc, policy, *, trace_sibling=False):
        if policy != "four-ranks" or trace_sibling:
            raise ValueError("The guarded context is only valid for the fixed four-window timing kernel")
        source = source.resolve()
        active = source.parents[4]
        rel = Path("csrc/operators/lightning_indexer_with_fetch")
        with TemporaryDirectory(prefix="kvip-guarded-li-") as tmp:
            tree = Path(tmp)
            copied = tree / source.relative_to(active)
            for name in ("include", "op_kernel"):
                shutil.copytree(active / rel / name, tree / rel / name)
            # Preserve relative includes into common/, lidu/, and the UBMem implementation.
            # Siblings are read-only compiler inputs from the selected native checkout.
            for sibling in (active / rel.parent).iterdir():
                if sibling.name != rel.name:
                    (tree / rel.parent / sibling.name).symlink_to(sibling, target_is_directory=sibling.is_dir())
            shutil.copytree(source, copied)
            h = tree / rel / "op_kernel/prefetch.h"
            text = h.read_text()
            anchor = "                DataCopy(jobs,ids,((count+7)/8)*8);"
            if text.count(anchor) != 1:
                raise ValueError("Unexpected native four-window queue construction")
            guard = GUARD.read_bytes()
            h.write_text(text.replace(anchor, guard.decode().rstrip()))
            # The isolated compiler cache hashes every file in this directory.
            (copied / "ordered_guard.inc").write_bytes(guard)
            launch = build(copied, normal_library, sdk, soc, policy)
            launch.frozen_guard_sha256 = hashlib.sha256(guard).hexdigest()
            return launch
    return isolated


def window_inputs(records, batch, history):
    ids = np.full((batch, 4, 256), SENTINEL, dtype="<i4")
    counts = np.zeros((batch, 4), dtype="<i4")
    needed = {f"r{r}_{t}{k}" for r in range(batch) for t in ("w", "q") for k in range(4)}
    if set(records) != needed:
        raise ValueError("Incomplete original per-row window or tail GET journal")
    for name, entries in records.items():
        if entries.dtype != np.dtype("<i4") or entries.ndim != 2 or entries.shape[1] != 5:
            raise ValueError(f"Invalid GET journal format in {name}")
    for r in range(batch):
        for k in range(4):
            src = records[f"r{r}_w{k}"][:, 0]
            if len(src) > 256 or np.any(src < 0) or np.any(src >= history) or np.unique(src).size != len(src):
                raise ValueError(f"Invalid source count, range or duplicate in row {r} window {k}")
            counts[r, k] = len(src)
            ids[r, k, :len(src)] = src
    return ids, counts


def extend_state(state, frozen, counts):
    torch, f = state.torch, state.forward
    if (state.history != 131328 or state.batch not in (20, 24, 28, 32)
            or state.rotating.runtime_shape.rounds != 4 or f.terminal is None
            or not getattr(state.large_window_launch, "frozen_guard_sha256", None)
            or f._context.numel() * f._context.element_size() != 352):
        raise ValueError("Unexpected URMA ABI or cold image for ordered prefetch control")
    fixed = torch.from_numpy(frozen).to(state.device)
    n = torch.from_numpy(counts).to(state.device)
    observed = torch.full_like(fixed, SENTINEL)
    status = torch.zeros((state.batch, 4, 16), dtype=torch.int32, device=state.device)
    trailer = torch.tensor([a.data_ptr() for a in (fixed, observed, n, status)],
                           dtype=torch.int64, device=state.device).view(torch.uint8)
    original = f._context
    bound = f.bindings.view(torch.int64)
    workspace = f.workspace.view(torch.int64)
    if bound.numel() != 16 or int(bound[14].item()) != original.data_ptr() or int(workspace[0].item()) != original.data_ptr():
        raise ValueError("List control would be attached to an unbound rotating context")
    extended = torch.cat((original, trailer))
    state.frozen_binding = (original, fixed, observed, n, status, extended)
    f._context = extended
    bound[14] = workspace[0] = extended.data_ptr()
    # Detailed terminal journaling is enabled for both controlled arms, including all timed calls.
    # It is mandatory to resolve every performed GET to its real source and destination.
    enable = f.set_diagnostics
    f.set_diagnostics = lambda enabled: enable(True)
    f.set_diagnostics(True)


def clear_evidence(state):
    f = state.forward
    state.frozen_binding[2].fill_(SENTINEL)
    state.frozen_binding[4].zero_()
    f.events_src.fill_(SENTINEL)
    f.events_dst.fill_(SENTINEL)
    f.terminal.times.zero_()


def list_digest(records, kind):
    h = hashlib.sha256()
    for name in sorted(records):
        if name.split("_")[1][0] != kind:
            continue
        item = records[name]
        h.update(name.encode() + b"\0")
        h.update(len(item).to_bytes(4, "little"))
        h.update(item.astype("<i4", copy=False).tobytes())
    return h.hexdigest()


def evidence(records, actual, counts, observed, status):
    """Compare every ordered tuple and separately audit pre-control membership and order."""
    result = dict(guard_failures=0, window_membership_mismatches=0, natural_window_order_mismatches=0,
                  ordered_prefetch_mismatches=0, ordered_tail_mismatches=0, first_mismatch=None)
    for name, ref in records.items():
        row, part = name.split("_")
        r, k = int(row[1:]), int(part[1:])
        is_window = part.startswith("w")
        if is_window:
            n = int(counts[r, k])
            if n < 0 or n > 256:
                raise ValueError("Window count cannot be represented by this four-window kernel")
            sources = observed[r, k, :n]
            # No jobs are issued for zero admissions, so the kernel never touches status in that case.
            expected_status = int(n != 0)
            result["guard_failures"] += int(status[r, k] != expected_status or n != len(ref))
            result["window_membership_mismatches"] += int(not np.array_equal(np.sort(sources), np.sort(ref[:, 0])))
            result["natural_window_order_mismatches"] += int(not np.array_equal(sources, ref[:, 0]))
        if np.array_equal(actual[name], ref):
            continue
        key = "ordered_prefetch_mismatches" if is_window else "ordered_tail_mismatches"
        result[key] += 1
        if result["first_mismatch"] is None:
            cmp = actual[name]
            m = min(len(cmp), len(ref))
            bad = np.flatnonzero((cmp[:m] != ref[:m]).any(axis=1))
            i = int(bad[0]) if len(bad) else m
            result["first_mismatch"] = dict(segment=name, ordinal=i, expected=ref[i].tolist() if i < len(ref) else None,
                                            actual=cmp[i].tolist() if i < len(cmp) else None)
    result["actual_window_sha256"] = list_digest(actual, "w")
    result["actual_final_sha256"] = list_digest(actual, "q")
    result["ordered_match"] = not any(result[k] for k in (
        "guard_failures", "window_membership_mismatches", "ordered_prefetch_mismatches", "ordered_tail_mismatches"))
    return result


def validate_physical_operands(records, inputs, pages):
    h, resident, width = (inputs.geometry[k] for k in ("history", "resident", "token_bytes"))
    for name, e in records.items():
        r = int(name.split("_")[0][1:])
        pos, token, page, offset, dst = e.T
        if (np.any(pos < 0) or np.any(pos >= h) or np.any(dst < 0) or np.any(dst >= resident)
                or np.any(token < 0) or np.any(token >= inputs.logical_address_valid.shape[1])
                or not np.array_equal(inputs.captured_layout[r, pos], token)
                or not np.all(inputs.logical_address_valid[r, token])
                or not np.array_equal(pages.page_indices[r, token // 128], page)
                or not np.array_equal(token % 128 * width, offset)):
            raise ValueError(f"Archived GET does not resolve to this capture's operands in {name}")


def validate_bound_addresses(state, records):
    """Compare the addresses at the actual native Get source-table indices before the first call."""
    torch = state.torch
    addresses = state.storage.addresses
    ctx = state.forward._context.view(torch.int64)
    row_owners = state.workspace.inputs.row_owners
    owners = row_owners.cpu().numpy().reshape(-1).astype(np.int64)
    if (int(ctx[23].item()) != row_owners.data_ptr()
            or int(ctx[24].item()) != addresses.data_ptr() or int(ctx[25].item()) != addresses.stride(0)
            or not addresses.is_contiguous() or owners.shape != (state.batch,)
            or np.any(owners < 0) or np.any(owners >= addresses.shape[0])):
        raise ValueError("Unexpected native per-owner Get address table binding")
    positions, expected = [], []
    for name, entries in records.items():
        r = int(name.split("_")[0][1:])
        pos = entries[:, 0].astype(np.int64)
        positions.append(owners[r] * addresses.stride(0) + pos)
        expected.append(state.logical_addresses[r, entries[:, 1]])
    index = torch.from_numpy(np.concatenate(positions)).to(state.device)
    if index.numel():
        actual = addresses.view(-1).index_select(0, index).cpu().numpy().view(np.uint64)
        if not np.array_equal(actual, np.concatenate(expected)) or np.any(actual == 0):
            raise ValueError("Archived GET would address different registered memory")
