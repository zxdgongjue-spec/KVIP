# SPDX-License-Identifier: Apache-2.0
"""Untimed post-state comparison for the physical-block scan experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from kvip.diagnostics.archive import array_digest


def configure_windows(archive, cold_windows):
    """Override scheduling only; never rewrite source arrays or input files."""
    from kvip.operators.lifetch.geometry import li_geometry, trigger_tokens

    if cold_windows != 8:
        raise ValueError("This audit fixes eight cold windows")
    original = archive.manifest["geometry"]
    g = dict(original, requested_windows=cold_windows + 1)
    layout = li_geometry(g["batch"], "auto", g["requested_windows"], g["aic_count"])
    if layout["intermediate_fetches"] != cold_windows + 1:
        raise ValueError("Insufficient LI cores for resident + eight cold windows")
    blocks = (g["history"] - g["resident"]) // 512
    half_blocks = (blocks + cold_windows - 1) // cold_windows
    if (2 * g["batch"] <= g["aic_count"] and g.get("heads", 32) == 32
            and g["resident"] % 512 == 0 and (g["history"] - g["resident"]) % 512 == 0
            and g["history"] - g["resident"] >= 512 * cold_windows
            and half_blocks * (2 * cold_windows - 1) < 2 * blocks):
        raise ValueError("Eight cold windows activate unsupported sequence-split scanning on this input; no NPU allocation performed")
    boundaries = trigger_tokens(g["batch"], g["history"], g["resident"],
                                "auto", cold_windows + 1, g["aic_count"])
    if len(boundaries) != 9 or boundaries[0] != g["resident"] or boundaries[-1] != g["history"]:
        raise ValueError("Unexpected native publication boundaries")
    g.update(rounds=cold_windows + 1, service_lanes=layout["service_aivs_per_window"])
    archive.manifest = dict(archive.manifest, geometry=g, layout_audit_schedule=dict(
        source_requested_windows=original["requested_windows"], cold_windows=8,
        resident_publications=1, native_rounds=9, boundaries=boundaries,
    ))


def _digest(parts):
    digest = hashlib.sha256()
    for part in parts:
        value = np.ascontiguousarray(part)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def summarize(state, oracle, address_delta):
    """Ignore invalid hot payloads and padding; compare live entries by slot and ID."""
    g = oracle.geometry
    owners = np.asarray(oracle.owners, dtype=np.int64)
    resident, width, stride = (g[k] for k in ("resident", "token_bytes", "output_stride"))
    ids = state["logical_ids"][owners].copy()
    valid = state["hot_valid"][owners] != 0
    ends = oracle.pre["query_ends"].reshape(len(owners), 4).max(axis=1)
    output = state["output"].reshape(-1, width)[:oracle.owner_count * stride]
    output = output.reshape(oracle.owner_count, stride, width)
    hot_map = np.where(valid, ids[:, :resident], -1)
    addresses = state["addresses"].view(np.uint64)[owners].copy()
    mask = addresses != 0
    base = np.uint64(oracle.ddr["region_base"] + address_delta)
    addresses[mask] -= base
    fields = {}
    for name, values in (("logical_ids", ids), ("keys", state["keys"][owners]),
                         ("key_scales", state["key_scales"][owners]), ("ddr_offsets", addresses),
                         ("address_valid", mask)):
        fields[name] = _digest([values[row, :int(end)] for row, end in enumerate(ends)])
    fields["hot_valid"] = array_digest(valid)
    fields["hot_slot_ids"] = array_digest(hot_map)
    physical_payload, canonical_ids, canonical_payload = [], [], []
    for row, owner in enumerate(owners):
        slots = np.flatnonzero(valid[row])
        logical = ids[row, slots]
        order = np.argsort(logical)
        physical_payload.append(output[owner, slots])
        canonical_ids.append(logical[order])
        canonical_payload.append(output[owner, slots[order]])
    fields["valid_hot_payload_by_slot"] = _digest(physical_payload)
    fields["hot_token_set"] = _digest(canonical_ids)
    fields["valid_hot_payload_by_token"] = _digest(canonical_payload)
    return dict(hashes=fields, ids=ids, valid=valid, hot_map=hot_map, ends=ends,
                hot_sets=canonical_ids, hot_valid_count=int(valid.sum()))


def compare(left, right):
    unequal = [key for key in left["hashes"] if left["hashes"][key] != right["hashes"][key]]
    changed_index = sum(int(np.count_nonzero(left["ids"][row, :int(end)] !=
                                             right["ids"][row, :int(end)]))
                        for row, end in enumerate(left["ends"]))
    set_delta = sum(len(np.setxor1d(a, b)) for a, b in zip(left["hot_sets"], right["hot_sets"]))
    return dict(
        persistent_state_equal=not unequal, different_fields=unequal,
        hot_tokens_equal="hot_token_set" not in unequal,
        hot_payload_by_token_equal="valid_hot_payload_by_token" not in unequal,
        changed_index_positions=changed_index,
        changed_hot_slots=int(np.count_nonzero(left["hot_map"] != right["hot_map"])),
        changed_hot_valid=int(np.count_nonzero(left["valid"] != right["valid"])),
        hot_token_symmetric_difference=int(set_delta),
    )


def capture(state, oracle, *, address_delta, variant, references):
    current = summarize(state, oracle, address_delta)
    comparisons = {name: compare(value, current) for name, value in references.items()}
    if variant in ("forward", "reverse"):
        references[variant] = current
    return dict(included_in_timing=False, hashes=current["hashes"],
                hot_valid_count=current["hot_valid_count"], comparisons=comparisons,
                ignored="Inactive owners, future Index positions, invalid hot payload bytes; no equality claim for ignored bytes")


def report(path):
    d = json.loads(Path(path).read_text())
    if d.get("status") != "pass":
        raise ValueError("Replay did not pass")
    cases = d["cases"]
    order = ["forward", "reverse", "reverse-end", "forward-end"]
    if d["execution_order"] != order:
        raise ValueError("Layout audit requires F/R/R/F order")
    print("NATIVE cold_windows=8 resident_publications=1 budget=64/512")
    print("case             mean_us    p50_us useful_pre  waste_MiB repair_MiB total_MiB")
    for name in order:
        c = cases[name]
        if c["geometry"]["rounds"] != 9:
            raise ValueError("Replay did not use eight cold windows")
        reads = c["reads_samples"]
        mean = lambda key: float(np.mean([r[key] for r in reads]))
        width = c["geometry"]["token_bytes"]
        print(f"{name:15} {c['timing']['mean_us']:9.3f} {c['timing']['p50_us']:9.3f} "
              f"{mean('prefetch')-mean('unused_prefetch'):10.1f} "
              f"{mean('unused_prefetch_bytes')/2**20:10.4f} "
              f"{mean('final')*width/2**20:10.4f} {mean('bytes')/2**20:9.4f}")
    for a, b in (("forward", "reverse"), ("forward", "forward-end"), ("reverse", "reverse-end")):
        c = cases[b]["layout_audit"]["comparisons"][a]
        print(f"STATE {a}/{b} equal={c['persistent_state_equal']} "
              f"hot_set_equal={c['hot_tokens_equal']} hot_slots_changed={c['changed_hot_slots']} "
              f"index_positions_changed={c['changed_index_positions']} "
              f"hot_set_delta={c['hot_token_symmetric_difference']}")
        if c["different_fields"]:
            print("  differing=" + ",".join(c["different_fields"]))
    for metric in ("mean_us", "p50_us"):
        f = [cases[n]["timing"][metric] for n in ("forward", "forward-end")]
        r = [cases[n]["timing"][metric] for n in ("reverse", "reverse-end")]
        print(f"TIMING {metric} conservative_gain={100*(1-max(r)/min(f)):+.2f}% "
              f"forward_drift={100*(f[1]/f[0]-1):+.2f}% reverse_drift={100*(r[1]/r[0]-1):+.2f}%")
    print("ORACLE_PASS ordered_TopK,thresholds,Index_ID_association,selected_KV_bytes,valid_hot_KV_bytes")
    if not cases["forward-end"]["layout_audit"]["comparisons"]["forward"]["persistent_state_equal"] or not cases["reverse-end"]["layout_audit"]["comparisons"]["reverse"]["persistent_state_equal"]:
        print("LAYOUT_NONDETERMINISTIC: same-direction repetitions differ; cross-direction differences alone are not causal proof")
    elif not cases["reverse"]["layout_audit"]["comparisons"]["forward"]["persistent_state_equal"]:
        print("LAYOUT_DIVERGED: frozen-input gains do not establish closed-loop cache evolution")
    else:
        print("LAYOUT_EQUAL_ON_THIS_INPUT: not a proof for all future service rounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    report(parser.parse_args().result)
