# SPDX-License-Identifier: Apache-2.0
"""Post-timing cross-transport snapshots; addresses are process-specific."""
import numpy as np


def save(path, ids, slots, output, mapping, lru, rows, addresses):
    ids = np.asarray(ids).reshape(len(mapping), 4, 2048)
    slots = np.asarray(slots).reshape(ids.shape)
    order = np.argsort(ids, axis=-1, kind="stable")
    sorted_ids = np.take_along_axis(ids, order, axis=-1)
    sorted_slots = np.take_along_axis(slots, order, axis=-1)
    # Canonicalize by logical token, because TopK order and physical hot slots
    # are not required to match between independent concurrent executions.
    payload = np.stack([output[b][sorted_slots[b]] for b in range(len(mapping))])
    np.savez_compressed(path, ids=sorted_ids, slots=sorted_slots, payload=payload,
                        mapping=mapping, lru=lru, epochs=rows[:, 16:48], orders=rows[:, 214:246], addresses=addresses)


def compare(urma_path, ubmem_path, resident):
    with np.load(urma_path) as a, np.load(ubmem_path) as b:
        if not np.array_equal(a['ids'], b['ids']):
            raise AssertionError('cross-transport logical TopK sets differ')
        if not np.array_equal(a['payload'], b['payload']):
            raise AssertionError('cross-transport selected KV bytes differ')
        lines = ['logical_topk=identical selected_kv_bytes=identical']
        for row in range(len(a['mapping'])):
            am, bm = a['mapping'][row], b['mapping'][row]
            at, bt = np.flatnonzero(am >= 0), np.flatnonzero(bm >= 0)
            common = np.intersect1d(at, bt, assume_unique=True)
            def slot(m, lru, tokens):
                virtual = m[tokens].astype(np.int64) - resident
                if np.any((virtual < 0) | (virtual >= lru.shape[0])):
                    raise ValueError('prefetch mapping outside candidate slots')
                return lru[virtual]
            sa, sb = slot(am, a['lru'][row], common), slot(bm, b['lru'][row], common)
            def epoch_of(mapping, tokens, counts, orders):
                used = np.flatnonzero(counts)
                if len(np.unique(orders[used])) != len(used) or np.any(orders[used] == 0):
                    raise ValueError("invalid prefetch epoch execution order")
                used = used[np.argsort(orders[used])]
                by_ordinal = np.repeat(used, counts[used].astype(np.int64))
                all_ordinals = mapping[mapping >= 0].astype(np.int64) - resident
                ordinals = mapping[tokens].astype(np.int64) - resident
                if (len(by_ordinal) != len(all_ordinals)
                        or not np.array_equal(np.sort(all_ordinals), np.arange(len(all_ordinals)))
                        or np.any((ordinals < 0) | (ordinals >= len(by_ordinal)))):
                    raise ValueError("prefetch counters do not identify each candidate epoch")
                return by_ordinal[ordinals]
            ea = epoch_of(am, common, a['epochs'][row], a['orders'][row])
            eb = epoch_of(bm, common, b['epochs'][row], b['orders'][row])
            for name, snapshot, mapping, tokens in (("urma", a, am, at), ("ubmem", b, bm, bt)):
                epochs = epoch_of(mapping, tokens, snapshot['epochs'][row], snapshot['orders'][row])
                used = np.isin(tokens, snapshot['ids'][row].reshape(-1))
                breakdown = []
                for epoch in np.unique(epochs):
                    mask = epochs == epoch
                    breakdown.append(dict(epoch=int(epoch), hot=int(np.count_nonzero(mask & (tokens < resident))),
                                          cold=int(np.count_nonzero(mask & (tokens >= resident))),
                                          unused=int(np.count_nonzero(mask & ~used))))
                lines.append(f"row={row} {name}_window_breakdown={breakdown}")
            lines.append(f'row={row} urma={len(at)} ubmem={len(bt)} common={len(common)} '
                         f'urma_only={len(at)-len(common)} ubmem_only={len(bt)-len(common)} '
                         f'common_slot_differences={np.count_nonzero(sa != sb)} common_epoch_differences={np.count_nonzero(ea != eb)} '
                         f'epochs_urma={a["epochs"][row].tolist()} epochs_ubmem={b["epochs"][row].tolist()}')
            changed = np.flatnonzero((sa != sb) | (ea != eb))[:3]
            for i in changed:
                token = int(common[i])
                lines.append(f"  token={token} block={token//128} block_offset={token%128} "
                             f"urma_epoch={ea[i]} ubmem_epoch={eb[i]} urma_slot={sa[i]} ubmem_slot={sb[i]} "
                             f"urma_addr=0x{int(a['addresses'][row, token]):x} "
                             f"ubmem_addr=0x{int(b['addresses'][row, token]):x}")
        lines.append('raw DDR VAs differ across allocations; token identity gives logical block=token//128 and offset=token%128')
        lines.append('per-token epoch reconstructed from allocation ordinal and recorded epoch execution order')
        return lines
