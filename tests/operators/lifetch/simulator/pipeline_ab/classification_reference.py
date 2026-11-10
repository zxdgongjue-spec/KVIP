"""Reference for the final TopK classification planned for LI's vector UB."""
MISSING = 0xFFFFFFFF


def classify(tokens, resident, history, valid, reserved, mapping, lru, used, tail=None):
    if len(valid) != resident or len(reserved) != resident or len(mapping) != history:
        raise ValueError('invalid metadata geometry')
    if not 0 <= used <= len(lru):
        raise ValueError('invalid prefetch count')
    if tail is not None and not 0 <= tail[0] <= tail[1] <= history:
        raise ValueError('invalid tail interval')
    slots, bitmap = [], [0] * ((resident + 31) // 32)
    for token in tokens:
        if token == MISSING:
            slots.append(MISSING)
            continue
        if token < 0 or token >= history:
            raise ValueError('token out of bounds')
        is_tail = tail is not None and tail[0] <= token < tail[1]
        if token < resident and not is_tail and valid[token] and reserved[token] < 0:
            slot = token
        else:
            slot = mapping[token] if used or tail is not None else MISSING
            if resident <= slot < resident + 2048:
                ordinal = slot - resident
                if ordinal >= len(lru):
                    raise ValueError('mapped LRU ordinal out of bounds')
                slot = lru[ordinal]
            elif resident + 8192 <= slot < 0x80000000:
                slot -= 8192
        slots.append(slot)
        if slot < resident:
            bitmap[slot // 32] |= 1 << (slot % 32)
    return slots, bitmap
