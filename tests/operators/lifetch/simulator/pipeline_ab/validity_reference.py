"""Model physical KV ownership vs the streaming Index permutation."""


def reconcile_hot_valid(rows, initial_valid, records, resident):
    """Invalidate changed hot slots without a matching post-fetch KV owner.

    `rows[slot]` is the original Index row identity at the end of the round.
    `records` are the final active producer records `(slot, original_row)`;
    the caller must already have replaced overwritten prefetch records.
    """
    if len(initial_valid) != resident or len(rows) < resident:
        raise ValueError('invalid geometry')
    written = {}
    for slot, token in records:
        if not 0 <= slot < resident or not 0 <= token < len(rows):
            raise ValueError('record outside storage')
        if slot in written:
            raise ValueError('duplicate destination')
        written[slot] = token
    result = list(initial_valid)
    for slot in range(resident):
        expected = written.get(slot, slot)
        if rows[slot] != expected:
            result[slot] = 0
            if slot in written:
                raise ValueError('written KV does not match final Index owner')
    return result
