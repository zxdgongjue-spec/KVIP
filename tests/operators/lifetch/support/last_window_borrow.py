"""Validate the normal LI mailbox and audit Index on an independent terminal-only replay."""
from collections import Counter
import hashlib

import numpy as np


DUAL = 'aa6889e773c57ff737fd5ddec75307613d8949dc'
CANDIDATE = 'e308c913d10a77117665cb53ad7b56412ee6dc00'
# CANN 9.2 linked images of the full-length five-case CPU model.
EXPECTED_TEXT = {
    'v9_kernel/0': '9bc6c3db01bd17dbf758f9bb470259bae4ab3e0755d1686a81dbe1903cf07ac1',
    'v9_kernel/1': '7863fcf3cfc86d7cbedbc5f901f6f5b7af58493f17fa99fa1bacf9f8bf7307f9',
    'parallel_terminal_kernel/1': '5054a60bd3822bd0790b781620f21cbfd9b8816f6d08684adc1f540a9f1083b4',
}


def check_device(path):
    from check_isolated_device import extract
    hashes = {}
    for name, sections in extract(path).items():
        for key, instructions in sections.items():
            if key[4] == '.text':
                hashes[f'{name}/{key[1]}'] = hashlib.sha256(instructions).hexdigest()
    if hashes != EXPECTED_TEXT:
        raise ValueError(f'Loaded LI/terminal code differs from the verified window-borrow build: {hashes}')
    return hashes


def check_flags(li_word, terminal_word, detail):
    if int(li_word) & (1 << 31) or int(terminal_word) != int(detail):
        raise ValueError(f'Wrong LI/terminal diagnostics flags: {li_word:#x}, {terminal_word:#x}, detail={detail}')


def options(forward):
    def word(tensor, lo):
        return int.from_bytes(bytes(tensor[lo:lo + 4].cpu().tolist()), 'little')
    return word(forward.workspace, 96), word(forward.terminal.workspace, 52)


def set_terminal_only(forward, enabled):
    # This runs before graph replay. The four addresses captured by the graph are stable.
    forward.set_options(diagnostics=False)
    if enabled:
        forward.terminal.workspace[52] = 1
    check_flags(*options(forward), enabled)


def check_workspace(forward):
    r = forward.rotating
    s = r.runtime_shape
    b, h = int(s.batch), int(s.history)
    if (b, int(s.splitQ), int(s.rounds), int(s.serviceLanes), int(s.serviceAics), int(s.liCores)) != (16, 1, 4, 16, 8, 32):
        raise ValueError('The live LI geometry does not select the B24 row-local helper')
    storage, ranks = r.window_topk_storage, r.rank_progress
    offset = b * (512 + 4*2048*6 + 1024 + ((h + 255)//256)*32)
    base = storage.data_ptr()
    # The live URMA context is a clone. Its incremental pointer is fourth from the end.
    incremental = int(forward._context.view(forward.torch.int64)[-4].item())
    if (incremental != base or ranks.data_ptr() != base + offset or tuple(ranks.shape) != (b, 4, 32)
            or ranks.element_size() != 8 or not ranks.is_contiguous() or ranks.data_ptr() % 64
            or storage.numel()*storage.element_size() != offset + b*4*32*8):
        raise ValueError('Helper mailbox and LI incremental binding do not refer to the same allocated storage')


def decode_mailbox(records, resident, history):
    raw = np.asarray(records)
    if raw.shape != (16, 4, 32) or raw.dtype.kind not in 'ui' or raw.itemsize != 8:
        raise ValueError('Expected 16 rows of 4 x 32 native 64-bit progress words')
    rows = []
    for b in range(16):
        decision, r2, r3, empty, owner_q = map(int, raw[b, 2, 16:21])
        offered, done = map(int, raw[b, 3, 16:18])
        if (decision not in (1, 2) or offered not in (0, 1) or done != int(decision == 1)
                or owner_q not in (0, 1) or (decision == 2 and owner_q != 1)):
            raise ValueError(f'Row {b}: inconsistent last-window decision/completion')
        if decision == 1:
            if not offered or empty not in (0, 1):
                raise ValueError(f'Row {b}: delegated job has no valid offer')
            if any(not resident <= (x & 0xffffffff) <= history or (x >> 32) > history for x in (r2, r3)):
                raise ValueError(f'Row {b}: invalid ready range for delegated queries')
        # Retained rows never set ready2/ready3; do not decode stale payloads.
        rows.append(dict(row=b, delegated=decision == 1, decision_q=owner_q, offered=offered,
                         ready=[r2, r3] if decision == 1 else None, complete=done))
    return rows


def summarize(calls):
    if set(calls) != set(range(20)) or any([r['row'] for r in rows] != list(range(16)) for rows in calls.values()):
        raise ValueError('Expected each of 20 captured calls with 16 rows')
    selected = Counter(r['decision_q'] for rows in calls.values() for r in rows if r['delegated'])
    return dict(calls=20, rows=320, delegated=selected.total(), retained=320-selected.total(),
                before_first_q=selected[0], before_second_q=selected[1])
