# SPDX-License-Identifier: Apache-2.0
"""One live BS1 graph, one frozen four-query GET plan, two doorbell policies."""
import hashlib
import numpy as np

from correctness import validate_terminal_receipts

TOP = 2048
LINE = 32
WINDOW_WORDS = LINE + TOP
QUERY_WORDS = LINE + 2 * TOP
FIXTURE_WORDS = LINE + 4 * WINDOW_WORDS + 4 * QUERY_WORDS


def completion_ends(count, row_total, single_batch):
    if count == 0:
        return []
    ends = [(count * k + 15) // 16 for k in (1, 6, 11)] if row_total >= 256 and not single_batch else []
    return sorted(set(ends + [count]))


def payload(raw, count, token_bytes, row_total, single_batch):
    """Strip only the SQ owner and completion control word. Verify actual posted cuts."""
    words = np.ascontiguousarray(raw).view("<u4").reshape(count, 16)
    expected = np.full(count, 0x30150000, dtype="<u4")
    for end in completion_ends(count, row_total, single_batch):
        expected[end - 1] = 0x30360000
    if np.any((words[:, 0] & 0x7fffffff) != expected) or np.any(words[:, 12] != token_bytes):
        raise ValueError("SQ control boundaries or actual GET length differ from the selected policy")
    if np.any((words[:, 10] == 0) & (words[:, 11] == 0)):
        raise ValueError("Null source in a submitted URMA GET")
    return words[:, 1:].copy()


class PostFixture:
    def __init__(self, state):
        self.state = state
        terminal = state.forward.terminal
        if state.batch != 1 or len(terminal.bindings) != 4 or state.rotating.rounds != 4:
            raise ValueError("Fixed GET comparison requires BS1, four windows and four private final SQ/CQ pairs")
        self.terminal = terminal
        self.stream = state.torch.npu.current_stream(state.device)
        self.snapshot = terminal.owner.lease.library.kvip_transport_snapshot
        self.data = state.torch.zeros(FIXTURE_WORDS, dtype=state.torch.int32, device=state.device)
        self.sq_copy = state.torch.empty(TOP * 64, dtype=state.torch.uint8, device=state.device)
        self.reference = None
        # The same device parameter remains bound after stream capture. Both
        # arms use the same real registered source/destination allocations.
        state.forward.workspace.view(state.torch.int64)[24] = self.data.data_ptr()
        self.record()

    def record(self):
        self.reference = None
        self.previous_heads = {}
        self.data.zero_()
        self.data[0] = 1
        self.state.single_post_probe = False
        self.stream.synchronize()

    def set_policy(self, policy):
        if self.reference is None or policy not in ("one", "four"):
            raise ValueError("Freeze a genuine full-graph seed before selecting its posting policy")
        self.state.single_post_probe = policy == "one"
        self.data[0] = 3 if policy == "one" else 2
        self.stream.synchronize()

    def _descriptors(self):
        state = self.state
        counts = state.rotating.final_posts[:, :, 0].cpu().numpy()
        statuses = self.terminal.owner.core_status.cpu().numpy().view("<u8")
        validate_terminal_receipts(counts, statuses, self.terminal.queue_views, single_batch=state.single_post_probe)
        n_total = int(counts.sum())
        lists, per_q, receipts = [], [], []
        for binding in self.terminal.bindings:
            count = int(counts[0, binding.query])
            report = statuses[binding.view]
            depth = binding.template[17]
            head, cq = map(int, report[4:6])
            previous = self.previous_heads.get(binding.view)
            if previous is not None and (head - previous[0], cq - previous[1]) != (count, int(report[3])):
                raise ValueError(f"Q{binding.query + 1} producer/consumer did not advance by the reported GETs/CQEs")
            self.previous_heads[binding.view] = (head, cq)
            if not 0 <= count <= TOP or count >= depth or head < count:
                raise ValueError("Invalid producer head/count in completed private SQ")
            start = (head - count) & (depth - 1)
            sq = binding.template[24] | (binding.template[25] << 32)
            first = min(count, depth - start)
            for index, length, dest in ((start, first, 0), (0, count - first, first)):
                if length:
                    code = self.snapshot(sq + index*64, self.sq_copy.data_ptr() + dest*64, length*64, self.stream.npu_stream)
                    if code:
                        raise RuntimeError(f"Cannot read completed SQ descriptors for Q{binding.query + 1}: rc={code}")
            raw = self.sq_copy[:count*64].cpu().numpy() if count else np.empty(0, dtype=np.uint8)
            lists.append(payload(raw, count, state.token_bytes, n_total, state.single_post_probe))
            per_q.append(count)
            receipts.append(int(report[3]))
        return lists, per_q, receipts

    def freeze(self):
        state = self.state
        fixture = self.data.cpu().numpy()
        windows = fixture[LINE:LINE+4*WINDOW_WORDS].reshape(4, WINDOW_WORDS)
        queries = fixture[LINE+4*WINDOW_WORDS:].reshape(4, QUERY_WORDS)
        if not np.all(windows[:, 1] == 1) or not np.all(queries[:, 5] == 1):
            raise ValueError("A real prefetch window or final query was missing from the seed")
        window_counts = windows[:, 0]
        if np.any(window_counts < 0) or np.any(window_counts > TOP):
            raise ValueError("Invalid seed window count")
        ordered_tokens = np.concatenate([row[LINE:LINE+n] for row, n in zip(windows, window_counts)])
        if len(np.unique(ordered_tokens)) != len(ordered_tokens) or np.any(
            (ordered_tokens < 0) | (ordered_tokens >= state.history)
        ):
            raise ValueError("Seed has repeated or out-of-range prefetch admissions")
        lists, per_q, receipts = self._descriptors()
        prior = len(ordered_tokens)
        if np.any(queries[:, 3] != prior) or list(queries[:, 0]) != per_q:
            raise ValueError("Recorded allocation or final SQ differs from the prefetch/terminal seed")
        slots = sorted(i for row in queries for i in range(int(row[1]), int(row[1] + row[0])))
        if slots != list(range(prior, prior + sum(per_q))):
            raise ValueError("Frozen query destinations do not partition the admitted slots")
        if np.any((queries[:, 4] < queries[:, 0]) | (queries[:, 4] > TOP)):
            raise ValueError("Invalid seed terminal candidate count")
        for row in queries:
            misses = row[LINE+TOP:LINE+TOP+int(row[4])]
            if np.any((misses < 0) | (misses >= state.history)):
                raise ValueError("Out-of-range physical position in seed misses")
        self.reference = lists
        self.prefetch = prior
        self.sha = hashlib.sha256()
        for descriptors in lists:
            self.sha.update(len(descriptors).to_bytes(4, "little"))
            self.sha.update(descriptors.tobytes())
        return dict(per_q_tokens=per_q, per_q_cqes=receipts, sha256=self.sha.hexdigest(), prefetch=prior)

    def verify(self):
        if self.reference is None:
            raise ValueError("Missing frozen reference")
        lists, counts, receipts = self._descriptors()
        for q, (actual, seed) in enumerate(zip(lists, self.reference), 1):
            if not np.array_equal(actual, seed):
                raise ValueError(f"Q{q} actual ordered SQ payload differs from frozen seed")
        prefetch = int(self.state.rotating.rows[0, 9].item())
        if prefetch != self.prefetch:
            raise ValueError("Prefetch traffic changed after freezing window admissions")
        return dict(per_q_tokens=counts, per_q_cqes=receipts, sha256=self.sha.hexdigest(), prefetch=prefetch)
