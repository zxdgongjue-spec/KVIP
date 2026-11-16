# SPDX-License-Identifier: Apache-2.0
"""Untimed control preparation and checks for device-owned route feedback.

No miss oracle or host mode decision is copied to the device. A frozen snapshot
is restored each replay; feedback alone evolves across successful steps.
"""

import numpy as np

from kvip.diagnostics.archive import TensorArchive, cpu_array

VALID_FEEDBACK = 0x4B565250


def requests_for(archive, *, generation=1, step=0):
    batch = archive.manifest["geometry"]["batch"]
    values = np.zeros((batch, 4), dtype=np.int64)
    values[:, 0] = archive.array("pre.metadata")[4].view(np.int32)[:batch]
    values[:, 1:3] = (generation, step)
    values[:, 3] = np.count_nonzero(archive.array("pre.query_ends").reshape(batch, 4) > 0, axis=1)
    return values


class RowPermutation:
    """Repack requests while preserving TND output order and owner storage."""

    tensor = TensorArchive.tensor

    def __init__(self, archive, order):
        self.original = archive
        self.manifest, self.directory = archive.manifest, archive.directory
        batch = self.manifest["geometry"]["batch"]
        self.order = np.asarray(order, dtype=np.int64)
        if self.order.shape != (batch,) or not np.array_equal(np.sort(self.order), np.arange(batch)):
            raise ValueError("Packed row order must be a complete batch permutation")
        self.inverse = np.argsort(self.order)

    def array(self, name):
        value = self.original.array(name)
        batch = len(self.order)
        if name == "pre.metadata":
            result = value.copy()
            for field in (0, 1):
                result[field] = value[field].reshape(batch, 4)[self.order].reshape(-1)
            active = value[3] != 0
            source = value[2, active]
            result[2, active] = self.inverse[source // 4] * 4 + source % 4
            result[4].view(np.int32)[:batch] = value[4].view(np.int32)[:batch][self.order]
            return result
        if name in ("pre.query", "pre.weights", "pre.query_scale"):
            return np.ascontiguousarray(value.reshape(batch, 4, *value.shape[1:])[self.order].reshape(value.shape))
        if name == "pre.query_ends":
            return np.ascontiguousarray(value.reshape(batch, 4)[self.order].reshape(value.shape))
        if name in ("pre.query_lengths", "pre.key_lengths", "pre.block_table", "pre.lru_slots",
                    "pre.reserved_hot", "pre.tail_bounds", "pre.threshold_bounds", "post.score_keys"):
            return np.ascontiguousarray(value[self.order])
        return value


class AutoReplay:
    """Prepare identities; observe, but never choose or overwrite device mode."""

    def __init__(self, torch, bundle, archive):
        self.torch, self.bundle, self.archive = torch, bundle, archive
        self.state = bundle.rotating.auto_route
        self.addresses = self._addresses()
        self.requests = requests_for(archive)
        # The freshly allocated device table is zero. Subsequent host shadows
        # come only from existing post-replay correctness readbacks; preparation
        # never waits for or reads the NPU to choose a mode.
        self.feedback_snapshot = np.zeros((self.state.owners, 16), dtype=np.uint64)
        self.before = None
        self.latest = None
        self.begin(step=0, learning=False)

    def _addresses(self):
        result = {name: getattr(self.state, name).data_ptr() for name in ("config", "requests", "feedback", "pending")}
        result["admission_controls"] = self.bundle.rotating.admission_controls.data_ptr()
        return result

    def begin(self, *, step, learning, requests=None):
        self.before = self.feedback_snapshot.copy()
        self.requests = (requests_for(self.archive, step=step) if requests is None else np.asarray(requests).copy())
        self.learning = learning
        self.state.set_requests(self.requests)
        self.state.set_learning(learning)
        if self._addresses() != self.addresses:
            raise AssertionError("Auto route changed captured state addresses")

    def observe(self):
        pending = cpu_array(self.torch, self.state.pending).astype(np.uint64)
        feedback = cpu_array(self.torch, self.state.feedback).astype(np.uint64)
        controls = cpu_array(self.torch, self.bundle.rotating.admission_controls)
        if not np.all(controls == controls[0]):
            raise AssertionError("Auto route did not choose a uniform graph mode")
        low = bool(controls[0, 0] == 0)
        if not np.all(pending[:, 4] == int(not low)):
            raise AssertionError("Device pending mode disagrees with effective prefetch controls")
        self.bundle.tuning["auto_low_active"] = low
        if not self.learning and not np.array_equal(feedback, self.before):
            raise AssertionError("Capture/warmup changed learned feedback")
        if not self.learning and low:
            raise AssertionError("Learning-disabled replay must use the conservative full path")
        changed = np.flatnonzero(np.any(feedback != self.before, axis=1))
        learned = []
        for row, request in enumerate(self.requests):
            owner, generation, step, _ = map(int, request)
            if pending[row, 3]:
                state = feedback[owner]
                if (int(state[7]) != VALID_FEEDBACK or int(state[0]) != generation
                        or int(state[1]) != step or int(state[2]) != int(pending[row, 9])):
                    raise AssertionError("Successful device feedback does not match its submitted identity/Get count")
                learned.append(owner)
            elif owner in changed:
                raise AssertionError("Rejected or duplicate request updated learned feedback")
        if not set(map(int, changed)) <= set(learned):
            raise AssertionError("Device feedback changed an unrelated owner")
        self.feedback_snapshot = feedback.copy()
        self.latest = dict(mode="low" if low else "high", controls=controls[0].tolist(),
                           steps=self.requests[:, 2].tolist(), reasons=pending[:, 5].tolist(),
                           learned_owners=learned, necessary_get=int(pending[:, 9].sum()),
                           useful_prefetch=int(pending[:, 10].sum()),
                           feedback_valid_owners=int(np.count_nonzero(feedback[:, 7] == VALID_FEEDBACK)))
        return self.latest

    def inject_prediction_fault(self, necessary_get):
        """Untimed fault test only; never called by normal replay learning."""
        self.state.feedback[:, 2].fill_(necessary_get)
        self.feedback_snapshot[:, 2] = necessary_get

    def record(self, reads):
        observed = dict(self.latest)
        # Independent traffic accounting is used only after replay for checks;
        # these values never enter metadata or seed the device predictor.
        observed.update(observed_prefetch=reads["prefetch"], observed_final=reads["final"],
                        observed_useful_get=reads["useful_transfers"])
        if observed["necessary_get"] != reads["useful_transfers"]:
            raise AssertionError("Device learned Get count differs from verified useful transfers")
        return observed
