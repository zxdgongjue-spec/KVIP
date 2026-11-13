# SPDX-License-Identifier: Apache-2.0
"""Replay state bound to the production LIFetch adapters, with no kernel fork."""
import numpy as np


class FusedReplayState:
    """Own one path's Index/hot/LRU state for an entire captured sequence.

    Runtime, access table and compiled library are owned by the caller. All
    input uploads and CPU checks occur outside timed graph execution. A closed
    state must not have live graphs or unfinished device work.
    """
    def __init__(self, torch, runtime, table, library, initial, pages, addresses,
                 *, device, aics, windows, prefetch_tokens, soc):
        from kvip.adapters.kvaf import KvafGetLease, KvafQueueView, UbmemLease
        from kvip.cache.index_storage import IndexCacheStorage
        from kvip.operators.lifetch.binding import LIFetchBinding
        from streaming_forward import StreamingForward
        from kvip.operators.lifetch.index_update import DirectIndexUpdate, UbmemIndexUpdate
        from kvip.operators.lifetch.workspace import LIFetchWorkspace, UbmemWorkspace
        from kvip.operators.lifetch_ubmem.binding import UbmemBinding
        from kvip.operators.lifetch_ubmem.build import UbmemGeometry

        self.torch, self.device, self.pages, self.initial = torch, device, pages, initial
        self.logical_addresses = addresses
        g = initial.geometry
        self.batch, self.history, self.resident = g['batch'], g['history'], g['resident']
        self.stride, self.token_bytes = g['output_stride'], g['token_bytes']
        self.storage = IndexCacheStorage(torch, self.batch + 1, self.history, self.resident, device)
        common = dict(batch=self.batch, history=self.history, aics=aics, windows=windows,
                      device=device, index_heads=g['heads'])
        ubmem = runtime.transport == 'ubmem'
        self.workspace = (UbmemWorkspace(torch, resident=self.resident, **common) if ubmem
                          else LIFetchWorkspace(torch, **common))
        self.output = torch.zeros((self.batch + 1, self.stride, self.token_bytes), dtype=torch.uint8, device=device)
        output_rows = self.output.view(-1, self.token_bytes)
        stream = torch.npu.current_stream(device)
        self.workspace.inputs.prepare([4] * self.batch, query_owners=list(range(1, self.batch + 1)))
        self.lease = (UbmemLease(torch, runtime, output_rows, stream) if ubmem
                      else KvafGetLease(torch, runtime, library, output_rows, stream))
        try:
            if not ubmem:
                self.workspace.allocate_rotation(
                    torch, batch=self.batch, history=self.history, resident=self.resident,
                    aics=aics, windows=windows, status_bytes=self.lease.sizes['status'])
            binding = dict(
                query_ends=self.workspace.inputs.query_ends.view(self.batch, 4),
                hot_valid=self.storage.hot_valid, row_owners=self.workspace.inputs.row_owners,
                token_addresses=self.storage.addresses, tail_bounds=self.workspace.tail_bounds,
                output_stride=self.stride, scratch=self.workspace.scratch,
            )
            if ubmem:
                self.lease.register_table(table)
                shape = UbmemGeometry(self.batch, self.history, self.resident, g['heads'],
                                      self.token_bytes, aics, windows, soc)
                self.rotating = UbmemBinding(torch, library, self.workspace.arguments(self.storage),
                                              self.lease, output_rows, shape=shape,
                                              prefetch_limit=prefetch_tokens, **binding)
            else:
                owner = KvafQueueView(self.lease, output_rows, table.addresses, table.valid)
                self.rotating = LIFetchBinding(
                    torch, library, self.workspace.arguments(self.storage), owner,
                    batch=self.batch, history=self.history, resident=self.resident,
                    intermediate_fetches=windows, prefetch_limit=prefetch_tokens,
                    aic_count=aics, token_bytes=self.token_bytes, **binding)
            update_type = UbmemIndexUpdate if ubmem else DirectIndexUpdate
            self.update = update_type(
                torch, keys=self.storage.keys, scales=self.storage.scales,
                addresses=self.storage.addresses, logical_ids=self.storage.logical_ids,
                row_owners=self.workspace.inputs.row_owners, rotating=self.rotating,
                query_inputs=self.workspace.inputs,
                status=torch.zeros(self.batch, dtype=torch.int32, device=device),
                outputs=torch.zeros((2, self.batch * 4, 1, 2048), dtype=torch.int32, device=device))
            self.forward = StreamingForward(self.rotating, self.update, torch, prefetch_tokens)
            self.reset()
        except BaseException:
            self.lease.close()
            raise

    def upload(self, target, array):
        self_value = self.torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
        target.copy_(self_value.reshape(target.shape))

    def reset(self):
        """Whole-sequence reset only. Never reset KVAF queues."""
        from kvip.cache.lru import SlotLRU
        initial = self.initial
        self.upload(self.storage.logical_ids[1:], initial.captured_layout.astype(np.int64))
        self.upload(self.storage.hot_valid[1:], initial.captured_hot_valid.astype(np.int32))
        self.output.zero_()
        self.policies = []
        for row in range(self.batch):
            self.upload(self.output[row + 1, :self.resident],
                        self.pages.token_payload(row, initial.captured_layout[row, :self.resident]))
            policy = SlotLRU()
            policy.resize(self.resident, self.resident)
            # The capture exports 2048 oldest candidates, not a full LRU chain.
            # Preserve that ordered prefix; initialize unknown remainder by slot.
            oldest = initial.captured_lru[row]
            remaining = np.setdiff1d(np.arange(self.resident), oldest, assume_unique=True)
            chain = np.concatenate((oldest, remaining)).astype(np.int32)
            policy.next[0, chain] = np.r_[chain[1:], -1]
            policy.previous[0, chain] = np.r_[-1, chain[:-1]]
            policy.head[0], policy.tail[0] = chain[0], chain[-1]
            policy._select()
            self.policies.append(policy)
        self.prepare(initial)

    def prepare(self, inputs):
        """Import real inputs while retaining this path's latest cache layout."""
        if inputs.identities != self.initial.identities or inputs.geometry != self.initial.geometry:
            raise ValueError('Replay request geometry changed after binding')
        layout = self.storage.logical_ids[1:].cpu().numpy()
        keys, scales, valid = inputs.index_for_layout(layout)
        self.upload(self.storage.keys[1:], keys)
        self.upload(self.storage.scales[1:], scales)
        rows = np.arange(self.batch)[:, None]
        addresses = self.logical_addresses[rows, layout].copy()
        addresses[valid == 0] = 0
        self.upload(self.storage.addresses[1:], addresses.astype(np.int64))
        self.upload(self.workspace.inputs.query.view(self.torch.uint8), inputs.query)
        # Reinterpret captured BF16 bits; numeric conversion changes the weights.
        self.upload(self.workspace.inputs.weights.view(self.torch.int16), inputs.weights.view(np.int16))
        self.upload(self.workspace.inputs.scale, inputs.query_scale)
        self.upload(self.workspace.inputs.query_ends, inputs.query_ends)
        self.upload(self.workspace.tail_bounds, inputs.tail_bounds)
        for row, (begin, end) in enumerate(inputs.tail_bounds):
            ids = np.arange(int(begin), int(end))
            capacity = self.stride - self.resident
            if capacity <= 0 or len(ids) > capacity:
                raise ValueError('Captured ABC range exceeds replay output stride')
            slots = self.resident + ids % capacity
            payload = self.torch.from_numpy(self.pages.token_payload(row, ids)).to(self.device)
            self.output[row + 1, self.torch.from_numpy(slots).to(self.device)] = payload
        candidates = np.stack([policy.slots[0] for policy in self.policies])
        reserved = np.full((self.batch, self.resident), -1, dtype=np.int32)
        for row, slots in enumerate(candidates):
            good = slots >= 0
            reserved[row, slots[good]] = np.arange(len(slots), dtype=np.int32)[good]
        self.upload(self.rotating.lru_slots, candidates)
        self.upload(self.rotating.reserved_hot, reserved)

    def validate_index(self, inputs, output):
        """Reject damaged persistent state before the next prepare can overwrite it."""
        layout = self.storage.logical_ids[1:].cpu().numpy()
        try:
            keys, scales, valid = inputs.index_for_layout(layout)
        except (ValueError, AssertionError) as error:
            raise RuntimeError(f'Replay Index layout invalid: {error}') from error
        addresses = self.logical_addresses[np.arange(self.batch)[:, None], layout].copy()
        addresses[valid == 0] = 0
        for name, expected in (('keys', keys), ('scales', scales), ('addresses', addresses)):
            actual = getattr(self.storage, name)[1:].cpu().numpy()
            if not np.array_equal(actual, expected):
                raise RuntimeError(f'Replay persistent Index {name} mismatch')
        hot_valid = self.storage.hot_valid[1:].cpu().numpy()
        if np.any((hot_valid != 0) & (hot_valid != 1)):
            raise RuntimeError('Replay hot validity outside boolean domain')
        for row in range(self.batch):
            slots = np.flatnonzero(hot_valid[row])
            expected = self.pages.token_payload(row, layout[row, slots])
            if not np.array_equal(output[row, slots], expected):
                raise RuntimeError(f'Replay persistent hot KV mismatch: request={row}')
        return layout, hot_valid

    def validate(self, inputs):
        """Check completion before using outputs; then full-byte KV identity."""
        status = self.update.status.cpu().numpy()
        if np.any(status):
            raise RuntimeError(f'Replay Index/Fetch completion failed: {status.tolist()}')
        ids = self.update.logical_output.cpu().numpy().reshape(self.batch, 4, 2048)
        slots = self.update.slot_output.cpu().numpy().reshape(self.batch, 4, 2048)
        if np.any(ids < 0) or np.any(ids >= self.history) or np.any(slots < 0) or np.any(slots >= self.stride):
            raise RuntimeError('Replay output token/slot outside allocated bounds')
        output = self.output[1:].cpu().numpy()
        layout, hot_valid = self.validate_index(inputs, output)
        for row in range(self.batch):
            mask = slots[row] < self.resident
            selected_slots = slots[row][mask]
            if (np.any(hot_valid[row, selected_slots] == 0)
                    or not np.array_equal(layout[row, selected_slots], ids[row][mask])):
                raise RuntimeError(f'Replay selected hot mapping invalid: request={row}')
        failures = []
        try:
            inputs.validate_topk(ids)
        except AssertionError as error:
            failures.append(str(error))
        for row in range(self.batch):
            expected = self.pages.token_payload(row, ids[row].reshape(-1))
            actual = output[row, slots[row].reshape(-1)]
            if not np.array_equal(actual, expected):
                failures.append(f'KV bytes: request={row}')
            hot = slots[row].reshape(-1)
            self.policies[row].commit(hot[hot < self.resident])
        assert not failures, f'Replay round={inputs.source_round}: ' + '; '.join(failures)

    def close(self):
        self.lease.close()


class NativeReplayState:
    """Original LI/LIDU/BatchGet, with independent persistent logical maps."""
    upload = FusedReplayState.upload

    def __init__(self, torch, torch_npu, runtime, table, initial, pages, *, device):
        from types import SimpleNamespace
        from native_mtp import NativeMtp
        from real_fetch_inputs import native_li

        self.torch, self.torch_npu, self.device = torch, torch_npu, device
        self.initial, self.pages, self.native_li = initial, pages, native_li
        g = initial.geometry
        self.batch, self.history, self.resident, self.stride = (
            g[name] for name in ('batch', 'history', 'resident', 'output_stride'))
        heads, width = g['heads'], g['token_bytes']
        if self.stride % 128:
            raise ValueError('Native replay output stride must align to a KV block')
        self.query = torch.empty((self.batch, 4, heads, 128), dtype=torch.float8_e4m3fn, device=device)
        self.keys = torch.empty((self.batch, self.history, 128), dtype=torch.uint8, device=device)
        self.scales = torch.empty((self.batch, self.history), dtype=torch.float32, device=device)
        self.weights = torch.empty((self.batch, 4, heads), dtype=torch.bfloat16, device=device)
        self.query_scale = torch.empty((self.batch, 4, heads), dtype=torch.float32, device=device)
        lengths = torch.full((self.batch,), self.history, dtype=torch.int32, device=device)
        self.fixture = dict(
            sparse_mode=3, query=self.query, key=self.keys.view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
            actual_query=torch.arange(1, self.batch + 1, dtype=torch.int32, device=device) * 4,
            actual_key=lengths, li_query_lengths=torch.full((self.batch,), 4, dtype=torch.int32, device=device),
            block_table=torch.arange(self.batch * self.history // 128, dtype=torch.int32, device=device).reshape(self.batch, -1),
            native_inputs=dict(weights=self.weights, query_scale=self.query_scale,
                               key_scale=self.scales.reshape(-1, 128, 1)))
        mapping = np.full((self.batch, self.history), -1, dtype=np.int32)
        for row in range(self.batch):
            hot = initial.captured_hot_valid[row] != 0
            mapping[row, initial.captured_layout[row, :self.resident][hot]] = np.flatnonzero(hot)
        session = SimpleNamespace(
            runtime=runtime, device_table=table, batch_get_attempted=False,
            cpu_payload=torch.from_numpy(np.stack([pages.token_payload(row, np.arange(self.history))
                                                  for row in range(self.batch)])))
        owner = SimpleNamespace(batch=self.batch, resident_budget=self.stride,
                                cache_map=torch.from_numpy(mapping).to(device), lease=None, session=session)
        self.native = NativeMtp(torch, device, owner, self.fixture, source_tokens=self.history,
                                padded_tokens=self.history, token_bytes=width)
        self.native.bitmap = torch.zeros((self.batch, 4176), dtype=torch.int32, device=device)
        self.table = table
        self.reset()

    def reset(self):
        from kvip.cache.lru import SlotLRU
        self.native.reset()
        self.policies = []
        for row in range(self.batch):
            policy = SlotLRU()
            policy.resize(self.resident, self.resident)
            oldest = self.initial.captured_lru[row]
            remainder = np.setdiff1d(np.arange(self.resident), oldest, assume_unique=True)
            order = np.concatenate((oldest, remainder)).astype(np.int32)
            policy.next[0, order] = np.r_[order[1:], -1]
            policy.previous[0, order] = np.r_[-1, order[:-1]]
            policy.head[0], policy.tail[0] = order[0], order[-1]
            policy._select()
            self.policies.append(policy)
        self.prepare(self.initial)

    def prepare(self, inputs):
        if inputs.identities != self.initial.identities or inputs.geometry != self.initial.geometry:
            raise ValueError('Native replay request geometry changed')
        self.upload(self.query.view(self.torch.uint8), inputs.query)
        self.upload(self.keys, inputs.logical_keys)
        self.upload(self.scales, inputs.logical_scales)
        self.upload(self.weights.view(self.torch.int16), inputs.weights.view(np.int16))
        self.upload(self.query_scale, inputs.query_scale)
        # Native sparse mode aligns Q4 to the final key length.
        if not np.all(np.diff(inputs.query_ends, axis=1) == 1):
            raise ValueError('Native replay requires consecutive causal Q4 positions')
        self.upload(self.fixture['actual_key'], inputs.query_ends[:, -1])
        forward = self.native.forward.cpu().numpy()
        reverse = self.native.reverse.cpu().numpy()
        output = self.native.output.view(self.batch, self.stride, -1)
        tail_capacity = self.stride - self.resident
        for row, (begin, end) in enumerate(inputs.tail_bounds):
            # Expired ring entries cannot remain as false hits after wraparound.
            forward[row, forward[row] >= self.resident] = -1
            reverse[row, self.resident:] = -1
            ids = np.arange(int(begin), int(end))
            if tail_capacity <= 0 or len(ids) > tail_capacity:
                raise ValueError('Native replay ABC exceeds bound output')
            slots = self.resident + ids % tail_capacity
            forward[row, ids] = slots
            reverse[row, slots] = ids
            output[row, self.torch.from_numpy(slots).to(self.device)] = self.torch.from_numpy(
                self.pages.token_payload(row, ids)).to(self.device)
        self.upload(self.native.forward, forward)
        self.upload(self.native.reverse, reverse)
        self.native.cursors.zero_()
        self.native.bitmap.zero_()
        for row, policy in enumerate(self.policies):
            self.upload(self.native.preferred[row], policy.slots[0])
            self.upload(self.native.bitmap[row, :policy.bitmap.shape[1]], policy.bitmap[0])
        published = inputs.logical_address_valid.reshape(self.batch, -1, 128)[:, :, 0]
        self.upload(self.table.valid, published.reshape(-1))

    def forward(self):
        self.selected = self.native_li(self.torch_npu, self.fixture)
        self.native.launch(self.selected)

    def validate(self, inputs):
        ids = self.selected.cpu().numpy().reshape(self.batch, 4, 2048)
        if np.any(ids < 0) or np.any(ids >= self.history):
            raise RuntimeError('Native replay output token outside allocated bounds')
        failures = []
        try:
            inputs.validate_topk(ids)
        except AssertionError as error:
            failures.append(str(error))
        try:
            self.native.verify(self.selected)
        except ValueError as error:
            if str(error).startswith(('native_mtp_payload_mismatch:', 'native_mtp_exported_slots_mismatch:')):
                failures.append(str(error))
            else:
                raise
        mapping = self.native.forward.cpu().numpy()
        for row, policy in enumerate(self.policies):
            slots = mapping[row, ids[row].reshape(-1)]
            policy.commit(slots[(slots >= 0) & (slots < self.resident)])
        assert not failures, '; '.join(failures)

    def close(self):
        pass  # No lease: caller owns the ordinary KVAF Runtime and table.
