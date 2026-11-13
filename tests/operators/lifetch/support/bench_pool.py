"""One immutable synthetic DDR pool per benchmark worker, never per graph."""
from types import SimpleNamespace


class BenchPool:
    def __init__(self, config):
        self.config = config
        self.runtime = None
        self.table = None
        self.active = False
        self.borrows = 0
        self.tables = {}

    def borrow(self, torch, device, batch, history, workers):
        import kvaf
        import numpy as np
        from contract_payload import payload

        if self.active or not 1 <= batch <= max(self.config.batches) or history != self.config.history:
            raise ValueError('Invalid shared bench pool borrow')
        if self.runtime is None:
            self.torch = torch
            self.workers = 40 if self.config.transport == 'urma' else 2
            build = self.config.kvaf_root / 'build-npu'
            options = dict(device_id=int(device.index or 0), npu_core_count=self.workers,
                           batch_put_kernel_object_path=str(build / 'bin/kvaf_execute_batch_put_kernels.o'))
            if self.config.transport == 'urma':
                options['jettys_per_core'] = 2
            else:
                options['transport'] = 'ubmem'
            self.runtime = kvaf.NpuRuntime(str(build / 'bin/kvaf_execute_batch_get_kernels.o'), 656, 128, **options)
            self.runtime.link_to(self.config.kvaf_server)
            capacity = max(self.config.batches) * history // 128
            block_bytes = 128 * 656
            slot_bytes = ((block_bytes + 4095) // 4096 * 4096
                          if self.config.transport == 'ubmem' else block_bytes)
            self.runtime.initialize((capacity + 2) * slot_bytes, slot_bytes)
            self.cpu_payload = torch.from_numpy(np.stack([
                payload(row, np.arange(history)) for row in range(max(self.config.batches))]))
            # Bound staging memory; every DDR page is written exactly once.
            flat = self.cpu_payload.reshape(capacity, block_bytes)
            for begin in range(0, capacity, 128):
                end = min(begin + 128, capacity)
                source = flat[begin:end].to(device)
                self.runtime.batch_put(list(range(begin, end)), source,
                                       [i * block_bytes for i in range(end - begin)],
                                       [block_bytes] * (end - begin))
                torch.npu.synchronize()
            self.table = self.runtime.prepare_access_table(list(range(capacity)))
            self.view = self.runtime.access_table_view(self.table, device='npu')
            self.tables[max(self.config.batches)] = (self.table, self.view)
            print(f'BENCH POOL INIT transport={self.config.transport} max_batch={max(self.config.batches)} '
                  f'history={history} put_bytes={capacity * block_bytes} runtime_workers={self.workers}', flush=True)
        if workers > self.workers:
            raise ValueError('Shared bench pool has insufficient queues')
        self.active = True
        self.borrows += 1
        count = batch * history // 128
        print(f'BENCH POOL REUSE case={self.borrows} batch={batch} additional_put_bytes=0', flush=True)
        if batch not in self.tables:
            table = self.runtime.prepare_access_table(list(range(count)))
            self.tables[batch] = (table, self.runtime.access_table_view(table, device='npu'))
        table, view = self.tables[batch]
        return SimpleNamespace(runtime=self.runtime, table=table,
                               cpu_payload=self.cpu_payload[:batch],
                               device_table=view)

    def release(self):
        if not self.active:
            raise RuntimeError('Shared bench pool released without a borrower')
        self.active = False

    def close(self):
        if self.active:
            raise RuntimeError('Cannot close shared pool before verified case completion')
        if self.runtime is not None:
            self.torch.npu.synchronize()
            for table, _ in self.tables.values():
                self.runtime.release_access_table(table)
            self.runtime.shutdown()
            self.runtime = None
