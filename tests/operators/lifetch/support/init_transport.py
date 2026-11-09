# SPDX-License-Identifier: Apache-2.0
"""Real CPU-service registration and transport lease lifecycle; no inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'python'))


def main(transport, argv=None):
    p = argparse.ArgumentParser(description=f'{transport} KVAF runtime and LIFetch lease initialization')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--kvaf-server', required=True)
    p.add_argument('--kvaf-root', type=Path)
    p.add_argument('--soc', help='CANN SoC; URMA only, detected from device when omitted')
    args = p.parse_args(argv)
    import acl
    import kvaf
    import torch
    import torch_npu  # noqa: F401
    from kvip.adapters.kvaf import KvafGetLease
    from kvip.adapters.ubmem import UbmemLease
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    state = ROOT / 'build/dependencies.json'
    root = (args.kvaf_root or (Path(json.loads(state.read_text())['kvaf']) if state.exists()
                               else ROOT / 'third_party/KVAF')).resolve()
    build = root / 'build-npu'
    if not Path(kvaf.__file__).resolve().is_relative_to(build / 'python/kvaf'):
        raise RuntimeError(f'Wrong KVAF package: {kvaf.__file__}; source scripts/env.sh for {root}')
    torch.npu.set_device(args.device)
    device = torch.device(f'npu:{args.device}')
    stream = torch.npu.current_stream(device)
    get, put = build / 'bin/kvaf_execute_batch_get_kernels.o', build / 'bin/kvaf_execute_batch_put_kernels.o'
    for path in (get, put):
        if not path.is_file():
            raise FileNotFoundError(f'KVAF kernel missing: {path}')
    kwargs = dict(device_id=args.device, npu_core_count=2, batch_put_kernel_object_path=str(put))
    if transport == 'urma':
        kwargs['jettys_per_core'] = 2
    else:
        kwargs['transport'] = 'ubmem'
    runtime = kvaf.NpuRuntime(str(get), 656, 128, **kwargs)
    lease = None
    try:
        print(f'START {transport}: connecting and allocating four KV blocks', flush=True)
        runtime.link_to(args.kvaf_server)
        block = 128 * 656
        slot = ((block + 4095) // 4096 * 4096) if transport == 'ubmem' else block
        runtime.initialize(4 * slot, slot)
        output = torch.zeros((128, 656), dtype=torch.uint8, device=device)
        if transport == 'urma':
            cann, headers = kvaf_build_paths(build)
            library, _ = build_libraries(cache_dir=build / 'lifetch', cann=cann, headers=headers,
                                         soc=args.soc or acl.get_soc_name())
            lease = KvafGetLease(torch, runtime, library, output, stream)
        else:
            lease = UbmemLease(torch, runtime, output, stream)
            lease.region(output)
        lease.check_stream()
        print(f'PASS {transport}: runtime, DDR registration and lease established; releasing', flush=True)
        lease.close()
        lease = None
        runtime.shutdown()
        print(f'PASS {transport}: lease released and runtime shut down', flush=True)
        return 0
    except Exception as error:
        print(f'FAIL {transport}: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        if lease is None and runtime.initialized:
            runtime.shutdown()
        raise
