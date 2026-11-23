"""Compare measured fused outputs to actual npu_quant_lightning_indexer outputs."""
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from reporting import atomic_json

MODES = ('strong_serial', 'strong_serial_repeat', 'same_layout_native')
CACHE_SCHEMA = 2


class NativeReplayLI:
    """The serial replay's BSND query and PA_BSND Index operands, without the fetch."""
    def __init__(self, torch, device, geometry):
        self.torch = torch
        self.device = device
        b, h, heads = (geometry[k] for k in ('batch', 'history', 'heads'))
        self.keys = torch.empty((b, h, 128), dtype=torch.uint8, device=device)
        self.scales = torch.empty((b, h), dtype=torch.float32, device=device)
        self.query = torch.empty((b, 4, heads, 128), dtype=torch.float8_e4m3fn, device=device)
        self.weights = torch.empty((b, 4, heads), dtype=torch.bfloat16, device=device)
        self.query_scale = torch.empty((b, 4, heads), dtype=torch.float32, device=device)
        self.fixture = dict(
            sparse_mode=geometry.get('sparse_mode', 3), query=self.query,
            key=self.keys.view(torch.float8_e4m3fn).reshape(-1, 128, 1, 128),
            actual_query=torch.arange(1, b + 1, dtype=torch.int32, device=device) * 4,
            actual_key=torch.full((b,), h, dtype=torch.int32, device=device),
            li_query_lengths=torch.full((b,), 4, dtype=torch.int32, device=device),
            block_table=torch.arange(b * h // 128, dtype=torch.int32, device=device).reshape(b, -1),
            native_inputs=dict(weights=self.weights, query_scale=self.query_scale,
                               key_scale=self.scales.reshape(-1, 128, 1)))

    def prepare(self, inputs, *, physical=False):
        torch = self.torch

        def upload(target, array):
            source = torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
            target.copy_(source.reshape(target.shape))

        if self.fixture['sparse_mode'] == 3 and not np.all(np.diff(inputs.query_ends, axis=1) == 1):
            raise ValueError('Native replay requires consecutive causal Q4 positions')
        keys, scales = inputs.logical_keys, inputs.logical_scales
        if physical:
            keys, scales, _ = inputs.index_for_layout(inputs.captured_layout)
        upload(self.query.view(torch.uint8), inputs.query)
        upload(self.keys, keys)
        upload(self.scales, scales)
        upload(self.weights.view(torch.int16), inputs.weights.view(np.int16))
        upload(self.query_scale, inputs.query_scale)
        upload(self.fixture['actual_key'], inputs.query_ends[:, -1])


def digest(array):
    data = np.ascontiguousarray(array)
    h = hashlib.sha256(f'{data.dtype.str} {data.shape}'.encode())
    h.update(memoryview(data).cast('B'))
    return h.hexdigest()


def native_ends(inputs):
    klen = inputs.query_ends[:, -1:]
    mode = inputs.geometry.get('sparse_mode', 3)
    if mode == 3:
        ends = klen + np.arange(-3, 1)
        if not np.array_equal(ends, inputs.query_ends):
            raise ValueError('Native causal bounds differ from the captured consecutive queries')
        return ends
    if mode == 0:
        return np.broadcast_to(klen, inputs.query_ends.shape)
    raise ValueError(f'Unsupported native sparse_mode={mode}')


def identities(inputs, torch, torch_npu, device):
    tensors = {name: digest(getattr(inputs, name)) for name in
               ('query', 'weights', 'query_scale', 'logical_keys', 'logical_scales',
                'query_ends', 'captured_layout')}
    package = Path(torch_npu.__file__).resolve().parent
    cann = Path(os.environ.get('ASCEND_CANN_PACKAGE_PATH', '/usr/local/Ascend/ascend-toolkit/latest'))
    files = [Path(torch_npu._C.__file__), package / 'lib/libtorch_npu.so',
             cann / 'lib64/libopapi.so', cann / 'version.info', cann / 'opp/version.info']
    versions = {}
    for path in files:
        if path.is_file():
            info = path.stat()
            versions[str(path.resolve())] = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
    boot = Path('/proc/sys/kernel/random/boot_id')
    return dict(schema=CACHE_SCHEMA, operator='torch_npu.npu_quant_lightning_indexer',
                torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                cann=str(getattr(torch.version, 'cann', '')), libraries=versions,
                opp=os.environ.get('ASCEND_OPP_PATH'), custom_opp=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
                host=platform.node(), boot=boot.read_text().strip() if boot.is_file() else None,
                npu=torch.npu.get_device_name(device), visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                sparse_mode=inputs.geometry.get('sparse_mode', 3), inputs=tensors)


def check_positions(positions, ends):
    if positions.dtype != np.int32 or positions.shape != (*ends.shape, 2048):
        raise ValueError('Native LI must return 2048 integer positions per query')
    if (np.any(positions < 0) or np.any(positions >= ends[..., None])
            or np.any(np.diff(np.sort(positions, axis=-1), axis=-1) == 0)):
        raise ValueError('Native LI output contains duplicate or out-of-domain positions')


def get_reference(inputs, torch, torch_npu, device, native_li, directory, *, strong_only=False):
    identity = identities(inputs, torch, torch_npu, device)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('KVIP_Q4_NATIVE_LI_CACHE', directory / 'native-li-reference'))
    cache.mkdir(parents=True, exist_ok=True)
    modes = ('strong_serial',) if strong_only else MODES
    path = cache / (f'{key}.strong-serial.npz' if strong_only else f'{key}.npz')
    ends = native_ends(inputs)
    positions = {}
    origin = 'fresh'
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                if json.loads(data['identity'].item()) != identity:
                    raise ValueError('Native reference cache input identity differs')
                for mode in modes:
                    value = data[mode]
                    check_positions(value, ends)
                    if digest(value) != data[f'{mode}_sha256'].item():
                        raise ValueError('Native reference output checksum differs')
                    positions[mode] = value
            origin = 'validated-cache'
        except (KeyError, OSError, EOFError, BadZipFile, ValueError):
            positions.clear()
    if not positions:
        operands = NativeReplayLI(torch, device, inputs.geometry)
        for mode in modes:
            if mode != 'strong_serial_repeat':
                operands.prepare(inputs, physical=(mode == 'same_layout_native'))
            selected = native_li(torch_npu, operands.fixture).cpu().numpy()
            if selected.dtype.kind not in 'iu':
                raise ValueError('Native LI output has a non-integer dtype')
            value = selected.reshape(*ends.shape, 2048)
            if np.any(value < 0) or np.any(value >= ends[..., None]):
                raise ValueError('Native LI returned an out-of-domain position')
            value = value.astype(np.int32)
            check_positions(value, ends)
            positions[mode] = value
        del operands
        torch.npu.empty_cache()
        with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            np.savez_compressed(temporary, **positions, identity=json.dumps(identity, sort_keys=True),
                                **{f'{mode}_sha256': digest(value) for mode, value in positions.items()})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    ids = positions.copy()
    if not strong_only:
        ids['same_layout_native'] = inputs.captured_layout[np.arange(len(ends))[:, None, None],
                                                           positions['same_layout_native']].astype(np.int32)
    metadata = dict(origin=origin, cache_path=str(path), cache_key=key, identity=identity,
                    layout_query='BSND', layout_key='PA_BSND', sparse_count=2048,
                    effective_native_ends=ends.tolist(), input_paths=list(map(str, inputs.source_paths)))
    return ids, metadata


def serial_stability(first, repeat):
    a = np.sort(first, axis=-1)
    b = np.sort(repeat, axis=-1)
    match = np.all(a == b, axis=-1)
    unstable = []
    for row, q in np.argwhere(~match):
        unstable.append(dict(row=int(row), query=int(q+1),
                             first_only=np.setdiff1d(a[row,q], b[row,q], assume_unique=True).tolist(),
                             repeat_only=np.setdiff1d(b[row,q], a[row,q], assume_unique=True).tolist()))
    return dict(stable_queries=int(match.sum()), total_queries=match.size, unstable=unstable)


def fused_li_output(state):
    positions = state.workspace.topk.cpu().numpy().reshape(state.batch, 4, 2048)
    unpack = state.workspace.inputs.unpack_indices.cpu().numpy().reshape(-1)
    if (positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= state.history)
            or unpack.dtype.kind not in 'iu' or not np.array_equal(np.sort(unpack), np.arange(state.batch*4))):
        raise ValueError('Invalid raw fused LI positions or packed query mapping')
    logical = state.input_layout[np.arange(state.batch)[:,None,None], positions]
    logical = logical.reshape(-1, 2048)[unpack].reshape(positions.shape)
    emitted = state.update.logical_output.cpu().numpy().reshape(positions.shape)
    if not np.array_equal(np.sort(logical, axis=-1), np.sort(emitted, axis=-1)):
        raise ValueError('Raw fused LI and the final output select different logical token sets')
    return logical


def describe_domains(inputs):
    ends = native_ends(inputs)
    positions = np.arange(inputs.geometry['history'])
    details = []
    for b, layout in enumerate(inputs.captured_layout):
        for q, bound in enumerate(ends[b]):
            serial = positions < bound
            physical = np.zeros_like(serial)
            physical[layout[:bound]] = True
            fused = inputs.logical_score_valid[b, q]
            details.append(dict(row=b, query=q+1, strong_only=int(np.count_nonzero(serial & ~fused)),
                                fused_only=int(np.count_nonzero(fused & ~serial)),
                                same_layout_native_only=int(np.count_nonzero(physical & ~fused)),
                                fused_only_vs_same_layout=int(np.count_nonzero(fused & ~physical))))
    return dict(strong_equals_fused_queries=sum(r['strong_only'] == r['fused_only'] == 0 for r in details),
                same_layout_equals_fused_queries=sum(
                    r['same_layout_native_only'] == r['fused_only_vs_same_layout'] == 0 for r in details),
                queries=details)


def cause(ids, keys, valid, cutoff):
    known = valid[ids]
    return dict(count=len(ids), outside_score_domain=int((~known).sum()),
                at_cutoff=int(np.count_nonzero(known & (keys[ids] == cutoff))),
                above_cutoff=int(np.count_nonzero(known & (keys[ids] > cutoff))),
                below_cutoff=int(np.count_nonzero(known & (keys[ids] < cutoff))),
                examples=[dict(token=int(t), captured_score_key=int(keys[t]) if valid[t] else None)
                          for t in ids[:8]])


def native_priority(inputs, reference):
    """Check the logical-ID cutoff rule against the independent native result."""
    reference = np.sort(reference, axis=-1)
    mismatches = []
    competing = 0
    for row, query in np.ndindex(reference.shape[:2]):
        ids = np.flatnonzero(inputs.logical_score_valid[row, query])
        scores = inputs.logical_score_keys[row, query, ids]
        k = reference.shape[-1]
        cutoff = np.partition(scores, len(scores) - k)[-k]
        above, equal = ids[scores > cutoff], ids[scores == cutoff]
        take = k - len(above)
        competing += int(len(equal) > take)
        expected = np.sort(np.concatenate((above, equal[:take])))
        actual = reference[row, query]
        if not np.array_equal(actual, expected):
            mismatches.append(dict(row=int(row), query=int(query+1), cutoff_key=int(cutoff),
                                   competing=int(len(equal)), needed=int(take),
                                   native_only=np.setdiff1d(actual, expected, assume_unique=True)[:16].tolist(),
                                   logical_first_only=np.setdiff1d(expected, actual, assume_unique=True)[:16].tolist()))
    total = reference.shape[0] * reference.shape[1]
    return dict(total_queries=total, matched_queries=total-len(mismatches),
                split_cutoff_queries=competing, mismatches=mismatches)


def write_comparison(directory, group, inputs, refs, metadata, outputs, *, eager=False):
    actual = np.asarray(outputs)
    expected_calls = 1 if eager else 20
    if actual.shape != (expected_calls, inputs.geometry['batch'], 4, 2048) or actual.dtype.kind not in 'iu':
        raise ValueError(f'Expected {expected_calls} complete Q4 TopK outputs')
    # Verify range before indexing the captured per-token score arrays.
    if np.any(actual < 0) or np.any(actual >= inputs.geometry['history']):
        raise ValueError('Fused output contains an out-of-bounds logical token')
    actual = actual.astype(np.int32)
    ordered = np.sort(actual, axis=-1)
    if np.any(np.diff(ordered, axis=-1) == 0):
        raise ValueError('Fused output contains duplicate logical tokens')
    cutoff = np.empty(actual.shape[1:3], dtype=np.uint16)
    for b, q in np.ndindex(cutoff.shape):
        valid = inputs.logical_score_valid[b, q]
        keys = inputs.logical_score_keys[b, q, valid]
        cutoff[b, q] = np.partition(keys, len(keys)-2048)[-2048]
    comparisons, samples = {}, {}
    for mode, values in refs.items():
        golden = np.sort(values, axis=-1)
        match = np.all(ordered == golden[None], axis=-1)
        records = []
        for i, b, q in np.argwhere(~match):
            keys = inputs.logical_score_keys[b, q]
            valid = inputs.logical_score_valid[b, q]
            missing = np.setdiff1d(golden[b, q], ordered[i,b,q], assume_unique=True)
            extra = np.setdiff1d(ordered[i,b,q], golden[b,q], assume_unique=True)
            loss = cause(missing, keys, valid, cutoff[b,q])
            gain = cause(extra, keys, valid, cutoff[b,q])
            records.append(dict(call=int(i), row=int(b), query=int(q+1), fused_cutoff_key=int(cutoff[b,q]),
                                cutoff_ties_only=(loss['count'] == loss['at_cutoff'] and
                                                  gain['count'] == gain['at_cutoff']),
                                missing=loss, extra=gain))
        comparisons[mode] = dict(matched_calls=int(match.all(axis=(1,2)).sum()),
                                 matched_queries=int(match.sum()), total_queries=match.size,
                                 cutoff_ties_only_mismatches=sum(r['cutoff_ties_only'] for r in records))
        samples[mode] = records
    np.savez_compressed(directory / 'native_li_ids.npz', fused=actual, **refs)
    stability = serial_stability(refs['strong_serial'], refs['strong_serial_repeat'])
    primary_match = comparisons['strong_serial']['matched_calls'] == len(outputs)
    result = dict(group=group, phase='eager' if eager else 'measured',
                  measured_calls=0 if eager else len(outputs), queries_per_call=cutoff.size,
                  primary_serial_sets_match=primary_match, serial_reference_stability=stability,
                  all_serial_sets_match=primary_match and not stability['unstable'],
                  all_same_layout_sets_match=comparisons['same_layout_native']['matched_calls'] == len(outputs),
                  native_reference=metadata, domains=describe_domains(inputs), comparisons=comparisons,
                  samples=samples, raw_ids='native_li_ids.npz')
    atomic_json(directory / 'native_li_compare.json', result)
    return result
