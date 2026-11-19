# SPDX-License-Identifier: Apache-2.0
"""Use the measured G2 production candidate in the real-input profile protocol."""

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


REVISION = 'cdfc4da4b53b80255740183c7a0a8db8bf350dcb'
OPTIONS = dict(selected_metadata=True, early_fetch_ready=True, ready_batch_fetch=True,
               low_miss_skip=True, auto_route=True, low_miss_tail=True,
               low_miss_sparse_init=True, auto_route_row_publish=True,
               low_miss_direct_index=True, low_miss_post_index=True, adaptive_admission=True)


def candidate_package(root):
    path = Path(root).resolve()
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', '-C', str(path), 'merge-base', '--is-ancestor', REVISION, revision], check=True)
    subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--', 'python', 'csrc'], check=True)
    return path / 'python', revision


def seed_bounds(snapshot, inputs):
    root = Path(snapshot).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    requests = manifest['metadata']['requests']
    ids = tuple((r['source_dp_rank'], r['source_request_id'], r['source_owner']) for r in requests)
    if ids != inputs.identities:
        raise ValueError('Recent seed archive identities do not match the compact G2 replay')
    def load(name):
        record = manifest['tensors']['pre.' + name]
        path = (root / record['file']).resolve()
        if not path.is_relative_to(root) or record['dtype'] != 'int32':
            raise ValueError(f'Invalid archived {name}')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != record['sha256']:
                raise ValueError(f'Checksum mismatch for {name}')
        return np.load(path, allow_pickle=False).view('<i4').reshape(record['shape'])
    ends = load('query_ends').reshape(-1, 4)
    bounds = load('threshold_bounds')
    if (bounds.shape != (len(ids), 2) or not np.array_equal(ends, inputs.query_ends)
            or not np.array_equal(bounds[:, 1], ends.max(axis=1))
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 0] > inputs.tail_bounds[:, 0])):
        raise ValueError('Invalid predecessor bounds for the peer G2 inputs')
    return bounds


def candidate_binding(torch, library, device, bounds):
    from kvip.operators.lifetch.auto_route import AutoRouteState
    from kvip.runtime.loader import load_lifetch_library

    handle = load_lifetch_library(library)
    for option in OPTIONS:
        getter = getattr(handle, 'kvip_' + option)
        getter.argtypes, getter.restype = [], ctypes.c_uint32
        if getter() != 1:
            raise RuntimeError(f'Candidate build option is inactive: {option}')
    batch = len(bounds)
    route = AutoRouteState(torch, device, batch, batch + 1)
    route.set_learning(False)
    set_step(route, 0)
    return dict(threshold_bounds=torch.from_numpy(bounds.copy()).to(device), auto_route=route,
                window_limit=64)


def set_step(route, step):
    requests = np.empty((route.batch, 4), dtype=np.int64)
    # Peer replay remaps DP request rows to stable, independent storage owners 1..16.
    requests[:, 0] = np.arange(1, route.batch + 1)
    requests[:, 1:] = [1, step, 4]
    route.set_requests(requests)
    if step:
        route.set_learning(True)


def checked_routes(states, step, *, low):
    counts = []
    for state in states:
        route = state.rotating.auto_route
        pending = route.pending.cpu().numpy().view(np.uint64)
        feedback = route.feedback.cpu().numpy().view(np.uint64)[1:]
        controls = state.rotating.admission_controls.cpu().numpy()
        expected = [0, route.full_controls[1], 0, 0] if low else list(route.full_controls)
        if (not np.all(controls == expected) or not np.all(pending[:, 4] == int(not low))
                or not np.all(pending[:, 3] == 1) or not np.all(feedback[:, 7] == 0x4B565250)
                or not np.all(feedback[:, 0] == 1) or not np.all(feedback[:, 1] == step)
                or not np.array_equal(feedback[:, 2], pending[:, 9])):
            raise AssertionError(f'Failed learned G2 route or committed feedback at step {step}')
        counts.append(int(pending[:, 9].sum()))
    return dict(step=step, low=len(states) if low else 0, high=0 if low else len(states),
                necessary_get_per_call=counts)
