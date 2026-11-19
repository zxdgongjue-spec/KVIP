#!/usr/bin/env bash
# Prepare exactly the eight G2 score captures used by the peer one-graph benchmark.
set -euo pipefail
work=${1:?usage: bash peer_g2_input.sh WORK_DIRECTORY}
mkdir -p "$work"
work=$(cd "$work" && pwd)
data=/home/c00982601/CacheOS/ubmem-validation
raw=${PEER_G2_RAW:-"$data/real-input-r51-g2g7.dKZUcF/trace"}
dest="$work/peer-inputs/case00002-r2-g2"
support=$(cd "$(dirname "$0")/support" && pwd)
mkdir -p "$(dirname "$dest")"
scratch=$(mktemp -d "$work/.peer-g2.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
existing=${PEER_G2_INPUT:-}
if [[ -z "$existing" ]]; then
    for p in "$dest" "$work/lifetch-real-suite/case00002-r2-g2" "$data/lifetch-real-suite/case00002-r2-g2"; do
        if [[ -s "$p/dp0/series.json" ]]; then existing=$p; break; fi
    done
fi
if [[ -z "$existing" && ! -s "$raw/dp0/series.json" ]]; then
    if [[ -n ${PEER_G2_RAW:-} ]]; then echo "Missing PEER_G2_RAW=$raw" >&2; exit 1; fi
    echo "INPUT_COPY copying only DP0–7 G2 round 2 from host24" >&2
    mkdir "$scratch/raw"
    ssh -n -T -o ServerAliveInterval=30 -o ServerAliveCountMax=6 root@141.62.38.24 \
        'cd /home/c00982601/CacheOS/ubmem-validation/real-input-r51-g2g7.dKZUcF/trace && tar -cf - dp[0-7]/series.json dp[0-7]/round0002/group002/manifest.json dp[0-7]/round0002/group002/scores.npz' \
        | tar -xf - -C "$scratch/raw"
    raw="$scratch/raw"
fi
PYTHONPATH="$support${PYTHONPATH:+:$PYTHONPATH}" python3 -B - "$raw" "$dest" "$scratch" "$existing" <<'PY'
import errno
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from workloads import checked_path, select_series

raw, dest, scratch = map(Path, sys.argv[1:4])
existing = sys.argv[4]
root = Path(existing).resolve() if existing else raw
refs = select_series(root, 2, (1, 1) if existing else (2, 2))
ids = [(ref.dp_rank, request, owner) for ref in refs for request, owner in ref.request_signature]
fingerprint = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
if fingerprint != '83bef8d67299723cbd9e04eaf14b9e270bbfdb2609bca771a8aee9c1eabd92bc':
    raise ValueError('G2 DP/request/owner order differs from peer replay input')

selected = cold = 0
out = scratch / 'export'
for ref in refs:
    manifest = json.loads(ref.manifest_path.read_text())
    geom = manifest['geometry']
    expected = dict(batch=2, heads=32, history=131328, resident=16384, output_stride=16896)
    if (manifest.get('real_inputs_saved') is not True or any(geom[k] != v for k, v in expected.items())
            or geom['token_bytes'] not in ((656,) if existing else (2624, 656))):
        raise ValueError(f'G2 geometry or real-input marker differs at {ref.manifest_path}')
    archive = checked_path(ref.manifest_path.parent, manifest['archive']['file'])
    with np.load(archive, allow_pickle=False) as z:
        top, layout, hot, bounds = (z[key] for key in ('post.topk', 'pre.logical_ids', 'pre.hot_valid', 'pre.tail_bounds'))
        for row in range(geom['batch']):
            pos = np.unique(top[row])
            hit = pos < geom['resident']
            hit[hit] = hot[row, pos[hit]] != 0
            token = layout[row, pos]
            lo, hi = bounds[row]
            selected += len(pos)
            cold += int(np.count_nonzero(~(hit | ((lo <= token) & (token < hi)))))
    if existing:
        continue
    dp = out / f'dp{ref.dp_rank}'
    group = dp / 'round0001/group002'
    group.mkdir(parents=True)
    payload = group / 'scores.npz'
    try:
        os.link(archive, payload)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(archive, payload)
    geom['token_bytes'] = 656
    manifest['archive']['file'] = payload.name
    (group / 'manifest.json').write_text(json.dumps(manifest))
    series = json.loads((root / dp.name / 'series.json').read_text())
    entry = series['rounds'][1]
    member, = [m for m in entry['groups'] if m['group'] == 2]
    member['manifest'] = 'round0001/group002/manifest.json'
    entry.update(round_index=1, groups=[member])
    series.update(rounds=[entry], expected_rounds=1, completed_rounds=1)
    (dp / 'series.json').write_text(json.dumps(series))
if (selected, cold) != (45138, 955):
    raise ValueError(f'G2 capture differs: demand={selected} cold={cold}')
if not existing:
    select_series(out, 2, (1, 1))
    # Concurrent users prepare identical immutable inputs in independent directories.
    try:
        out.rename(dest)
    except OSError as exc:
        if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not (dest/'dp7/series.json').is_file():
            raise
print(root if existing else dest)
PY
