#!/usr/bin/env bash
# Prepare one matched 8-DP, Q4, 656-byte case. Both arms consume this same directory.
set -euo pipefail
work=${1:?usage: bash peer_g2_input.sh WORK_DIRECTORY [GROUP]}
group_id=${2:-2}
case "$group_id" in
    2|7) capture=real-input-r51-g2g7.dKZUcF; source_round=2 ;;
    4|17) capture=real-input-r51-miss5-miss10.hj_6zeym; source_round=2 ;;
    10) capture=score-trace-141.62.38.24-7lg40xtj; source_round=51 ;;
    *) echo "Unsupported capture group: $group_id" >&2; exit 2 ;;
esac
printf -v group_name 'group%03d' "$group_id"
printf -v raw_round 'round%04d' "$source_round"
mkdir -p "$work"
work=$(cd "$work" && pwd)
data=/home/c00982601/CacheOS/ubmem-validation
suffix="r$source_round-g$group_id"
case_dir="case-$suffix"
raw="$data/$capture/trace"
existing=
if [[ $group_id == 2 ]]; then
    case_dir=case00002-r2-g2
    raw=${PEER_G2_RAW:-$raw}
    existing=${PEER_G2_INPUT:-}
fi
dest="$work/peer-inputs/$case_dir"
support=$(cd "$(dirname "$0")/support" && pwd)
mkdir -p "$(dirname "$dest")"
scratch=$(mktemp -d "$work/.peer-g${group_id}.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
complete_input() {
    local root=$1 round=$2 rank group
    for rank in {0..7}; do
        group="$root/dp$rank/$round/$group_name"
        [[ -s "$root/dp$rank/series.json" && -s "$group/manifest.json" && -s "$group/scores.npz" ]] || return 1
    done
}
if [[ -z "$existing" ]]; then
    for p in "$dest" "$work/lifetch-real-suite"/case*-"$suffix" \
             "${PEER_SUITE_ROOT:-$data/lifetch-real-suite}"/case*-"$suffix" \
             "$data/recent-index-threshold/peer-inputs"/case*-"$suffix" \
             /home/x00965276/KVIP/lifetch-real-suite/case*-"$suffix"; do
        if complete_input "$p" round0001; then existing=$p; break; fi
    done
fi
if [[ -z "$existing" ]] && ! complete_input "$raw" "$raw_round"; then
    if [[ $group_id == 2 && -n ${PEER_G2_RAW:-} ]]; then echo "Incomplete PEER_G2_RAW=$raw" >&2; exit 1; fi
    host=${PEER_TRACE_SOURCE_HOST:-root@141.62.38.24}
    if [[ $group_id == 2 ]]; then host=${PEER_G2_SOURCE_HOST:-$host}; fi
    command -v ssh >/dev/null || { echo "Need ssh or a local 8-DP input under PEER_SUITE_ROOT" >&2; exit 1; }
    echo "INPUT_COPY G$group_id DP0–7 only, source=$host" >&2
    mkdir "$scratch/raw"
    ssh -T -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -- "$host" \
        bash -s -- "$suffix" "$capture" "$group_name" "$raw_round" <<'REMOTE' | tar -xf - -C "$scratch/raw"
set -eu
data=/home/c00982601/CacheOS/ubmem-validation
suffix=$1 capture=$2 group_name=$3 raw_round=$4
for dir in "$data/recent-index-threshold/peer-inputs"/case*-"$suffix" "$data/recent-index-threshold/lifetch-real-suite"/case*-"$suffix" \
           "$data/lifetch-real-suite"/case*-"$suffix" /home/x00965276/KVIP/lifetch-real-suite/case*-"$suffix" "$data/$capture/trace"; do
    round=round0001
    [[ "$dir" != */trace ]] || round=$raw_round
    members=()
    for rank in {0..7}; do
        group=dp$rank/$round/$group_name
        members+=(dp$rank/series.json "$group/manifest.json" "$group/scores.npz")
    done
    for file in "${members[@]}"; do [[ -s "$dir/$file" ]] || continue 2; done
    exec tar -C "$dir" -chf - "${members[@]}"
done
echo "Missing $suffix on source host. Set PEER_TRACE_SOURCE_HOST to the capture host, or PEER_SUITE_ROOT to the local case collection." >&2
exit 1
REMOTE
    if complete_input "$scratch/raw" round0001; then
        existing="$scratch/raw"
    else
        raw="$scratch/raw"
    fi
fi
PYTHONPATH="$support${PYTHONPATH:+:$PYTHONPATH}" python3 -B - "$raw" "$dest" "$scratch" "$existing" "$group_id" "$source_round" <<'PY'
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
downloaded = root.resolve() == (scratch / 'raw').resolve()
group_id, source_round = map(int, sys.argv[5:7])
capture_round = 1 if existing else source_round
refs = select_series(root, group_id, (capture_round, capture_round))
if [ref.dp_rank for ref in refs] != list(range(8)):
    raise ValueError(f'G{group_id} needs DP0–7')
ids = [(ref.dp_rank, request, owner) for ref in refs for request, owner in ref.request_signature]
fingerprint = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
if group_id == 2 and fingerprint != '83bef8d67299723cbd9e04eaf14b9e270bbfdb2609bca771a8aee9c1eabd92bc':
    raise ValueError('G2 DP/request/owner order differs from peer replay input')

selected = cold = 0
archives = []
out = scratch / 'export'
for ref in refs:
    manifest = json.loads(ref.manifest_path.read_text())
    geom = manifest['geometry']
    expected = dict(batch=2, heads=32, history=131328, resident=16384, output_stride=16896)
    if (manifest.get('real_inputs_saved') is not True or any(geom[k] != v for k, v in expected.items())
            or geom['token_bytes'] not in ((656,) if existing else (2624, 656))):
        raise ValueError(f'G{group_id} geometry or real-input marker differs at {ref.manifest_path}')
    archive = checked_path(ref.manifest_path.parent, manifest['archive']['file'])
    description = manifest['archive']
    with archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if archive.stat().st_size != description['bytes'] or digest != description['sha256']:
        raise ValueError(f'G{group_id} input transfer checksum differs: {archive}')
    archives.append(digest)
    with np.load(archive, allow_pickle=False) as z:
        required = {'real.keys', 'real.key_scales', 'real.query', 'real.query_scale', 'real.weights'}
        if not required.issubset(z.files):
            raise ValueError(f'G{group_id} missing captured LI operands: {archive}')
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
    group = dp / f'round0001/group{group_id:03d}'
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
    entry = series['rounds'][source_round - 1]
    member, = [m for m in entry['groups'] if m['group'] == group_id]
    member['manifest'] = f'round0001/group{group_id:03d}/manifest.json'
    entry.update(round_index=1, groups=[member])
    series.update(rounds=[entry], expected_rounds=1, completed_rounds=1)
    (dp / 'series.json').write_text(json.dumps(series))
expected_counts = {2: (45138, 955), 7: (57293, 11355), 10: (69708, 2319),
                   17: (86406, 4982), 4: (48088, 4717)}
if (selected, cold) != expected_counts[group_id]:
    raise ValueError(f'G{group_id} capture differs: selected={selected} cold={cold}')
if not existing or downloaded:
    prepared = root if existing else out
    select_series(prepared, group_id, (1, 1))
    # Concurrent users prepare identical immutable inputs in independent directories.
    try:
        prepared.rename(dest)
    except OSError as exc:
        if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not (dest/'dp7/series.json').is_file():
            raise
input_id = hashlib.sha256(json.dumps([group_id, source_round, ids, archives], separators=(',', ':')).encode()).hexdigest()
print(f'INPUT G{group_id} source_round={source_round} replay_round=1 selected={selected} cold={cold} '
      f'Q4_BS16_token_bytes=656 sha256={input_id}', file=sys.stderr)
print(root if existing and not downloaded else dest)
PY
