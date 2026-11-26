#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One original ten-call measurement block. The parent supplies a fresh local endpoint.
set -euo pipefail
root=$1 work=$2 out=$3 group=$4 trace=$5 sdk=$6 endpoint=$7 arm=$8 label=${9:-candidate}
card=${PEER_PHYSICAL_NPU:-4}
reader=9f5e2b837b36dd090940205756ffbc6e0c033dd9
paced=b261b4885edd79dabac822a2e6e3790c7f226ad6
remote=https://gitcode.com/CacheOS/KVIP.git
if ! git -C "$root" cat-file -e "${reader}^{commit}" 2>/dev/null; then
    git -C "$root" fetch -q --no-tags "$remote" refs/heads/test/lifetch-fixed-q4-packed-device
fi
historical="$out/pr30/reader-$reader"
if [[ ! -d $historical ]]; then
    tmp=$(mktemp -d "$out/pr30/reader.XXXXXX")
    git -C "$root" archive "$reader" tests/operators/lifetch/support | tar -x -C "$tmp"
    mv "$tmp" "$historical"
fi
support="$historical/tests/operators/lifetch/support"
archive=${PEER_PR30_PUBLISHED_ARCHIVE:-$work/peer-baseline/pr30-feature-live-main-ten.aYyvMS}
owner=
if [[ $card == 4 ]]; then
    owner="$archive/oracle/g$group"
    if [[ ! -f "$owner/native_score_oracle.json" ]]; then
        owner=$("$PYTHON" -B "$support/feature_host_reference.py" "$work/peer-baseline" "$group" owner)
    fi
fi
if [[ -z $owner ]]; then
    owner="$out/pr30/oracle/g$group"
    if [[ ! -f "$owner/native_score_oracle.json" ]]; then
        KVIP_Q4_NATIVE_LI_CACHE="$work/peer-baseline/native-li-reference-npu$card" \
            "$PYTHON" -B "$support/feature_score_oracle.py" "$trace" "$group" "$owner"
    fi
fi
case $label in
    candidate) production="$root" ;;
    pr30_reference)
        production="$out/pr30/paced-$paced"
        if [[ ! -d $production ]]; then
            if ! git -C "$root" cat-file -e "${paced}^{commit}" 2>/dev/null; then
                git -C "$root" fetch -q --no-tags "$remote" refs/heads/perf/lifetch-main-dualcut-paced
            fi
            git -C "$root" worktree add -q --detach "$production" "$paced"
        fi
        [[ $(git -C "$production" rev-parse HEAD) == "$paced" ]]
        ;;
    *) echo "Unexpected packed replay arm: $label" >&2; exit 2 ;;
esac
printf '%s\n' "$owner" > "$arm/oracle.txt"
printf '%s\n' "$archive" > "$out/pr30/published_archive.txt"
printf '%s\n' "$support" > "$out/pr30/reader_path.txt"
cd "$arm"
export PYTHONPATH="$production/python:${PYTHONPATH:-}"
timeout --kill-after=30s 7200s "$PYTHON" -B "$support/main_five_run.py" \
    --main "$production" --revision "$(git -C "$production" rev-parse HEAD)" --trace "$trace" \
    --owner-archive "$owner" --group "$group" --kvaf-root "$sdk" --kvaf-server "$endpoint" \
    --output "$arm" --ablation-policy paced
