#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One original ten-call measurement block. The parent supplies a fresh local endpoint.
set -euo pipefail
root=$1 work=$2 out=$3 group=$4 trace=$5 sdk=$6 endpoint=$7 arm=$8
reader=b9edb1514b2278d7a15411f4ff7a7b2b9f87749c
if ! git -C "$root" cat-file -e "${reader}^{commit}" 2>/dev/null; then
    git -C "$root" fetch -q --no-tags https://gitcode.com/CacheOS/KVIP.git refs/heads/test/lifetch-pr30-live-main-ten
fi
historical="$out/pr30/reader-$reader"
if [[ ! -d $historical ]]; then
    tmp=$(mktemp -d "$out/pr30/reader.XXXXXX")
    git -C "$root" archive "$reader" tests/operators/lifetch/support | tar -x -C "$tmp"
    mv "$tmp" "$historical"
fi
support="$historical/tests/operators/lifetch/support"
archive=${PEER_PR30_PUBLISHED_ARCHIVE:-$work/peer-baseline/pr30-feature-live-main-ten.aYyvMS}
owner="$archive/oracle/g$group"
if [[ ! -f "$owner/native_score_oracle.json" ]]; then
    owner=$("$PYTHON" -B "$support/feature_host_reference.py" "$work/peer-baseline" "$group" owner)
fi
if [[ -z $owner ]]; then
    owner="$out/pr30/oracle/g$group"
    if [[ ! -f "$owner/native_score_oracle.json" ]]; then
        KVIP_Q4_NATIVE_LI_CACHE="$work/peer-baseline/native-li-reference-g2" \
            "$PYTHON" -B "$support/feature_score_oracle.py" "$trace" "$group" "$owner"
    fi
fi
printf '%s\n' "$owner" > "$arm/oracle.txt"
printf '%s\n' "$archive" > "$out/pr30/published_archive.txt"
printf '%s\n' "$support" > "$out/pr30/reader_path.txt"
cd "$arm"
timeout --kill-after=30s 7200s "$PYTHON" -B "$support/main_five_run.py" \
    --main "$root" --revision "$(git -C "$root" rev-parse HEAD)" --trace "$trace" \
    --owner-archive "$owner" --group "$group" --kvaf-root "$sdk" --kvaf-server "$endpoint" \
    --output "$arm" --ablation-policy paced
