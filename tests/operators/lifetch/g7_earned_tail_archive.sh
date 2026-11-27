#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Read completed same-host NPU4 journals. No card initialization or reruns.
set -euo pipefail
repo=${1:?expected KVIP repository}; rev=${2:?expected archive-reader commit}
work=${PEER_G2_WORK:-/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold}
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
support=tests/operators/lifetch/support
git -C "$repo" archive "$rev" "$support" | tar -x -C "$tmp"
py="$work/peer-baseline/python-da318bdb/bin/python"
if ! "$py" -B -c 'import hashlib,numpy; assert hasattr(hashlib,"file_digest")' 2>/dev/null; then py=/usr/local/python3.12.13/bin/python3; fi
"$py" -B "$tmp/$support/g7_earned_tail_archive.py" "$work/peer-baseline"
