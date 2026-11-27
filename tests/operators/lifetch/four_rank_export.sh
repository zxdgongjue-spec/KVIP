#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=${1:?expected KVIP checkout}
rev=$(git -C "$repo" rev-parse --verify "${2:?expected exact exporter commit}^{commit}")
shift 2
work=/home/c00982601/CacheOS/ubmem-validation/recent-index-threshold
stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
git -C "$repo" archive "$rev" tests/operators/lifetch/support | tar -xf - -C "$stage"
for py in "$work/peer-baseline/python-da318bdb/bin/python" /usr/local/python3.12.13/bin/python3 python3; do
    if "$py" -B -c 'import hashlib,numpy; assert hasattr(hashlib,"file_digest")' 2>/dev/null; then
        "$py" -B "$stage/tests/operators/lifetch/support/export_four_rank_inputs.py" "$@"
        exit 0
    fi
done
echo 'CPU exporter requires Python 3.11+ with NumPy' >&2
exit 1
