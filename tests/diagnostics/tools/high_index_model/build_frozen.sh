#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
if [[ $# != 3 ]]; then echo 'Usage: build_frozen.sh PINNED_KVIP KVAF_INCLUDE NEW_OUTPUT' >&2; exit 2; fi
repo=$(cd "$1" && pwd -P)
inc=$(cd "$2" && pwd -P)
dst=$3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cann=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.2.0}
source "$cann/set_env.sh"
python3 -B "$here/frozen_source.py" "$repo" "$dst"
if ! { cmake -S "$dst" -B "$dst/build" -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend950DT_9582 \
           -DASCEND_CANN_PACKAGE_PATH="$cann" -DKVAF_INCLUDE="$inc" &&
       cmake --build "$dst/build" --target kvip_frozen_tail --parallel 2; } >"$dst/build.log" 2>&1; then
    tail -24 "$dst/build.log" >&2; exit 1
fi
test -s "$dst/build/libkvip_frozen_tail.so"
ln -s build/libkvip_frozen_tail.so "$dst/libkvip_frozen_tail.so"
