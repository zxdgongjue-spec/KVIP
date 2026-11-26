#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
a5_root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}
export KVIP_KVAF_ROOT="${KVIP_KVAF_ROOT:-"$a5_root/KVAF-unified-813f1750"}"
python=${KVIP_PYTHON:-python3}
for arg in "$@"; do
    if [[ $arg == --comparison-dry-run || $arg == --help || $arg == -h ]]; then
        exec "$python" -B "$here/run_mr28_unified_score.py" "$@"
    fi
done

if [[ ${1:-} == --prepare-deps ]]; then
    revision=813f1750a50b7e6c92d5f5397f2fbc05befef6da
    if [[ ! -e $KVIP_KVAF_ROOT ]]; then
        git init "$KVIP_KVAF_ROOT"
    fi
    if ! git -C "$KVIP_KVAF_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
        git -C "$KVIP_KVAF_ROOT" fetch --depth=1 https://gitcode.com/CacheOS/KVAF.git "refs/keep-around/$revision"
        git -C "$KVIP_KVAF_ROOT" switch --detach FETCH_HEAD
    fi
    "$python" -B "$repo/scripts/build.py" --prepare-only --transport urma --kvaf-root "$KVIP_KVAF_ROOT"
    umdk="$KVIP_KVAF_ROOT/third_party/umdk"
    if [[ ! -e $umdk ]]; then
        mkdir -p "$(dirname "$umdk")"
        git clone --depth=1 --branch br_UMDK_24.12.0_LTS https://gitcode.com/openeuler/umdk.git "$umdk"
    fi
    test -f "$umdk/include/urma_api.h"
    exit
fi

if [[ ${1:-} == --build-deps ]]; then
    test -f "$KVIP_KVAF_ROOT/third_party/umdk/include/urma_api.h" || {
        printf '%s\n' 'Prepare the dependency sources on the host with --prepare-deps first.' >&2
        exit 1
    }
    shift
    cann=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}}
    if [[ ! -f "$cann/set_env.sh" ]]; then
        printf 'CANN environment missing at %s/set_env.sh\n' "$cann" >&2
        exit 1
    fi
    set +u
    source "$cann/set_env.sh"
    set -u
    "$python" -m pybind11 --cmakedir >/dev/null
    PYTHON="$python" PIP_NO_INDEX=1 bash "$repo/scripts/build.sh" \
        --transport urma --kvaf-root "$KVIP_KVAF_ROOT" --cann "$cann" "$@"
    exit
fi

exec bash "$here/run_mr28_container.sh" --mr28-unified-score "$@"
