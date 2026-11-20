#!/usr/bin/env bash
# Prepare the host-local G2 dependencies used by all URMA replay arms.

peer_node_pin() {
    local repo=$1 dest=$2 revision=$3 branch=$4
    if [[ ! -e "$dest" ]]; then
        if ! git -C "$repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
            git -C "$repo" fetch -q origin "$branch" || return
        fi
        mkdir -p "$(dirname "$dest")" || return
        git -C "$repo" worktree add -q --detach "$dest" "$revision" || return
    fi
    if [[ $(git -C "$dest" rev-parse HEAD) != "$revision" ]] || ! git -C "$dest" diff --quiet HEAD; then
        printf 'PREPARE_FAILED unexpected source checkout: %s (expected %s)\n' "$dest" "$revision" >&2
        return 1
    fi
}

peer_node_logged() {
    local log=$1
    shift
    local rc=0
    "$@" >"$log" 2>&1 || rc=$?
    if (( rc )); then
        printf 'PREPARE_FAILED rc=%d LOG=%s\n' "$rc" "$log" >&2
        tail -12 "$log" >&2
    fi
    return "$rc"
}

peer_prepare_node() {
    local root=$1 baseline=$2 clean=$3 work=$4 out=$5
    local serial_revision=da318bdb81344672b5d9013e04d517669d3eb142
    local kvaf_revision=de4fd37e02fa86357f71fe4fcbbbecc3e75d1d58
    local area=${clean%/*} repo cann cache headers lib prefix cpu_cache venv lidu
    mkdir -p "$area" || return
    if [[ ${6:-} != fused-only ]]; then
        peer_node_pin "$root" "$baseline" "$serial_revision" test/host20-peer-serial-only || return
    fi
    if [[ ! -e "$clean" ]]; then
        repo="$work/KVAF"
        if [[ $(git -C "$repo" remote get-url origin 2>/dev/null) == *CacheOS/KVAF* ]]; then
            peer_node_pin "$repo" "$clean" "$kvaf_revision" main || return
        else
            git clone -q --single-branch --branch main --no-checkout https://gitcode.com/CacheOS/KVAF.git "$clean" || return
            git -C "$clean" checkout -q --detach "$kvaf_revision" || return
        fi
    fi
    peer_node_pin "$clean" "$clean" "$kvaf_revision" main || return

    # Use the same CANN resolver as the actual KVAF NPU build.
    source "$clean/scripts/cann_root.sh" || return
    cann=$(kvaf_find_cann_root) || return
    if [[ ! -f "$cann/include/acl/acl.h" ]]; then
        printf 'PREPARE_FAILED missing ACL header under %s\n' "$cann" >&2
        return 1
    fi
    export ASCEND_HOME_PATH="$cann" ASCEND_CANN_PACKAGE_PATH="$cann"
    export ASCEND_RT_VISIBLE_DEVICES=4 SOC_VERSION=Ascend950DT_9582 ASCEND_LAUNCH_BLOCKING=0
    export MAX_JOBS=2 OMP_NUM_THREADS=1 NANOVLLM_A5_OPS_BUILD_JOBS=2

    venv="$area/python-${serial_revision:0:8}"
    if [[ ! -x "$venv/bin/python" ]]; then
        peer_node_logged "$out/venv.log" /usr/local/python3.12.13/bin/python3 -m venv --system-site-packages "$venv" || return
    fi
    export PYTHON="$venv/bin/python" NANOVLLM_A5_OPS_PYTHON="$venv/bin/python"
    if ! "$PYTHON" -m pybind11 --cmakedir >/dev/null 2>&1; then
        peer_node_logged "$out/pybind11.log" "$PYTHON" -m pip install pybind11 || return
    fi

    cpu_cache="$clean/build-cpu/CMakeCache.txt"
    if [[ ! -x "$clean/build-cpu/runtime/kvaf_cpu_service" ]] ||
       ! grep -qsx 'KVAF_CPU_ENABLE_URMA:BOOL=ON' "$cpu_cache"; then
        # Reuse local UMDK when already configured; build.sh handles new hosts.
        for cache in "$cpu_cache" "$work/KVAF/build-cpu/CMakeCache.txt"; do
            [[ -f "$cache" ]] || continue
            headers=$(sed -n 's/^KVAF_URMA_ROOT:[^=]*=//p' "$cache")
            lib=$(sed -n 's/^KVAF_URMA_LIBRARY:[^=]*=//p' "$cache")
            if [[ -z ${KVAF_UMDK_ROOT:-} && -f "$headers/include/urma_api.h" ]]; then export KVAF_UMDK_ROOT="$headers"; fi
            if [[ -z ${KVAF_URMA_LIBRARY:-} && -f "$lib" ]]; then export KVAF_URMA_LIBRARY="$lib"; fi
        done
        if [[ -z ${KVAF_UMDK_ROOT:-} ]]; then
            for prefix in /usr /usr/local; do
                if [[ -f "$prefix/include/urma_api.h" && ( -f "${KVAF_URMA_LIBRARY:-}" ||
                    -f "$prefix/lib64/liburma.so" || -f "$prefix/lib/liburma.so" || -f /lib64/liburma.so ) ]]; then
                    export KVAF_UMDK_ROOT="$prefix"
                    break
                fi
            done
        fi
        peer_node_logged "$out/build-cpu.log" bash "$clean/build.sh" cpu --jobs 2 || return
    fi

    if [[ ${6:-} == fused-only ]]; then
        printf 'LOCAL_DEPS_READY clean_KVAF=%.8s CANN=%s\n' "$kvaf_revision" "$cann"
        return
    fi
    lidu="$baseline/csrc/operators/lidu"
    (
        export PYTHONPATH="$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
        local vendors=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
        if [[ ${#vendors[@]} != 1 || ! -s ${vendors[0]} ]] || ! "$PYTHON" -c 'from pathlib import Path; import sys; import nanovllm_dsa_lidu_fused_batch_get_prepare_mtp as m; assert Path(m.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()); m.lidu_fused_batch_get_prepare_mtp_out' "$lidu" >"$out/lidu-import.log" 2>&1; then
            peer_node_logged "$out/build-lidu.log" bash "$lidu/build_lidu_fused_batch_get_prepare_mtp.sh" || exit
        fi
    ) || return
    printf 'LOCAL_DEPS_READY clean_KVAF=%.8s LIDU=%.8s CANN=%s\n' "$kvaf_revision" "$serial_revision" "$cann"
}
