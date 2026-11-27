#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from the already configured kvaf_vllm_ascend container on a5-12.
set -eo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
mode=${1:---help}
if (( $# )); then shift; fi
runner="$repo/tests/operators/lifetch/support/fixed_q4_a5_12.py"

case "$mode" in
  --dry-run)
    exec "${KVIP_PYTHON:-python3}" -B "$runner" --dry-run "$@"
    ;;
  packed-synthetic)
    if [[ "${1:-}" == --dry-run ]]; then
      shift
      exec "${KVIP_PYTHON:-python3}" -B "$runner" --synthetic --dry-run "$@"
    fi
    packed_input=(--synthetic)
    packed_name=synthetic
    ;;
  packed-q1-pair|packed-large-q1-pair)
    trial=--window-q1-pair
    packed_name=window-q1
    if [[ "$mode" == packed-large-q1-pair ]]; then
      trial=--large-window-q1-pair
      packed_name=large-window-q1
    fi
    if [[ "${1:-}" == --dry-run ]]; then
      shift
      exec "${KVIP_PYTHON:-python3}" -B "$runner" --dry-run --synthetic "$trial" "$@"
    fi
    packed_input=(--synthetic "$trial")
    ;;
  packed|q1-w2) ;;
  *)
    printf '%s\n' 'Usage:' '  run_fixed_q4_a5_12.sh packed G2_PREPARED_TRACE [runner options]' \
      '  run_fixed_q4_a5_12.sh packed-synthetic [runner options]' \
      '  run_fixed_q4_a5_12.sh packed-q1-pair [runner options]' \
      '  run_fixed_q4_a5_12.sh packed-large-q1-pair [runner options]' \
      '  run_fixed_q4_a5_12.sh q1-w2 MR35_CHECKOUT MAIN_CHECKOUT' \
      '  run_fixed_q4_a5_12.sh --dry-run'
    if [[ "$mode" == --help || "$mode" == -h ]]; then exit 0; fi
    exit 2
    ;;
esac

root=${KVIP_A5_ROOT:-/home/w00568585/li-aligned-a5}
kvaf=${KVIP_KVAF_ROOT:-"$root/KVAF-unified-813f1750"}
lidu=${KVIP_LIDU_ROOT:-"$root/KVIP-mr28/csrc/operators/lidu"}
py=${KVIP_PYTHON:-/usr/local/python3.12.13/bin/python3}
build="$kvaf/build-npu"
results=${KVIP_A5_RESULTS:-"$root/results"}

if [[ "$mode" == packed ]]; then
  trace=${1:?Pass the prepared real G2 capture directory in the container}
  shift
  if [[ ! -d "$trace" ]]; then
    printf 'Prepared G2 capture is missing in the container: %s\n' "$trace" >&2
    exit 2
  fi
  packed_input=(--trace "$trace")
  packed_name=g2
elif [[ "$mode" == q1-w2 ]]; then
  if (( $# != 2 )); then
    printf '%s\n' 'q1-w2 needs the MR35 and main checkouts.' >&2
    exit 2
  fi
  q1=$1
  main=$2
  test "$(git -C "$q1" rev-parse HEAD)" = a9f4fe3b7531f32f4f9faa9da65697e5ab43aa30
  test "$(git -C "$main" rev-parse HEAD)" = df8ad2598e70e4b5e266b15028ab26c601a2bdf2
  git -C "$q1" diff --quiet HEAD --
  git -C "$main" diff --quiet HEAD --
fi

git -C "$repo" diff --quiet HEAD --
if [[ "$mode" == packed-q1-pair || "$mode" == packed-large-q1-pair ]]; then
  git -C "$repo" merge-base --is-ancestor 555e1e51c3639fad47e0132eaab426582cddc543 HEAD
else
  git -C "$repo" diff --quiet af52a02b97daddadd5d93b9d100f2ff1b03b499d HEAD -- \
    csrc python patches scripts pyproject.toml
fi
test "$(git -C "$kvaf" rev-parse HEAD)" = 813f1750a50b7e6c92d5f5397f2fbc05befef6da
for kind in get put; do
  test -s "$build/bin/kvaf_execute_batch_${kind}_kernels.o"
done

cann=$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$build/CMakeCache.txt")
test -n "$cann"
source "$cann/set_env.sh"
mkdir -p "$results"
exec 9>"$results/.lifetch-a5-12-physical2.lock"
flock -n 9 || { printf 'Another LIFetch run holds the physical NPU 2 lock.\n' >&2; exit 1; }

probe=$("$py" - <<'PY'
import ctypes as c
import sys

acl = c.CDLL("libascendcl.so")
acl.aclGetRecentErrMsg.restype = c.c_char_p

def check(name, code):
    if code:
        msg = (acl.aclGetRecentErrMsg() or b"").decode(errors="replace")
        raise SystemExit(f"{name}: rc={code}\n{msg}")

check("aclInit", acl.aclInit(None))
count = c.c_uint32()
check("aclrtGetDeviceCount", acl.aclrtGetDeviceCount(c.byref(count)))
matches = []
for device in range(count.value):
    logical, physical = c.c_int32(), c.c_int32()
    check("user to logical", acl.aclrtGetLogicDevIdByUserDevId(device, c.byref(logical)))
    check("logical to physical", acl.aclrtGetPhyDevIdByLogicDevId(logical.value, c.byref(physical)))
    print(f"device={device} logical={logical.value} physical={physical.value}", file=sys.stderr)
    if physical.value == 2:
        matches.append(device)
if len(matches) != 1:
    raise SystemExit(f"Expected one mapping for physical NPU 2, found {matches}")
check("aclrtSetDevice on physical NPU 2", acl.aclrtSetDevice(matches[0]))
print(f"BENCH_DEVICE={matches[0]}")
PY
) || { printf '%s\n' "$probe" >&2; exit 1; }
device=$(sed -n 's/^BENCH_DEVICE=//p' <<< "$probe")
[[ "$device" =~ ^[0-9]+$ ]]
printf 'ACL OK: physical=2 device=%s\n' "$device"

libs=("$lidu"/_custom_opp/vendors/*/op_api/lib/libcust_opapi.so)
if [[ ${#libs[@]} -ne 1 || ! -f "${libs[0]}" ]]; then
  printf 'Expected built LIDU OPP under %s\n' "$lidu" >&2
  exit 1
fi
export KVIP_LIDU_OPP="${libs[0]%/op_api/lib/libcust_opapi.so}"
export ASCEND_CUSTOM_OPP_PATH="$KVIP_LIDU_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export PYTHONPATH="$build/python:$lidu/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
PYTHONPATH="$repo/python:$PYTHONPATH" "$py" - "$build" <<'PY'
import sys
import acl, kvaf, torch, torch_npu
from kvip.runtime.build import kvaf_build_paths
from nanovllm_dsa_lidu_fused_batch_get_prepare_mtp import lidu_fused_batch_get_prepare_mtp_out

kvaf_build_paths(sys.argv[1])
getattr(kvaf.NpuRuntime, "get_graph_urma_resources")
if not callable(lidu_fused_batch_get_prepare_mtp_out):
    raise SystemExit("LIDU registration is unavailable")
print("CANN, KVAF and LIDU OK")
PY

if [[ "$mode" != q1-w2 ]]; then
  out=$(mktemp -d "$results/packed-$packed_name-card2.XXXXXX")
  printf 'OUTPUT_PARENT=%s\n' "$out"
  PYTHONPATH="$repo/python:$PYTHONPATH" "$py" -B "$runner" "$@" "${packed_input[@]}" \
    --device "$device" --physical-npu 2 --kvaf-root "$kvaf" --output "$out/run"
  exit
fi

export KVIP_A5_ROOT="$root" KVIP_A5_DEVICE="$device" KVIP_PYTHON="$py" KVIP_KVAF_ROOT="$kvaf"
export KVIP_AB_OUTPUT="$results/q1-w2-card2"
mkdir -p "$KVIP_AB_OUTPUT"
export KVIP_KVAF_SERVER=127.0.0.1:28986
port_open() { ( : >/dev/tcp/127.0.0.1/28986 ) 2>/dev/null; }
if ! port_open; then
  service_log=$(mktemp "$KVIP_AB_OUTPUT/kvaf-service.XXXXXX")
  "$kvaf/build-cpu/runtime/kvaf_cpu_service" --listen 127.0.0.1 --port 28986 \
    </dev/null >"$service_log" 2>&1 &
  service_pid=$!
  cleanup() { kill "$service_pid" 2>/dev/null || true; wait "$service_pid" 2>/dev/null || true; }
  trap cleanup EXIT
  for attempt in {1..15}; do
    if port_open; then break; fi
    sleep 1
  done
  port_open || { tail -30 "$service_log"; exit 1; }
fi
bash "$q1/experiments/lifetch_mr23_service/run_full32_a5.sh" \
  --window-q-wide --main-root "$main" --warmup 2 --iterations 5 --repeats 2
