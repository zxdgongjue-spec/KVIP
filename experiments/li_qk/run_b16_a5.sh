#!/usr/bin/env bash
# Direct repository entry for the eight B16 30/32-core controls.
set -Eeuo pipefail
for LI_B16_ARG in "$@"; do
    [[ $LI_B16_ARG != --suite ]] || { echo 'ERROR: this entry fixes --suite batch16_cores' >&2; exit 2; }
done
LI_B16_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$LI_B16_HERE/run_history_a5.sh" --suite batch16_cores "$@"
