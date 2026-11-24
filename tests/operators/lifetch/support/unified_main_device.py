"""Check the loaded CANN 9.2 upstream images against the separately built B24 target."""
import hashlib
from pathlib import Path

from check_isolated_device import extract


REVISION = "d6cfd392f722dd3f96ef6e05175f503aac511dbb"
EXPECTED_TEXT = {
    "v9_kernel/0": "effe20d53f971b793a365edcd52725a07a257463c4b916f0fe4882cd7eb8c8a3",
    "v9_kernel/1": "8865289b694c64ca7f667e5c9e68a7b3c693664ac1b107aaa58dab08382e6fc1",
    "parallel_terminal_kernel/1": "c9c7837ca2fbf532c7a785de2f73276c278b1de3078aec33bbba91a9a93caf28",
}


def check(library):
    result = {}
    for name, sections in extract(Path(library)).items():
        for key, code in sections.items():
            tag = f"{name}/{key[1]}"
            if key[4] != ".text" or tag in result:
                raise ValueError(f"Unexpected executable section in upstream {name}: {key}")
            result[tag] = hashlib.sha256(code).hexdigest()
    if result != EXPECTED_TEXT:
        raise ValueError(f"Loaded upstream LI or terminal differs from the verified NPU4 build: {result}")
    return result
