"""Compile the actual C++ device descriptor and compare every Python ABI offset."""

import ctypes
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from kvip.v9 import ABI_VERSION, native_source
from kvip.v9.contract import GEOMETRY_FIELDS, POINTER_FIELDS, DeviceContract

SOURCE = r"""
#include <iostream>
#include "contract_abi.h"
int main() {
  std::cout << rotating::kContractVersion << " " << sizeof(rotating::Contract);
#define FIELD(name) std::cout << " " << offsetof(rotating::Contract, name);
  V9_CONTRACT_POINTERS(FIELD)
  V9_CONTRACT_GEOMETRY(FIELD)
#undef FIELD
  std::cout << std::endl;
}
"""


def main():
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise SystemExit("a C++ compiler is required for the ABI contract check")
    with tempfile.TemporaryDirectory(prefix="kvip-abi-") as root:
        executable = Path(root) / "abi"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-x",
                "c++",
                "-I",
                str(native_source()),
                "-o",
                str(executable),
                "-",
            ],
            input=SOURCE,
            text=True,
            check=True,
        )
        actual = list(
            map(int, subprocess.check_output([str(executable)], text=True).split())
        )
    expected = [ABI_VERSION, ctypes.sizeof(DeviceContract)]
    expected += [
        getattr(DeviceContract, name).offset
        for name in POINTER_FIELDS + GEOMETRY_FIELDS
    ]
    if actual != expected:
        raise SystemExit("C++/Python ABI size or offset mismatch")
    print(
        json.dumps(
            {
                "device_abi": ABI_VERSION,
                "descriptor_bytes": expected[1],
                "matched_offsets": len(expected) - 2,
                "status": "pass",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
