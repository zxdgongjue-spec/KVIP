"""Build a separately marked test source tree; never edit production kernels."""

import re
import shutil
from pathlib import Path


def prepare_source(source, target):
    target = Path(target)
    if target.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError("test source must be separate from production source")
    shutil.copytree(source, target, dirs_exist_ok=True)
    shutil.copy2(Path(__file__).with_name("memory_transport.h"), target / "memory_transport.h")
    lifecycle = target / "lifecycle.cpp"
    text = lifecycle.read_text()
    anchor = "__aicore__ inline CqDrainResult DrainQueue("
    if text.count(anchor) != 1:
        raise ValueError("mock transport drain anchor changed")
    text = text.replace(anchor, '#include "memory_transport.h"\n' + anchor)
    signature = "    if (!pending) return {0,0,0};"
    if text.count(signature) != 1:
        raise ValueError("mock transport drain body changed")
    text = text.replace(signature, "    return TestDrainQueue(entity,queue,pending);\n" + signature)
    text = text.replace("kvaf_device::KvafBatchGetDevice(workerArgs,", "TestBatchGet(workerArgs,")
    lifecycle.write_text(text)
    # Doorbells are owned NPU scratch, never MMIO in this test binary.
    for path in target.glob("*.h"):
        text = path.read_text()
        text = re.sub(
            r"st_dev\((uint32_t\([^;]*?\)),reinterpret_cast<__gm__ uint32_t\*>\((\w+)\),0\);",
            (
                r"*reinterpret_cast<__gm__ uint32_t*>(\2)=\1;\n"
                r"kvaf_device::CleanCacheLines(reinterpret_cast<__gm__ uint8_t*>(\2),64);"
            ),
            text,
        )
        if "st_dev(" in text:
            raise ValueError(f"unhandled hardware doorbell in {path.name}")
        path.write_text(text, encoding="utf-8", newline="\n")
    bridge = target / "bridge.cpp"
    with bridge.open("a") as out:
        out.write('\nextern "C" uint32_t v9_test_transport() { return 1; }\n')
    return target
