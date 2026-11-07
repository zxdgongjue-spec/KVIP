#!/usr/bin/env python3
"""Compile/run actual UBmem ring and ABI host models; no NPU validation claim."""
from pathlib import Path
import os
import subprocess
import tempfile

root=Path(__file__).resolve().parents[1]
compiler=os.environ.get("CXX","c++")
common={"SIM_HISTORY":33024,"SIM_RESIDENT":8192,"INDEX_HEADS":64,"SIM_Q":4,
        "V9_AIC_COUNT":32,"V9_SPLIT_Q":1,"V9_INTERMEDIATE_FETCHES":8,"V9_FINAL_AIVS":0,
        "V9_SERVICE_AICS":0,"V9_FINAL_REPLICATED":1,"V9_DEDICATED_SERVICE":0,
        "V9_UBPIPE_DEPTH":8,"V9_UBPIPE_ROWS":4,"V9_WINDOW_UBPIPE_DEPTH":8,
        "V9_USER_UB_BYTES":223232}
with tempfile.TemporaryDirectory(prefix="kvip-ubmem-model-") as directory:
    for batch in (1,2):
        for width in (656,2624):
            definitions=common|{"SIM_BATCH":batch,"TOKEN_BYTES":width}
            options=[compiler,"-std=c++17","-O1","-g","-fsanitize=address,undefined",
                     *[f"-D{k}={v}" for k,v in definitions.items()],
                     f"-I{root/'include'}",f"-I{root/'op_kernel/copy'}"]
            for name,sources in (("copy",[root/"tests/copy_model.cpp"]),
                                 ("abi",[root/"tests/abi_test.cpp",root/"op_host/launch.cpp"])):
                binary=Path(directory)/f"{name}-{batch}-{width}"
                subprocess.run([*options,*map(str,sources),"-o",str(binary)],check=True)
                subprocess.run([str(binary)],check=True)
print("Static ABI and real copy-header models passed all B1/B2 × 656/2624 specializations")
