"""Build and bind the fixed-shape two-row terminal only when this experimental arm is requested."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile

from terminal_two_row_native.lease_guard import BoundJointTerminal


def build(repo, normal_library, sdk_build, soc):
    from kvip.runtime.build import build_failure, kvaf_build_paths
    normal_library=Path(normal_library)
    op=Path(repo)/"csrc/operators/lightning_indexer_with_fetch"
    src=Path(__file__).with_name("terminal_two_row_native")
    cann,headers=kvaf_build_paths(sdk_build)
    digest=hashlib.sha256(f"{soc} {normal_library.parent.name}".encode())
    for path in sorted((*src.glob("*.h"),src/"kernel.cpp",src/"bridge.cpp",src/"CMakeLists.txt",src/"prepare_overlay.py",
                        *op.joinpath("op_kernel/terminal").glob("*.h"))):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    folder=normal_library.parent/("two-row-tail-"+digest.hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    library=folder/"libtwo_row_bridge.so"
    log=folder/"build.log"
    with (folder/"build.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=folder) as tmp,log.open("w") as output:
                steps=[["cmake","-S",str(src),"-B",tmp,f"-DSOC_VERSION={soc}",
                        f"-DASCEND_CANN_PACKAGE_PATH={cann}",f"-DOP={op}",f"-DKVAF_INCLUDE={headers}"],
                       ["cmake","--build",tmp,"--target","two_row_bridge","--parallel","1"]]
                for step in steps:
                    rc=subprocess.run(step,stdout=output,stderr=subprocess.STDOUT).returncode
                    if rc:
                        output.flush()
                        raise build_failure(log,"two-row terminal",rc)
                (Path(tmp)/library.name).replace(library)
    return ctypes.CDLL(str(library))


def prepare_tail(repo, normal_library, sdk_build, soc, forward, counter_mhz=None):
    return BoundJointTerminal(forward,build(repo,normal_library,sdk_build,soc),counter_mhz)
