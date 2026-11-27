"""Bind the archived single-row terminal with its private doorbell timing option."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile

from functools import lru_cache
from four_rank_tail_image import tail_text


@lru_cache(maxsize=None)
def build(repo, normal_library, sdk_build, soc):
    from kvip.runtime.build import build_failure, kvaf_build_paths
    normal_library=Path(normal_library)
    op=Path(repo)/"csrc/operators/lightning_indexer_with_fetch"
    src=Path(__file__).with_name("terminal_stagger_native")
    cann,headers=kvaf_build_paths(sdk_build)
    digest=hashlib.sha256(f"{soc} {normal_library.parent.name}".encode())
    for path in sorted((*src.glob("*.h"),src/"kernel.cpp",src/"bridge.cpp",src/"CMakeLists.txt",src/"prepare_overlay.py",
                        *op.joinpath("op_kernel/terminal").glob("*.h"))):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    folder=normal_library.parent/("stagger-tail-"+digest.hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    library=folder/"libterminal_stagger_bridge.so"
    log=folder/"build.log"
    with (folder/"build.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=folder) as tmp,log.open("w") as output:
                steps=[["cmake","-S",str(src),"-B",tmp,f"-DSOC_VERSION={soc}",
                        f"-DASCEND_CANN_PACKAGE_PATH={cann}",f"-DOP={op}",f"-DKVAF_INCLUDE={headers}"],
                       ["cmake","--build",tmp,"--target","terminal_stagger_bridge","--parallel","1"]]
                for step in steps:
                    rc=subprocess.run(step,stdout=output,stderr=subprocess.STDOUT).returncode
                    if rc:
                        output.flush()
                        raise build_failure(log,"stagger terminal",rc)
                (Path(tmp)/library.name).replace(library)
    return ctypes.CDLL(str(library))



def install(module, repo, sdk, soc, enabled):
    previous = module.attach

    def attach(state, launch):
        previous(state, launch)
        if launch.window_policy != "four-ranks" or launch.trace_only:
            raise ValueError("Expected timed four-window LI")
        normal = Path(state.rotating.library._name).resolve()
        lib = build(repo, normal, sdk, soc)
        hot = ctypes.cast(state.rotating.library.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
        tail = ctypes.cast(lib.kvip_test_terminal_stagger, ctypes.c_void_p)
        # The last reserved word is independent of profiling and the frozen LI buffers.
        binding = state.forward.terminal.workspace
        if binding.numel() != 128 or binding[120:128].count_nonzero().item():
            raise ValueError("Terminal binding is incompatible with the doorbell gate")
        binding[120] = int(enabled)
        def forward(a, b, stream, shape):
            return launch(a, b, stream, shape, hot, tail)
        state.forward.terminal_function = forward
        state.terminal_gate_build = Path(lib._name)
    module.attach = attach


@lru_cache(maxsize=None)
def images(library):
    return tail_text(library, "terminal_stagger_kernel")


def identity(state, enabled):
    word = int.from_bytes(bytes(state.forward.terminal.workspace[120:128].cpu().tolist()), "little")
    if word != int(enabled):
        raise ValueError("Measured state lost its terminal gate setting")
    return dict(tail_text=images(state.terminal_gate_build), gate_word=word,
                cold_build=state.large_window_launch.source_fingerprint)
