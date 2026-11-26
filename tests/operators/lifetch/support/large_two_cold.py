"""Build and select the optional fixed-Q4 large-batch three-window kernel."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile

from kvip.runtime.build import build_failure
from kvip.runtime.build import kvaf_build_paths


def build(experiment_source, normal_library, sdk_build, soc, *, trace_sibling=False):
    cann, headers = kvaf_build_paths(sdk_build)
    # The normal build ID covers CANN, compiler, KVAF and all included operator sources.
    digest = hashlib.sha256()
    digest.update(bytes([trace_sibling]))
    for source in sorted(experiment_source.iterdir()):
        if source.is_file():
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    folder = normal_library.parent / ("two-cold-" + digest.hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    library = folder / "libtwo_cold_bridge.so"
    log = folder / "build.log"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            # CANN's nested build cannot resume a failed intermediate device link.
            with tempfile.TemporaryDirectory(dir=folder, prefix="compile-") as tmp, log.open("w") as output:
                configure = ["cmake", "-S", str(experiment_source), "-B", tmp,
                             f"-DSOC_VERSION={soc}", f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                             f"-DTRACE_SIBLING={'ON' if trace_sibling else 'OFF'}",
                             f"-DKVAF_INCLUDE={headers}"]
                compile_kernel = ["cmake", "--build", tmp, "--target", "two_cold_bridge", "--parallel", "2"]
                for command in (configure, compile_kernel):
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                    if result.returncode:
                        output.flush()
                        raise build_failure(log, "optional large-batch two-cold kernel", result.returncode)
                (Path(tmp) / library.name).replace(library)
    experimental = ctypes.CDLL(str(library))
    launch = experimental.kvip_test_two_cold_forward
    launch.argtypes = [ctypes.c_void_p] * 6
    launch.restype = ctypes.c_uint32
    return launch


def attach(state, launch):
    regular = state.rotating.library
    if state.forward.terminal is None:
        raise ValueError("Large-batch two-cold experiment needs the four-Q private terminal path")
    hot = ctypes.cast(regular.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
    tail = ctypes.cast(regular.aclrtlaunch_v9_parallel_terminal_large, ctypes.c_void_p)
    state.forward.terminal_function = lambda a, b, stream, shape: launch(a, b, stream, shape, hot, tail)
    state.two_cold_windows = True
