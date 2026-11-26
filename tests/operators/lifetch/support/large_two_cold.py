"""Build isolated cold kernels for the fixed-Q4 large-batch replay."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile

from kvip.runtime.build import build_failure
from kvip.runtime.build import kvaf_build_paths


WINDOW_GRANTS = {"two-cold": (256, 384, 384), "split-role": (256, 256, 256, 256)}
COLD_SYMBOL = {"two-cold": "v9_li_large_two_cold", "split-role": "v9_li_large_bs32_split_role"}


def build(experiment_source, normal_library, sdk_build, soc, policy="two-cold", *, trace_sibling=False):
    if policy not in WINDOW_GRANTS:
        raise ValueError(f"Unknown large-batch window policy {policy!r}")
    cann, headers = kvaf_build_paths(sdk_build)
    # The normal build ID covers CANN, compiler, KVAF and all included operator sources.
    digest = hashlib.sha256()
    digest.update(f"{policy}:{soc}:trace_sibling={int(trace_sibling)}".encode())
    if policy == "split-role":
        source = (experiment_source.resolve().parents[4] / "csrc/operators/lightning_indexer_with_fetch"
                  / "op_kernel/indexer_large_dualrole.cpp")
        if not source.is_file():
            raise FileNotFoundError(f"Missing split-role device implementation: {source}")
        # The isolated TU also consumes the regular Cube, vector and score scheduling code.
        for path in (source, source.with_name("indexer_large.cpp"), source.parent / "indexer/indexer_cube.h",
                     source.parent / "indexer/indexer_vector.h", source.parent / "indexer/large_bs_split_owner.h"):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    for source in sorted(experiment_source.iterdir()):
        if source.is_file():
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    build_id = digest.hexdigest()
    folder = normal_library.parent / (policy + "-" + build_id[:16])
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
                             f"-DKVIP_EXPERIMENT_SPLIT_ROLE={'ON' if policy == 'split-role' else 'OFF'}",
                             f"-DTRACE_SIBLING={'ON' if trace_sibling else 'OFF'}",
                             f"-DKVAF_INCLUDE={headers}"]
                compile_kernel = ["cmake", "--build", tmp, "--target", "two_cold_bridge", "--parallel", "2"]
                for command in (configure, compile_kernel):
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                    if result.returncode:
                        output.flush()
                        raise build_failure(log, f"large-batch {policy} kernel", result.returncode)
                (Path(tmp) / library.name).replace(library)
    experimental = ctypes.CDLL(str(library))
    launch = (experimental.kvip_test_bs32_split_role_forward if policy == "split-role"
              else experimental.kvip_test_two_cold_forward)
    launch.argtypes = [ctypes.c_void_p] * 6
    launch.restype = ctypes.c_uint32
    launch.window_policy = policy
    launch.window_limit = max(WINDOW_GRANTS[policy])
    launch.source_fingerprint = f"{normal_library.parent.name}/{build_id}"
    return launch


def attach(state, launch):
    regular = state.rotating.library
    if state.forward.terminal is None:
        raise ValueError("Large-batch window experiment requires the four-Q private terminal path")
    hot = ctypes.cast(regular.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
    tail = ctypes.cast(regular.aclrtlaunch_v9_parallel_terminal_large, ctypes.c_void_p)
    state.forward.terminal_function = lambda a, b, stream, shape: launch(a, b, stream, shape, hot, tail)
    state.large_window_launch = launch
