# SPDX-License-Identifier: Apache-2.0
"""Attach a private large-batch tail after the unchanged hot and ranked cold launches."""
import ctypes
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile


@lru_cache(maxsize=None)
def build(normal, sdk, soc):
    from kvip.runtime.build import kvaf_build_paths, build_failure
    from four_rank_tail_image import tail_text
    source = Path(__file__).with_name("medium_terminal_native")
    op = source.parents[4] / "csrc/operators/lightning_indexer_with_fetch"
    cann, headers = kvaf_build_paths(sdk)
    digest = hashlib.sha256(repr((normal.parent.name, soc)).encode())
    for path in [*sorted(source.iterdir()), op/"CMakeLists.txt", *sorted((op/"op_kernel/terminal").glob("*.*"))]:
        digest.update(path.read_bytes())
    folder = normal.parent / ("medium-tail-" + digest.hexdigest()[:16])
    folder.mkdir(exist_ok=True)
    binary = folder / "libmedium_tail_bridge.so"
    log = folder / "build.log"
    with (folder/"build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            with tempfile.TemporaryDirectory(dir=folder) as tmp, log.open("w") as out:
                for cmd in (["cmake", "-S", str(source), "-B", tmp, f"-DOP={op}", f"-DSOC_VERSION={soc}",
                             f"-DASCEND_CANN_PACKAGE_PATH={cann}", f"-DKVAF_INCLUDE={headers}"],
                            ["cmake", "--build", tmp, "--target", "medium_tail_bridge", "--parallel", "2"]):
                    rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
                    if rc:
                        out.flush()
                        raise build_failure(log, "medium GET tail", rc)
                (Path(tmp)/binary.name).replace(binary)
    old, new = tail_text(normal), tail_text(binary)
    if old == new:
        raise ValueError("The two terminal images are identical")
    return ctypes.CDLL(str(binary)), old, new


def install(large_two_cold, sdk):
    import acl
    import correctness
    import execution
    from functools import partial

    # The pinned native checkout supplies the old replay modules. Both its eager
    # and graph validators must require this kernel's private CQ count.
    spec = importlib.util.spec_from_file_location("medium_tail_checks", Path(__file__).with_name("correctness.py"))
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    validator = partial(checks.validate_terminal_receipts, medium_rows=True)
    correctness.validate_terminal_receipts = execution.validate_terminal_receipts = validator
    native_attach = large_two_cold.attach

    def attach(state, launch):
        native_attach(state, launch)
        if state.batch not in (20, 24, 28, 32) or launch.window_policy != "four-ranks":
            raise ValueError("Medium tail requires a large Q4 replay")
        regular = state.rotating.library
        candidate, old, new = build(Path(regular._name).resolve(), sdk, acl.get_soc_name())
        hot = ctypes.cast(regular.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
        tail = ctypes.cast(candidate.kvip_private_medium_tail, ctypes.c_void_p)
        def forward(bindings, terminal, stream, shape):
            return launch(bindings, terminal, stream, shape, hot, tail)
        state.forward.terminal_function = forward
        state.medium_images = old, new
        state.medium_tail_pointer = tail.value
    large_two_cold.attach = attach


def image(state):
    old, new = state.medium_images
    fun = state.forward.terminal_function
    refs = dict(zip(fun.__code__.co_freevars, (cell.cell_contents for cell in fun.__closure__)))
    if refs["tail"].value != state.medium_tail_pointer:
        raise ValueError("Timed tail is not the tested terminal")
    return dict(cold_build=state.large_window_launch.source_fingerprint, tail_text=new, old_tail_text=old)


def receipts(states):
    import numpy as np
    observed = []
    for state in states:
        term = state.forward.terminal
        status = term.owner.core_status.cpu().numpy().view(np.uint64)
        observed.append(status[np.asarray(term.queue_views)].reshape(state.batch, 4, -1)[:, :, 3].tolist())
    return observed
