"""Bind the single-row terminal with two pilot queries per long row."""
import ctypes
import fcntl
import hashlib
from pathlib import Path
import subprocess
import tempfile

from functools import lru_cache
from four_rank_tail_image import tail_text


@lru_cache(maxsize=None)
def build(repo, normal_library, sdk_build, soc, natural=False):
    from kvip.runtime.build import build_failure, kvaf_build_paths
    normal_library=Path(normal_library)
    op=Path(repo)/"csrc/operators/lightning_indexer_with_fetch"
    src=Path(__file__).with_name("terminal_stagger_native")
    cann,headers=kvaf_build_paths(sdk_build)
    digest=hashlib.sha256(f"{soc} {normal_library.parent.name} natural={natural}".encode())
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
                        f"-DASCEND_CANN_PACKAGE_PATH={cann}",f"-DOP={op}",f"-DNATURAL_GET={'ON' if natural else 'OFF'}",f"-DKVAF_INCLUDE={headers}"],
                       ["cmake","--build",tmp,"--target","terminal_stagger_bridge","--parallel","1"]]
                for step in steps:
                    rc=subprocess.run(step,stdout=output,stderr=subprocess.STDOUT).returncode
                    if rc:
                        output.flush()
                        raise build_failure(log,"stagger terminal",rc)
                (Path(tmp)/library.name).replace(library)
    return ctypes.CDLL(str(library))



def install(module, repo, sdk, soc, enabled, *, natural=False):
    previous = module.attach

    def attach(state, launch):
        previous(state, launch)
        if launch.window_policy != "four-ranks" or launch.trace_only:
            raise ValueError("Expected timed four-window LI")
        normal = Path(state.rotating.library._name).resolve()
        lib = build(repo, normal, sdk, soc, natural)
        hot = ctypes.cast(state.rotating.library.aclrtlaunch_v9_li_large_hot, ctypes.c_void_p)
        tail = ctypes.cast(lib.kvip_test_terminal_first_cq, ctypes.c_void_p)
        # The last reserved word is independent of profiling and the frozen LI buffers.
        binding = state.forward.terminal.workspace
        if binding.numel() != 128 or binding[120:128].count_nonzero().item():
            raise ValueError("Terminal binding is incompatible with the doorbell gate")
        import struct
        import torch
        permit_ptr = binding[104:112]
        if permit_ptr.count_nonzero().item():
            raise ValueError("First-CQ permits conflict with an existing graph binding")
        permits = torch.zeros((state.batch, 2, 32), dtype=torch.int32, device=state.device)
        permit_ptr.copy_(torch.tensor(list(struct.pack("<Q", permits.data_ptr())), dtype=torch.uint8, device=state.device))
        state.first_cq_permits = permits
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
    permit_values = state.first_cq_permits[:, :, 0].cpu().tolist()
    if any(a != b or a not in ((0,1) if enabled else (0,)) for a,b in permit_values):
        raise ValueError("Missing or failed release on one of the two Q pairs")
    return dict(tail_text=images(state.terminal_gate_build), gate_word=word,
                cold_build=state.large_window_launch.source_fingerprint)


def final_plan(records, slots, archived_starts=None):
    import numpy as np
    batch, depth = slots.shape
    data = np.zeros((batch, 4, 32 + 4096), dtype="<i4")
    for row in range(batch):
        by_q = [records[f"r{row}_q{q}"] for q in range(4)]
        before = sum(len(records[f"r{row}_w{w}"]) for w in range(4))
        spans = []
        for q, entries in enumerate(by_q):
            n = len(entries)
            data[row,q,:2] = (n,4)
            if n == 0:
                continue
            ids = entries[:,0]
            if n > 2048 or len(np.unique(ids)) != n:
                raise ValueError("Duplicate or oversize private final GET list")
            data[row,q,32:32+n] = ids
            data[row,q,32+2048:32+2048+n] = np.sort(ids)
            if archived_starts is None:
                starts = np.flatnonzero(slots[row] == entries[0,4])
                if (len(starts) != 1 or starts[0]+n > depth or
                        not np.array_equal(slots[row,starts[0]:starts[0]+n],entries[:,4])):
                    raise ValueError("Cannot recover reservation order from this archived LRU")
                begin = int(starts[0])
            else:
                begin = archived_starts[row][q]
            spans.append((begin,n,q))
        pred = 4
        for begin,n,q in sorted(spans):
            if begin != before:
                raise ValueError("Archived final reservation spans have gaps or overlap")
            data[row,q,1] = pred
            pred = q
            before += n
    return data


def bind_final_plan(state, records, archived_starts):
    import struct
    import torch
    plan = final_plan(records, state.rotating.lru_slots.cpu().numpy(), archived_starts)
    device = torch.from_numpy(plan).to(state.device)
    slot = state.forward.terminal.workspace[112:120]
    if slot.count_nonzero().item():
        raise ValueError("Private final plan pointer conflicts with an active workspace")
    slot.copy_(torch.tensor(list(struct.pack("<Q", device.data_ptr())), dtype=torch.uint8, device=state.device))
    state.captured_final_plan = device
