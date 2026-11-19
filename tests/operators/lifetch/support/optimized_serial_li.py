"""Numeric direct LI adapter. LIDU/Get remain the existing NativeMtp path."""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "experiments/li_qk"


def build_serial_library(*, cache_dir, cann, headers, soc):
    from kvip.runtime.build import build_failure
    options = dict(RUN_MODE="npu", ASCEND_CANN_PACKAGE_PATH=str(cann), KVAF_INCLUDE=str(headers),
                   SOC_VERSION=soc, LI_SERIAL_BRIDGE="ON")
    digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
    directories = (HERE, ROOT / "csrc/operators/lightning_indexer_with_fetch",
                   ROOT / "csrc/operators/common", Path(headers))
    for directory in directories:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix in (".cpp", ".h", ".py") or path.name == "CMakeLists.txt"):
                if not any(part in ("build", "build_out", "__pycache__") for part in path.relative_to(directory).parts):
                    digest.update(str(path.relative_to(directory)).encode())
                    digest.update(path.read_bytes())
    preparer = ROOT / "experiments/split_core/prepare_score_source.py"
    digest.update(preparer.read_bytes())
    for relative in ("version.info", "compiler/ccec_compiler/bin/bisheng"):
        path = Path(cann) / relative
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    build = Path(cache_dir) / digest.hexdigest()[:20]
    build.mkdir(parents=True, exist_ok=True)
    library = build / "build/libli_serial_bridge.so"
    with (build / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (build / "complete.json").is_file() or not library.is_file():
            source = build / "source"
            if not (source / "score-source.json").is_file():
                # A failed snapshot is not a valid source cache.
                if source.exists():
                    import shutil
                    shutil.rmtree(source)
                spec = importlib.util.spec_from_file_location("li_serial_source_snapshot", preparer)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.prepare(source)
            log = build / "build.log"
            with log.open("w") as output:
                commands = (
                    ["cmake", "-S", str(HERE), "-B", str(build / "build"),
                     f"-DSCORE_SOURCE={source}", *[f"-D{k}={v}" for k, v in options.items()]],
                    ["cmake", "--build", str(build / "build"), "--target", "li_serial_bridge", "--parallel", "2"],
                )
                for command in commands:
                    output.write(json.dumps(command) + "\n")
                    output.flush()
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                    if result.returncode:
                        output.flush()
                        raise build_failure(log, "numeric serial LI", result.returncode)
            if not library.is_file():
                raise RuntimeError(f"Serial LI bridge not produced; see {log}")
            (build / "complete.json").write_text(json.dumps(options, indent=2) + "\n")
    return library


def task_plan(batch, history, capacity, requested_parts=0):
    if not 1 <= batch <= 16 or not 1 <= capacity <= 32:
        raise ValueError("Numeric serial LI requires B1..16 and 1..32 score groups")
    if requested_parts:
        parts = requested_parts
    else:
        available = max(1, capacity // batch)
        parts = max(p for p in (1, 2, 4, 8, 16) if p <= available)
    if (not 1 <= batch <= 16 or not 1 <= capacity <= 32 or parts not in (1, 2, 4, 8, 16, 32)
            or history % parts or history // parts < 2048 or (history // parts) % 2048):
        raise ValueError("Numeric serial LI requires B1..16 and equal 2048-aligned history shards")
    stride = history // parts
    tasks = [(0, 4, p * stride, (p + 1) * stride, owner) for owner in range(batch) for p in range(parts)]
    return parts, stride, [tasks[i:i + capacity] for i in range(0, len(tasks), capacity)]


class OptimizedSerialLI:
    def __init__(self, torch, fixture, library, *, selected_shape, batch, history, heads, aics, aivs, parts=0):
        if aivs < 4 or not bool((fixture["actual_key"] == history).all().item()):
            raise ValueError("Numeric serial control currently requires full valid history and >=4 AIVs")
        self.torch = torch
        self.batch, self.history, self.heads, self.aivs = batch, history, heads, aivs
        self.parts, self.stride, tasks = task_plan(batch, history, min(aics, aivs // 2), parts)
        self.library = ctypes.CDLL(str(library))
        u32, ptr = ctypes.c_uint32, ctypes.c_void_p
        self.library.li_serial_context_bytes.argtypes = []
        self.library.li_serial_context_bytes.restype = u32
        self.library.li_serial_context_init.argtypes = [ptr, u32, u32, u32, u32]
        self.library.li_serial_context_init.restype = u32
        self.library.li_serial_score.argtypes = [ptr, ptr, ptr, ptr, u32, u32, u32]
        self.library.li_serial_score.restype = u32
        self.library.li_serial_topk.argtypes = [ptr, ptr, ptr, ptr, u32, u32, u32, u32]
        self.library.li_serial_topk.restype = u32
        device = fixture["query"].device
        self.references = [fixture]

        def raw(value):
            tensor = torch.from_numpy(np.asarray(value).view(np.uint8).reshape(-1).copy()).to(device)
            self.references.append(tensor)
            return tensor

        self.selected = torch.empty(selected_shape, dtype=torch.int32, device=device)
        self.canonical = torch.empty((batch, 4, history), dtype=torch.int16, device=device)
        size = self.library.li_serial_context_bytes()
        context = ctypes.create_string_buffer(size)
        self.check(self.library.li_serial_context_init(context, size, batch, history, heads))
        self.context = raw(np.frombuffer(context, np.uint8))
        rows = max(map(len, tasks))
        packed_q = torch.empty((rows, 4, heads, 128), dtype=torch.uint8, device=device)
        packed_w = torch.empty((rows, 4, heads), dtype=torch.int16, device=device)
        packed_s = torch.empty((rows, 4, heads), dtype=torch.float32, device=device)
        packed_p = torch.empty((rows, self.stride // 128), dtype=torch.int32, device=device)
        qlen = torch.full((rows,), 4, dtype=torch.int32, device=device)
        klen = torch.full((rows,), self.stride, dtype=torch.int32, device=device)
        dummy_ids = torch.empty((rows, 4, 2048), dtype=torch.int32, device=device)
        scratch = torch.empty((rows, 4, self.stride), dtype=torch.int16, device=device)
        self.references.extend([packed_q, packed_w, packed_s, packed_p, qlen, klen, dummy_ids, scratch])
        native = fixture["native_inputs"]
        self.waves = []
        for wave in tasks:
            task = raw(np.asarray(wave, dtype=np.uint32))
            bindings = [0] * 16
            for index, tensor in ((0, packed_q), (1, fixture["key"]), (2, packed_w), (3, packed_s),
                                  (4, native["key_scale"]), (5, qlen), (6, klen), (7, packed_p),
                                  (9, dummy_ids), (12, scratch)):
                bindings[index] = tensor.data_ptr()
            bindings = raw(np.asarray(bindings, dtype=np.uint64))
            output_blob = np.asarray([self.canonical.data_ptr(), task.data_ptr(), 0, 0, 0,
                                      self.selected.data_ptr()], dtype=np.uint64).tobytes() + np.asarray([history, 1], dtype=np.uint32).tobytes()
            output = raw(np.frombuffer(output_blob, np.uint8))
            # Match the uint64[9] + uint32[4] Inputs ABI without alignment guesses.
            pointers = [fixture["query"].data_ptr(), native["weights"].data_ptr(), native["query_scale"].data_ptr(),
                        fixture["block_table"].data_ptr(), packed_q.data_ptr(), packed_w.data_ptr(),
                        packed_s.data_ptr(), packed_p.data_ptr(), task.data_ptr()]
            blob = np.asarray(pointers, np.uint64).tobytes() + np.asarray([heads, history, self.stride, len(wave)], np.uint32).tobytes()
            inputs = raw(np.frombuffer(blob, np.uint8))
            self.waves.append((len(wave), inputs, bindings, output))
        self.geometry = dict(route="numeric_direct", batch=batch, history=history, history_parts=self.parts,
                             stride=self.stride, score_waves=len(self.waves), score_aics_peak=rows,
                             score_aivs_peak=2*rows, prepare_launches=len(self.waves), score_launches=len(self.waves),
                             topk_launches=(4*batch + (aivs//4)*4 - 1)//((aivs//4)*4), pack_in_timing=True,
                             scope="numeric LI only; then unchanged NativeMtp LIDU/Get",
                             binary_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"Numeric serial LI launch failed: {code}")

    def __call__(self):
        stream = self.torch.npu.current_stream().npu_stream
        for rows, inputs, bindings, output in self.waves:
            self.check(self.library.li_serial_score(stream, inputs.data_ptr(), bindings.data_ptr(), output.data_ptr(),
                                                    rows, self.stride, self.heads))
        self.check(self.library.li_serial_topk(stream, self.canonical.data_ptr(), self.selected.data_ptr(),
                                             self.context.data_ptr(), self.history, self.heads, self.batch, self.aivs))
        return self.selected
