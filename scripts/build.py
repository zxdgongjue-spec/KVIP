#!/usr/bin/env python3
"""Build pinned KVAF, original LIDU and the complete LIFetch/Index update library."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1c1310387ebfb6776a07316874e924265b209ada"
PATCHES = ("kvaf-reuse-device-ra", "kvaf-fused-context", "kvaf-fused-output-views")
UBMEM_PATCH = "kvaf-ubmem"
BACKGROUND_PATCH = "kvaf-background-put"
URMA_POST32_PATCH = "kvaf-urma-get-post32"
URMA_BULK_PATCH = "kvaf-urma-get-bulk"
UBMEM_DISPATCH_PATCH = "kvaf-ubmem-get-dispatch"


def call(command, cwd=None, env=None):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def runtime_preflight():
    if sys.version_info < (3, 12):
        raise RuntimeError(
            f"KVIP builds require Python >= 3.12; selected {sys.executable} "
            f"({sys.version.split()[0]}). Set PYTHON to the serving interpreter."
        )
    for name in ("torch", "torch_npu"):
        try:
            importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(
                f"Cannot load {name} with {sys.executable}: {error}. "
                "Select the serving Python with torch/torch_npu and load its CANN environment."
            ) from error


def build_lidu(env, skip=False):
    source = ROOT / "csrc/operators/lidu"
    if skip:
        libraries = list((source / "_custom_opp/vendors").glob("*/op_api/lib/libcust_opapi.so"))
        extensions = list((source / "torch_extension/nanovllm_dsa_lidu_fused_batch_get_prepare_mtp").glob("_C*.so"))
        if len(libraries) != 1 or len(extensions) != 1:
            raise RuntimeError("--skip-lidu-build requires one existing LIDU vendor library and Python extension")
        print(f"REUSE LIDU: {libraries[0]} {extensions[0]}", flush=True)
        return
    call(["bash", source / "build_lidu_fused_batch_get_prepare_mtp.sh"], env=env)


def patch_path(name):
    directory = (
        ROOT / "patches"
        if name in (BACKGROUND_PATCH, UBMEM_PATCH, URMA_POST32_PATCH, URMA_BULK_PATCH, UBMEM_DISPATCH_PATCH)
        else ROOT / "scripts/patches"
    )
    return directory / (name + ".patch")


def applies(root, name, reverse=False):
    return (
        subprocess.run(
            ["git", "-C", str(root), "apply", *(["--reverse"] if reverse else []), "--check", str(patch_path(name))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def apply_patch(root, name):
    if not applies(root, name):
        raise RuntimeError(f"KVAF {name} conflicts with existing edits; checkout preserved")
    call(["git", "apply", patch_path(name)], cwd=root)


def verify_applied_patch_stack(root, names):
    """Verify oldest-to-newest patches using a private index, preserving user edits.

    A later patch can alter an earlier patch's added files, preventing a direct
    reverse check. Stage only this stack's paths in a disposable index and
    reverse there; never reset or edit the checkout or its real index.
    """
    paths = set()
    for name in names:
        stat = subprocess.check_output(["git", "apply", "--numstat", str(patch_path(name))], text=True)
        paths.update(line.split("\t", 2)[2] for line in stat.splitlines())
    with tempfile.TemporaryDirectory(prefix="kvip-patch-check-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        commands = [["read-tree", "HEAD"], ["add", "--", *sorted(paths)]]
        commands += [["apply", "--cached", "--reverse", str(patch_path(name))] for name in reversed(names)]
        for command in commands:
            if subprocess.run(
                ["git", "-C", str(root), *command],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode:
                return False
    return True


def prepare_dependency(root, transport="urma", urma_get_post32=False, ubmem_get_tiled=False, urma_get_bulk=False):
    if transport not in ("urma", "ubmem"):
        raise ValueError("transport must be urma or ubmem")
    if urma_get_post32 and transport != "urma":
        raise ValueError("--urma-get-post32 requires --transport urma")
    if ubmem_get_tiled and transport != "ubmem":
        raise ValueError("--ubmem-get-tiled requires --transport ubmem")
    if urma_get_bulk and transport != "urma":
        raise ValueError("--urma-get-bulk requires --transport urma")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", "https://gitcode.com/CacheOS/KVAF.git", root])
        call(["git", "checkout", "--detach", REVISION], cwd=root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise RuntimeError(f"Expected KVAF {REVISION}, found {head}; existing checkout left unchanged")
    if applies(root, UBMEM_DISPATCH_PATCH, True):
        if transport != "ubmem":
            raise RuntimeError("KVAF checkout already has the local UBmem patch; use a separate URMA checkout")
        if not verify_applied_patch_stack(root, (UBMEM_PATCH, UBMEM_DISPATCH_PATCH)):
            raise RuntimeError("KVAF UBmem patch stack conflicts with existing edits; checkout preserved")
        print(f"KVAF UBmem tiled dependency already patched: {root} @ {head}", flush=True)
        return
    # The UBmem patch changes common-patch contexts. Recognize it first, while
    # keeping the default URMA checkout on its original three-patch chain.
    if applies(root, UBMEM_PATCH, True):
        if transport != "ubmem":
            raise RuntimeError("KVAF checkout already has the local UBmem patch; use a separate URMA checkout")
        if ubmem_get_tiled:
            apply_patch(root, UBMEM_DISPATCH_PATCH)
        print(f"KVAF UBmem dependency already patched: {root} @ {head}", flush=True)
        return
    bulk_applied = applies(root, URMA_BULK_PATCH, True)
    if bulk_applied:
        if transport != "urma":
            raise RuntimeError("KVAF checkout has the URMA bulk patch; use a separate UBmem checkout")
        if not verify_applied_patch_stack(root, (URMA_POST32_PATCH, URMA_BULK_PATCH)):
            raise RuntimeError("KVAF URMA Get patch stack conflicts with existing edits; checkout preserved")
    # Applying output-views changes the parent patch's context: check newest first.
    if not applies(root, PATCHES[0], True):
        apply_patch(root, PATCHES[0])
    if not applies(root, PATCHES[-1], True):
        for name in PATCHES[1:]:
            if not applies(root, name, True):
                apply_patch(root, name)
    # Preserve the existing three-patch checkout when only its background Put
    # upgrade is missing; never reverse a patch or overwrite local source.
    if not applies(root, BACKGROUND_PATCH, True):
        apply_patch(root, BACKGROUND_PATCH)
    if transport == "ubmem":
        apply_patch(root, UBMEM_PATCH)
        if ubmem_get_tiled:
            apply_patch(root, UBMEM_DISPATCH_PATCH)
    elif not bulk_applied:
        if (urma_get_post32 or urma_get_bulk) and not applies(root, URMA_POST32_PATCH, True):
            apply_patch(root, URMA_POST32_PATCH)
        if urma_get_bulk:
            apply_patch(root, URMA_BULK_PATCH)
    print(f"KVAF {transport} dependency ready: {root} @ {head}", flush=True)


def build_get_object(build_dir, jobs, env, name, target):
    if not (build_dir / "CMakeCache.txt").is_file():
        raise RuntimeError(f"{name} requires an existing configured NPU build")
    # An existing Make generator cannot build a newly added target until it is
    # reconfigured. Keep its toolchain/options in the existing cache.
    call(["cmake", "-S", build_dir.parent, "-B", build_dir], env=env)
    call(["cmake", "--build", build_dir, "--target", target, "--parallel", jobs], env=env)
    if not (build_dir / "bin" / (name + ".o")).is_file():
        raise RuntimeError(f"Optional Get kernel was not produced: {name}")


def build_urma_post32(build_dir, jobs, env):
    build_get_object(build_dir, jobs, env, "kvaf_execute_batch_get_post32", "kvaf_execute_batch_get_post32_kernel")


def build_ubmem_tiled(build_dir, jobs, env):
    build_get_object(build_dir, jobs, env, "kvaf_ubmem_batch_get_tiled", "kvaf_ubmem_batch_get_tiled_kernel")


def build_urma_bulk(build_dir, jobs, env):
    build_get_object(build_dir, jobs, env, "kvaf_execute_batch_get_bulk", "kvaf_execute_batch_get_bulk_kernel")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvaf-root", type=Path, help="existing checkout; otherwise download to third_party/KVAF")
    parser.add_argument("--prepare-only", action="store_true", help="prepare pinned source, without CANN builds")
    parser.add_argument("--skip-kvaf-build", action="store_true", help="use an already rebuilt KVAF")
    parser.add_argument("--skip-lidu-build", action="store_true", help="reuse existing LIDU vendor/Python libraries")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--soc", default="Ascend950PR_9599", help="compile target architecture")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--transport", choices=("urma", "ubmem"), default="urma")
    parser.add_argument("--urma-get-post32", action="store_true", help="Build the optional URMA 32-WQE Get object")
    parser.add_argument("--urma-get-bulk", action="store_true", help="Build the optional URMA bulk-WQE Get object")
    parser.add_argument(
        "--ubmem-get-tiled", action="store_true", help="Build the optional UBmem tiled-dispatch Get object"
    )
    parser.add_argument(
        "--get-kernel-only",
        action="store_true",
        help="Build only the selected optional Get object in an existing build; keep the selected runtime unchanged",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.urma_get_post32 and args.transport != "urma":
        parser.error("--urma-get-post32 requires --transport urma")
    if args.ubmem_get_tiled and args.transport != "ubmem":
        parser.error("--ubmem-get-tiled requires --transport ubmem")
    if args.urma_get_bulk and args.transport != "urma":
        parser.error("--urma-get-bulk requires --transport urma")
    if args.get_kernel_only and not (args.ubmem_get_tiled or args.urma_get_post32 or args.urma_get_bulk):
        parser.error("--get-kernel-only requires --ubmem-get-tiled, --urma-get-post32 or --urma-get-bulk")
    if not args.prepare_only and not args.get_kernel_only:
        runtime_preflight()
    dependency = (args.kvaf_root or ROOT / "third_party/KVAF").resolve()
    prepare_dependency(
        dependency,
        args.transport,
        urma_get_post32=args.urma_get_post32,
        ubmem_get_tiled=args.ubmem_get_tiled,
        urma_get_bulk=args.urma_get_bulk,
    )
    (ROOT / "build").mkdir(exist_ok=True)
    build_dir = dependency / ("build-ubmem-npu" if args.transport == "ubmem" else "build-npu")
    state = dict(kvaf=str(dependency), revision=REVISION, transport=args.transport, kvaf_build_dir=str(build_dir))
    if args.urma_get_post32:
        state["urma_get_post32"] = True
    if args.ubmem_get_tiled:
        state["ubmem_get_tiled"] = True
    if args.urma_get_bulk:
        state["urma_get_bulk"] = True
    if args.prepare_only:
        # Preparing source must not switch a working Python environment to an
        # unbuilt dependency. Select it only after the build succeeds.
        return
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        ASCEND_CANN_PACKAGE_PATH=str(args.cann),
        SOC_VERSION=args.soc,
        KVAF_UMDK_ROOT=str(dependency / "third_party/umdk"),
        NANOVLLM_A5_OPS_BUILD_JOBS=str(args.jobs),
        NANOVLLM_A5_OPS_PYTHON=sys.executable,
    )
    if args.get_kernel_only:
        if args.ubmem_get_tiled:
            build_ubmem_tiled(build_dir, args.jobs, env)
        if args.urma_get_post32:
            build_urma_post32(build_dir, args.jobs, env)
        if args.urma_get_bulk:
            build_urma_bulk(build_dir, args.jobs, env)
        print("PASS: optional Get kernel built; selected runtime and other operators unchanged.", flush=True)
        return
    if args.transport == "ubmem":
        if not args.skip_kvaf_build:
            call(
                ["bash", dependency / "build.sh", "npu", "--transport", "ubmem", "--jobs", args.jobs],
                cwd=dependency,
                env=env,
            )
        for name in ("kvaf_ubmem_batch_get.o", "kvaf_ubmem_batch_put.o"):
            if not (build_dir / "bin" / name).is_file():
                raise RuntimeError(f"UBmem kernel missing: {build_dir / 'bin' / name}")
        if not list((build_dir / "python/kvaf").glob("*.so")):
            raise RuntimeError(f"UBmem Python extension missing: {build_dir / 'python/kvaf'}")
        if args.ubmem_get_tiled:
            build_ubmem_tiled(build_dir, args.jobs, env)
        build_lidu(env, args.skip_lidu_build)
        call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", ROOT])
        sys.path.insert(0, str(ROOT / "python"))
        from kvip.operators.lifetch_ubmem.specialization import Specialization, build_specialization

        # Initial no-weight smoke shapes. Serving prepares its exact history,
        # head count and device AIC budget once before memory profiling.
        for batch in (1, 2):
            for token_bytes in (656, 2624):
                build_specialization(
                    cache_dir=build_dir / "lifetch_ubmem",
                    cann=args.cann,
                    shape=Specialization(batch, 33024, 8192, 64, token_bytes, 32, 8, args.soc),
                    jobs=args.jobs,
                )
        (ROOT / "build/dependencies.json").write_text(json.dumps(state, indent=2))
        print("PASS: KVAF UBmem Put/Get + LIDU + UBmem LIFetch/Index update built. Source scripts/env.sh.", flush=True)
        return
    if not args.skip_kvaf_build:
        for target in ("cpu", "npu"):
            call(
                [
                    "bash",
                    dependency / "build.sh",
                    target,
                    "--jobs",
                    args.jobs,
                    *(["--without-aicpu"] if target == "npu" else []),
                ],
                cwd=dependency,
                env=env,
            )
    if args.urma_get_post32:
        build_urma_post32(build_dir, args.jobs, env)
    if args.urma_get_bulk:
        build_urma_bulk(build_dir, args.jobs, env)
    build_lidu(env, args.skip_lidu_build)
    # Editable install intentionally retains csrc alongside the Python package.
    call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", ROOT])
    sys.path.insert(0, str(ROOT / "python"))
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    cann, headers = kvaf_build_paths(dependency / "build-npu")
    print(f"BUILD universal LIFetch; logs under {ROOT / 'build/lifetch'}", flush=True)
    paths = build_libraries(cache_dir=ROOT / "build/lifetch", cann=cann, headers=headers, soc=args.soc)
    (ROOT / "build/dependencies.json").write_text(json.dumps(state, indent=2))
    print(f"READY: {paths}", flush=True)
    print("PASS: KVAF + LIDU + LIFetch/Index update built. Source scripts/env.sh before serving.", flush=True)


if __name__ == "__main__":
    main()
