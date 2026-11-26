#!/usr/bin/env python3
"""Build pinned KVAF, original LIDU and the complete LIFetch/Index update library."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "87e261ae4d07d4868cfdbb306d15f3f137edfc3d"


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


def prepare_dependency(root, transport):
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", "--branch", "main",
              "https://gitcode.com/CacheOS/KVAF.git", root])
        call(["git", "checkout", "--detach", REVISION], cwd=root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise RuntimeError(f"Expected KVAF main revision {REVISION}, found {head}")
    patches = [
        (ROOT / "scripts/patches/kvaf-ddr-create-timeout.patch", ()),
        (ROOT / "scripts/patches/kvaf-main-fused-device.patch", ()),
    ]
    lease_fix = ROOT / "scripts/patches/kvaf-prepared-get-lease.patch"
    lease_command = ["git", "-C", str(root), "apply"]
    lease_fixed = subprocess.run(
        [*lease_command, "--reverse", "--check", lease_fix], capture_output=True
    ).returncode == 0
    if not lease_fixed:
        patches.append((ROOT / "scripts/patches/kvaf-main-fused-resources.patch", ()))
    if transport == "ubmem":
        patches.append((ROOT / "patches/kvaf-ubmem-device-get.patch", ("--include=access_operator/include/ubmem/*",)))
    pending = []
    for path, options in patches:
        command = ["git", "-C", str(root), "apply", *options]
        reverse = subprocess.run([*command, "--reverse", "--check", path], capture_output=True)
        if reverse.returncode != 0:
            call([*command, "--check", path])
            pending.append([*command, path])
    for command in pending:
        call(command)
    if not lease_fixed:
        call([*lease_command, lease_fix])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvaf-root", type=Path, help="existing checkout; otherwise download to third_party/KVAF")
    parser.add_argument("--prepare-only", action="store_true", help="prepare pinned source, without CANN builds")
    parser.add_argument("--with-aicpu", action="store_true", help="build and install KVAF AICPU backends")
    parser.add_argument("--skip-kvaf-build", action="store_true", help="use an already rebuilt KVAF")
    parser.add_argument("--skip-lidu-build", action="store_true", help="reuse existing LIDU vendor/Python libraries")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--soc", default="Ascend950PR_9599", help="compile target architecture")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--transport", choices=("urma", "ubmem"), default="urma")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.prepare_only:
        runtime_preflight()
    dependency = (args.kvaf_root or ROOT / "third_party/KVAF").resolve()
    prepare_dependency(dependency, args.transport)
    (ROOT / "build").mkdir(exist_ok=True)
    build_dir = dependency / "build-npu"
    state = dict(kvaf=str(dependency), revision=REVISION, transport=args.transport, kvaf_build_dir=str(build_dir))
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
    if not args.skip_kvaf_build:
        call(["bash", dependency / "build.sh", "cpu", "--jobs", args.jobs], cwd=dependency, env=env)
        # Deployment builds do not need KVAF's host model tests. The pinned
        # script hardcodes BUILD_TESTING, so override that CMake argument only.
        npu_build = """
cmake() {
    local -a args=()
    local arg
    for arg in "$@"; do
        if [[ "$arg" == -DBUILD_TESTING=ON ]]; then arg=-DBUILD_TESTING=OFF; fi
        args+=("$arg")
    done
    command cmake "${args[@]}"
}
export -f cmake
exec bash "$@"
"""
        call(
            [
                "bash", "-c", npu_build, "kvaf-npu-build",
                dependency / "build.sh", "npu", "--jobs", args.jobs,
                *([] if args.with_aicpu else ["--without-aicpu"]),
            ],
            cwd=dependency, env=env,
        )
    for kind in ("get", "put"):
        kernel = build_dir / "bin" / f"kvaf_execute_batch_{kind}_kernels.o"
        if not kernel.is_file():
            raise RuntimeError(f"Unified KVAF kernel missing: {kernel}")
    package = build_dir / "python/kvaf"
    if not (package / "__init__.py").is_file() or not list(package.glob("*.so")):
        raise RuntimeError(f"Unified KVAF Python extension missing: {package}")
    bindings = ["get_graph_urma_resources"]
    if args.transport == "ubmem":
        bindings.append("get_graph_mapped_base")
    call(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import kvaf; "
         "[getattr(kvaf.NpuRuntime, name) for name in sys.argv[2:]]",
         build_dir / "python", *bindings],
        env=env,
    )
    verifier = dependency / "bindings/verify_python_binding.py"
    call(
        [sys.executable, "-I", "-c",
         "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
         "sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name='__main__')",
         build_dir / "python", verifier, "--require-npu"],
        env=env,
    )
    build_lidu(env, args.skip_lidu_build)
    call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", ROOT])
    sys.path.insert(0, str(ROOT / "python"))
    if args.transport == "ubmem":
        from kvip.operators.lifetch_ubmem.build import UbmemGeometry, build_ubmem_library

        # One library contains both KV widths and accepts runtime geometry.
        build_ubmem_library(
            cache_dir=build_dir / "lifetch_ubmem",
            cann=args.cann,
            shape=UbmemGeometry(1, 33024, 8192, 64, 656, 32, 8, args.soc),
            jobs=args.jobs,
        )
        (ROOT / "build/dependencies.json").write_text(json.dumps(state, indent=2))
        print("PASS: KVAF UBmem Put/Get + LIDU + UBmem LIFetch/Index update built. Source scripts/env.sh.", flush=True)
        return
    from kvip.runtime.build import build_libraries, kvaf_build_paths

    cann, headers = kvaf_build_paths(dependency / "build-npu")
    print(f"BUILD universal LIFetch; logs under {ROOT / 'build/lifetch'}", flush=True)
    paths = build_libraries(cache_dir=ROOT / "build/lifetch", cann=cann, headers=headers, soc=args.soc)
    (ROOT / "build/dependencies.json").write_text(json.dumps(state, indent=2))
    print(f"READY: {paths}", flush=True)
    print("PASS: KVAF + LIDU + LIFetch/Index update built. Source scripts/env.sh before serving.", flush=True)


if __name__ == "__main__":
    main()
