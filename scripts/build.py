#!/usr/bin/env python3
"""Build pinned KVAF, original LIDU and the complete LIFetch/Index update library."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c4375e13d7a9519d907ab6a5d11227795f207286"
PATCHES = ("kvaf-reuse-device-ra", "kvaf-fused-context", "kvaf-fused-output-views")
UBMEM_PATCH = "kvaf-ubmem-runtime"


def call(command, cwd=None, env=None):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def applies(root, name, reverse=False):
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "apply",
                *(["--reverse"] if reverse else []),
                "--check",
                str(ROOT / "scripts/patches" / (name + ".patch")),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def prepare_dependency(root, transport="urma"):
    if transport not in ("urma", "ubmem"):
        raise ValueError("transport must be urma or ubmem")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", "https://gitcode.com/CacheOS/KVAF.git", root])
        call(["git", "checkout", "--detach", REVISION], cwd=root)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise RuntimeError(f"Expected KVAF {REVISION}, found {head}; existing checkout left unchanged")
    # The UBmem patch changes shared Runtime/binding contexts from earlier
    # patches. Recognize the complete final patch before examining its parents.
    if (ROOT / "scripts/patches" / (UBMEM_PATCH + ".patch")).is_file() and applies(root, UBMEM_PATCH, True):
        if transport != "ubmem":
            raise RuntimeError("This KVAF tree contains UBmem changes; use a separate URMA baseline tree")
        print(f"KVAF UBmem dependency ready: {root} @ {head}", flush=True)
        return
    # Applying output-views changes the parent patch's context: check newest first.
    if not applies(root, PATCHES[0], True):
        if not applies(root, PATCHES[0]):
            raise RuntimeError("KVAF base patch conflicts with existing edits; checkout preserved")
        call(["git", "apply", ROOT / "scripts/patches" / (PATCHES[0] + ".patch")], cwd=root)
    if not applies(root, PATCHES[-1], True):
        for name in PATCHES[1:]:
            if applies(root, name, True):
                continue
            if not applies(root, name):
                raise RuntimeError(f"KVAF {name} conflicts with existing edits; no forced reset")
            call(["git", "apply", ROOT / "scripts/patches" / (name + ".patch")], cwd=root)
    if transport == "ubmem":
        if not applies(root, UBMEM_PATCH):
            raise RuntimeError("KVAF UBmem patch conflicts with existing edits; checkout preserved")
        call(["git", "apply", ROOT / "scripts/patches" / (UBMEM_PATCH + ".patch")], cwd=root)
    print(f"KVAF dependency ready: {root} @ {head}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kvaf-root", type=Path, help="existing checkout; otherwise download to third_party/KVAF")
    parser.add_argument("--prepare-only", action="store_true", help="prepare patches, without CANN builds")
    parser.add_argument("--skip-kvaf-build", action="store_true", help="use an already rebuilt, patched KVAF")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--soc", default="Ascend950PR_9599", help="compile target architecture")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--transport", choices=("urma", "ubmem"), default="urma")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    dependency = (args.kvaf_root or ROOT / "third_party/KVAF").resolve()
    prepare_dependency(dependency, args.transport)
    (ROOT / "build").mkdir(exist_ok=True)
    build_dir = dependency / ("build-ubmem-npu" if args.transport == "ubmem" else "build-npu")
    state = dict(kvaf=str(dependency), revision=REVISION, transport=args.transport, kvaf_build_dir=str(build_dir))
    if args.prepare_only:
        # Preparing source must not switch a working Python environment to an
        # unbuilt dependency. Select it only after the build succeeds.
        return
    env = dict(
        os.environ,
        ASCEND_CANN_PACKAGE_PATH=str(args.cann),
        SOC_VERSION=args.soc,
        KVAF_UMDK_ROOT=str(dependency / "third_party/umdk"),
        NANOVLLM_A5_OPS_BUILD_JOBS=str(args.jobs),
        NANOVLLM_A5_OPS_PYTHON=sys.executable,
    )
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
        (ROOT / "build/dependencies.json").write_text(json.dumps(state, indent=2))
        print("PASS: KVAF UBmem Put/Get runtime built. UBmem LIFetch is not integrated yet.", flush=True)
        return
    if not args.skip_kvaf_build:
        for target in ("cpu", "npu"):
            call(["bash", dependency / "build.sh", target, "--jobs", args.jobs], cwd=dependency, env=env)
    call(["bash", ROOT / "csrc/operators/lidu/build_lidu_fused_batch_get_prepare_mtp.sh"], env=env)
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
