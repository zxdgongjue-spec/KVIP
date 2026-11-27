# SPDX-License-Identifier: Apache-2.0
"""Resume a candidate-only four-rank replay from its original archive and timed code."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ENTRY = Path("tests/operators/lifetch")
BATCHES = (32, 28, 24, 20)
GROUPS = (7, 2, 10, 4, 17)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def checkout(repo, sources, revision, pattern):
    for path in sorted(sources.glob(pattern + ".*")):
        if (path / ".git").is_file() and git(path, "rev-parse", "HEAD") == revision:
            if not git(path, "status", "--porcelain", "--untracked-files=no"):
                return path
    path = Path(tempfile.mkdtemp(prefix=pattern + ".", dir=sources))
    git(repo, "worktree", "add", "-q", "--detach", str(path), revision)
    return path


def replay(root, control, timed, native):
    env = os.environ.copy()
    for name in ("KVAF_CPU_UB_DEVICE", "PEER_BS32_LI", "PEER_PACKED_REFERENCE"):
        env.pop(name, None)
    card = (root / "physical_npu.txt").read_text().strip()
    env.update(PEER_G2_WORK=str(root.parent.parent), PEER_PACKED_RESUME=str(root),
               PEER_TIMED_SOURCE=str(timed), PEER_LOCAL_ONLY="1", PEER_PHYSICAL_NPU=card,
               ASCEND_RT_VISIBLE_DEVICES=card, PEER_PACKED_BLOCKS="1", PEER_LARGE_COLD_WINDOWS="3",
               PEER_FOUR_WINDOW_RANKS="1", PEER_FOUR_WINDOW_CARRY="0",
               FIXED_Q4_GROUPS=" ".join(map(str, GROUPS)), FIXED_Q4_BATCHES=" ".join(map(str, BATCHES)))
    sys.path.insert(0, str(control / ENTRY / "support"))
    from replay_interrupt_evidence import snapshot

    def save_evidence(when):
        try:
            with (root / "interruption.jsonl").open("a") as stream:
                stream.write(json.dumps({"when": when, "evidence": snapshot(root)}) + "\n")
        except (OSError, ValueError) as error:
            print(f"EVIDENCE_UNAVAILABLE {error}", flush=True)

    save_evidence("before_resume")
    try:
        subprocess.run(["bash", str(control / ENTRY / "large_batch_npu4.sh"), "--four-ranks-candidate-only"],
                       stdin=subprocess.DEVNULL, env=env, check=True)
    finally:
        save_evidence("after_replay")
    from report_large_li_phases import report
    rows = report(root, native, batches=BATCHES, groups=GROUPS)
    expected = {(g, b) for g in GROUPS for b in BATCHES}
    if len(rows) != len(expected) or {(r["group"], r["batch"]) for r in rows} != expected:
        raise ValueError("Incomplete twenty-cell phase report")
    report_file = root / "li_phase_comparison.json"
    stage = report_file.with_suffix(".tmp")
    stage.write_text(json.dumps(rows, indent=2) + "\n")
    stage.replace(report_file)
    print(f"RESUME_COMPLETE cells={len(rows)} PHASE_REPORT={report_file}", flush=True)


def start(repo, revision, root):
    if root.parent.name != "peer-baseline" or not root.name.startswith("fixed-q4-four-ranks-one-block."):
        raise ValueError(f"Expected an existing one-block four-rank archive, got {root}")
    def field(name):
        return (root / (name + ".txt")).read_text().strip()
    if field("candidate_blocks") != "1" or field("selection_oracle") != "captured_logical_scores":
        raise ValueError("This entry point resumes a candidate-only ten-call graph")
    card = field("physical_npu")
    if card not in ("3", "4") or field("host") != os.uname().nodename:
        raise ValueError("Archive hostname or physical card does not match")
    local_machine = Path("/etc/machine-id")
    machine = local_machine.read_text().strip() if local_machine.is_file() else ""
    archived_machine = field("machine_id") if (root / "machine_id.txt").is_file() else ""
    if machine and archived_machine:
        same_host = machine == archived_machine
    else:
        same_host = field("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not same_host:
        raise ValueError("Archive belongs to a different machine")
    native = field("native")
    if git(repo, "rev-parse", "--verify", native + "^{commit}") != native:
        raise ValueError(f"Missing original timed revision {native}")
    if git(repo, "rev-parse", "--verify", revision + "^{commit}") != revision:
        raise ValueError(f"Expected a full controller commit ID, got {revision}")
    sources = root.parent.parent / "fused-scan-sources"
    log = root / "resume.log"
    with (root / ".four-ranks-resume.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"ALREADY_RUNNING LOG={log}")
            return
        controller = checkout(repo, sources, revision, "four-ranks-resume")
        timed = checkout(repo, sources, native, f"four-ranks-npu{card}")
        with log.open("a") as output:
            print(f"RESUME_BEGIN utc={datetime.now(timezone.utc).isoformat()} physical_NPU={card} "
                  f"native={native} controller={revision} result={root}", file=output, flush=True)
            # Only the supervisor inherits the archive lock. Its workers take the existing physical-card lock.
            child = subprocess.Popen(
                [sys.executable, "-Bu", str(controller / ENTRY / "resume_four_ranks.py"),
                 "--worker", str(controller), str(timed), native, str(root)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),), start_new_session=True)
    print(f"RESUME_STARTED pid={child.pid} LOG={log}")


def main():
    if sys.argv[1:2] == ["--worker"]:
        control, timed, native, root = sys.argv[2:]
        try:
            replay(Path(root), Path(control), Path(timed), native)
        except subprocess.CalledProcessError as error:
            print(f"RESUME_FAILED child_returncode={error.returncode} result={root}", flush=True)
            return error.returncode if error.returncode >= 0 else 128 - error.returncode
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("revision")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    start(args.repo.resolve(strict=True), args.revision, args.archive.resolve(strict=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
