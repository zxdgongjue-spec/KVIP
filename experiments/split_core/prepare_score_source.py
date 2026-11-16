"""Stage separate current/historical score builds without rewriting pinned replay artifacts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OP = Path("csrc/operators/lightning_indexer_with_fetch")
PIN = "81b1472809b7a5f3469e26467feee0ca1c61d71a"


def extract_source(archive, output):
    if hasattr(tarfile, "data_filter"):
        archive.extractall(output, filter="data")
    else:
        # Python 3.9 host: Git source files/directories only, never links/devices.
        for member in archive.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unsupported source archive entry: " + member.name)
        archive.extractall(output)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source anchor changed: " + old[:80])
    return text.replace(old, new)


def adapt_historical(root):
    path = root / OP / "op_kernel/indexer/indexer.h"
    s = path.read_text()
    s = replace_once(
        s,
        "    rotating::RuntimeShape runtimeShape_;",
        "    rotating::RuntimeShape runtimeShape_;\n    bool scoreOnly_ = false;\n    uint32_t scoreQueryGroup_ = 4;",
    )
    s = replace_once(
        s,
        "if (tilingData->splitQ && constInfo.tSize <= 2 * tilingData->aicCount)",
        "if ((scoreOnly_ && scoreQueryGroup_ == 2) || (tilingData->splitQ && constInfo.tSize <= 2 * tilingData->aicCount))",
    )
    s = replace_once(
        s,
        "    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);",
        """    if (scoreOnly_) {
        splitCoreInfo.bN2Start = splitCoreInfo.bN2End = aiCoreIdx;
        splitCoreInfo.s2End = actualSeqLengthsGm.GetValue(aiCoreIdx) / S2_BASE_SIZE - 1;
        splitCoreInfo.isCoreEnable = aiCoreIdx < usedCoreNum;
    } else { SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo); }""",
    )
    anchor = "    CalcRunInfo(loop, s2LoopIdx, runInfo, qScaleLoop, kScaleLoop);\n"
    s = replace_once(
        s,
        anchor,
        anchor
        + """    if (scoreOnly_) {
        if ASCEND_IS_AIC { matmulService.ComputeMm1(runInfo); }
        else { vectorService.ProcessVec1(runInfo); }
        return;
    }
""",
    )
    path.write_text(s)
    # Retain historical arithmetic. Return a scale bank after its last consumer,
    # including short range tails; the archived replay source remains unchanged.
    path = root / OP / "op_kernel/indexer/indexer_vector.h"
    s = path.read_text()
    s = replace_once(
        s,
        "    if ((info.s2Idx - info.s2Start) % 16 == 0) {\n        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);",
        "    if ((info.s2Idx - info.s2Start + 1) % 16 == 0 || (info.s2Idx + 1) * s2BaseSize_ >= info.validS2Len || info.isLastS2InnerLoop) {\n        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);",
    )
    path.write_text(s)


def prepare(output, version):
    if version not in ("current", "historical"):
        raise ValueError("unknown operator version")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if version == "current":
        shutil.copytree(ROOT / "csrc/operators", output / "csrc/operators")
        base = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    else:
        snap = ROOT / "experiments/split_core/historical"
        manifest = json.loads((snap / "source_manifest.json").read_text())
        if manifest["base"] != PIN:
            raise ValueError("pinned source mismatch")
        for rel, sha in manifest["files"].items():
            if hashlib.sha256((snap / rel).read_bytes()).hexdigest() != sha:
                raise ValueError("archive source modified: " + rel)
        raw = subprocess.check_output(["git", "-C", str(ROOT), "archive", PIN, "csrc/operators"])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            extract_source(archive, output)
        for variant in ("base", "history"):
            shutil.copytree(snap / "overlay" / variant, output, dirs_exist_ok=True)
        adapt_historical(output)
        base = PIN
    records = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((output / "csrc/operators").rglob("*"))
        if p.is_file()
    }
    evidence = dict(
        version=version,
        base_commit=base,
        source_files=records,
        scope="score-only adaptation, not full streaming",
        historical_archive_unchanged=True,
    )
    (output / "score-source.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--version", required=True, choices=("current", "historical"))
    a = p.parse_args()
    result = prepare(a.output, a.version)
    print(
        json.dumps(
            dict(version=result["version"], base_commit=result["base_commit"], files=len(result["source_files"]))
        )
    )
