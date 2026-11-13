"""Build scoped visualizations from actual msprof JSON and native Model footers.

Raw instruction traces remain local. Exported occupancy is the exact interval
union per pipe; it is a derived view, not an end-to-end hardware timeline.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile

FOOTER = re.compile(
    r"core:\s*(\d+),\s*subcore:\s*(\d+),\s*start:\s*(\d+),\s*tick:\s*(\d+),.*?"
    r"blkID:\s*(\d+),\s*blkDim:\s*(\d+),.*?done\++"
)
ARCHIVE_SHA256 = {
    "mainv4-baseline-zero-b1-h32-r1.tar.gz": "70c1529c99d62e0d476dd904fbf4c151ab3805c0e7559f02be67cb1a96f698c4",
    "mainv4-candidate-zero-b1-h32-r1.tar.gz": "b01b81860a69c3d7d1904ccf1dddd270861e72718ce5a2a9282b9e3aee9a3fc4",
}
STAGES = ("Init", "LI", "Resolve", "PublishPlan", "Apply", "FetchUpdate", "Fusion3 Tail")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def footer(data):
    found = FOOTER.findall(data.decode(errors="replace") if isinstance(data, bytes) else data)
    if len(found) != 1:
        raise ValueError(f"expected one complete native footer, got {len(found)}")
    row = dict(zip(("core", "subcore", "start", "tick", "block", "blocks"), map(int, found[0])))
    if not (row["tick"] > row["start"] >= 0 and 0 <= row["block"] < row["blocks"]):
        raise ValueError("invalid native envelope")
    return row


def interval_union(intervals):
    result = []
    for a, b in sorted(intervals):
        if not (math.isfinite(a) and math.isfinite(b) and b >= a):
            raise ValueError("invalid trace interval")
        if b == a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def occupancy(events):
    lanes = collections.defaultdict(list)
    count = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        start, duration = float(e["ts"]), float(e.get("dur", 0))
        if duration < 0:
            raise ValueError("negative instruction duration")
        lanes[(str(e["pid"]), str(e["tid"]))].append((start, start + duration))
        count += 1
    result = [{"core": core, "pipe": pipe, "intervals": interval_union(v)} for (core, pipe), v in sorted(lanes.items())]
    if not result or not any(r["intervals"] for r in result):
        raise ValueError("no instruction intervals")
    origin = min(a for r in result for a, b in r["intervals"])
    end = max(b for r in result for a, b in r["intervals"])
    # Keep vendor timestamps intact; the viewer subtracts origin only for display.
    return dict(x_events=count, origin=origin, end=end, lanes=result)


def stage(name):
    if "fusion3_tail" in name:
        return "Fusion3 Tail"
    for key, label in (
        ("init", "Init"),
        ("li_kernel", "LI"),
        ("resolve", "Resolve"),
        ("update_outputs", "PublishPlan"),
        ("apply_owned", "Apply"),
        ("fetch_update", "FetchUpdate"),
    ):
        if key in name:
            return label
    raise ValueError(f"unknown historical kernel {name}")


def archive(path, expected):
    if ARCHIVE_SHA256.get(Path(path).name) != sha(path):
        raise ValueError("historical import is pinned to the audited zero-miss archives")
    rows = []
    with tarfile.open(path, "r|gz") as stream:
        for member in stream:
            if (
                "_model_raw_snapshot/" not in member.name
                or not member.name.endswith(".instr_log.dump")
                or member.size == 0
            ):
                continue
            data = stream.extractfile(member).read()
            record = footer(data)
            kernel = member.name.split("_model_raw_snapshot/", 1)[1].split("/", 1)[0]
            rows.append(
                dict(
                    kernel=kernel,
                    stage=stage(kernel),
                    raw_member=member.name,
                    raw_sha256=hashlib.sha256(data).hexdigest(),
                    **record,
                )
            )
    if len(rows) != expected:
        raise ValueError(f"unexpected canonical footer count {len(rows)}")
    origin, end = min(r["start"] for r in rows), max(r["tick"] for r in rows)
    return dict(
        archive=Path(path).name,
        archive_sha256=sha(path),
        canonical_rows=rows,
        native_origin=origin,
        native_end=end,
        native_span=end - origin,
        clock="native Model start/tick; not vendor trace us or hardware time",
    )


def collect(current, repo, output, baseline, fused):
    version = json.loads((repo / ".aiinfra/tasks/SPLIT-CORE-007/tool-version-observation.json").read_text())
    source = json.loads((repo / ".aiinfra/tasks/SPLIT-CORE-007/capture-source.json").read_text())
    progress = json.loads((repo / ".aiinfra/tasks/SPLIT-CORE-007/capture-progress.json").read_text())
    assert len(progress) == 2 and all(r["exit_code"] == 0 and not r["error"] for r in progress)
    profiles, variants = [], []
    export = current / "exports"
    export.mkdir(exist_ok=True)
    for t in (1, 8):
        folder = current / f"t{t}"
        log = (folder / "capture.log").read_text()
        oracle = json.loads((folder / "oracle.json").read_text())
        assert oracle["batch"] == 1 and oracle["history"] == oracle["valid"] == 131072
        assert oracle["shards"] == t and oracle["aiv_budget"] == 56 and oracle["aic_workers"] == 0
        assert oracle["status"] == "passed" and all(
            oracle[k] for k in ("guards", "selected_set_and_scores", "all_local_candidates", "input_unchanged")
        )
        ticks = re.findall(r"Total tick:\s*(\d+)", log)
        assert len(ticks) == log.count("Model stopped successfully.") == 1 and "All task success." in log
        variants.append(
            dict(
                shards=t,
                whole_model_ticks=int(ticks[0]),
                oracle_sha256=sha(folder / "oracle.json"),
                capture_log_sha256=sha(folder / "capture.log"),
                warnings=[s for s in log.splitlines() if "[WARN]" in s],
            )
        )
        paths = sorted(folder.glob("profile/**/simulator/trace.json"))
        assert len(paths) == (1 if t == 1 else 2)
        for p in paths:
            phase = "merge" if "split_topk_merge" in str(p) else "local"
            data = json.loads(p.read_text())
            events = data["traceEvents"] if isinstance(data, dict) else data
            native = []
            dump = p.parent.parent / "dump"
            for raw in sorted(dump.glob("*.instr_log.dump")):
                if raw.stat().st_size:
                    record = footer(raw.read_bytes())
                    assert record["core"] == 0 and record["subcore"] in (1, 2)
                    expected_blocks = min(4 * t, 56) if phase == "local" else 4
                    assert record["blocks"] == expected_blocks
                    native.append(dict(file=raw.name, sha256=sha(raw), **record))
            assert len(native) == 2 and {r["subcore"] for r in native} == {1, 2}
            fixed = export / f"topk-t{t}-{phase}.trace.json"
            shutil.copyfile(p, fixed)
            item = dict(
                label=f"T{t} {phase}",
                shards=t,
                phase=phase,
                raw_trace=str(fixed.relative_to(repo)),
                raw_trace_sha256=sha(p),
                raw_trace_bytes=p.stat().st_size,
                native_core_footers=native,
                sampled_blocks=len(native),
                launched_blocks=expected_blocks,
                vendor_clock="original msprof simulator ts/dur (us fields); no hardware calibration",
                **occupancy(events),
            )
            profiles.append(item)
    historical = {"serial6": archive(baseline, 12), "Fusion3": archive(fused, 5)}
    evidence = dict(
        schema_version=1,
        source=source,
        tool=version,
        current_scope="NEW Model capture: TopK only B1/Q4/H131072/G32/m205/V56, log mask0x1, core0 V0/V1 only",
        historical_scope="IMPORTED raw capture: B1/Q4/H8192/R4096/G32/zero-MISS/Service0/GM-copy; LI splitQ0, 1C/2V",
        historical_collector_exit_code=-15,
        historical_collector_status="terminated; Model/oracle and raw coverage passed; not clean collector exit",
        physical_npu=False,
        variants=variants,
        profiles=[{k: v for k, v in p.items() if k != "lanes"} for p in profiles],
        historical=historical,
        limitations=[
            "Current complete streaming has no matching full-chain capture",
            "P32/T8 is not integrated into production Final/Service",
            "Only selected core0 V0/V1, not all active blocks",
            "Vendor trace and native Model counters shown in separate domains",
            "Pipe occupancy is an exact interval union, not instruction details or stall attribution",
            "Raw full Chrome trace stays local",
            "Historical smoke is not the128K standard, and was not rerun in this task",
        ],
    )
    (output / "results/scheduling-trace-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return profiles, historical, evidence


def figures(output, profiles, historical):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({"font.family": "Arial Unicode MS", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(15.2, 6.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, 4.3)
    ax.axis("off")
    paths = [
        (
            "旧 serial6",
            ["Init", "LI combined\nQ2/AIC 或 Q4/AIC", "Resolve", "Outputs / Plan", "Apply", "Fetch / Update"],
        ),
        ("历史 Fusion3", ["Init", "LI combined", "Tail：Resolve → Publish\n→ Plan/Apply → Finish"]),
        (
            "当前生产 P2",
            [
                "Init",
                "Hot：两history owner",
                "Resident threshold",
                "Cold：分window + Service",
                "Final：全H TopK T1\n+ classification",
                "同kernel尾部\n3B + 4W 静态角色",
            ],
        ),
        (
            "P/T 研究入口",
            [
                "Init",
                "Q4 × P history score",
                "每query T片 local-K",
                "T>1：merge T×K → K",
                "旧 mock 尾部\nService0，非生产",
            ],
        ),
    ]
    for i, (label, boxes) in enumerate(paths):
        y = 3.3 - i
        ax.text(0.05, y + 0.05, label, fontsize=11, va="center", fontweight="bold")
        width = 13.6 / len(boxes)
        for j, text in enumerate(boxes):
            x = 2 + j * width
            color = "#dceaf5" if j in (1, 2, 3, 4) else "#e0eee9"
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.31), width - 0.16, 0.64, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#526579"
                )
            )
            ax.text(x + (width - 0.16) / 2, y + 0.015, text, ha="center", va="center", fontsize=9)
            if j < len(boxes) - 1:
                ax.annotate(
                    "",
                    xy=(x + width - 0.02, y),
                    xytext=(x + width - 0.15, y),
                    arrowprops={"arrowstyle": "->", "color": "#526579"},
                )
    ax.text(8, 4.05, "全通路启动/依赖示意：宽度和间隙不代表时间", ha="center", fontsize=16, fontweight="bold")
    ax.text(
        8,
        -0.35,
        "当前 P2 来自上游13d；本分支独立预算/F4不等于生产P32/T8。当前六launch与旧serial6含义不同。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output / "scheduling-sequence.png", dpi=170)
    fig.savefig(output / "scheduling-sequence.svg")
    plt.close(fig)
    colors = {
        "SCALAR": "#63758d",
        "SCALARLDST": "#a1afc1",
        "VECTOR": "#138f87",
        "MTE2": "#e9a84a",
        "MTE3": "#d76b63",
        "RVECEX": "#5a68bd",
    }
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.8))
    max_end = max(p["end"] - p["origin"] for p in profiles)
    for ax, p in zip(axes, profiles):
        lanes = [r for r in p["lanes"] if r["core"] == "core0.veccore0" and r["pipe"] in colors]
        for i, r in enumerate(lanes):
            ax.broken_barh(
                [(a - p["origin"], b - a) for a, b in r["intervals"]], (i - 0.33, 0.66), facecolors=colors[r["pipe"]]
            )
        ax.set_yticks(range(len(lanes)), [r["pipe"] for r in lanes], fontsize=8)
        ax.set_ylim(-0.7, len(lanes) - 0.3)
        ax.set_xlim(0, max_end * 1.02)
        ax.set_title(
            f"{p['label']}  | core0.veccore0 | {p['x_events']:,} 原始X事件（双子核合计）", fontsize=10, loc="left"
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("各kernel首个记录事件归零的 msprof 模拟时间（原始us字段；非真机时间）")
    fig.suptitle("本次新采集：TopK T1 / T8（含额外merge），仅单core family采样", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "每个panel独立归零，不跨kernel拼接；色块是同PIPE指令占用区间union，不能当整条流水耗时或stall归因。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output / "topk-msprof-pipes.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.8))
    palette = dict(zip(STAGES, ["#98a4b4", "#4385b5", "#ad75b5", "#43978d", "#e6a044", "#c96c66", "#38826f"]))
    for ax, (name, d) in zip(axes, historical.items()):
        lanes = sorted({(r["core"], r["subcore"]) for r in d["canonical_rows"]})
        for r in d["canonical_rows"]:
            y = lanes.index((r["core"], r["subcore"]))
            a = r["start"] - d["native_origin"]
            dur = r["tick"] - r["start"]
            ax.broken_barh([(a, dur)], (y - 0.33, 0.66), facecolors=palette[r["stage"]])
            ax.text(a + dur / 2, y, r["stage"], ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(lanes)), [f"core{c}/sub{s}" for c, s in lanes], fontsize=8)
        ax.set_ylim(-0.7, len(lanes) - 0.3)
        ax.set_xlim(0, 245000)
        ax.set_title(
            f"{name}：{len(d['canonical_rows'])}份canonical完成记录，native包络{d['native_span']:,} ticks",
            loc="left",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    axes[-1].set_xlabel("各run首个kernel归零的 native Model counter ticks（不是vendor us，不是真机）")
    fig.suptitle("历史raw导入：serial6 / Fusion3 | B1/Q4/H8192/R4096/G32/zero-miss", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "两侧LI均splitQ0、1C/2V；来源为旧msprof raw footer，非本次重跑、非128K性能；collector退出-15保留。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output / "historical-fusion3-native-envelope.png", dpi=170)
    plt.close(fig)


HTML = """<!doctype html><meta charset="utf-8"><title>分核逻辑与msprof证据</title>
<style>body{font:15px system-ui;background:#f5f7fa;color:#182b42;margin:24px auto;max-width:1450px;padding:0 20px}h1{font-size:27px}section{background:white;border:1px solid #d5dfe8;border-radius:12px;padding:16px;margin:18px 0}canvas{width:100%;display:block;border-top:1px solid #e4e9ef}p{line-height:1.65}.note{color:#53667c}button,select,input{margin:5px;padding:5px}img{width:100%}.warn{background:#fff3d8;padding:12px;border-radius:8px}small{color:#53667c}</style>
<h1>当前全通路分核：静态依赖与实测证据</h1>
<p class="warn">当前生产是条件P2、Final T1和静态3B+4W尾部；P32/T8未完整接入。下方分成新TopK仿真、旧Fusion3小形状和静态示意三种证据，不能拼成当前全链性能。</p>
<section><h2>① 本次新抓 msprof：TopK T1 / T8</h2><p>B1 / Q4 / H131072 / G32 / miss205 / V56，同driver和score；只有core0的V0/V1。色块是原始指令占用的同PIPE区间union，非逐指令视图。原始Chrome trace保留本机，路径和SHA见scheduling-trace-evidence.json。</p>
<div id="profiles"></div><p class="note">每个kernel独立归零，原始vendor ts/dur的us字段只表示仿真时间。没有跨kernel拼接，也不能从选中core估算全部核或真机TPOT。鼠标悬停可看原始区间。</p></section>
<section><h2>② 历史 msprof raw footer：旧serial6 / Fusion3</h2><p>B1 / Q4 / H8192 / R4096 / G32 / zero-miss / Service0；LI均1C/2V。旧collector退出-15，Model/oracle和完整raw覆盖通过；不是本次重跑或128K性能。</p><img src="historical-fusion3-native-envelope.png"><p>这个panel只用native Model start/tick，与上方vendor us分域展示。</p></section>
<section><h2>③ 静态启动与依赖示意（不是trace）</h2><img src="scheduling-sequence.svg"><p><a href="end_to_end_schedule.md">完整源码审计</a> · <a href="results/scheduling-trace-evidence.json">证据索引</a></p></section>
<script>const DATA=__DATA__;
const colors={SCALAR:'#63758d',SCALARLDST:'#9aadc3',VECTOR:'#168f85',MTE1:'#e2ba6b',MTE2:'#e9a84a',MTE3:'#d76b63',RVECEX:'#596cbd',RVECLD:'#8f80c2',RVECST:'#ae76a7',FLOWCTRL:'#bd9b6d'};
for(const p of DATA){const box=document.createElement('div');box.innerHTML=`<h3>${p.label} — ${p.x_events.toLocaleString()} 原始X事件</h3><label>缩放 <select><option>1</option><option>2</option><option>4</option><option>8</option><option>16</option></select></label><label>平移 <input type="range" min="0" max="1000" value="0"></label><label><input type="checkbox" checked> 同时显示V1</label><canvas></canvas><small></small>`;document.querySelector('#profiles').append(box);const cv=box.querySelector('canvas'),zoom=box.querySelector('select'),pan=box.querySelector('input[type=range]'),both=box.querySelector('input[type=checkbox]'),status=box.querySelector('small');let regions=[];
function draw(){const lanes=p.lanes.filter(r=>(both.checked||r.core.endsWith('veccore0'))&&r.intervals.length);const h=lanes.length*20+34,w=cv.clientWidth,dpr=devicePixelRatio||1;cv.width=w*dpr;cv.height=h*dpr;cv.style.height=h+'px';const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);const left=210,span=(p.end-p.origin)/Number(zoom.value),offset=p.origin+(p.end-p.origin-span)*Number(pan.value)/1000,scale=(w-left-12)/span;regions=[];ctx.font='11px system-ui';for(let i=0;i<lanes.length;i++){const r=lanes[i],y=12+i*20;ctx.fillStyle='#43566c';ctx.fillText(r.core+'/'+r.pipe,3,y+9);ctx.fillStyle=colors[r.pipe]||'#9aa9ba';for(const [a,b] of r.intervals){if(b<offset||a>offset+span)continue;const x=left+Math.max(0,a-offset)*scale,dx=Math.max(.8,(Math.min(b,offset+span)-Math.max(a,offset))*scale);ctx.fillRect(x,y,dx,12);regions.push({x,y,dx,a,b,label:r.core+'/'+r.pipe});}}ctx.fillStyle='#43566c';for(let n=0;n<6;n++){const x=left+n*(w-left-12)/5;ctx.fillText(((offset-p.origin)+n*span/5).toFixed(3),x, h-5);}status.textContent=`当前窗口：Δ ${(offset-p.origin).toFixed(3)} … ${(offset+span-p.origin).toFixed(3)} vendor模拟us；原始origin=${p.origin}`;}
for(const c of [zoom,pan,both])c.oninput=draw;window.addEventListener('resize',draw);cv.onmousemove=e=>{const r=cv.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,v=regions.find(v=>x>=v.x&&x<=v.x+v.dx&&y>=v.y&&y<=v.y+12);cv.title=v?`${v.label}: 原始[${v.a}, ${v.b}], union占用=${v.b-v.a} vendor us`:''};draw();}
</script>"""


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--fused-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    profiles, historical, evidence = collect(
        args.current.resolve(), repo, output, args.baseline_archive, args.fused_archive
    )
    figures(output, profiles, historical)
    (output / "scheduling-traces.html").write_text(
        HTML.replace(
            "__DATA__",
            json.dumps(profiles, separators=(",", ":"))
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026"),
        )
    )
    print(
        json.dumps(
            dict(
                status="passed",
                new_Model_runs=2,
                new_official_trace_files=3,
                canonical_new_core_logs=6,
                historical_core_logs=17,
                physical_npu=False,
            )
        )
    )


if __name__ == "__main__":
    main()
