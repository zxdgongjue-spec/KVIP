# SPDX-License-Identifier: Apache-2.0
"""Summarize the five completed natural window cases and package small matching evidence."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean, median, correlation, StatisticsError
from tempfile import mkdtemp

from b24_window_report import cached, values, CONTROL
from four_rank_followup import final_lists
from large_batch_placement import field, placement

READER = "e602ff4a688a733664aaabe84aa30732b9e68062"
CELLS = ((2, 28, 2), (2, 32, 2), (10, 32, 2), (7, 32, 2), (7, 32, 4))
CLOCK = "g2-window-clocks"


def recorded(ref, g, b, x):
    mode = f"window-x{x}-early"
    arm = cached(READER, ref, g, b, mode)
    if not arm:
        raise ValueError(f"Missing the checked G{g} BS{b} {mode} archive")
    result, orig, _ = values(arm, ref, g, b, mode)
    return arm, result, orig


def intervals(r):
    series = {}
    for k in ("LI1", "LI23", "LI", "tail"):
        v = [s[k] for s in r["samples"]]
        mid = median(v)
        series[k] = dict(values=v, median=mid, mad=median(abs(t-mid) for t in v), minimum=min(v), maximum=max(v))
    return series


def saved_frequency_fields(arm):
    # Preserve the original field name: a nominal clock in device metadata does
    # not establish the run-time frequency of an individual invocation.
    result={}
    for name in ("info.json", "info.json.*", "profiler_info*.json", "host_info.json"):
        for meta in arm.rglob(name):
            if meta.stat().st_size > 2_000_000:
                continue
            try:
                data=json.loads(meta.read_bytes())
            except (ValueError, OSError):
                continue
            def walk(obj, prefix):
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        if ("freq" in k.lower() or "mhz" in k.lower()) and isinstance(v,(float,int,str)):
                            result[f"{meta.relative_to(arm)}:{prefix}{k}"]=v
                        if isinstance(v,(list,dict)):
                            walk(v,prefix+k+".")
                elif isinstance(obj,list):
                    for n,v in enumerate(obj):
                        walk(v,prefix+str(n)+".")
            walk(data,"")
    return result


def snapshot(r):
    d = r["window_density"]
    t = r["traffic"]
    f = final_lists(t, r["identity"]["batch"])
    return dict(cold=r["cold_build_fingerprint"], diagnostic_cold=r["diagnostic_cold_build_fingerprint"],
                actual_ranks=r.get("actual_ranks") or r["diagnostic_ranks"]["ranks"],
                graph=r["graph_mean_us"], stage_means=r["mean_us"], samples=intervals(r),
                traffic=t, final_by_row=f, epochs=d["epochs"],
                first_useful=d["first_useful_post_us"], last_post=d["last_post_us"],
                issue_lead=d["mean_useful_lead_to_li_end_us"], li3_span=d["li3_span_us"],
                max_inter_post=d["longest_inter_post_us"], overlap=d["adjacent_epochs_distinct_row_service_overlap_us"],
                fifo_model=d["ideal_link_by_payload_GBps"])


def lines(g,b,x,result):
    tag=f"G{g} BS{b} 1+{x}"
    view=snapshot(result)
    st = view["samples"]
    print(tag, "P50/MAD_us", *(f"{k}={s['median']:.2f}/{s['mad']:.2f}" for k,s in st.items()))
    print(tag, f"LI3_diag={view['li3_span']:.2f} last_Post={view['last_post']:.2f} "
          f"useful_GET_mean_lead={view['issue_lead']:.2f} us",
          "rank_plus_service_mean_us=" + "/".join(f"{r['mean_rank_and_service_us']:.2f}" for r in view["epochs"]))
    v=view["final_by_row"]
    print(tag, f"final_GET_mean={mean(s['final'] for s in view['traffic']):.1f} "
          f"largest_per_row_Q_mean={max(max(q) for q in v['mean_final_by_row_q']):.1f} "
          f"predicted_final_batches={mean(v['predicted_final_cq_batches_per_call']):.1f}")
    try:
        corr=correlation(st["LI1"]["values"],st["LI23"]["values"])
    except StatisticsError:
        corr=None
    view["within_graph_LI1_LI23_correlation"]=corr
    print(tag,f"within_graph_LI1_LI23_corr={corr:.3f}" if corr is not None else "within_graph_corr=NA")
    return view


def clock_hit(ref,reader,b):
    from four_rank_accept import matched
    for root in sorted(ref.parent.glob(f"fixed-q4-audit-{CLOCK}.*"), key=lambda p:p.stat().st_mtime_ns, reverse=True):
        arm=root / f"g2/bs{b}/block1/{CLOCK}"
        if (field(root,"audit_reader")==reader and field(root,"audit_reference")==str(ref) and matched(root,ref)
                and (arm/"placement_verified").is_file() and (arm/"single_diagnostic.json").is_file()
                and placement(arm/"thp.json")==placement(ref/f"g2/bs{b}/block1/fused/thp.json")):
            return arm


def show_clocks(arm,ref,b):
    data=json.loads((arm/"single_diagnostic.json").read_bytes())
    oldpath,new,original=recorded(ref,2,b,2)
    diag=data["li_windows"]
    if (data["input_archive"]!=str(oldpath/"packed_result.json") or data["identity"]!=new["identity"]
            or data["diagnostic_calls"]!=1 or data["measured_calls"]!=0 or data["correctness"]!="PASS"
            or diag["diagnostic_cold_build_fingerprint"]!=new["diagnostic_cold_build_fingerprint"]):
        raise ValueError("Incorrect short-window diagnostic source")
    def raw(r):
        keys=("separate_LI_diagnostic","separate_score_span_cycles","separate_sibling_predecessor_cycles","separate_window_event_u64")
        return {k:r[k] for k in keys}

    def last_row(r):
        rows=r["separate_LI_diagnostic"]["per_query_cycle_stamps"]
        return max(range(len(rows)),key=lambda n:max(q[3] for q in rows[n]))
    cells=(original, diag)
    critical=sorted({last_row(r) for r in cells})
    for r in cells:
        events=r["separate_window_event_u64"]
        mhz=r["separate_LI_diagnostic"]["clock_mhz"]
        clocks=r["separate_LI_diagnostic"]["per_query_cycle_stamps"]
        cols=lambda vs: "/".join(f"{v:.2f}" for v in vs)
        tag=f"G2 BS{b} 1+{len(events)-1} last_row={last_row(r)}"
        counters=[[event[15] for event in epoch] for epoch in events[1:]]
        print(tag, "batch_incremental_merge_calls=" + "/".join(str(sum(v&65535 for v in ep)) for ep in counters),
              "batch_candidates=" + "/".join(str(sum(v>>40 for v in ep)) for ep in counters))
        for rid in critical:
            end_prev=events[0][rid][1]
            phases=[]
            for epoch in events[1:]:
                v=epoch[rid]
                cycle=(v[4]-end_prev,v[13]-v[4],v[14]-v[13],v[1]-v[14])
                if min(cycle[1:])<0:
                    raise ValueError("Nonmonotonic stages on the window owner")
                phases.append(cycle)
                end_prev=v[1]
            scores=r["separate_score_span_cycles"]
            def spans(lane):
                return cols((z-a)/mhz for a,z in scores[lane][rid])
            names=("arrival_minus_previous_end","predecessor_check","rank_and_join","service_after_join")
            stages=" ".join(f"{name}="+cols(z[i]/mhz for z in phases) for i,name in enumerate(names))
            print(tag,f"row={rid} owner_score_us={spans('post_owner')} sibling_score_us={spans('sibling')} "
                  f"first_window_us={(events[0][rid][1]-clocks[rid][0][2])/mhz:.2f} {stages} "
                  f"final_exit_us={(clocks[rid][0][3]-end_prev)/mhz:.2f}")
    print(f"G2 BS{b} matched_rows_and_Q_counts={data['archive_matches_window_and_final_counts']}/10")
    return dict(original=raw(original), two_cold=raw(diag),
                archive_matches=data["archive_matches_window_and_final_counts"], input_archive=data["input_archive"])


def make(ref, reader, directory, include_clocks):
    package = {"reference":str(ref),"cells":{},"G2_diagnostic":{}}
    for g,b,x in CELLS:
        arm,r,old=recorded(ref,g,b,x)
        key=f"G{g}-BS{b}"
        out=package["cells"].setdefault(key, {})
        if "1+3" not in out:
            out["1+3"]=lines(g,b,3,old)
            out["1+3"]["profiler_frequency_fields"]=saved_frequency_fields(ref/f"g{g}/bs{b}/block1/fused")
        out[f"1+{x}"]=lines(g,b,x,r)
        out[f"1+{x}"]["archive"]=str(arm)
        out[f"1+{x}"]["profiler_frequency_fields"]=saved_frequency_fields(arm)
    if include_clocks:
        for b in (28,32):
            arm=clock_hit(ref,reader,b)
            if not arm:
                raise ValueError(f"Missing completed G2 BS{b} two-window diagnostic")
            package["G2_diagnostic"][str(b)]=show_clocks(arm,ref,b)
    dest=Path(mkdtemp(prefix="window-evidence-",dir=directory))/"window-evidence.json.gz"
    with gzip.open(dest,"wt",encoding="utf-8") as fp:
        json.dump(package,fp,separators=(",",":"),allow_nan=False)
    print(f"RETURN_FILE={dest} bytes={dest.stat().st_size}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("archive","package","show","cached"))
    p.add_argument("directory",type=Path)
    p.add_argument("ref",type=Path)
    p.add_argument("arg")
    p.add_argument("batch",type=int,nargs="?")
    p.add_argument("mode",nargs="?")
    a=p.parse_args()
    if a.action in ("archive","package"):
        make(a.ref,a.arg,a.directory,a.action=="package")
    elif a.action=="show":
        show_clocks(a.directory,a.ref,a.batch)
    else:
        reader=field(a.directory,"audit_reader")
        if reader:
            arm=clock_hit(a.ref,reader,a.batch)
            if arm:print(arm)
