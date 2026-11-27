# SPDX-License-Identifier: Apache-2.0
"""Compare one/two private prefixes at fixed work and service cost. Standard Python only."""
import argparse
import hashlib
import heapq
import itertools
import json
import struct
import tarfile
from pathlib import Path


def load_json(archive, name):
    return json.load(archive.extractfile(name))


def chunks(own, total):
    if not own:
        return ()
    ends = [(own*i+15)//16 for i in (1,6,11)] if total >= 256 and own > 1 else []
    bounds = sorted(set([0, own, *ends]))
    return tuple(b-a for a,b in zip(bounds, bounds[1:]))


def exact_query_assignments(sample):
    """Recover Q ownership from its actual atomic slot allocation, which can be out of Q order."""
    tail, traffic = sample["tail"], sample["traffic"]
    cells = []
    digest = hashlib.sha256()
    for r, (assigned, stamps) in enumerate(zip(tail["final_index_assignments_by_row"],
                                                tail["separate_terminal_query_stamps"])):
        prior = sum(traffic["row_window_prefetch"][r])
        tokens, slots = assigned["tokens"], assigned["hot_slots"]
        assert len(tokens) == len(slots) == len(set(tokens)) == len(set(slots))
        visited = []
        cell = []
        for q, t in enumerate(stamps):
            own, offset = t[20], t[22] - prior
            assert own == traffic["row_query_final"][r][q]
            if own:
                assert 0 <= offset and offset+own <= len(tokens)
            offsets = list(range(offset, offset+own))
            visited.extend(offsets)
            pairs = tuple((tokens[i], slots[i]) for i in offsets)
            cell.append(pairs)
            digest.update(struct.pack("<III", r,q,len(pairs)))
            for token, slot in pairs:
                digest.update(struct.pack("<II",token,slot))
        assert sorted(visited) == list(range(len(tokens)))
        cells.append(cell)
    return cells, digest.hexdigest()


def shapes(sample, formal):
    tail = sample["tail"]
    t = tail["separate_terminal_query_stamps"]
    times = tail["decoded"]["qs"]
    hz = tail["clock_mhz"]
    batch = formal["identity"]["batch"]
    same = batch == len(t)
    rows = []
    for r, donor in enumerate(formal["identity"]["donor_rows"]):
        src = r if same else donor
        raw = t[src]
        start = min(j[0] for j in raw)
        first = [(j[4]-start)/hz for j in raw]
        index = [(j[16]-start)/hz for j in raw]
        tail_cost = (max(j[10] for j in raw) - max(j[17] for j in raw))/hz
        x = times[src*4:src*4+4]
        rows.append(dict(first=first,index=index,retire=tail_cost,
                         start=(start-tail["decoded"]["origin"])/hz,
                         exit=(max(j[10] for j in raw)-tail["decoded"]["origin"])/hz,
                         rtt=[[(k["drain_end"]-k["post_end"])/hz for k in q["batches"]] for q in x]))
    return rows


def run(rows, windows, counts, order, bandwidth, ack, credit, *, fixed_rtt=False):
    # One FCFS link is shared by four prefetch windows and all final reads. CQ delay is
    # additional, nonblocking propagation. Each Q's CQ remains ordered even if RTTs differ.
    events = []
    unique = itertools.count()
    def later(t, kind, *args):
        heapq.heappush(events, (t,next(unique),kind,args))

    work = [[chunks(n,sum(row)) for n in row] for row in counts]
    issued, retired, due, qdone = {}, {}, {}, {}
    pref = [-1e9]*len(rows)
    begun = {}
    ends = []
    posts = []
    link = -1e9
    backlog = 0
    queue = iter(order)
    def begin(t):
        r = next(queue, None)
        if r is None:
            return
        begun[r] = t
        for q in range(4):
            key = (r,q)
            issued[key] = retired[key] = 0
            due[key] = -1e9
            if work[r][q]:
                later(t+rows[r]["first"][q], "send", r,q)
                if credit == 2 and len(work[r][q]) > 1:
                    later(t+rows[r]["index"][q], "fill", r,q)
            else:
                later(t+rows[r]["index"][q], "done", r,q)
    for t,r,n in windows:
        later(t, "prefetch", r,n)
    for _ in range(16):
        begin(0.)
    while events:
        now,_,kind,args = heapq.heappop(events)
        if kind == "prefetch":
            r,n = args
            link = max(link,now) + n*0.656/bandwidth
            pref[r] = max(pref[r],link+ack)
            continue
        if kind == "finish":
            ends.append(now)
            begin(now)
            continue
        r,q = args
        key = (r,q)
        plan = work[r][q]
        index = begun[r]+rows[r]["index"][q]
        if kind == "done":
            assert key not in qdone
            qdone[key] = now
            if all((r,k) in qdone for k in range(4)):
                later(max(pref[r],*(qdone[r,k] for k in range(4)))+rows[r]["retire"], "finish", r)
        elif kind == "fill":
            # The G7 capture places the next Post about 0.09 us after the CQ,
            # with a 0.009 us publication. Charge 0.10 us for each later doorbell.
            if issued[key] < len(plan) and issued[key]-retired[key] < credit:
                later(now+0.10, "send", r,q)
        elif kind == "send":
            # Several runnable events can see the same credit. Recheck at the actual Post.
            i = issued[key]
            if i == len(plan) or i-retired[key] >= credit:
                continue
            assert i == 0 or now >= index
            issued[key] += 1
            posts.append((r,q,i,plan[i]))
            backlog = max(backlog,sum(issued.values())-sum(retired.values()))
            link = max(link,now)+plan[i]*0.656/bandwidth
            observed = max(link+ack, due[key])
            if fixed_rtt:
                observed = max(observed,now+rows[r]["rtt"][q][i])
            due[key] = observed
            later(max(index,observed), "receipt", r,q)
        else:
            assert kind == "receipt" and retired[key] < issued[key]
            retired[key] += 1
            if retired[key] == len(plan):
                later(now, "done", r,q)
            else:
                later(now, "fill", r,q)
    expected = sorted((r,q,i,n) for r,row in enumerate(work) for q,plan in enumerate(row)
                      for i,n in enumerate(plan))
    assert sorted(posts) == expected and len(ends) == len(rows) and issued == retired
    return max(ends), backlog


def analyze(diagnostics, formal_archive):
    out = {}
    with tarfile.open(diagnostics) as diag, tarfile.open(formal_archive) as archive:
        captures = {b:load_json(diag,f"g7/bs{b}/single_diagnostic.json") for b in (28,32)}
        for name in archive.getnames():
            if not name.endswith("packed_result.json"):
                continue
            f = load_json(archive,name)
            b,g = f["identity"]["batch"],f["identity"]["group"]
            traffic = f["traffic"][0]
            assert f["calls"] == f["full_KV_Index_checked_calls"] == 10 and f["correctness"] == "PASS"
            assert all(v["row_query_final"] == traffic["row_query_final"] and
                       v["row_window_prefetch"] == traffic["row_window_prefetch"] for v in f["traffic"])
            counts = traffic["row_query_final"]
            row_pref = traffic["row_window_prefetch"]
            change = sum(sum(row)>=256 and max(row)>1 for row in counts)
            case = dict(graph_us=f["mean_us"]["graph"],tail_us=f["mean_us"]["tail"],
                        final_tokens=traffic["final"],eligible_rows=change,
                        prefetch_tokens=traffic["prefetch"])
            out[f"G{g}_BS{b}"] = case
            if not change:
                case["changed_Q"]=0
                continue
            assert g == 7
            sample = captures.get(b,captures[32])
            case["changed_Q"]=sum(len(chunks(n,sum(row)))>1 for row in counts for n in row)
            rows = shapes(sample, f)
            if b in captures:
                assert sample["archive_matches_window_and_final_counts"] == 10
                assert counts == sample["traffic"]["row_query_final"] and row_pref == sample["traffic"]["row_window_prefetch"]
                assignments, case["ordered_token_slot_sha256"] = exact_query_assignments(sample)
                for r,row in enumerate(assignments):
                    for q,items in enumerate(row):
                        pieces = chunks(counts[r][q],sum(counts[r]))
                        ends = list(itertools.accumulate(pieces))
                        captured = sample["tail"]["decoded"]["qs"][r*4+q]["batches"]
                        assert ends == [c["prefix_end"] for c in captured]
                        # Each published prefix always denotes this fixed sequence of source/slot pairs.
                        assert len(items)==sum(pieces)
                case["compute_source"]="same-size NPU4 G7 terminal diagnostic"
                case["diagnostic_tail_us"]=sample["tail"]["decoded"]["terminal_span_us"]
                order = sorted(range(b),key=lambda r:rows[r]["start"])
            else:
                case["compute_source"]="NPU4 G7 BS32 donor row local Post/Index/retire durations"
                # Equal-prefetch rows use the original donor's BS32 entry order as tie-break proxy.
                order = sorted(range(b),key=lambda r:(-sum(row_pref[r]),rows[r]["start"],r))
            li = f["separate_LI_diagnostic"]
            origin = max(q[3] for row in li["per_query_cycle_stamps"] for q in row)
            windows=[]
            for e,epoch in enumerate(f["separate_window_event_u64"]):
                for r,ev in enumerate(epoch):
                    assert ev[10] == row_pref[r][e]
                    if ev[10]:
                        windows.append(((ev[12]-origin)/li["clock_mhz"],r,ev[10]))
            sensitivity=[]
            for speed in (10,25,50,100):
                for delay in (1,4,20,70):
                    for observed in ([False,True] if b in captures else [False]):
                        old = run(rows,windows,counts,order,speed,delay,1,fixed_rtt=observed)
                        new = run(rows,windows,counts,order,speed,delay,2,fixed_rtt=observed)
                        gain=old[0]-new[0]
                        sensitivity.append(dict(GBps=speed,ack_us=delay,archived_rtt_floor=observed,
                            old_us=old[0],new_us=new[0],saved_us=gain,pct_graph=gain*100/case["graph_us"],
                            peak_pending=[old[1],new[1]]))
            case["models"]=sensitivity
    assert len(out)==20
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("diagnostics",type=Path)
    p.add_argument("formal",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result=analyze(args.diagnostics,args.formal)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for k,c in sorted(result.items()):
        if "models" in c:
            plain=[s for s in c["models"] if not s["archived_rtt_floor"]]
            print(k,"eligible",c["eligible_rows"],"FIFO graph % range",round(min(x["pct_graph"] for x in plain),2),round(max(x["pct_graph"] for x in plain),2))
