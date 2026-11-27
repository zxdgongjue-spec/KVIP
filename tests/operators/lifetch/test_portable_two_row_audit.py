# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from unittest import TestCase

from portable_two_row_audit import NATIVE, audit, evaluate, local


def case():
    routes=[[i,i+16] for i in range(16)]
    windows=[[0]*4 for _ in range(32)]
    rows=[[0]*4 for _ in range(32)]
    rows[0]=[250, 2, 2, 2]
    rows[1]=[254, 2, 2, 1]
    rows[2]=[249, 2, 2, 2]
    tr=dict(row_window_prefetch=windows,row_query_final=rows,prefetch=0,final=sum(map(sum,rows)))
    sample=dict(tail=22, LI=50)
    base=dict(identity=dict(native=NATIVE,physical_npu=4,batch=32,protocol="ten_independent_states_one_graph_one_event_v1"),
              arm="fused",correctness="PASS",get_workers=32,full_KV_Index_checked_calls=10,calls=10,traffic=[tr]*10,
              samples=[sample]*10,mean_us=dict(tail=22,LI=50),graph_mean_us=80,graph_event_us=800,
              checked_queries=1280,checked_rows=320,cold_build_fingerprint="cold",measured_cold_symbol="v9_li_large_four_ranks")
    new=deepcopy(base)
    symbol="v9_test_two_row_terminal"
    plain=dict(accepted=True,ordered_pairs=[],team_rows=routes,normal_handoffs=16,early_handoffs=0,parked_rows=0)
    new.update(arm="two-row-natural",measured_tail_symbol=symbol,
               calls_with_row_pair=1,parked_rows_per_call=[1]+[0]*9,
               terminal_build=dict(cold_build="cold",measured_tail_symbol=symbol,original_tail_text=["A"],tail_text=["B"]),
               terminal_decisions=[dict(plain,ordered_pairs=[(0,16)],normal_handoffs=15,early_handoffs=1,parked_rows=1)]+[plain]*9)
    return new,base


class PortableAudit(TestCase):
    def test_assignment_and_matched_counts(self):
        new,base=case()
        result=evaluate(new,base)
        self.assertEqual(result["eligible"],[1]*10)
        self.assertEqual(result["eligible_in_first_wave"],[1]*10)
        self.assertEqual(result["matched_counts_calls"],10)
        self.assertEqual(result["parked"],[1]+[0]*9)
        self.assertEqual(result["final_Q_zero_single_multi_per_call"],[116,5,7])
        self.assertEqual(result["tail_by_activation"]["paired"],[1,22,22])

    def test_invalid_pick(self):
        new,base=case()
        new["terminal_decisions"][0]["ordered_pairs"]=[(1,17)]
        with self.assertRaisesRegex(ValueError,"early assignment"):
            evaluate(new,base)

    def test_per_row_traffic(self):
        new,base=case()
        new["traffic"][0]["row_query_final"][1][1] += 1
        with self.assertRaisesRegex(ValueError,"traffic counts"):
            evaluate(new,base)

    def test_empty_machine_and_cross_host(self):
        with tempfile.TemporaryDirectory() as out:
            root=Path(out)
            (root/"peer-baseline").mkdir()
            with redirect_stdout(StringIO()) as console:
                audit(root,("l","abc","boot"))
            self.assertEqual(console.getvalue().count("NO_LOCAL_COMPLETED"),5)
            for key,v in (("host","l"),("physical_npu","4"),("machine_id","abc"),("boot_id","boot")):
                (root/(key+".txt")).write_text(v)
            self.assertTrue(local(root,("l","abc","newboot")))
            self.assertFalse(local(root,("l","other","boot")))

    def test_first_and_following_batches_are_counted_once(self):
        from peer_tail_clocks import pressure
        def batch(end, start):
            return dict(prefix_end=end,post_begin=start,post_end=start+1,
                        drain_begin=start+2,drain_end=start+5)
        qs = [dict(row=0,query=q,own=0,batches=[]) for q in range(4)]
        qs[0].update(own=7, batches=[batch(2,10),batch(7,20)])
        qs[1].update(own=1, batches=[batch(1,10)])
        r=dict(tail=dict(clock_mhz=2,decoded=dict(qs=qs)),identity=dict(batch=1),traffic=dict(final=8))
        self.assertEqual(pressure(r),dict(peak_unretired=2,batches=[2,1],tokens=[3,5],post_to_retire_p50_us=[2.0,2.0],drain_p50_us=[1.5,1.5]))
        qs[0]["batches"][1]["drain_begin"]=5
        with self.assertRaisesRegex(ValueError,"clock order"):
            pressure(r)

    def test_teams_require_distinct_rows_and_include_touching_private_waits(self):
        from peer_tail_clocks import simultaneous_drains
        waits = [(0,10,20),(0,15,30),(1,16,23),(1,23,26),(2,17,19),(2,22,25)]
        self.assertEqual(simultaneous_drains(waits,3), [(17,19),(22,25)])
        self.assertEqual(simultaneous_drains(waits,4), [])
