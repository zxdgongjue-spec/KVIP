# SPDX-License-Identifier: Apache-2.0
"""CPU checks of real partition header, global theta and trace accounting.

These checks are not CANN instruction simulation or evidence of NPU speedup.
"""
import ctypes
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests/operators/lifetch/support')]
from kvip.operators.lifetch.score_policy import validate_score_t2
from diagnostics import account
from performance import split_graph_invocations
from workloads import configuration
from run_a5 import BASE, baseline, checksum, summarize, save
from correctness import audit_score_row, audit_cold_predicate


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        path=Path(cls.directory.name)
        source=path/'partition.cpp'
        source.write_text('''#include "csrc/operators/common/lifetch/score_partition.h"
extern "C" bool valid(unsigned b,unsigned h,unsigned p,unsigned a) {return kvip_score_t2::Valid(b,h,p,a);}
extern "C" void slice(unsigned task,unsigned p,unsigned h,unsigned* out) {
 auto s=kvip_score_t2::Partition(task,p,h);out[0]=s.owner;out[1]=s.beginTile;out[2]=s.endTile;
}
''')
        compiler=shutil.which('c++') or shutil.which('clang++')
        if compiler is None:
            raise RuntimeError('A host C++ compiler is required for the real partition-header check')
        lib=path/'partition.so'
        subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-I',str(ROOT),str(source),'-o',str(lib)],check=True)
        cls.lib=ctypes.CDLL(str(lib))
        cls.lib.valid.argtypes=[ctypes.c_uint]*4;cls.lib.valid.restype=ctypes.c_bool
        cls.lib.slice.argtypes=[ctypes.c_uint]*3+[ctypes.POINTER(ctypes.c_uint)]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def partitions(self,b,h,p):
        self.assertTrue(self.lib.valid(b,h,p,32))
        result=[]
        for task in range(b*p):
            values=(ctypes.c_uint*3)()
            self.lib.slice(task,p,h,values)
            result.append(tuple(values))
        return result

    def test_exact_cover_and_canonical_rows(self):
        for b,p in [(1,p) for p in (1,2,4,8,16,30,32)]+[(16,1),(16,2)]:
            h=131072
            out=np.zeros((b,4,h),np.uint16)
            visits=np.zeros((b,h//128),np.uint8)
            # Distinct row/Q data catches a physical-AIC/owner addressing swap.
            source=(np.arange(b*4*h,dtype=np.uint32).reshape(b,4,h)*37%65521).astype(np.uint16)
            lengths=[]
            for owner,start,end in self.partitions(b,h,p):
                self.assertLess(start,end)
                self.assertEqual(start%2,0);self.assertEqual(end%2,0)
                out[owner,:,start*128:end*128]=source[owner,:,start*128:end*128]
                visits[owner,start:end]+=1;lengths.append(end-start)
            self.assertTrue(np.all(visits==1))
            np.testing.assert_array_equal(out,source)
            self.assertLessEqual(max(lengths)-min(lengths),2)

    def test_reject_oversubscription_and_empty_slice(self):
        for b,h,p,a in [(16,131072,3,32),(1,256,2,32),(1,131073,16,32),(1,131072,0,32)]:
            self.assertFalse(self.lib.valid(b,h,p,a))
        shape=SimpleNamespace(batch=16,history=131072,resident=16384,rounds=4,
                              indexHeads=32,splitQ=1,liCores=32,aicCount=32)
        validate_score_t2(shape,2)
        with self.assertRaises(ValueError):validate_score_t2(shape,3)
        shape.liCores=16
        with self.assertRaises(ValueError):validate_score_t2(shape,2)

    def test_global_theta_and_ties(self):
        h,r,k=131072,16384,2048
        # A resident shard with all high values must not set its local theta as
        # the shared theta; exact ties also exercise the strict > predicate.
        scores=np.zeros((4,h),np.uint16)
        scores[:,:r//2]=50000
        scores[:,r//2:r]=np.arange(r//2,dtype=np.uint16)
        theta=np.partition(scores[:,:r],r-k,axis=1)[:,r-k]
        scores[:,r:]=49999+np.arange(h-r,dtype=np.uint16)%3
        reference=scores[:,r:]>theta[:,None]
        bitmap=np.zeros_like(reference)
        for owner,start,end in self.partitions(1,h,30):
            a=max(start*128,r);z=end*128
            if z>a:bitmap[:,a-r:z-r]=scores[:,a:z]>theta[:,None]
        np.testing.assert_array_equal(bitmap,reference)
        self.assertFalse(np.any(bitmap[scores[:,r:]==theta[:,None]]))
        local_theta=np.partition(scores[:,r//2:r],r//2-k,axis=1)[:,r//2-k]
        self.assertTrue(np.all(local_theta < theta))

    def test_new_kernels_counted_in_whole_graph(self):
        names=('v9_li_kernel_score_t2_4','v9_li_kernel_1','v9_li_kernel_cold_t2_2',
               'v9_li_kernel_3','v9_streaming_resolve_update')
        events=[dict(name='replay',ph='X',ts=0,dur=100)]
        for call in range(2):
            for i,name in enumerate(names):
                events.append(dict(name=name,ph='X',cat='kernel',ts=call*40+i*6+1,dur=5))
        result=account(events,'replay',100,full=True,fused_final=True)
        self.assertNotIn('error',result)
        self.assertEqual(len(result['intervals_us']['li']),8)
        graph=split_graph_invocations(result,2)
        self.assertEqual(len(graph['samples']),2)
        self.assertEqual(graph['timing']['mean_us'],50)
        self.assertEqual(graph['timing']['stage_mean_us']['li'],20)

    def test_cli_default_and_guard(self):
        self.assertEqual(configuration(['bench']).li_score_parts,0)
        self.assertEqual(configuration(['bench','--li-score-parts','16','--batches','1']).li_score_parts,16)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            configuration(['bench','--li-score-parts','16','--transport','ubmem'])

    def test_baseline_only_overrides_startup_builder(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            source = baseline(output)
            relative = 'csrc/operators/lightning_indexer_with_fetch/op_kernel/indexer.cpp'
            original = subprocess.check_output(['git', 'show', BASE+':'+relative], cwd=ROOT)
            self.assertEqual((source/relative).read_bytes(), original)
            relative = 'python/kvip/runtime/build.py'
            original = subprocess.check_output(['git', 'show', BASE+':'+relative], cwd=ROOT)
            self.assertEqual((output/'baseline-build-original.py').read_bytes(), original)
            self.assertEqual((source/relative).read_bytes(), (ROOT/relative).read_bytes())
            metadata = json.loads((output/'baseline_build_override.json').read_text())
            self.assertEqual(metadata['effective_sha256'], checksum(ROOT/relative))

    def test_score_audit_decodes_signed_numeric_reference(self):
        keys = np.full(4096, 15088, np.uint16)  # encoded BF16(-2288)
        keys[0] = 32768  # +0
        reference = np.full(4096, -2288, np.float32); reference[0] = 0
        valid = np.ones(4096, bool)
        self.assertEqual(audit_score_row(keys, reference, valid, np.arange(2048))['score_mismatches'], 0)
        self.assertEqual(audit_score_row(keys, keys, valid, np.arange(2048))['score_mismatches'], 0)
        bad_selection = np.arange(1,2049)
        self.assertEqual(audit_score_row(keys, reference, valid, bad_selection)['missing_above'], 1)
        reference[2] = -2208
        result = audit_score_row(keys, reference, valid, np.arange(2048))
        self.assertEqual(result['score_mismatches'], 1)
        self.assertEqual(result['examples'][0]['device_score'], -2288)

    def test_cold_predicate_audit_strict_threshold_and_bits(self):
        keys = np.resize(np.asarray([15087,15088,15089,32768], np.uint16), 512)
        expected = keys > 15088
        packed = np.zeros(128, np.uint8)
        for i in np.flatnonzero(expected): packed[i//4] |= 1 << (2*(i%4))
        self.assertEqual(audit_cold_predicate(keys, packed, 15088, 128)['cold_predicate_mismatches'], 0)
        packed[42] = 0
        self.assertEqual(audit_cold_predicate(keys, packed, 15088, 128)['cold_predicate_mismatches'], 2)

    def test_summary_uses_graph_event_and_rejects_operand_change(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            save(output/'manifest.json',dict(device_resources=dict(effective_aic=32,effective_aiv=64)))
            jobs=[]
            for case,parts,us in [('mr23',0,100.),('score_t2_p16',16,80.)]:
                folder=output/case/'miss10'/'case-b1';folder.mkdir(parents=True)
                fixture=folder.parent/'fixture';fixture.mkdir()
                (fixture/'fixture.json').write_text('{}');(fixture/'arg0.bin').write_bytes(b'same inputs')
                save(folder/'report.json',dict(batch=1,miss_percent=10,correctness='PASS',execution='COMPLETED'))
                for path,event in [('serial',120.),('fused',us)]:
                    (folder/path).mkdir()
                    save(folder/path/'path_result.json',dict(all_rounds=dict(
                        timing_protocol='one_profiled_graph',event_mean_us=event,p50_us=event-5,
                        stage_mean_us=dict(li=event/2,resolve_update=event/3),inter_kernel_gap_mean_us=1)))
                jobs.append(dict(name=case,case=case,parts=parts,pf=1024,round=1,
                                 output=str(output/case),log='unit-test only'))
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()): table=summarize(output,jobs,1)
            optimized=next(r for r in table if r['case']=='score_t2_p16')
            self.assertEqual(optimized['over_mr23'],.8)
            self.assertEqual(optimized['effective_aic'],32)
            self.assertAlmostEqual(optimized['over_serial'],2/3)
            (output/'score_t2_p16/miss10/fixture/arg0.bin').write_bytes(b'different')
            with self.assertRaisesRegex(RuntimeError,'Operand mismatch'):
                summarize(output,jobs,1)


if __name__=='__main__':unittest.main()
