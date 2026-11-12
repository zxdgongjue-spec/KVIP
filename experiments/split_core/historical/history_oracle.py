"""Independent exact local-candidate checks in original token coordinates."""
import hashlib
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_gather_geometry():
    layouts=trunks=0
    for h in (8192,131072):
        pages=h//128
        for p in range(1,33):
            if pages//p*128<=2048:continue
            stride=((pages+p-1)//p*128+255)//256*256
            for t in range(1,33):
                if pages//t*128<=2048:continue
                layouts+=1
                for selection in range(t):
                    begin=pages*selection//t*128;end=pages*(selection+1)//t*128
                    for trunk_begin in range(begin,end,16384):
                        trunk_end=min(end,trunk_begin+16384);cursor=trunk_begin
                        first=((trunk_begin//128+1)*p-1)//pages
                        for s in range(first,p):
                            pb=pages*s//p*128;pe=pages*(s+1)//p*128
                            if pb>=trunk_end:break
                            lo=max(trunk_begin,pb);hi=min(trunk_end,pe)
                            if lo<hi:
                                assert lo==cursor and 0<=lo-pb<hi-pb<=pe-pb<=stride
                                assert 0<=lo-trunk_begin<hi-trunk_begin<=trunk_end-trunk_begin
                                cursor=hi
                        assert cursor==trunk_end
                        trunks+=1
    assert (layouts,trunks)==(1033,18226)
    return dict(layouts=layouts,trunks=trunks,scope='CPU address arithmetic only; no Model acceptance')


def verify_local_candidates(folder,geometry,reference_bytes=None):
    folder=Path(folder);q=geometry['real_batch']*4;h=geometry['history'];t=geometry['topk_shards'];k=2048
    data=reference_bytes if reference_bytes is not None else (folder/'score-keys.bin').read_bytes()
    assert len(data)==q*h*2
    reference=np.frombuffer(data,np.uint16).reshape(q,h)
    ids=np.fromfile(folder/'local-ids.bin',np.uint32)
    scores=np.fromfile(folder/'local-scores.bin',np.uint16)
    assert ids.size==scores.size==q*t*k+128
    assert (ids[-128:]==0x5a5a5a5a).all() and (scores[-128:]==0x5a5a).all()
    ids=ids[:-128].reshape(q,t,k);scores=scores[:-128].reshape(q,t,k)
    groups=0
    for query in range(q):
        for shard in range(t):
            begin=h//128*shard//t*128;end=h//128*(shard+1)//t*128
            assert end-begin>=k
            chosen=ids[query,shard];values=scores[query,shard]
            assert ((chosen>=begin)&(chosen<end)).all() and np.unique(chosen).size==k
            assert np.array_equal(values,reference[query,chosen])
            cutoff=np.partition(reference[query,begin:end],-k)[-k]
            assert (values>=cutoff).all()
            strictly_better=np.flatnonzero(reference[query,begin:end]>cutoff)+begin
            assert np.isin(strictly_better,chosen).all()
            groups+=1
    return dict(status='passed',queries=q,selection_shards=t,groups=groups,
                source_score_sha256=hashlib.sha256(data).hexdigest(),
                ids_sha256=sha(folder/'local-ids.bin'),scores_sha256=sha(folder/'local-scores.bin'),
                ranges=True,unique=True,scores_match_original_tokens=True,kth_threshold=True,
                all_strict_winners=True,guards=True,local_ties='any legal threshold-equivalent set')
