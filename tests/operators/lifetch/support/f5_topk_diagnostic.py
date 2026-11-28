"""F5 first-eager post-completion snapshot; no NPU operations in this module."""
import hashlib,json
from pathlib import Path
import numpy as np

def query_stats(ids, native, scores):
    ids=np.asarray(ids).reshape(-1)
    native=np.asarray(native).reshape(-1)
    k=len(ids)
    cutoff=float(np.partition(scores,-k)[-k])
    tol=max(abs(cutoff)*2**-7,float(np.finfo(np.float32).tiny))
    valid=ids[(ids>=0)&(ids<len(scores))]
    vals=scores[valid]
    below=vals < cutoff-tol
    mandatory=np.flatnonzero(scores>cutoff+tol)
    return dict(cutoff=cutoff,tolerance=tol,minimum_selected=float(vals.min()) if len(vals) else None,
        invalid=int(k-len(valid)),duplicates=int(len(valid)-len(np.unique(valid))),
        below_cutoff=int(below.sum()),mandatory_missing=int((~np.isin(mandatory,valid)).sum()),
        native_overlap=int(np.intersect1d(ids,native).size),
        below_examples=[dict(token=int(t),score=float(scores[t])) for t in valid[below][:8]])

def sanitize_meta(meta, base, nbytes):
    # Remove raw pointer words; only bounded offsets into score buffer retained.
    m=meta.copy()
    ptr=m[:,:,0].copy()
    offsets=np.full(ptr.shape,-1,np.int64)
    for i in np.ndindex(ptr.shape):
        x=int(ptr[i])
        if base<=x<base+nbytes and (x-base)%2==0: offsets[i]=(x-base)//2
    m[:,:,0]=0
    return m,offsets

def save_snapshot(folder,lengths,oracle_bf16,native,ids,selected,logical_output,
                  raw,base,meta,live_meta,events,rows_no_ptr,progress,completion,reads):
    folder=Path(folder)
    scores=(oracle_bf16.astype(np.uint32)<<16).view(np.float32)
    assert scores.shape==(4,131072) and ids.shape==(16,4,2048)
    clean,offsets=sanitize_meta(meta,base,raw.nbytes)
    arrays=dict(actual_k=lengths,native=native,fused=ids,physical_selected=selected,
                service_logical_output=logical_output,oracle_bf16=oracle_bf16,raw_score_u16=raw,
                incremental_words=clean,score_offsets_u16=offsets,events=events,
                row_words_8_onward=rows_no_ptr,progress=progress)
    if live_meta is not None:
        clean_live,live_offsets=sanitize_meta(live_meta,base,raw.nbytes)
        arrays.update(live_incremental_words=clean_live,live_score_offsets_u16=live_offsets)
    npz=folder/"f5_topk_diagnostic.npz"
    np.savez_compressed(npz,**arrays)
    queries=[]
    for b,k in enumerate(lengths):
        for q in range(4):
            row=query_stats(ids[b,q],native[b,q],scores[q,:int(k)])
            row.update(request=b,query=q,actual_k=int(k),
                       logical_output_equal=bool(np.array_equal(ids[b,q],logical_output[b,q])),
                       score_offset_u16=int(offsets[b,q]),cursor_word=int(clean[b,q,1]))
            if live_meta is not None:
                row.update(live_score_offset_u16=int(live_offsets[b,q]),live_cursor_word=int(clean_live[b,q,1]))
            queries.append(row)
    w=events.view("<u4").reshape(8,8,32)
    summary=dict(batch=16,actual_k=lengths.tolist(),completion=completion,reads=reads,
        events=dict(complete=int(np.count_nonzero(w[:,:,6]==1)),skipped=int(np.count_nonzero(w[:,:,7]==1)),
            skipped_by_epoch=np.count_nonzero(w[:,:,7]==1,axis=1).tolist()),
        failing_queries=sum(bool(x["invalid"] or x["duplicates"] or x["below_cutoff"] or x["mandatory_missing"]) for x in queries),
        queries=queries,npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
        scope="post full-forward and CQ check, before TopK oracle; metadata may have been retired; no per-stage causality")
    (folder/"f5_topk_diagnostic.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
